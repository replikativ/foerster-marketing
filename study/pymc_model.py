"""Matched public PyMC-Marketing MMM with explicit temporal lift evidence.

Usage: ``python study/pymc_model.py <fixture.json> <output-directory> --profile full``.
The fixed-point audit of the same model lives in ``pymc_audit.py``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytensor
import pytensor.xtensor as ptx
from common import (
    CHANNELS,
    CONTROLS,
    LAG,
    SCALE,
    digest,
    gate_fields,
    passes_gates,
    write_json,
)
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, MichaelisMentenSaturation
from pymc_marketing.mmm.scaling import FixedScaling, Scaling

PROFILES = {
    "pilot": {
        "draws": 100,
        "tune": 100,
        "chains": 2,
        "target_accept": 0.9,
        "max_depth": 8,
    },
    "full": {
        "draws": 4000,
        "tune": 2000,
        "chains": 4,
        "target_accept": 0.95,
        "max_depth": 10,
    },
}
CORES = 2
SEED = 731
# The 15 model parameters with their dimensions, in the order of common.TRUTH.
PARAMETERS = [
    ("adstock_alpha", ["channel"]),
    ("saturation_alpha", ["channel"]),
    ("saturation_lam", ["channel"]),
    ("intercept_contribution", []),
    ("y_sigma", []),
    ("gamma_control", ["control"]),
]
PARAMETER_NAMES = [name for name, _ in PARAMETERS]


def dataframe(data):
    dates = pd.date_range("2023-01-01", periods=len(data["x"]), freq="W-SUN")
    frame = pd.DataFrame(np.asarray(data["x"]), columns=CHANNELS)
    frame[CONTROLS] = np.asarray(data["controls"])
    frame["date"] = dates
    return frame


def build(fixture):
    X = dataframe(fixture["training"])
    y = pd.Series(fixture["training"]["y"], name="target")
    mmm = MMM(
        date_column="date",
        target_column="target",
        channel_columns=CHANNELS,
        control_columns=CONTROLS,
        adstock_first=True,
        adstock=GeometricAdstock(
            l_max=LAG,
            normalize=True,
            priors={"alpha": Prior("Beta", alpha=2, beta=2, dims="channel")},
        ),
        saturation=MichaelisMentenSaturation(
            priors={
                "alpha": Prior("LogNormal", mu=0, sigma=0.6, dims="channel"),
                "lam": Prior("LogNormal", mu=-0.5, sigma=0.6, dims="channel"),
            }
        ),
        scaling=Scaling(
            channel=FixedScaling(dims=(), value=1000.0),
            target=FixedScaling(dims=(), value=SCALE),
        ),
        model_config={
            "intercept": Prior("Normal", mu=1, sigma=0.6),
            "gamma_control": Prior("Normal", mu=0, sigma=0.3, dims="control"),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.2)),
        },
    )
    mmm.build_model(X, y)
    add_temporal_lift_measurements(mmm, fixture["training"].get("experiments", []))
    return mmm, X, y


def add_temporal_lift_measurements(mmm, experiments):
    """Condition the built model on lift experiments with their own spend paths.

    Each experiment records weekly spend for a baseline and a treatment plan,
    starting ``LAG - 1`` weeks before the test so that earlier spend carries
    over, and weights selecting the weeks in which the lift was measured. The
    model-implied lift is the weighted sum of the treatment-minus-baseline
    media response, computed with the model's own adstock and saturation
    variables, and the measured estimate is a Normal observation of it.

    We do not use ``MMM.add_lift_test_measurements``. It compares the
    saturation curve at two steady spend levels, ``saturation(x + delta_x) -
    saturation(x)``, and PyMC-Marketing's lift-test notebook
    (``docs/source/notebooks/mmm/mmm_lift_test.ipynb``) notes that this static
    likelihood does not model a campaign's adstock time path. For this study's
    four-week video test at the true parameters, the temporal lift is EUR 4,122
    while the steady-state approximation gives EUR 3,550.

    The term is added after ``build_model``, so it is not part of the model
    configuration that ``MMM.save`` and ``MMM.load`` restore; a reloaded model
    lacks it. Decisions only need the posterior draws, so that is harmless
    there, but a refit must call ``build`` again.
    """
    model = mmm.model
    # Both scales are fixed numbers. Reading them as numbers keeps the
    # compiled density bit for bit identical to the one that produced the
    # published posterior (a symbolic sigma rounds the gradient differently in
    # the last bit). One experiment per term, for the same reason: stacking
    # them on an `experiment` dimension changes the summation order.
    channel_scale = float(model["channel_scale"].eval())
    target_scale = float(model["target_scale"].eval())
    with model:
        for i, experiment in enumerate(experiments):

            def response(path):
                spend = ptx.as_xtensor(
                    np.asarray(path) / channel_scale,
                    dims=("experiment_date", "channel"),
                )
                carryover = mmm.adstock(spend, core_dim="experiment_date")
                return mmm.saturation(carryover, core_dim="experiment_date").sum(
                    "channel"
                )

            weights = ptx.as_xtensor(
                np.asarray(experiment["weights"]), dims=("experiment_date",)
            )
            lift = pmd.Deterministic(
                f"external_lift_{i}_model_estimated",
                (
                    (
                        response(experiment["treatment"])
                        - response(experiment["baseline"])
                    )
                    * weights
                ).sum("experiment_date"),
            )
            pmd.Normal(
                f"external_lift_{i}",
                mu=lift,
                sigma=experiment["sigma"] / target_scale,
                observed=experiment["estimate"] / target_scale,
            )


def provenance(fixture_path, runner):
    home = str(Path.home())
    return {
        "fixture-sha256": digest(fixture_path),
        "runner-sha256": digest(runner),
        "common-sha256": digest(Path(__file__).with_name("common.py")),
        "model-sha256": digest(__file__),
        "compiler-cache": {
            "directory": str(pytensor.config.compiledir).replace(home, "~", 1),
            "policy": "existing persistent cache; not a fresh-cache compilation benchmark",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["pymc-marketing", "pymc", "pytensor", "numpy", "scipy"]
        },
    }


def parameter_draws(posterior):
    """Return a (chain, draw, 15) array in the parameter order of common.TRUTH."""
    if (
        list(posterior.coords["channel"].values) != CHANNELS
        or list(posterior.coords["control"].values) != CONTROLS
    ):
        raise ValueError("Posterior coordinate order does not match the fixture")
    columns = []
    for name, dims in PARAMETERS:
        value = posterior[name].transpose("chain", "draw", *dims).values
        columns.append(value.reshape(value.shape[0], value.shape[1], -1))
    return np.concatenate(columns, axis=2)


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + usage.ru_utime + usage.ru_stime


def fit(fixture_path, directory, profile):
    fixture = json.loads(fixture_path.read_text())
    options = PROFILES[profile]
    if (directory / "posterior.nc").exists():
        raise FileExistsError(directory / "posterior.nc")
    started = time.perf_counter()
    cpu = cpu_seconds()
    mmm, X, y = build(fixture)
    built = time.perf_counter()
    # PyMC's own NUTS sampler. init="adapt_diag" starts every chain at the
    # model's initial point without jitter; the Foerster engine starts from the
    # same point, so both engines begin matched (pymc_audit.py checks this).
    mmm.fit(
        X,
        y,
        draws=options["draws"],
        tune=options["tune"],
        chains=options["chains"],
        cores=CORES,
        target_accept=options["target_accept"],
        nuts_sampler="pymc",
        nuts={"max_treedepth": options["max_depth"]},
        init="adapt_diag",
        random_seed=SEED,
        progressbar=False,
    )
    # fit() reuses the built model; make sure the lift terms took part.
    observed_names = {rv.name for rv in mmm.model.observed_RVs}
    for i, _ in enumerate(fixture["training"].get("experiments", [])):
        if f"external_lift_{i}" not in observed_names:
            raise RuntimeError("Fit discarded external experiment observations")
    sampled = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    # Keep parameters and sampler statistics; the deterministic time series
    # are reconstructed from the fixture, avoiding a large redundant file.
    portable = mmm.idata.copy()
    portable["posterior"] = mmm.idata["posterior"].to_dataset()[PARAMETER_NAMES]
    portable.to_netcdf(directory / "posterior.nc", engine="h5netcdf")
    summary = az.summary(mmm.idata, var_names=PARAMETER_NAMES, round_to="none")
    stats = mmm.idata.sample_stats
    report = {
        "engine": "pymc-marketing",
        "case": fixture["case"],
        "profile": profile,
        "stored-posterior-variables": PARAMETER_NAMES,
        "observed-variables": sorted(observed_names),
        **gate_fields(summary, divergences=stats["diverging"].sum()),
        "depth-limit-visits": int(stats["reached_max_treedepth"].sum()),
        "timing": {
            "build-seconds": built - started,
            "fit-call-seconds": sampled - built,
            "cpu-including-exited-children": cpu_seconds() - cpu,
        },
        "sampler": {**options, "cores": CORES, "seed": SEED},
        **provenance(fixture_path, __file__),
    }
    write_json(directory / "pymc-report.json", report)
    if profile == "full" and not passes_gates(report):
        raise RuntimeError(
            "Full PyMC fit failed diagnostics; inspect retained artifacts"
        )
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    args = parser.parse_args()
    fit(args.fixture, args.output, args.profile)


if __name__ == "__main__":
    main()
