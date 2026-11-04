"""Compare independently sampled parameter posteriors using Monte Carlo error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from analysis import totals
from common import (
    MAX_MEAN_ERROR_MCSE,
    N_PARAMETERS,
    digest,
    gate_fields,
    passes_gates,
    summary,
    write_json,
)
from foerster_evaluation import decode
from pymc_model import parameter_draws


def diagnostics(chains):
    dataset = xr.Dataset(
        {"theta": (("chain", "draw", "parameter"), chains)},
        coords={"parameter": np.arange(chains.shape[2])},
    )
    data = xr.DataTree.from_dict({"posterior": dataset})
    return az.summary(data, var_names=["theta"], round_to="none")


def resource_cpu(path):
    fields = {}
    for line in path.read_text().splitlines():
        for key in [
            "User time (seconds)",
            "System time (seconds)",
            "Maximum resident set size (kbytes)",
        ]:
            prefix = key + ": "
            if line.strip().startswith(prefix):
                fields[key] = float(line.strip()[len(prefix) :])
    return {
        "fit-process-cpu-seconds": fields["User time (seconds)"]
        + fields["System time (seconds)"],
        "maximum-rss-kbytes": int(fields["Maximum resident set size (kbytes)"]),
    }


def compare(directory):
    fixture_path = directory.parent / "fixture.json"
    fixture = json.loads(fixture_path.read_text())
    native = json.loads((directory / "foerster/posterior.json").read_text())
    posterior = decode(native["posterior"])
    foerster = np.asarray(posterior["chains"])
    with xr.open_datatree(directory / "pymc/posterior.nc", engine="h5netcdf") as data:
        pymc = parameter_draws(data["posterior"].to_dataset())
    f_report = json.loads((directory / "foerster/foerster-report.json").read_text())
    p_report = json.loads((directory / "pymc/pymc-report.json").read_text())
    if not (
        f_report["statistically-qualified"]
        and p_report["profile"] == "full"
        and passes_gates(p_report)
    ):
        raise ValueError("Full per-engine statistical gates must pass first")
    assert (
        f_report["provenance"]["fixture-sha256"]
        == p_report["fixture-sha256"]
        == digest(fixture_path)
    )
    assert foerster.shape == pymc.shape == (4, 4000, N_PARAMETERS)
    expected_names = [
        [group, channel]
        for group in ["alpha", "beta", "kappa"]
        for channel in fixture["spec"]["channels"]
    ]
    expected_names += [["intercept"], ["sigma"]] + [
        ["control", c] for c in fixture["spec"]["controls"]
    ]
    assert posterior["parameter-names"] == expected_names
    fs, ps = diagnostics(foerster), diagnostics(pymc)
    for stats in [fs, ps]:
        assert list(stats.index) == [f"theta[{j}]" for j in range(N_PARAMETERS)]
        if not np.isfinite(
            stats[
                ["mean", "sd", "mcse_mean", "r_hat", "ess_bulk", "ess_tail"]
            ].to_numpy()
        ).all():
            raise ValueError("Shared posterior diagnostics contain nonfinite values")
        # Divergences are gated from each engine's own sampler statistics.
        if not passes_gates(gate_fields(stats)):
            raise ValueError(
                "Independently recomputed posterior diagnostics fail the declared gates"
            )
    rows = []
    for j, name in enumerate(expected_names):
        a, b = fs.iloc[j], ps.iloc[j]
        error = float(abs(a["mean"] - b["mean"]))
        combined = float(np.hypot(a["mcse_mean"], b["mcse_mean"]))
        score = (
            error / combined if combined > 0 else (0.0 if error == 0 else float("inf"))
        )
        rows.append(
            {
                "parameter": name,
                "foerster-mean": float(a["mean"]),
                "pymc-mean": float(b["mean"]),
                "foerster-sd": float(a["sd"]),
                "pymc-sd": float(b["sd"]),
                "foerster-quantiles": np.quantile(
                    foerster[:, :, j], [0.05, 0.5, 0.95]
                ).tolist(),
                "pymc-quantiles": np.quantile(
                    pymc[:, :, j], [0.05, 0.5, 0.95]
                ).tolist(),
                "combined-mcse-mean": combined,
                "mean-error-in-combined-mcse": score,
                "passes": score <= MAX_MEAN_ERROR_MCSE,
            }
        )
    decision = json.loads((directory / "pymc/pymc-decision.json").read_text())
    candidates = {
        "foerster": f_report["allocation"]["allocation"],
        "pymc": decision["evaluation"]["allocation"],
    }
    baseline = fixture["decision"]["baseline"]
    functional = []
    flat_f = foerster.reshape(-1, N_PARAMETERS)
    flat_p = pymc.reshape(-1, N_PARAMETERS)
    baseline_f = totals(flat_f, fixture, baseline)
    baseline_p = totals(flat_p, fixture, baseline)
    for label, allocation in candidates.items():
        gain_f = totals(flat_f, fixture, allocation) - baseline_f
        gain_p = totals(flat_p, fixture, allocation) - baseline_p
        a = diagnostics(gain_f.reshape(4, 4000, 1)).iloc[0]
        b = diagnostics(gain_p.reshape(4, 4000, 1)).iloc[0]
        combined = float(np.hypot(a["mcse_mean"], b["mcse_mean"]))
        score = float(abs(a["mean"] - b["mean"]) / combined)
        functional.append(
            {
                "plan": label + "-optimized",
                "weekly-allocation": allocation,
                "foerster-gain": summary(gain_f),
                "pymc-gain": summary(gain_p),
                "mean-error-in-combined-mcse": score,
                "passes": bool(score <= MAX_MEAN_ERROR_MCSE),
            }
        )
    resources = {
        engine: resource_cpu(directory / engine / "fit-resources.txt")
        for engine in ["pymc", "foerster"]
    }
    # Descriptive only. PyMC ran its default sampler (nuts_sampler="pymc",
    # PyMC's own NUTS on the PyTensor C backend), not nutpie, NumPyro or
    # BlackJAX, which can be considerably faster.
    resources["pymc"]["sampler"] = "PyMC default NUTS (nuts_sampler='pymc')"
    for engine, stats in [("foerster", fs), ("pymc", ps)]:
        resources[engine]["min-bulk-ess-per-fit-process-cpu-second"] = float(
            stats["ess_bulk"].min() / resources[engine]["fit-process-cpu-seconds"]
        )
    report = {
        "case": fixture["case"],
        "passes": all(r["passes"] for r in rows + functional),
        "scope": "One synthetic three-channel model; full diagnostic gates and mean agreement within five combined MCSE. Quantiles reported descriptively. No broad calibration or performance-parity claim.",
        "parameter-comparison": rows,
        "common-plan-comparison": functional,
        "resources": resources,
        "fixture-sha256": digest(fixture_path),
        "posterior-sha256": {
            "foerster": digest(directory / "foerster/posterior.json"),
            "pymc": digest(directory / "pymc/posterior.nc"),
        },
        "comparison-source-sha256": digest(__file__),
    }
    write_json(directory / "posterior-comparison.json", report)
    if not report["passes"]:
        raise AssertionError(
            "Posterior mean agreement gate failed; inspect retained comparison"
        )
    print(
        json.dumps(
            {
                "case": fixture["case"],
                "passes": True,
                "max-mean-error-combined-mcse": max(
                    r["mean-error-in-combined-mcse"] for r in rows
                ),
                "resources": resources,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    compare(p.parse_args().directory)
