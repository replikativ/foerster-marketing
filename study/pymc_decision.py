"""Replay a saved PyMC-Marketing retailer fit through its public budget API.

Usage: ``python study/pymc_decision.py <fixture.json> <fit-directory>``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from analysis import evaluate
from common import (
    CHANNELS,
    HORIZON,
    N_PARAMETERS,
    OPTIMIZATION_DRAWS_PER_CHAIN,
    digest,
    passes_gates,
    write_json,
)
from pymc_marketing.mmm import MMM
from pymc_model import parameter_draws


def replay(fixture_path, directory):
    fixture = json.loads(fixture_path.read_text())
    fit_report = json.loads((directory / "pymc-report.json").read_text())
    if fit_report["fixture-sha256"] != digest(fixture_path):
        raise ValueError("Wrong fixture for saved fit")
    if fit_report["profile"] == "full" and not passes_gates(fit_report):
        raise ValueError("Full fit failed statistical gates")
    with xr.open_datatree(directory / "posterior.nc", engine="h5netcdf") as data:
        idata = data.load()
    chains = parameter_draws(idata["posterior"].to_dataset())
    stride = max(1, int(np.ceil(chains.shape[1] / OPTIMIZATION_DRAWS_PER_CHAIN)))
    idata["posterior"] = idata["posterior"].isel(draw=slice(None, None, stride))
    # Rebuild the model from the configuration stored with the fit. The
    # temporal lift term is not part of that configuration (see
    # pymc_model.add_temporal_lift_measurements); the allocation only needs the
    # media parameters' posterior draws, so the reloaded model suffices.
    mmm = MMM.load_from_idata(idata)
    decision = fixture["decision"]
    last_date = pd.Timestamp(mmm.idata.fit_data["date"].values[-1])
    started = time.perf_counter()
    # The allocation is in EUR/week and is repeated in each of the HORIZON
    # weeks; the four-week commitment is HORIZON * weekly-budget.
    optimizer = mmm.budget_optimizer(
        start_date=last_date + pd.Timedelta(weeks=1),
        end_date=last_date + pd.Timedelta(weeks=HORIZON),
    )
    result = optimizer.allocate_budget(
        total_budget=decision["weekly-budget"],
        budget_bounds={
            channel: (lo, hi)
            for channel, lo, hi in zip(
                CHANNELS, decision["lower"], decision["upper"], strict=True
            )
        },
        # Tighter than the library default (1e-9 with rescaling), which lands
        # within EUR 0.0002 of this; ftol=1e-10 reproduces the published runs.
        minimize_kwargs={"options": {"ftol": 1e-10}},
    )
    allocation = result.budgets.sel(channel=CHANNELS).values
    optimized = time.perf_counter()
    evaluation = evaluate(
        chains.reshape(-1, N_PARAMETERS),
        chains[:, ::stride].reshape(-1, N_PARAMETERS),
        fixture,
        allocation,
    )
    report = {
        "engine": "pymc-marketing",
        "case": fixture["case"],
        "profile": fit_report["profile"],
        "statistically-qualified": fit_report["profile"] == "full",
        "evaluation": evaluation,
        "optimizer": {
            "success": bool(result.scipy_result.success),
            "iterations": int(result.scipy_result.nit),
        },
        "timing": {
            "public-allocation-seconds": optimized - started,
            "independent-evaluation-seconds": time.perf_counter() - optimized,
        },
        "provenance": {
            "fixture-sha256": digest(fixture_path),
            "posterior-sha256": digest(directory / "posterior.nc"),
            "runner-sha256": digest(__file__),
            "analysis-sha256": digest(Path(__file__).with_name("analysis.py")),
        },
    }
    write_json(directory / "pymc-decision.json", report)
    print(
        json.dumps({k: report[k] for k in ["case", "profile", "optimizer", "timing"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    replay(args.fixture, args.directory)


if __name__ == "__main__":
    main()
