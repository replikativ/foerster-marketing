"""Synthetic fulfilment scenario and independent references for the budget decision.

`generate` writes the warehouse scenario that the Clojure runner reads
(examples/kettle_and_pine/data/warehouse.json, from the calibrated fixture).
`check` independently recomputes the capacity posterior by quadrature and every
exported decision run's profit from its revenue path and capacities.

Usage:
    python study/warehouse_check.py generate <calibrated.json> <warehouse.json>
    python study/warehouse_check.py check <warehouse.json> <decision-dir> <calibrated.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from common import HORIZON, LAG, TRUTH, forward, future_plan

SEED = 20261104
ECONOMICS = {
    "average-order-value": 45.0,
    "margin": 0.40,
    "delay-cost-per-order-week": 4.0,
    "weekly-cancellation-rate": 0.15,
}
CAPACITY = {
    "normal": 900.0,
    "true-reduced": 560.0,
    "prior-mean": 650.0,
    "prior-sd": 120.0,
    "weekly-sd": 30.0,
    "report-sd": 5.0,
    "remaining-reduced-weeks": 3,
}
BUDGETS = [0.0, 1250.0, 2500.0, 3750.0, 5000.0, 6250.0, 7500.0]
DISRUPTION_START_BACKLOG = 300.0


def ship_week(backlog, arrivals, capacity, cancel):
    available = backlog + arrivals
    shipped = min(available, max(0.0, capacity))
    remaining = available - shipped
    cancelled = cancel * remaining
    return remaining - cancelled, shipped, cancelled


def generate(fixture_path: Path, output: Path):
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    fixture = json.loads(fixture_path.read_text())
    rng = np.random.default_rng(SEED)
    arrivals = (
        np.asarray(fixture["training"]["y"][-3:]) / ECONOMICS["average-order-value"]
    )
    backlog = DISRUPTION_START_BACKLOG
    weeks = []
    for week, a in zip([-3, -2, -1], arrivals):
        capacity = CAPACITY["true-reduced"] + CAPACITY["weekly-sd"] * rng.normal()
        backlog, shipped, _ = ship_week(
            backlog, a, capacity, ECONOMICS["weekly-cancellation-rate"]
        )
        reported = shipped + CAPACITY["report-sd"] * rng.normal()
        weeks.append(
            {"week": week, "arrivals": float(a), "reported-shipped": float(reported)}
        )
    scenario = {
        "fixture-case": fixture["case"],
        "seed": SEED,
        "economics": ECONOMICS,
        "capacity": {k: v for k, v in CAPACITY.items() if k != "true-reduced"},
        "hidden-true-reduced-capacity": CAPACITY["true-reduced"],
        "budgets": BUDGETS,
        "disruption-start": {"backlog": DISRUPTION_START_BACKLOG},
        "disruption-weeks": weeks,
        "current": {"backlog": float(backlog)},
        "horizon": HORIZON,
        "carryover": LAG - 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenario, indent=2) + "\n")
    print(json.dumps({"current-backlog": backlog, "weeks": weeks}, indent=2))


def capacity_posterior(scenario):
    """Grid quadrature over the reduced mean capacity and each week's capacity."""
    c = scenario["capacity"]
    cancel = scenario["economics"]["weekly-cancellation-rate"]
    grid = np.linspace(
        c["prior-mean"] - 6 * c["prior-sd"], c["prior-mean"] + 6 * c["prior-sd"], 4001
    )
    weekly = np.linspace(-6, 6, 2401)
    log_post = -0.5 * ((grid - c["prior-mean"]) / c["prior-sd"]) ** 2
    # The backlog path depends on each week's realised capacity; integrate
    # weeks jointly by tracking the backlog per (grid, weekly) state would be
    # exponential, so condition on the reported shipments being saturated and
    # verify saturation numerically below.
    backlog = scenario["disruption-start"]["backlog"]
    saturated = True
    for week in scenario["disruption-weeks"]:
        available = backlog + week["arrivals"]
        caps = grid[:, None] + c["weekly-sd"] * weekly[None, :]
        shipped = np.minimum(available, np.maximum(0.0, caps))
        like = np.exp(
            -0.5 * ((week["reported-shipped"] - shipped) / c["report-sd"]) ** 2
        )
        prior = np.exp(-0.5 * weekly**2)
        log_post += np.log(np.trapezoid(like * prior[None, :], weekly, axis=1) + 1e-300)
        saturated &= week["reported-shipped"] < available - 10 * c["report-sd"]
        backlog, _, _ = ship_week(
            backlog, week["arrivals"], week["reported-shipped"], cancel
        )
    w = np.exp(log_post - log_post.max())
    w /= np.trapezoid(w, grid)
    mean = np.trapezoid(grid * w, grid)
    sd = np.sqrt(np.trapezoid((grid - mean) ** 2 * w, grid))
    return {"mean": float(mean), "sd": float(sd), "saturated": bool(saturated)}


def profit(path, backlog, capacities, economics, budget):
    aov, cancel = (
        economics["average-order-value"],
        economics["weekly-cancellation-rate"],
    )
    shipped_total, delay = 0.0, 0.0
    for revenue, capacity in zip(path, capacities):
        backlog, shipped, _ = ship_week(backlog, revenue / aov, capacity, cancel)
        shipped_total += shipped
        delay += backlog
    weeks = len(path) - (LAG - 1)
    return (
        economics["margin"] * aov * shipped_total
        - economics["delay-cost-per-order-week"] * delay
        - weeks * budget
    )


def truth_profits(scenario, fixture, report):
    """Hidden-truth expected profit of each reported plan (reporting only)."""
    out = {}
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(2000, HORIZON + LAG - 1))
    for state in report["states"]:
        rows = []
        for plan in state["plans"]:
            x, controls = future_plan(fixture, np.asarray(plan["weekly-allocation"]))
            path = forward(TRUTH, x, controls)[LAG - 1 :]
            caps = np.where(
                np.arange(HORIZON + LAG - 1)[None, :] < state["reduced-weeks"],
                state["reduced-mean-truth"] + scenario["capacity"]["weekly-sd"] * noise,
                scenario["capacity"]["normal"],
            )
            values = [
                profit(
                    path,
                    state["start-backlog"],
                    cap,
                    scenario["economics"],
                    plan["budget"],
                )
                for cap in caps
            ]
            rows.append(
                {
                    "budget": plan["budget"],
                    "truth-expected-profit": float(np.mean(values)),
                }
            )
        out[state["name"]] = rows
    return out


def check(scenario_path: Path, directory: Path, fixture_path: Path):
    scenario = json.loads(scenario_path.read_text())
    fixture = json.loads(fixture_path.read_text())
    report = json.loads((directory / "report.json").read_text())
    runs = json.loads((directory / "runs.json").read_text())
    results = {"passes": True}

    reference = capacity_posterior(scenario)
    results["capacity-reference"] = reference
    smc = report["capacity"]["seeds"]
    errors = [abs(s["mean"] - reference["mean"]) / s["mcse"] for s in smc]
    results["capacity-mean-error-in-mcse"] = errors
    results["capacity-passes"] = reference["saturated"] and max(errors) < 3.0
    results["passes"] &= results["capacity-passes"]

    worst = 0.0
    for state in runs["states"]:
        paths = np.asarray(runs["demand-paths"])  # [plan][draw][week]
        for r in state["runs"]:
            for k, plan in enumerate(state["budgets"]):
                expected = profit(
                    paths[k][r["draw"]],
                    state["start-backlog"],
                    r["capacities"],
                    scenario["economics"],
                    plan,
                )
                worst = max(
                    worst, abs(expected - r["profits"][k]) / (1.0 + abs(expected))
                )
    results["profit-max-relative-error"] = worst
    results["profit-passes"] = worst < 1e-8
    results["passes"] &= results["profit-passes"]
    results["truth"] = truth_profits(scenario, fixture, report)
    (directory / "independent-check.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in results.items() if k != "truth"}, indent=2))
    if not results["passes"]:
        raise SystemExit("Independent fulfilment check failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("generate")
    g.add_argument("fixture", type=Path)
    g.add_argument("output", type=Path)
    c = sub.add_parser("check")
    c.add_argument("scenario", type=Path)
    c.add_argument("directory", type=Path)
    c.add_argument("fixture", type=Path)
    a = p.parse_args()
    if a.mode == "generate":
        generate(a.fixture, a.output)
    else:
        check(a.scenario, a.directory, a.fixture)


if __name__ == "__main__":
    main()
