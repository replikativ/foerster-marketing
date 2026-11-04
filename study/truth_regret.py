"""Score each engine's recommended plan against the hidden truth.

Reads a publication study directory and writes truth-regret.json next to it.
The truth is used only after both engines have chosen their plans: it never
influences a recommendation. For every case, the true revenue gain of each
recommended plan over the current plan is compared with the truth-optimal plan
under the same budget and channel bounds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from common import CASES, LAG, TRUTH, digest, forward, future_plan, write_json
from scipy.optimize import minimize


def true_total(fixture, weekly):
    """Truth conditional-mean revenue over the campaign and its carryover."""
    x, controls = future_plan(fixture, weekly)
    return float(forward(TRUTH, x, controls)[LAG - 1 :].sum())


def truth_optimum(fixture):
    decision = fixture["decision"]
    budget = decision["weekly-budget"]
    result = minimize(
        lambda w: -true_total(fixture, w),
        np.asarray(decision["baseline"], float),
        bounds=list(zip(decision["lower"], decision["upper"], strict=True)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - budget}],
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"truth optimisation failed: {result.message}")
    return result.x


def score_case(study, case):
    fixture_path = study / case / "fixture.json"
    comparison_path = study / case / "full" / "posterior-comparison.json"
    fixture = json.loads(fixture_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    if not comparison["passes"]:
        raise RuntimeError(f"{case}: posterior comparison did not pass")
    baseline = np.asarray(fixture["decision"]["baseline"], float)
    base_total = true_total(fixture, baseline)
    optimum = truth_optimum(fixture)
    optimum_gain = true_total(fixture, optimum) - base_total
    plans = []
    for entry in comparison["common-plan-comparison"]:
        weekly = np.asarray(entry["weekly-allocation"], float)
        gain = true_total(fixture, weekly) - base_total
        plans.append(
            {
                "plan": entry["plan"],
                "weekly-allocation": weekly.tolist(),
                "true-gain": gain,
                "regret": optimum_gain - gain,
            }
        )
    return {
        "case": case,
        "fixture-sha256": digest(fixture_path),
        "posterior-comparison-sha256": digest(comparison_path),
        "current-plan": baseline.tolist(),
        "truth-optimal-plan": optimum.tolist(),
        "truth-optimal-gain": optimum_gain,
        "plans": plans,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    output = args.study / "truth-regret.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    report = {
        "scope": "Hidden-truth scoring after the decision; truth never selects a plan.",
        "generator-sha256": digest(__file__),
        "cases": [score_case(args.study, c) for c in args.cases],
    }
    write_json(output, report)
    for c in report["cases"]:
        print(
            c["case"],
            "optimum",
            np.round(c["truth-optimal-plan"]),
            round(c["truth-optimal-gain"]),
        )
        for p in c["plans"]:
            print(
                "  ",
                p["plan"],
                np.round(p["weekly-allocation"]),
                "gain",
                round(p["true-gain"]),
                "regret",
                round(p["regret"]),
            )


if __name__ == "__main__":
    main()
