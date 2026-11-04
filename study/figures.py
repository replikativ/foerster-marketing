"""Render decision-focused figures only from the qualified paired study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import CHANNELS, digest


def render(directory):
    comparison = json.loads((directory / "posterior-comparison.json").read_text())
    if not comparison["passes"]:
        raise ValueError("Paired posterior check must pass before rendering")
    fixture_path = directory.parent / "fixture.json"
    if digest(fixture_path) != comparison["fixture-sha256"]:
        raise ValueError("Figure fixture differs")
    fixture = json.loads(fixture_path.read_text())
    reports = {
        engine: json.loads((directory / engine / file).read_text())
        for engine, file in [
            ("foerster", "independent-evaluation.json"),
            ("pymc", "pymc-decision.json"),
        ]
    }
    if not all(r["statistically-qualified"] for r in reports.values()):
        raise ValueError("Unqualified result")
    for engine, report in reports.items():
        if (
            report["provenance"]["fixture-sha256"] != comparison["fixture-sha256"]
            or report["provenance"]["posterior-sha256"]
            != comparison["posterior-sha256"][engine]
        ):
            raise ValueError("Figure inputs differ from the qualified comparison")
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    colours = {"foerster": "#387b56", "pymc": "#29649a"}

    def save(fig, name):
        fig.savefig(output / (name + ".svg"))
        fig.savefig(output / (name + ".png"), dpi=140)
        plt.close(fig)

    train = len(fixture["training"]["y"])
    weeks = np.arange(train + 1, len(fixture["all-data"]["y"]) + 1)
    observed = np.asarray(fixture["all-data"]["y"])[train:]
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (engine, r) in zip(axes, reports.items(), strict=True):
        h = r["evaluation"]["holdout"]
        colour = colours[engine]
        ax.fill_between(
            weeks,
            h["predictive-p05"],
            h["predictive-p95"],
            color=colour,
            alpha=0.17,
            label="90% predictive interval",
        )
        ax.plot(
            weeks,
            h["conditional-mean-p50"],
            color=colour,
            label="Median conditional mean",
        )
        ax.scatter(
            weeks,
            observed,
            color="#222222",
            s=16,
            label="Held-out observations",
            zorder=3,
        )
        ax.set_title("Foerster Marketing" if engine == "foerster" else "PyMC-Marketing")
        ax.set_ylabel("Revenue (EUR/week)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="best", ncols=3, fontsize=8)
    axes[-1].set_xlabel("Week; conditioning on observed exogenous controls")
    fig.suptitle("26 held-out weeks; targets were excluded from fitting")
    save(fig, "holdout")
    labels = ["Current plan", "Foerster plan", "PyMC plan"]
    allocations = np.array(
        [
            fixture["decision"]["baseline"],
            reports["foerster"]["evaluation"]["allocation"],
            reports["pymc"]["evaluation"]["allocation"],
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    left = np.zeros(3)
    for j, (channel, colour) in enumerate(
        zip(CHANNELS, ["#29649a", "#cb6f26", "#387b56"], strict=True)
    ):
        ax.barh(labels, allocations[:, j], left=left, color=colour, label=channel)
        for i, value in enumerate(allocations[:, j]):
            ax.text(
                left[i] + value / 2,
                i,
                f"{value:,.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        left += allocations[:, j]
    ax.set_xlabel("Advertising spend (EUR/week)")
    ax.set_xlim(0, fixture["decision"]["weekly-budget"])
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3)
    ax.set_title(
        f"A fixed EUR {fixture['decision']['weekly-budget'] * fixture['decision']['horizon']:,.0f} commitment across {fixture['decision']['horizon']} weeks"
    )
    save(fig, "allocation")
    # Evaluate one common proposed plan under each independently fitted belief.
    plan = next(
        p
        for p in comparison["common-plan-comparison"]
        if p["plan"] == "foerster-optimized"
    )
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    for row, (engine, name) in enumerate(
        [("foerster", "Foerster posterior"), ("pymc", "PyMC posterior")]
    ):
        gain = plan[engine + "-gain"]
        ax.plot(
            [gain["p05"], gain["p95"]], [row, row], color=colours[engine], linewidth=4
        )
        ax.scatter(gain["p50"], row, color=colours[engine], s=60, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks([0, 1], ["Foerster posterior", "PyMC posterior"])
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xlabel("Modeled incremental revenue (EUR); four weeks + five-week carryover")
    ax.set_title(
        "One proposed allocation, two posterior assessments: median and 90% interval"
    )
    ax.grid(axis="x", alpha=0.2)
    save(fig, "plan-gain")
    print("Rendered qualified comparison figures:", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    render(p.parse_args().directory)
