"""Shared business data and independent numerical references for the retailer study.

The two inference implementations read identical serialized fixtures. Neither
uses this NumPy reference to construct its sampling graph.

Run ``python study/common.py [output-directory]`` to regenerate the three
fixtures; by default they go to ``examples/kettle_and_pine/data/<case>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples/kettle_and_pine/data"
CASES = ["staggered", "co-scheduled", "calibrated"]
SEED = 20261101
CHANNELS = ["search", "social", "video"]
CONTROLS = ["demand", "promotion", "annual_sin", "annual_cos"]
TRAIN, PERIODS, HORIZON, LAG = 130, 156, 4, 6
SCALE = 10000.0
TRUTH = np.array(
    [
        0.25,
        0.50,
        0.75,
        0.8,
        1.1,
        0.9,
        0.8,
        0.7,
        0.55,
        1.2,
        0.07,
        0.20,
        0.12,
        0.18,
        -0.08,
    ]
)
# alpha[3], beta[3], kappa[3], intercept, sigma, control[4]: the parameter
# vector both engines share, in this order.
N_PARAMETERS = len(TRUTH)
# Per-fit statistical gates applied identically to both engines.
MAX_RHAT = 1.01
MIN_ESS = 100
# Posterior means of the two engines must agree within this many combined MCSE.
MAX_MEAN_ERROR_MCSE = 5
# Both engines optimize over at most this many evenly thinned draws per chain.
OPTIMIZATION_DRAWS_PER_CHAIN = 256
SPEC = {
    "channels": CHANNELS,
    "controls": CONTROLS,
    "saturation": "michaelis-menten",
    "l-max": LAG,
    "scaling": {"channel": [1000.0] * 3, "target": SCALE},
    "priors": {
        "alpha": {"distribution": "beta", "alpha": 2.0, "beta": 2.0},
        "beta": {"distribution": "log-normal", "mu": 0.0, "sigma": 0.6},
        "kappa": {"distribution": "log-normal", "mu": -0.5, "sigma": 0.6},
        "intercept": {"distribution": "normal", "mu": 1.0, "sigma": 0.6},
        "sigma": {"distribution": "half-normal", "sigma": 0.2},
        "control": {"distribution": "normal", "mu": 0.0, "sigma": 0.3},
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gate_fields(summary, divergences=0):
    """Condense an ArviZ summary into the fields that `passes_gates` reads."""
    values = summary[["r_hat", "ess_bulk", "ess_tail"]].to_numpy()
    return {
        "finite-diagnostics": bool(np.isfinite(values).all()),
        "max-rhat": float(summary["r_hat"].max()),
        "min-ess-bulk": float(summary["ess_bulk"].min()),
        "min-ess-tail": float(summary["ess_tail"].min()),
        "divergences": int(divergences),
    }


def passes_gates(report):
    """Apply the study's statistical gates to a fit report's diagnostic fields."""
    return bool(
        report.get("finite-diagnostics", False)
        and report["max-rhat"] < MAX_RHAT
        and report["min-ess-bulk"] >= MIN_ESS
        and report["min-ess-tail"] >= MIN_ESS
        and report["divergences"] == 0
    )


def forward(theta, x, controls):
    theta, x, controls = np.asarray(theta), np.asarray(x), np.asarray(controls)
    weights = theta[:3, None] ** np.arange(LAG)
    weights /= weights.sum(axis=1, keepdims=True)
    carry = np.column_stack(
        [np.convolve(x[:, j] / 1000, weights[j])[: len(x)] for j in range(3)]
    )
    media = theta[3:6] * carry / (theta[6:9] + carry)
    return SCALE * (theta[9] + media.sum(axis=1) + controls @ theta[11:])


def _ar(innovations, persistence):
    output = np.empty_like(innovations)
    previous = np.zeros(innovations.shape[1:])
    for t, noise in enumerate(innovations):
        previous = persistence * previous + noise
        output[t] = previous
    return output


def make_fixture(case):
    assert case in CASES
    rng = np.random.default_rng(SEED)
    noise = _ar(rng.normal(size=(PERIODS, 3)), 0.70)
    demand_raw = _ar(rng.normal(size=PERIODS), 0.75)
    demand_center, demand_scale = demand_raw[:TRAIN].mean(), demand_raw[:TRAIN].std()
    demand = (demand_raw - demand_center) / demand_scale
    quarter_strength = rng.uniform(0.85, 1.15, size=12)
    observation_noise = rng.normal(size=PERIODS)
    t = np.arange(PERIODS)
    phase = t % 13
    promotion = ((phase == 4) | (phase == 5) | ((t % 52) >= 48)).astype(float)
    social_flight = ((phase >= 2) & (phase <= 6)).astype(float)
    video_flight = ((phase >= 7) & (phase <= 11)).astype(float)
    strength = quarter_strength[t // 13]
    search = np.clip(
        1700 + 250 * demand + 300 * promotion + 100 * noise[:, 0], 800, 2800
    )
    social = np.clip(
        700 + 1400 * strength * social_flight + 400 * promotion + 100 * noise[:, 1],
        300,
        3500,
    )
    video = np.clip(
        100 + 1800 * strength * video_flight + 400 * promotion + 80 * noise[:, 2],
        0,
        3000,
    )
    if case != "staggered":
        video = np.clip(0.8 * social + 0.2 * video + 20 * noise[:, 2], 0, 3000)
    x = np.column_stack([search, social, video])
    controls = np.column_stack(
        [demand, promotion, np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)]
    )
    y = forward(TRUTH, x, controls) + SCALE * TRUTH[10] * observation_noise
    data = {
        "x": x[:TRAIN].tolist(),
        "controls": controls[:TRAIN].tolist(),
        "time-index": t[:TRAIN].tolist(),
        "y": y[:TRAIN].tolist(),
        # The library's optional steady-state lift rows; this study supplies
        # temporal experiments instead. Kept so fixture bytes stay unchanged.
        "lift": [],
    }
    if case == "calibrated":
        baseline = np.tile([2000.0, 1800.0, 1200.0], (LAG - 1 + HORIZON + LAG - 1, 1))
        treatment = baseline.copy()
        treatment[LAG - 1 : LAG - 1 + HORIZON, 2] += 800
        weights = np.r_[np.zeros(LAG - 1), np.ones(HORIZON + LAG - 1)]
        zero_controls = np.zeros((len(baseline), len(CONTROLS)))
        delta = np.sum(
            weights
            * (
                forward(TRUTH, treatment, zero_controls)
                - forward(TRUTH, baseline, zero_controls)
            )
        )
        sigma = 1500.0
        estimate = delta + sigma * np.random.default_rng(SEED + 1).normal()
        data["experiments"] = [
            {
                "id": "independent-video-lift",
                "independent-of-training?": True,
                "baseline": baseline.tolist(),
                "treatment": treatment.tolist(),
                "weights": weights.tolist(),
                "estimate": float(estimate),
                "sigma": sigma,
            }
        ]
    future_t = np.arange(TRAIN, TRAIN + HORIZON)
    future_controls = np.column_stack(
        [
            demand[TRAIN - 1] * 0.75 ** np.arange(1, HORIZON + 1),
            promotion[TRAIN : TRAIN + HORIZON],
            np.sin(2 * np.pi * future_t / 52),
            np.cos(2 * np.pi * future_t / 52),
        ]
    )
    q = unconstrain(TRUTH)
    points = [
        q.tolist(),
        (q + 0.07 * np.sin(np.arange(len(q)) + 1)).tolist(),
        (q - 0.11 * np.cos(np.arange(len(q)) + 1)).tolist(),
    ]
    fixture = {
        "format": "retailer-study-1",
        "case": case,
        "seed": SEED,
        "spec": SPEC,
        "training": data,
        "all-data": {
            "x": x.tolist(),
            "controls": controls.tolist(),
            "time-index": t.tolist(),
            "y": y.tolist(),
        },
        "decision": {
            "weekly-budget": 5000.0,
            "horizon": HORIZON,
            "lower": [1000.0, 500.0, 0.0],
            "upper": [2600.0, 3000.0, 2500.0],
            "baseline": [2000.0, 1800.0, 1200.0],
            "controls": future_controls.tolist(),
        },
        "truth": TRUTH.tolist(),
        "unconstrained-points": points,
        "design": {
            "synthetic": True,
            "spending": "persistent multiweek flights; not IID weekly shocks",
            "train-weeks": TRAIN,
            "holdout-weeks": PERIODS - TRAIN,
            "demand-scaling": {"mean": float(demand_center), "sd": float(demand_scale)},
            "spend-correlations": np.corrcoef(x[:TRAIN].T).tolist(),
            "future-demand": "decays from final training observation; no held-out demand used",
            "holdout-scoring": "conditional on exogenous controls; held-out target never fitted",
        },
    }
    assert np.min(y) > 0
    assert np.isfinite(x).all() and (x >= 0).all()
    return fixture


def constrain(q):
    q = np.asarray(q)
    return np.r_[1 / (1 + np.exp(-q[:3])), np.exp(q[3:9]), q[9], np.exp(q[10]), q[11:]]


def unconstrain(theta):
    theta = np.asarray(theta)
    return np.r_[
        np.log(theta[:3] / (1 - theta[:3])),
        np.log(theta[3:9]),
        theta[9],
        np.log(theta[10]),
        theta[11:],
    ]


def reference_lift(theta, experiment):
    """Measured-window revenue lift of an experiment's treatment over its baseline."""
    controls = np.zeros((len(experiment["baseline"]), len(CONTROLS)))
    return np.sum(
        np.asarray(experiment["weights"])
        * (
            forward(theta, experiment["treatment"], controls)
            - forward(theta, experiment["baseline"], controls)
        )
    )


def reference_logp(q, fixture):
    from scipy.stats import beta, halfnorm, lognorm, norm

    q = np.asarray(q)
    p = constrain(q)
    data = fixture["training"]
    lp = beta.logpdf(p[:3], 2, 2).sum()
    lp += lognorm.logpdf(p[3:6], s=0.6, scale=1).sum()
    lp += lognorm.logpdf(p[6:9], s=0.6, scale=np.exp(-0.5)).sum()
    lp += norm.logpdf(p[9], 1, 0.6) + halfnorm.logpdf(p[10], scale=0.2)
    lp += norm.logpdf(p[11:], 0, 0.3).sum()
    mu = forward(p, data["x"], data["controls"]) / SCALE
    lp += norm.logpdf(np.asarray(data["y"]) / SCALE, mu, p[10]).sum()
    for experiment in data.get("experiments", []):
        delta = reference_lift(p, experiment)
        lp += norm.logpdf(
            experiment["estimate"] / SCALE, delta / SCALE, experiment["sigma"] / SCALE
        )
    lp += np.log(p[:3]).sum() + np.log1p(-p[:3]).sum() + q[3:9].sum() + q[10]
    return float(lp)


def future_plan(fixture, allocation):
    history = fixture["training"]
    tail = LAG - 1
    x = np.vstack(
        [history["x"][-tail:], np.tile(allocation, (HORIZON, 1)), np.zeros((tail, 3))]
    )
    controls = np.vstack(
        [
            history["controls"][-tail:],
            fixture["decision"]["controls"],
            np.zeros((tail, len(CONTROLS))),
        ]
    )
    return x, controls


def summary(values):
    return {
        "mean": float(np.mean(values)),
        **{
            n: float(v)
            for n, v in zip(
                ["p05", "p50", "p95"],
                np.quantile(values, [0.05, 0.5, 0.95]),
                strict=True,
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=FIXTURES)
    args = parser.parse_args()
    for case in CASES:
        path = args.output / f"{case}.json"
        if path.exists():
            raise FileExistsError(path)
        fixture = make_fixture(case)
        write_json(path, fixture)
        print(case, fixture["design"]["spend-correlations"], flush=True)


if __name__ == "__main__":
    main()
