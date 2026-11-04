"""Independent business-unit evaluation of saved posterior draws.

This module does not build either engine's inference or allocation graph.
"""

from __future__ import annotations

import numpy as np
from common import (
    LAG,
    N_PARAMETERS,
    SCALE,
    forward,
    future_plan,
    summary,
)
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


def predictions(draws, x, controls):
    draws = np.asarray(draws, dtype=float)
    x = np.asarray(x, dtype=float) / 1000
    controls = np.asarray(controls, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != N_PARAMETERS:
        raise ValueError(f"Expected rows of {N_PARAMETERS} constrained parameters")
    weights = draws[:, :3, None] ** np.arange(LAG)
    weights /= weights.sum(axis=2, keepdims=True)
    carry = np.zeros((len(draws), len(x), 3))
    for lag in range(min(LAG, len(x))):
        carry[:, lag:, :] += weights[:, :, lag][:, None, :] * x[None, : len(x) - lag, :]
    media = draws[:, None, 3:6] * carry / (draws[:, None, 6:9] + carry)
    return SCALE * (draws[:, 9, None] + media.sum(axis=2) + draws[:, 11:] @ controls.T)


def validate_reference(draws, fixture):
    xs = np.asarray(draws)
    selected = xs[np.linspace(0, len(xs) - 1, min(8, len(xs)), dtype=int)]
    data = fixture["all-data"]
    actual = predictions(selected, data["x"], data["controls"])
    expected = np.array([forward(p, data["x"], data["controls"]) for p in selected])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)


def totals(draws, fixture, allocation):
    x, controls = future_plan(fixture, allocation)
    return predictions(draws, x, controls)[:, LAG - 1 :].sum(axis=1)


def oracle(draws, fixture):
    decision = fixture["decision"]
    budget = decision["weekly-budget"]
    lower, upper = np.asarray(decision["lower"]), np.asarray(decision["upper"])
    objective = lambda fractions: (
        -totals(draws, fixture, budget * fractions).mean() / SCALE
    )
    result = minimize(
        objective,
        np.asarray(decision["baseline"]) / budget,
        method="SLSQP",
        bounds=list(zip(lower / budget, upper / budget, strict=True)),
        constraints={"type": "eq", "fun": lambda f: f.sum() - 1},
        options={"maxiter": 500, "ftol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(result.message)
    allocation = result.x * budget
    grid = []
    for search in np.arange(lower[0], upper[0] + 0.01, 100):
        for social in np.arange(lower[1], upper[1] + 0.01, 100):
            video = budget - search - social
            if lower[2] <= video <= upper[2]:
                x = np.array([search, social, video])
                grid.append((float(totals(draws, fixture, x).mean()), x))
    best_value, best_allocation = max(grid, key=lambda pair: pair[0])
    expected = float(totals(draws, fixture, allocation).mean())
    if expected + 0.01 < best_value:
        raise AssertionError("Continuous oracle worse than feasible grid")
    return {
        "allocation": allocation.tolist(),
        "expected-response": expected,
        "grid-step": 100.0,
        "grid-best-allocation": best_allocation.tolist(),
        "grid-best-response": best_value,
        "grid-cells": len(grid),
        "solver-iterations": int(result.nit),
    }


def holdout(draws, fixture):
    n = len(fixture["training"]["y"])
    data = fixture["all-data"]
    # Batch to bound memory independently of the posterior's draw count.
    means = np.concatenate(
        [
            predictions(batch, data["x"], data["controls"])[:, n:]
            for batch in np.array_split(draws, max(1, int(np.ceil(len(draws) / 256))))
        ]
    )
    observed = np.asarray(data["y"])[n:]
    sigma = np.asarray(draws)[:, 10, None] * SCALE
    log_density = norm.logpdf(observed[None, :], means, sigma)
    log_predictive = logsumexp(log_density, axis=0) - np.log(len(draws))
    # Predictive intervals contain observation noise. Conditional-mean
    # intervals are separate; the seeded simulation affects evaluation only.
    predictive = means + np.random.default_rng(90210).normal(size=means.shape) * sigma
    lower, upper = np.quantile(predictive, [0.05, 0.95], axis=0)
    return {
        "weeks": len(observed),
        "conditional-on-observed-controls": True,
        "rmse": float(np.sqrt(np.mean((means.mean(axis=0) - observed) ** 2))),
        "mean-log-predictive-density": float(log_predictive.mean()),
        "predictive-90-coverage": float(
            np.mean((observed >= lower) & (observed <= upper))
        ),
        "conditional-mean-p05": np.quantile(means, 0.05, axis=0).tolist(),
        "conditional-mean-p50": np.quantile(means, 0.5, axis=0).tolist(),
        "conditional-mean-p95": np.quantile(means, 0.95, axis=0).tolist(),
        "predictive-p05": lower.tolist(),
        "predictive-p95": upper.tolist(),
    }


def evaluate(draws, optimization_draws, fixture, allocation):
    validate_reference(draws, fixture)
    allocation = np.asarray(allocation)
    decision = fixture["decision"]
    np.testing.assert_allclose(
        allocation.sum(), decision["weekly-budget"], rtol=0, atol=0.01
    )
    if not (
        np.all(allocation >= np.asarray(decision["lower"]) - 0.01)
        and np.all(allocation <= np.asarray(decision["upper"]) + 0.01)
    ):
        raise AssertionError("Allocation violates business bounds")
    independent = oracle(optimization_draws, fixture)
    value = totals(optimization_draws, fixture, allocation).mean()
    # EUR 0.10 tolerance applies to the whole four-week-plus-tail mean response.
    if independent["expected-response"] - value > 0.10:
        raise AssertionError(
            {"allocation-value": float(value), "independent-oracle": independent}
        )
    gain = totals(draws, fixture, allocation) - totals(
        draws, fixture, decision["baseline"]
    )
    return {
        "holdout": holdout(draws, fixture),
        "allocation": allocation.tolist(),
        "oracle": independent,
        "gain-full-posterior": summary(gain),
        "probability-positive-conditional-mean-gain": float(np.mean(gain > 0)),
        "retained-evaluation-draws": len(draws),
        "optimization-draws": len(optimization_draws),
    }
