"""Fixed-point audit of the PyMC-Marketing model against the NumPy reference.

Usage: ``python study/pymc_audit.py <fixture.json> <pymc-expanded-check.json>``.

At each of the fixture's unconstrained check points this compares PyMC's log
density with ``common.reference_logp``, its gradient with central finite
differences of the reference, and the training mean and pointwise likelihood
with ``common.forward``. It also checks the sampler's start (the model's
initial point, shared with the Foerster engine) and that the compiled graph
follows changed observations. ``compare_checks.py`` then compares this output
with the Foerster engine's audit.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from common import (
    N_PARAMETERS,
    SCALE,
    constrain,
    forward,
    reference_lift,
    reference_logp,
    write_json,
)
from pymc_model import build, provenance
from scipy.stats import norm

# PyMC value variables (unconstrained) for each block of the parameter vector.
VALUE_VARIABLES = [
    ("adstock_alpha_logodds__", slice(0, 3)),
    ("saturation_alpha_log__", slice(3, 6)),
    ("saturation_lam_log__", slice(6, 9)),
    ("intercept_contribution", 9),
    ("y_sigma_log__", 10),
    ("gamma_control", slice(11, N_PARAMETERS)),
]
# The prior-centred start both engines use: alpha 0.5, beta exp(0.18),
# kappa exp(-0.32), intercept 1, sigma 0.2, controls 0 (unconstrained here).
MATCHED_START = np.r_[
    np.zeros(3), np.full(3, 0.18), np.full(3, -0.32), 1.0, np.log(0.2), np.zeros(4)
]
FINITE_DIFFERENCE_STEP = 1e-5


def value_point(q):
    q = np.asarray(q)
    return {name: q[index] for name, index in VALUE_VARIABLES}


def finite_difference_gradient(q, fixture):
    q = np.asarray(q)
    step = FINITE_DIFFERENCE_STEP * np.eye(len(q))
    return np.array(
        [
            (reference_logp(q + e, fixture) - reference_logp(q - e, fixture))
            / (2 * FINITE_DIFFERENCE_STEP)
            for e in step
        ]
    )


def check(fixture_path, output):
    fixture = json.loads(fixture_path.read_text())
    mmm, _, _ = build(fixture)
    model = mmm.model
    initial = model.initial_point()
    assert set(initial) == {name for name, _ in VALUE_VARIABLES}, list(initial)
    for name, expected in value_point(MATCHED_START).items():
        np.testing.assert_allclose(initial[name], expected, rtol=1e-12, atol=1e-12)
    logp, gradient = model.compile_logp(), model.compile_dlogp()
    mean = model.compile_fn(
        model.replace_rvs_by_values([model["mu"].values])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    pointwise = model.compile_fn(
        model.logp(vars=[model[mmm.output_var]], sum=False)[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    experiments = fixture["training"].get("experiments", [])
    model_lifts = [
        model.compile_fn(
            model.replace_rvs_by_values(
                [model[f"external_lift_{i}_model_estimated"].values]
            )[0],
            inputs=model.value_vars,
            on_unused_input="ignore",
        )
        for i in range(len(experiments))
    ]
    # PyMC concatenates the gradient in its own value-variable order.
    sizes = {name: np.size(value) for name, value in initial.items()}
    offsets = np.cumsum([0] + [sizes[v.name] for v in model.value_vars])
    gradient_slices = {
        v.name: slice(offsets[k], offsets[k + 1])
        for k, v in enumerate(model.value_vars)
    }
    rows = []
    for q in fixture["unconstrained-points"]:
        p = value_point(q)
        expected = reference_logp(q, fixture)
        actual = float(logp(p))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
        raw = np.asarray(gradient(p))
        ordered = np.concatenate(
            [raw[gradient_slices[name]] for name, _ in VALUE_VARIABLES]
        )
        np.testing.assert_allclose(
            ordered, finite_difference_gradient(q, fixture), rtol=1e-5, atol=1e-5
        )
        natural = constrain(q)
        mu = np.asarray(mean(p)) * SCALE
        ll = np.asarray(pointwise(p)) - np.log(SCALE)
        expected_mu = forward(
            natural, fixture["training"]["x"], fixture["training"]["controls"]
        )
        expected_ll = norm.logpdf(
            fixture["training"]["y"], expected_mu, natural[10] * SCALE
        )
        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(ll, expected_ll, rtol=1e-11, atol=1e-10)
        lifts = [float(f(p)) * SCALE for f in model_lifts]
        np.testing.assert_allclose(
            lifts,
            [reference_lift(natural, e) for e in experiments],
            rtol=1e-12,
            atol=1e-8,
        )
        rows.append(
            {
                "q": q,
                "theta": natural.tolist(),
                "logp": actual,
                "gradient": ordered.tolist(),
                "mean": mu.tolist(),
                "pointwise": ll.tolist(),
                "external-lift": lifts,
                "reference_logp": expected,
            }
        )
    # Reuse the compiled graph with changed observations, without rebuilding
    # the model structure.
    target = model["target_data"]
    original = target.get_value().copy()
    start = value_point(fixture["unconstrained-points"][0])
    try:
        target.set_value(original + 100.0)
        changed_value = float(logp(start))
        changed_fixture = copy.deepcopy(fixture)
        changed_fixture["training"]["y"] = (np.asarray(original) + 100.0).tolist()
        np.testing.assert_allclose(
            changed_value,
            reference_logp(fixture["unconstrained-points"][0], changed_fixture),
            atol=1e-8,
            rtol=1e-10,
        )
        assert changed_value != rows[0]["logp"]
    finally:
        target.set_value(original)
    write_json(
        output,
        {
            "engine": "pymc-marketing",
            "case": fixture["case"],
            "points": rows,
            "changed-data": {"logp": changed_value},
            "initial-natural-position": constrain(MATCHED_START).tolist(),
            **provenance(fixture_path, __file__),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.fixture, args.output)


if __name__ == "__main__":
    main()
