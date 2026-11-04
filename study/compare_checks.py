"""Compare independently executed model audits; never substitute for sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
from common import SCALE, constrain, digest, forward, reference_logp, write_json
from scipy.stats import norm


def compare(directory):
    fixture_path = directory / "fixture.json"
    fixture = json.loads(fixture_path.read_text())
    python = json.loads((directory / "pymc-expanded-check.json").read_text())
    clojure = json.loads((directory / "foerster-check.json").read_text())
    assert (
        python["fixture-sha256"]
        == clojure["provenance"]["fixture-sha256"]
        == digest(fixture_path)
    )
    assert (
        len(python["points"])
        == len(clojure["points"])
        == len(fixture["unconstrained-points"])
    )
    maxima = {"logp": 0.0, "gradient": 0.0, "mean": 0.0, "pointwise": 0.0}
    for q, p, f in zip(
        fixture["unconstrained-points"],
        python["points"],
        clojure["points"],
        strict=True,
    ):
        np.testing.assert_array_equal(p["q"], q)
        np.testing.assert_array_equal(f["q"], q)
        theta = constrain(q)
        np.testing.assert_allclose(f["theta"], theta, rtol=1e-12, atol=1e-12)
        mu = forward(theta, fixture["training"]["x"], fixture["training"]["controls"])
        ll = norm.logpdf(fixture["training"]["y"], mu, theta[10] * SCALE)
        for row in [p, f]:
            np.testing.assert_allclose(
                row["logp"], reference_logp(q, fixture), rtol=1e-10, atol=1e-8
            )
            np.testing.assert_allclose(row["mean"], mu, rtol=1e-12, atol=1e-8)
            np.testing.assert_allclose(row["pointwise"], ll, rtol=1e-11, atol=1e-10)
        for key, previous in maxima.items():
            a, b = np.asarray(p[key]), np.asarray(f[key])
            error = (
                np.max(np.abs(a - b) / (1 + np.abs(a)))
                if key == "gradient"
                else np.max(np.abs(a - b))
            )
            maxima[key] = max(previous, float(error))
            np.testing.assert_allclose(
                a, b, rtol=1e-9, atol=1e-8 if key != "gradient" else 1e-7
            )
    np.testing.assert_allclose(
        python["changed-data"]["logp"],
        clojure["changed-data"]["logp"],
        rtol=1e-10,
        atol=1e-8,
    )
    report = {
        "case": fixture["case"],
        "passes": True,
        "scope": "fixed-point density, gradients, original-unit means/likelihood and changed observations; no posterior claim",
        "points": len(python["points"]),
        "max-errors": maxima,
        "fixture-sha256": digest(fixture_path),
        "python-audit-sha256": digest(directory / "pymc-expanded-check.json"),
        "foerster-audit-sha256": digest(directory / "foerster-check.json"),
    }
    write_json(directory / "numerical-comparison.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    compare(p.parse_args().directory)
