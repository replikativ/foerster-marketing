"""Evaluate the library's portable tagged JSON posterior independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analysis import evaluate
from common import N_PARAMETERS, OPTIMIZATION_DRAWS_PER_CHAIN, digest, write_json


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {"$keyword"}:
            return value["$keyword"]
        if set(value) == {"$nonfinite"}:
            return float(value["$nonfinite"])
        output = {}
        for key, item in value.items():
            if key[:1] not in {":", "="}:
                raise ValueError("Unknown portable posterior key")
            output[key[1:]] = decode(item)
        return output
    return value


def replay(fixture_path, directory):
    fixture = json.loads(fixture_path.read_text())
    report = json.loads((directory / "foerster-report.json").read_text())
    if report["provenance"]["fixture-sha256"] != digest(fixture_path):
        raise ValueError("Wrong fixture for saved fit")
    if (
        report["profile"] == "full"
        and not report["diagnostics"]["passes-tutorial-gates?"]
    ):
        raise ValueError("Full fit failed statistical gates")
    raw = json.loads((directory / "posterior.json").read_text())
    if raw["wire-format"] != "foerster-marketing/tagged-json-1":
        raise ValueError("Unknown posterior format")
    posterior = decode(raw["posterior"])
    chains = np.asarray(posterior["chains"])
    if (
        posterior["spec"]["channels"] != fixture["spec"]["channels"]
        or posterior["spec"]["controls"] != fixture["spec"]["controls"]
    ):
        raise ValueError("Wrong model coordinates")
    if chains.ndim != 3 or chains.shape[-1] != N_PARAMETERS:
        raise ValueError("Unexpected posterior dimensions")
    stride = max(1, int(np.ceil(chains.shape[1] / OPTIMIZATION_DRAWS_PER_CHAIN)))
    evaluation = evaluate(
        chains.reshape(-1, N_PARAMETERS),
        chains[:, ::stride].reshape(-1, N_PARAMETERS),
        fixture,
        report["allocation"]["allocation"],
    )
    output = {
        "engine": "foerster-marketing",
        "case": fixture["case"],
        "profile": report["profile"],
        "statistically-qualified": report["statistically-qualified"],
        "evaluation": evaluation,
        "provenance": {
            "fixture-sha256": digest(fixture_path),
            "posterior-sha256": digest(directory / "posterior.json"),
            "runner-sha256": digest(__file__),
            "analysis-sha256": digest(Path(__file__).with_name("analysis.py")),
        },
    }
    write_json(directory / "independent-evaluation.json", output)
    print(
        json.dumps(
            {
                "case": fixture["case"],
                "profile": report["profile"],
                "evaluation-draws": evaluation["retained-evaluation-draws"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fixture", type=Path)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    replay(a.fixture, a.directory)
