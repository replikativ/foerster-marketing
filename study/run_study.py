"""Run the paired PyMC-Marketing / Foerster study in sequential, bounded workers.

Run from the study's Python environment (study/requirements.txt), with
``clojure`` on the PATH. Each output directory is immutable evidence: a failed
run needs a fresh output directory.

Stages per case: fixed-point audits of both engines and their comparison; then
full fits, decisions, replays, posterior comparison and figures. Once every
case has a posterior comparison, the hidden-truth regret is scored, and when
the calibrated case is included, the warehouse decision runs and is checked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from common import (
    CASES,
    FIXTURES,
    MAX_MEAN_ERROR_MCSE,
    MAX_RHAT,
    MIN_ESS,
    ROOT,
    digest,
    write_json,
)

WAREHOUSE = FIXTURES / "warehouse.json"
SOURCE_DIRECTORIES = ["src", "examples", "study"]
SOURCE_SUFFIXES = {".clj", ".edn", ".json", ".py", ".txt"}


def relative(path):
    """Path as seen from the repository root, where every stage runs."""
    return os.path.relpath(path, ROOT)


def sources():
    paths = [ROOT / "deps.edn", ROOT / "build.clj"]
    for directory in SOURCE_DIRECTORIES:
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
        )
    return {relative(p): digest(p) for p in sorted(set(paths))}


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage-timeout-seconds", type=int, default=7800)
    parser.add_argument(
        "--clojure-aliases",
        default="",
        help="extra deps aliases appended to :jvm:study, e.g. :local",
    )
    args = parser.parse_args()
    if args.stage_timeout_seconds <= 0:
        parser.error("Stage timeout must be positive")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty_at_start = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / ("process-" + "-".join(args.cases) + ".json")
    if receipt_path.exists():
        parser.error("Refusing to overwrite a process receipt; select a fresh output")
    hashes = sources()
    manifest = output / "inputs.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["source-sha256"] != hashes:
            parser.error(
                "Source changed since the first study worker; select a fresh output"
            )
    else:
        write_json(
            manifest,
            {
                "source-sha256": hashes,
                "profile": "full",
                "gates": {
                    "max-rhat-exclusive": MAX_RHAT,
                    "min-ess-bulk-tail": MIN_ESS,
                    "divergences": 0,
                    "max-mean-error-combined-mcse": MAX_MEAN_ERROR_MCSE,
                },
            },
        )
        for name in hashes:
            target = output / "executed-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    for case in args.cases:
        original = FIXTURES / f"{case}.json"
        target = output / case / "fixture.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(original):
            parser.error("Existing fixture differs: " + relative(target))
        if (target.parent / "full").exists():
            parser.error("Refusing to overwrite an existing full-study directory")
        shutil.copyfile(original, target)

    receipt = {
        "started-at": datetime.now(UTC).isoformat(),
        "passes": False,
        "cases": args.cases,
        "profile": "full",
        "steps": [],
        "revision": revision,
        "dirty-at-start": dirty_at_start,
        "clojure-aliases": args.clojure_aliases,
        "source-manifest-sha256": digest(manifest),
    }
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    # Every stage runs from the repository root with relative paths, so that
    # logs, receipts and resource files carry no machine-specific locations.
    # A bare interpreter name is looked up on the PATH.
    python = [relative(args.python) if os.sep in args.python else args.python]
    aliases = "-M:jvm:study" + args.clojure_aliases
    foerster = ["clojure", aliases, "-m", "foerster-study"]

    def script(name):
        return [*python, f"study/{name}"]

    def run(case, stage, command, directory):
        if sources() != hashes:
            raise RuntimeError("Numerical source changed during qualification")
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / (stage + ".log")
        resources = directory / (stage + "-resources.txt")
        started = time.monotonic()
        print("Starting", case, stage, flush=True)
        process = None
        timed_out = False
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(
                    ["/usr/bin/time", "-v", "-o", relative(resources), *command],
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > args.stage_timeout_seconds:
                        timed_out = True
                        terminate(process)
                        break
                    try:
                        process.wait(
                            timeout=min(60, args.stage_timeout_seconds - elapsed)
                        )
                    except subprocess.TimeoutExpired:
                        print(
                            "Running",
                            case,
                            stage,
                            f"({time.monotonic() - started:.0f}s)",
                            flush=True,
                        )
        finally:
            if process is not None:
                terminate(process)
                receipt["steps"].append(
                    {
                        "case": case,
                        "stage": stage,
                        "command": command,
                        "exit": process.returncode,
                        "timed-out": timed_out,
                        "wall-seconds": time.monotonic() - started,
                        "log": str(log.relative_to(output)),
                        "resources": str(resources.relative_to(output)),
                    }
                )
                write_json(receipt_path, receipt)
        if process.returncode or timed_out:
            raise RuntimeError(f"{case} {stage} failed; see {relative(log)}")

    try:
        # Establish target agreement for every requested case before fitting.
        for case in args.cases:
            directory = output / case
            fixture = relative(directory / "fixture.json")
            check = relative(directory / "pymc-expanded-check.json")
            run(
                case,
                "pymc-check",
                [*script("pymc_audit.py"), fixture, check],
                directory,
            )
            run(
                case,
                "foerster-check",
                [*foerster, "check", fixture, relative(directory)],
                directory,
            )
            run(
                case,
                "compare-checks",
                [*script("compare_checks.py"), relative(directory)],
                directory,
            )
        for case in args.cases:
            directory = output / case / "full"
            fixture = relative(directory.parent / "fixture.json")
            pymc, native = directory / "pymc", directory / "foerster"
            run(
                case,
                "fit",
                [
                    *script("pymc_model.py"),
                    fixture,
                    relative(pymc),
                    "--profile",
                    "full",
                ],
                pymc,
            )
            run(
                case,
                "decision",
                [*script("pymc_decision.py"), fixture, relative(pymc)],
                pymc,
            )
            run(
                case,
                "fit",
                [*foerster, "fit", fixture, relative(native), "full"],
                native,
            )
            for stage in ["decision", "replay"]:
                run(
                    case,
                    "native-replay" if stage == "replay" else stage,
                    [*foerster, stage, fixture, relative(native)],
                    native,
                )
            run(
                case,
                "independent-evaluation",
                [*script("foerster_evaluation.py"), fixture, relative(native)],
                native,
            )
            run(
                case,
                "compare-posteriors",
                [*script("compare_posteriors.py"), relative(directory)],
                directory,
            )
            run(
                case,
                "figures",
                [*script("figures.py"), relative(directory)],
                directory,
            )
        # Cases may run in separate invocations; score once all are present.
        if all(
            (output / case / "full/posterior-comparison.json").exists()
            for case in CASES
        ):
            run(
                "all",
                "truth-regret",
                [*script("truth_regret.py"), relative(output)],
                output,
            )
        if "calibrated" in args.cases:
            calibrated = output / "calibrated"
            warehouse = output / "warehouse"
            run(
                "warehouse",
                "decision",
                [
                    "clojure",
                    "-J-Xmx3g",
                    aliases,
                    "-m",
                    "warehouse-study",
                    relative(WAREHOUSE),
                    relative(calibrated / "fixture.json"),
                    relative(calibrated / "full/foerster"),
                    relative(warehouse),
                ],
                warehouse,
            )
            run(
                "warehouse",
                "independent-check",
                [
                    *script("warehouse_check.py"),
                    "check",
                    relative(WAREHOUSE),
                    relative(warehouse),
                    relative(calibrated / "fixture.json"),
                ],
                warehouse,
            )
        receipt["passes"] = True
    finally:
        receipt["finished-at"] = datetime.now(UTC).isoformat()
        write_json(receipt_path, receipt)
    print("Qualified", args.cases, relative(output), flush=True)


if __name__ == "__main__":
    main()
