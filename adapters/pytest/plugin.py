"""
pytest plugin hook for margin.

Add to conftest.py:
    pytest_plugins = ["adapters.pytest.plugin"]

Or install margin with pytest entry point and it auto-registers.

After the test suite completes, prints a typed health expression
and optionally writes it to a JSON file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_results: dict = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": 0,
    "flaky": 0,
    "durations": [],
    "start_time": 0.0,
}


def pytest_configure(config) -> None:
    _results["start_time"] = time.time()


def pytest_runtest_logreport(report) -> None:
    if report.when == "call":
        if report.passed:
            _results["passed"] += 1
        elif report.failed:
            _results["failed"] += 1
        elif report.skipped:
            _results["skipped"] += 1
        _results["durations"].append(report.duration)
    elif report.when == "setup" and report.failed:
        _results["errors"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    from .suite import suite_expression

    total = _results["passed"] + _results["failed"] + _results["skipped"] + _results["errors"]
    if total == 0:
        return

    duration = time.time() - _results["start_time"]
    durations = _results["durations"]

    metrics = {
        "pass_rate": _results["passed"] / total if total > 0 else 0.0,
        "flake_rate": _results["flaky"] / total if total > 0 else 0.0,
        "skip_rate": _results["skipped"] / total if total > 0 else 0.0,
        "duration_seconds": duration,
        "new_failures": float(_results["failed"]),
    }

    if durations:
        metrics["mean_test_duration"] = sum(durations) / len(durations)

    expr = suite_expression(metrics, suite_id=str(config.rootpath.name))

    terminalreporter.section("margin health")
    terminalreporter.write_line(expr.to_string())

    # Write JSON if configured
    output = config.getoption("--margin-output", default=None)
    if output:
        Path(output).write_text(expr.to_json())


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--margin-output",
        default=None,
        help="Write margin health expression to this JSON file after the run.",
    )
