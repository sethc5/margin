"""
pytest plugin hook for margin.

Add to conftest.py:
    pytest_plugins = ["adapters.pytest.plugin"]

Features:
    - Prints typed health after every test run
    - CI gate: --margin-fail-below=DEGRADED fails the build if any metric hits that state
    - Diff: --margin-baseline=path.json compares against a previous run
    - Per-file breakdown: --margin-per-file shows health per test file
    - Output: --margin-output=path.json saves results for next diff
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from margin.health import Health, SEVERITY
from margin.confidence import Confidence

_results: dict = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": 0,
    "flaky": 0,
    "durations": [],
    "test_durations": [],  # (nodeid, duration) pairs for slowest callout
    "start_time": 0.0,
    "per_file": defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "durations": []}),
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
        _results["test_durations"].append((report.nodeid, report.duration))

        # Per-file tracking
        fpath = report.fspath if hasattr(report, "fspath") else str(report.nodeid).split("::")[0]
        pf = _results["per_file"][str(fpath)]
        if report.passed:
            pf["passed"] += 1
        elif report.failed:
            pf["failed"] += 1
        elif report.skipped:
            pf["skipped"] += 1
        pf["durations"].append(report.duration)

    elif report.when == "setup" and report.failed:
        _results["errors"] += 1


def _build_suite_metrics() -> dict[str, float]:
    total = _results["passed"] + _results["failed"] + _results["skipped"] + _results["errors"]
    if total == 0:
        return {}

    duration = time.time() - _results["start_time"]
    durations = _results["durations"]

    metrics = {
        "pass_rate": _results["passed"] / total,
        "flake_rate": _results["flaky"] / total,
        "skip_rate": _results["skipped"] / total,
        "duration_seconds": duration,
        "new_failures": float(_results["failed"]),
    }
    if durations:
        metrics["mean_test_duration"] = sum(durations) / len(durations)

    # Coverage — read from pytest-cov if available
    try:
        cov_path = Path(".coverage")
        if cov_path.exists():
            import coverage
            cov = coverage.Coverage()
            cov.load()
            import io
            total = cov.report(file=io.StringIO())
            metrics["coverage"] = total / 100.0
    except Exception:
        pass  # no pytest-cov or no .coverage file — skip

    return metrics


def _build_file_metrics() -> dict[str, dict[str, float]]:
    """Build per-file metrics dict."""
    file_metrics = {}
    for fpath, counts in _results["per_file"].items():
        total = counts["passed"] + counts["failed"] + counts["skipped"]
        if total == 0:
            continue
        m = {
            "pass_rate": counts["passed"] / total,
            "skip_rate": counts["skipped"] / total,
            "new_failures": float(counts["failed"]),
        }
        if counts["durations"]:
            m["mean_test_duration"] = sum(counts["durations"]) / len(counts["durations"])
            m["duration_seconds"] = sum(counts["durations"])
        file_metrics[fpath] = m
    return file_metrics


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    from .suite import suite_expression, parse_suite

    metrics = _build_suite_metrics()
    if not metrics:
        return

    expr = suite_expression(metrics, suite_id=str(config.rootpath.name))

    # ── Print suite health ──
    terminalreporter.section("margin health")
    terminalreporter.write_line(expr.to_string())

    # ── Slowest tests ──
    n_slow = config.getoption("--margin-slowest", default=0)
    if n_slow and n_slow > 0:
        from .suite import TEST_METRICS
        slow_thresh = TEST_METRICS["mean_test_duration"].thresholds
        sorted_tests = sorted(_results["test_durations"], key=lambda x: x[1], reverse=True)
        terminalreporter.write_line("")
        terminalreporter.write_line(f"Slowest {min(n_slow, len(sorted_tests))} tests:")
        for nodeid, dur in sorted_tests[:n_slow]:
            from margin.health import classify
            h = classify(dur, Confidence.HIGH, thresholds=slow_thresh)
            terminalreporter.write_line(f"  {h.value:12s} {dur:7.3f}s  {nodeid}")

    # ── Per-file breakdown ──
    if config.getoption("--margin-per-file", default=False):
        terminalreporter.write_line("")
        terminalreporter.write_line("Per-file breakdown:")
        file_metrics = _build_file_metrics()

        # Sort by worst health (most failures first)
        for fpath, fm in sorted(file_metrics.items(), key=lambda kv: kv[1].get("pass_rate", 1.0)):
            file_expr = suite_expression(fm, suite_id=fpath)
            worst = None
            for obs in file_expr.observations:
                if worst is None or SEVERITY.get(obs.health, 0) > SEVERITY.get(worst.health, 0):
                    worst = obs
            health_str = worst.health.value if worst else "INTACT"
            terminalreporter.write_line(f"  {health_str:12s} {fpath}")

    # ── Diff against baseline ──
    baseline_path = config.getoption("--margin-baseline", default=None)
    if baseline_path and Path(baseline_path).exists():
        terminalreporter.write_line("")
        try:
            from margin.diff import diff
            from margin.observation import Expression

            baseline_data = json.loads(Path(baseline_path).read_text())
            baseline_expr = Expression.from_dict(baseline_data)
            d = diff(baseline_expr, expr)

            if d.any_health_changed:
                terminalreporter.write_line("Changes since baseline:")
                for change in d.changes:
                    if change.health_changed:
                        b = change.health_before.value if change.health_before else "absent"
                        a = change.health_after.value if change.health_after else "absent"
                        terminalreporter.write_line(f"  {change.name}: {b} → {a}")
                    elif change.sigma_delta is not None and abs(change.sigma_delta) > 0.01:
                        sign = "+" if change.sigma_delta >= 0 else ""
                        terminalreporter.write_line(
                            f"  {change.name}: {change.health_after.value} ({sign}{change.sigma_delta:.2f}σ)")
            else:
                terminalreporter.write_line("No health changes since baseline.")
        except Exception as e:
            terminalreporter.write_line(f"Baseline diff failed: {e}")

    # ── Save output ──
    output = config.getoption("--margin-output", default=None)
    if output:
        Path(output).write_text(expr.to_json())

    # ── CI gate ──
    fail_below = config.getoption("--margin-fail-below", default=None)
    if fail_below:
        threshold = _parse_health_state(fail_below)
        if threshold is not None:
            threshold_sev = SEVERITY.get(threshold, 0)
            for obs in expr.observations:
                if SEVERITY.get(obs.health, 0) >= threshold_sev:
                    terminalreporter.write_line(
                        f"\nmargin CI gate FAILED: {obs.name} is {obs.health.value} "
                        f"(threshold: {fail_below})")
                    config._margin_gate_failed = True
                    return


def pytest_sessionfinish(session, exitstatus) -> None:
    """Override exit code if the margin gate failed."""
    if getattr(session.config, "_margin_gate_failed", False):
        session.exitstatus = 1


def pytest_addoption(parser) -> None:
    group = parser.getgroup("margin", "Margin health classification")
    group.addoption(
        "--margin-output",
        default=None,
        help="Write margin health JSON to this file after the run.",
    )
    group.addoption(
        "--margin-baseline",
        default=None,
        help="Path to a previous --margin-output JSON file. Shows diff.",
    )
    group.addoption(
        "--margin-fail-below",
        default=None,
        help="Fail the build if any metric reaches this state or worse. "
             "Values: INTACT, DEGRADED, ABLATED.",
    )
    group.addoption(
        "--margin-per-file",
        action="store_true",
        default=False,
        help="Show per-file health breakdown.",
    )
    group.addoption(
        "--margin-slowest",
        type=int,
        default=0,
        help="Show N slowest tests with health classification.",
    )


def _parse_health_state(name: str) -> Optional[Health]:
    name = name.upper().strip()
    for h in Health:
        if h.value == name:
            return h
    return None
