"""
Persistence: save and restore Monitor state across process restarts.

Also provides batch replay — feed historical data through a Monitor
and get typed analysis of all drift periods, anomalies, and correlations.

    monitor.save("state.json")
    monitor = Monitor.load("state.json", parser)
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from .observation import Observation, Expression, Parser
from .streaming import Monitor, DriftTracker, AnomalyTracker, CorrelationTracker


def save_monitor(monitor: Monitor, path: str) -> None:
    """Save Monitor state to a JSON file."""
    # Read anomaly_min_reference from one of the trackers (all share the same value)
    _any_tracker = next(iter(monitor._anomaly_trackers.values()), None)
    _anomaly_min_ref = _any_tracker._min_reference if _any_tracker else 10

    state = {
        "step": monitor.step,
        "window": monitor.window,
        "drift_window": monitor.drift_window,
        "anomaly_window": monitor.anomaly_window,
        "correlation_window": monitor.correlation_window,
        "anomaly_min_reference": _anomaly_min_ref,
        "drift": {},
        "anomaly": {},
        "correlation": {
            "components": monitor._correlation_tracker.components,
            "series": {c: list(d) for c, d in monitor._correlation_tracker._series.items()},
            "n_updates": monitor._correlation_tracker.n_updates,
        },
    }

    for name, dt in monitor._drift_trackers.items():
        state["drift"][name] = {
            "observations": [o.to_dict() for o in dt._observations],
        }

    for name, at in monitor._anomaly_trackers.items():
        state["anomaly"][name] = {
            "values": list(at._values),
        }

    Path(path).write_text(json.dumps(state, indent=2))


def load_monitor(path: str, parser: Parser, **kwargs) -> Monitor:
    """
    Restore a Monitor from a saved state file.

    Args:
        path:    path to the state JSON
        parser:  the Parser to use (must match the saved components)
        **kwargs: passed to Monitor constructor (window, min_correlation, etc.)
    """
    data = json.loads(Path(path).read_text())

    window = data.get("window", kwargs.pop("window", 100))
    drift_window = data.get("drift_window", kwargs.pop("drift_window", None))
    anomaly_window = data.get("anomaly_window", kwargs.pop("anomaly_window", None))
    correlation_window = data.get("correlation_window", kwargs.pop("correlation_window", None))
    anomaly_min_reference = data.get("anomaly_min_reference", kwargs.pop("anomaly_min_reference", 10))
    monitor = Monitor(
        parser, window=window,
        drift_window=drift_window,
        anomaly_window=anomaly_window,
        correlation_window=correlation_window,
        anomaly_min_reference=anomaly_min_reference,
        **kwargs,
    )
    monitor._step = data.get("step", 0)

    # Restore drift tracker observations
    for name, dt_data in data.get("drift", {}).items():
        if name in monitor._drift_trackers:
            tracker = monitor._drift_trackers[name]
            for od in dt_data.get("observations", []):
                obs = Observation.from_dict(od)
                tracker._observations.append(obs)
            # Reclassify from restored observations
            if tracker._observations:
                from .drift import classify_drift
                tracker._classification = classify_drift(list(tracker._observations))

    # Restore anomaly tracker values
    for name, at_data in data.get("anomaly", {}).items():
        if name in monitor._anomaly_trackers:
            tracker = monitor._anomaly_trackers[name]
            for v in at_data.get("values", []):
                tracker._values.append(v)

    # Restore correlation series
    corr_data = data.get("correlation", {})
    for c in monitor._correlation_tracker.components:
        series = corr_data.get("series", {}).get(c, [])
        for v in series:
            monitor._correlation_tracker._series[c].append(v)
    monitor._correlation_tracker._n_updates = corr_data.get("n_updates", 0)

    return monitor


def replay(
    parser: Parser,
    data: list[dict[str, float]],
    timestamps: Optional[list[datetime]] = None,
    window: int = 100,
    drift_window: Optional[int] = None,
    anomaly_window: Optional[int] = None,
    correlation_window: Optional[int] = None,
    **kwargs,
) -> tuple[Monitor, list[dict]]:
    """
    Replay historical data through a Monitor.

    Args:
        parser:            Parser for health classification
        data:              list of {component: value} dicts, one per step
        timestamps:        optional timestamps per step (defaults to 1-second intervals)
        window:            base tracker window size (all trackers inherit if per-tracker not set)
        drift_window:      override window for DriftTracker
        anomaly_window:    override window for AnomalyTracker
        correlation_window: override window for CorrelationTracker
        **kwargs:          passed to Monitor

    Returns (monitor, snapshots) where snapshots is a list of status dicts,
    one per step.
    """
    from datetime import timedelta

    monitor = Monitor(
        parser, window=window,
        drift_window=drift_window,
        anomaly_window=anomaly_window,
        correlation_window=correlation_window,
        **kwargs,
    )
    snapshots = []

    t0 = timestamps[0] if timestamps else datetime(2000, 1, 1)

    for i, values in enumerate(data):
        t = timestamps[i] if timestamps and i < len(timestamps) else t0 + timedelta(seconds=i)
        monitor.update(values, now=t)
        snapshots.append(monitor.status())

    return monitor, snapshots


def replay_csv(
    parser: Parser,
    path: str,
    timestamp_column: Optional[str] = None,
    window: int = 100,
    drift_window: Optional[int] = None,
    anomaly_window: Optional[int] = None,
    correlation_window: Optional[int] = None,
    **kwargs,
) -> tuple[Monitor, list[dict]]:
    """
    Replay from a CSV file.

    The CSV should have one column per component, matching the parser's
    baselines. An optional timestamp column provides timing.

    Requires no external dependencies — uses stdlib csv module.
    """
    import csv
    from datetime import timedelta

    rows: list[dict[str, float]] = []
    timestamps: list[datetime] = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = {}
            for key, val in row.items():
                if key == timestamp_column:
                    try:
                        timestamps.append(datetime.fromisoformat(val))
                    except (ValueError, TypeError):
                        pass
                    continue
                try:
                    values[key] = float(val)
                except (ValueError, TypeError):
                    continue
            if values:
                rows.append(values)

    return replay(
        parser, rows, timestamps or None, window,
        drift_window=drift_window,
        anomaly_window=anomaly_window,
        correlation_window=correlation_window,
        **kwargs,
    )
