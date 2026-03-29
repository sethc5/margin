"""
Database health metrics as margin observations.

Standard thresholds for connection pools, query performance, replication.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class DBMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


DB_METRICS: dict[str, DBMetric] = {
    "pool_usage": DBMetric(
        name="pool_usage", display_name="Connection Pool Usage",
        thresholds=Thresholds(intact=0.60, ablated=0.95, higher_is_better=False),
        baseline=0.30, unit="ratio",
    ),
    "pool_available": DBMetric(
        name="pool_available", display_name="Available Connections",
        thresholds=Thresholds(intact=5.0, ablated=1.0, higher_is_better=True),
        baseline=15.0, unit="connections",
    ),
    "query_latency_p50": DBMetric(
        name="query_latency_p50", display_name="Query P50 Latency",
        thresholds=Thresholds(intact=10.0, ablated=200.0, higher_is_better=False),
        baseline=3.0, unit="ms",
    ),
    "query_latency_p99": DBMetric(
        name="query_latency_p99", display_name="Query P99 Latency",
        thresholds=Thresholds(intact=100.0, ablated=2000.0, higher_is_better=False),
        baseline=25.0, unit="ms",
    ),
    "slow_query_rate": DBMetric(
        name="slow_query_rate", display_name="Slow Query Rate",
        thresholds=Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        baseline=0.002, unit="ratio",
    ),
    "replication_lag": DBMetric(
        name="replication_lag", display_name="Replication Lag",
        thresholds=Thresholds(intact=1.0, ablated=30.0, higher_is_better=False),
        baseline=0.1, unit="seconds",
    ),
    "deadlock_rate": DBMetric(
        name="deadlock_rate", display_name="Deadlock Rate",
        thresholds=Thresholds(intact=0.001, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="per_second",
    ),
    "cache_hit_rate": DBMetric(
        name="cache_hit_rate", display_name="Buffer Cache Hit Rate",
        thresholds=Thresholds(intact=0.95, ablated=0.80, higher_is_better=True),
        baseline=0.99, unit="ratio",
    ),
    "disk_usage": DBMetric(
        name="disk_usage", display_name="Storage Usage",
        thresholds=Thresholds(intact=0.70, ablated=0.90, higher_is_better=False),
        baseline=0.40, unit="ratio",
    ),
    "active_transactions": DBMetric(
        name="active_transactions", display_name="Active Transactions",
        thresholds=Thresholds(intact=50.0, ablated=200.0, higher_is_better=False),
        baseline=10.0, unit="count",
    ),
}


def parse_db(
    metrics: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, DBMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify
    d = defs or DB_METRICS
    observations = {}
    for name, value in metrics.items():
        m = d.get(name)
        if m is None:
            continue
        health = classify(value, confidence, thresholds=m.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value, baseline=m.baseline,
            confidence=confidence, higher_is_better=m.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def db_expression(
    metrics: dict[str, float],
    db_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, DBMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_db(metrics, confidence, defs, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=db_id,
    )
