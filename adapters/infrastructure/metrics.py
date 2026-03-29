"""
Infrastructure metrics as margin observations.

Standard thresholds for server and service health.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class InfraMetric:
    """Threshold profile for an infrastructure metric."""
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


INFRA_METRICS: dict[str, InfraMetric] = {
    # ── Lower is better ──
    "cpu_usage": InfraMetric(
        name="cpu_usage", display_name="CPU Usage",
        thresholds=Thresholds(intact=60.0, ablated=95.0, higher_is_better=False),
        baseline=30.0, unit="%",
    ),
    "memory_usage": InfraMetric(
        name="memory_usage", display_name="Memory Usage",
        thresholds=Thresholds(intact=70.0, ablated=95.0, higher_is_better=False),
        baseline=45.0, unit="%",
    ),
    "disk_usage": InfraMetric(
        name="disk_usage", display_name="Disk Usage",
        thresholds=Thresholds(intact=75.0, ablated=95.0, higher_is_better=False),
        baseline=40.0, unit="%",
    ),
    "error_rate": InfraMetric(
        name="error_rate", display_name="Error Rate",
        thresholds=Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        baseline=0.001, unit="ratio",
    ),
    "p99_latency": InfraMetric(
        name="p99_latency", display_name="P99 Latency",
        thresholds=Thresholds(intact=200.0, ablated=2000.0, higher_is_better=False),
        baseline=50.0, unit="ms",
    ),
    "p50_latency": InfraMetric(
        name="p50_latency", display_name="P50 Latency",
        thresholds=Thresholds(intact=50.0, ablated=500.0, higher_is_better=False),
        baseline=15.0, unit="ms",
    ),

    # ── Higher is better ──
    "uptime": InfraMetric(
        name="uptime", display_name="Uptime",
        thresholds=Thresholds(intact=0.999, ablated=0.95, higher_is_better=True),
        baseline=0.9999, unit="ratio",
    ),
    "throughput": InfraMetric(
        name="throughput", display_name="Throughput",
        thresholds=Thresholds(intact=1000.0, ablated=200.0, higher_is_better=True),
        baseline=2000.0, unit="rps",
    ),
    "available_connections": InfraMetric(
        name="available_connections", display_name="Available Connections",
        thresholds=Thresholds(intact=50.0, ablated=5.0, higher_is_better=True),
        baseline=200.0, unit="count",
    ),
    "cache_hit_rate": InfraMetric(
        name="cache_hit_rate", display_name="Cache Hit Rate",
        thresholds=Thresholds(intact=0.80, ablated=0.40, higher_is_better=True),
        baseline=0.95, unit="ratio",
    ),
}


def parse_metrics(
    readings: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    metrics: Optional[dict[str, InfraMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse infrastructure metric readings into margin Observations.

    Args:
        readings: {"cpu_usage": 78, "error_rate": 0.03, "throughput": 1500}
        confidence: measurement confidence
        metrics: override metric definitions
        measured_at: timestamp
    """
    from margin.health import classify

    defs = metrics or INFRA_METRICS
    observations = {}
    for name, value in readings.items():
        metric = defs.get(name)
        if metric is None:
            continue
        health = classify(value, confidence, thresholds=metric.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value, baseline=metric.baseline,
            confidence=confidence, higher_is_better=metric.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def service_expression(
    readings: dict[str, float],
    service_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    metrics: Optional[dict[str, InfraMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a service-wide Expression from metric readings."""
    obs = parse_metrics(readings, confidence, metrics, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=service_id,
    )
