"""
HTTP endpoint health as margin observations.

Metrics per endpoint or service-wide. Standard thresholds for web APIs.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class EndpointMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


ENDPOINT_METRICS: dict[str, EndpointMetric] = {
    "p50_ms": EndpointMetric(
        name="p50_ms", display_name="P50 Latency",
        thresholds=Thresholds(intact=100.0, ablated=1000.0, higher_is_better=False),
        baseline=30.0, unit="ms",
    ),
    "p99_ms": EndpointMetric(
        name="p99_ms", display_name="P99 Latency",
        thresholds=Thresholds(intact=500.0, ablated=5000.0, higher_is_better=False),
        baseline=150.0, unit="ms",
    ),
    "error_rate": EndpointMetric(
        name="error_rate", display_name="Error Rate (5xx)",
        thresholds=Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        baseline=0.002, unit="ratio",
    ),
    "rps": EndpointMetric(
        name="rps", display_name="Requests/sec",
        thresholds=Thresholds(intact=50.0, ablated=5.0, higher_is_better=True),
        baseline=200.0, unit="rps",
    ),
    "queue_depth": EndpointMetric(
        name="queue_depth", display_name="Request Queue Depth",
        thresholds=Thresholds(intact=10.0, ablated=100.0, higher_is_better=False),
        baseline=2.0, unit="requests",
    ),
    "timeout_rate": EndpointMetric(
        name="timeout_rate", display_name="Timeout Rate",
        thresholds=Thresholds(intact=0.005, ablated=0.05, higher_is_better=False),
        baseline=0.001, unit="ratio",
    ),
    "success_rate": EndpointMetric(
        name="success_rate", display_name="Success Rate (2xx)",
        thresholds=Thresholds(intact=0.99, ablated=0.90, higher_is_better=True),
        baseline=0.998, unit="ratio",
    ),
}


def parse_endpoint(
    metrics: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, EndpointMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify
    d = defs or ENDPOINT_METRICS
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


def endpoint_expression(
    metrics: dict[str, float],
    endpoint: str = "",
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, EndpointMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_endpoint(metrics, confidence, defs, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=endpoint,
    )
