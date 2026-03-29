"""
Task queue health as margin observations.

Standard thresholds for Celery, RQ, Dramatiq, or any task queue.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class TaskMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


TASK_METRICS: dict[str, TaskMetric] = {
    "queue_depth": TaskMetric(
        name="queue_depth", display_name="Queue Depth",
        thresholds=Thresholds(intact=100.0, ablated=10000.0, higher_is_better=False),
        baseline=20.0, unit="tasks",
    ),
    "worker_utilization": TaskMetric(
        name="worker_utilization", display_name="Worker Utilization",
        thresholds=Thresholds(intact=0.80, ablated=0.99, higher_is_better=False),
        baseline=0.50, unit="ratio",
    ),
    "failure_rate": TaskMetric(
        name="failure_rate", display_name="Task Failure Rate",
        thresholds=Thresholds(intact=0.02, ablated=0.15, higher_is_better=False),
        baseline=0.005, unit="ratio",
    ),
    "retry_rate": TaskMetric(
        name="retry_rate", display_name="Retry Rate",
        thresholds=Thresholds(intact=0.05, ablated=0.25, higher_is_better=False),
        baseline=0.01, unit="ratio",
    ),
    "task_latency_p50": TaskMetric(
        name="task_latency_p50", display_name="Task P50 Latency",
        thresholds=Thresholds(intact=5.0, ablated=60.0, higher_is_better=False),
        baseline=1.0, unit="seconds",
    ),
    "task_latency_p99": TaskMetric(
        name="task_latency_p99", display_name="Task P99 Latency",
        thresholds=Thresholds(intact=30.0, ablated=300.0, higher_is_better=False),
        baseline=5.0, unit="seconds",
    ),
    "throughput": TaskMetric(
        name="throughput", display_name="Tasks/sec Completed",
        thresholds=Thresholds(intact=10.0, ablated=1.0, higher_is_better=True),
        baseline=50.0, unit="tasks/sec",
    ),
    "dead_letter_rate": TaskMetric(
        name="dead_letter_rate", display_name="Dead Letter Rate",
        thresholds=Thresholds(intact=0.001, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
}


def parse_queue(
    metrics: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, TaskMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify
    d = defs or TASK_METRICS
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


def queue_expression(
    metrics: dict[str, float],
    queue_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, TaskMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_queue(metrics, confidence, defs, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=queue_id,
    )
