"""
Data quality metrics as margin observations.

Column-level and dataset-level quality checks for any tabular pipeline.
No pandas dependency — pass in the metrics you've already computed.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class DQMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


DQ_METRICS: dict[str, DQMetric] = {
    "completeness": DQMetric(
        name="completeness", display_name="Completeness (non-null ratio)",
        thresholds=Thresholds(intact=0.99, ablated=0.80, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "null_rate": DQMetric(
        name="null_rate", display_name="Null Rate",
        thresholds=Thresholds(intact=0.01, ablated=0.20, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "duplicate_rate": DQMetric(
        name="duplicate_rate", display_name="Duplicate Rate",
        thresholds=Thresholds(intact=0.001, ablated=0.05, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "schema_match": DQMetric(
        name="schema_match", display_name="Schema Match",
        thresholds=Thresholds(intact=1.0, ablated=0.90, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "row_count_ratio": DQMetric(
        name="row_count_ratio", display_name="Row Count vs Expected",
        thresholds=Thresholds(intact=0.90, ablated=0.50, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "freshness_hours": DQMetric(
        name="freshness_hours", display_name="Data Freshness",
        thresholds=Thresholds(intact=1.0, ablated=24.0, higher_is_better=False),
        baseline=0.25, unit="hours",
    ),
    "value_drift": DQMetric(
        name="value_drift", display_name="Value Distribution Drift",
        thresholds=Thresholds(intact=0.05, ablated=0.30, higher_is_better=False),
        baseline=0.01, unit="KL divergence",
    ),
    "outlier_rate": DQMetric(
        name="outlier_rate", display_name="Outlier Rate",
        thresholds=Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
        baseline=0.005, unit="ratio",
    ),
    "type_error_rate": DQMetric(
        name="type_error_rate", display_name="Type Error Rate",
        thresholds=Thresholds(intact=0.0, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
}


def parse_quality(
    metrics: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, DQMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify
    d = defs or DQ_METRICS
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


def pipeline_expression(
    metrics: dict[str, float],
    pipeline_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, DQMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_quality(metrics, confidence, defs, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=pipeline_id,
    )
