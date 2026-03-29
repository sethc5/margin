"""
Test suite health as margin observations.

Metrics computed from pytest results — pass these after your CI run.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class TestMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


TEST_METRICS: dict[str, TestMetric] = {
    "pass_rate": TestMetric(
        name="pass_rate", display_name="Pass Rate",
        thresholds=Thresholds(intact=0.98, ablated=0.80, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "flake_rate": TestMetric(
        name="flake_rate", display_name="Flake Rate",
        thresholds=Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "skip_rate": TestMetric(
        name="skip_rate", display_name="Skip Rate",
        thresholds=Thresholds(intact=0.05, ablated=0.20, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "duration_seconds": TestMetric(
        name="duration_seconds", display_name="Suite Duration",
        thresholds=Thresholds(intact=120.0, ablated=600.0, higher_is_better=False),
        baseline=60.0, unit="seconds",
    ),
    "coverage": TestMetric(
        name="coverage", display_name="Code Coverage",
        thresholds=Thresholds(intact=0.80, ablated=0.50, higher_is_better=True),
        baseline=0.90, unit="ratio",
    ),
    "coverage_delta": TestMetric(
        name="coverage_delta", display_name="Coverage Change",
        thresholds=Thresholds(intact=-0.01, ablated=-0.05, higher_is_better=True),
        baseline=0.0, unit="delta",
    ),
    "new_failures": TestMetric(
        name="new_failures", display_name="New Failures This Run",
        thresholds=Thresholds(intact=0.0, ablated=5.0, higher_is_better=False),
        baseline=0.0, unit="count",
    ),
    "mean_test_duration": TestMetric(
        name="mean_test_duration", display_name="Mean Test Duration",
        thresholds=Thresholds(intact=0.5, ablated=5.0, higher_is_better=False),
        baseline=0.1, unit="seconds",
    ),
}


def parse_suite(
    metrics: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, TestMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify
    d = defs or TEST_METRICS
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


def suite_expression(
    metrics: dict[str, float],
    suite_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    defs: Optional[dict[str, TestMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_suite(metrics, confidence, defs, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=suite_id,
    )
