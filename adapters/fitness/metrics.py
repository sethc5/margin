"""
Personal fitness metrics as margin observations.

Standard ranges for healthy adults. Override for age, sex, fitness level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence
from adapters.healthcare.vitals import BandThresholds, classify_band


@dataclass
class FitnessMetric:
    name: str
    display_name: str
    thresholds: Optional[Thresholds] = None
    band: Optional[BandThresholds] = None
    unit: str = ""


FITNESS_METRICS: dict[str, FitnessMetric] = {
    "resting_hr": FitnessMetric(
        name="resting_hr", display_name="Resting Heart Rate",
        band=BandThresholds(
            normal_low=50, normal_high=80,
            critical_low=38, critical_high=100,
            baseline=62, unit="bpm",
        ),
    ),
    "hrv": FitnessMetric(
        name="hrv", display_name="Heart Rate Variability",
        thresholds=Thresholds(intact=30.0, ablated=15.0, higher_is_better=True),
        unit="ms RMSSD",
    ),
    "sleep_hours": FitnessMetric(
        name="sleep_hours", display_name="Sleep Duration",
        band=BandThresholds(
            normal_low=7.0, normal_high=9.0,
            critical_low=4.0, critical_high=12.0,
            baseline=8.0, unit="hours",
        ),
    ),
    "sleep_quality": FitnessMetric(
        name="sleep_quality", display_name="Sleep Quality Score",
        thresholds=Thresholds(intact=70.0, ablated=40.0, higher_is_better=True),
        unit="score",
    ),
    "steps": FitnessMetric(
        name="steps", display_name="Daily Steps",
        thresholds=Thresholds(intact=7000.0, ablated=2000.0, higher_is_better=True),
        unit="steps",
    ),
    "stress": FitnessMetric(
        name="stress", display_name="Stress Score",
        thresholds=Thresholds(intact=40.0, ablated=80.0, higher_is_better=False),
        unit="score",
    ),
    "body_battery": FitnessMetric(
        name="body_battery", display_name="Body Battery / Recovery",
        thresholds=Thresholds(intact=50.0, ablated=15.0, higher_is_better=True),
        unit="score",
    ),
    "spo2_overnight": FitnessMetric(
        name="spo2_overnight", display_name="Overnight SpO2",
        band=BandThresholds(
            normal_low=95.0, normal_high=100.0,
            critical_low=90.0, critical_high=100.0,
            baseline=97.0, unit="%",
        ),
    ),
}


def parse_fitness(
    readings: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    metrics: Optional[dict[str, FitnessMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    from margin.health import classify

    defs = metrics or FITNESS_METRICS
    observations = {}
    for name, value in readings.items():
        metric = defs.get(name)
        if metric is None:
            continue
        if metric.band:
            health = classify_band(value, metric.band, confidence)
            higher_is_better = value < metric.band.baseline
            baseline = metric.band.baseline
        elif metric.thresholds:
            health = classify(value, confidence, thresholds=metric.thresholds)
            higher_is_better = metric.thresholds.higher_is_better
            baseline = (metric.thresholds.intact + metric.thresholds.ablated) / 2
        else:
            continue
        observations[name] = Observation(
            name=name, health=health, value=value, baseline=baseline,
            confidence=confidence, higher_is_better=higher_is_better,
            measured_at=measured_at,
        )
    return observations


def daily_expression(
    readings: dict[str, float],
    user_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    metrics: Optional[dict[str, FitnessMetric]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_fitness(readings, confidence, metrics, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=user_id,
    )
