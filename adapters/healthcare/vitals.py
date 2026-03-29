"""
Clinical vital signs as margin observations.

Each vital has a normal range (band) — both too high and too low are
unhealthy. This is modeled as two thresholds per vital: a low boundary
(higher_is_better=True) and a high boundary (higher_is_better=False).

Ranges from WHO, AHA, and standard clinical references for resting adults.
Pediatric, geriatric, and condition-specific ranges should override these.

NOT A MEDICAL DEVICE. This is a typed vocabulary for clinical data.
Clinical decisions require licensed practitioners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from margin.health import Health, Thresholds, classify, SEVERITY
from margin.observation import Observation, Expression
from margin.confidence import Confidence
from margin.composite import CompositeObservation, AggregateStrategy


@dataclass
class BandThresholds:
    """
    A clinical range with both low and high boundaries.

    normal_low:    lower end of normal range (below = too low)
    normal_high:   upper end of normal range (above = too high)
    critical_low:  life-threatening low value
    critical_high: life-threatening high value
    baseline:      expected healthy value (midpoint of normal range)
    unit:          measurement unit for display
    """
    normal_low: float
    normal_high: float
    critical_low: float
    critical_high: float
    baseline: float
    unit: str = ""

    @property
    def low_thresholds(self) -> Thresholds:
        """Thresholds for the low boundary: higher_is_better=True."""
        return Thresholds(
            intact=self.normal_low,
            ablated=self.critical_low,
            higher_is_better=True,
        )

    @property
    def high_thresholds(self) -> Thresholds:
        """Thresholds for the high boundary: higher_is_better=False."""
        return Thresholds(
            intact=self.normal_high,
            ablated=self.critical_high,
            higher_is_better=False,
        )


def classify_band(
    value: float,
    band: BandThresholds,
    confidence: Confidence = Confidence.HIGH,
    correcting: bool = False,
) -> Health:
    """
    Classify a value against a band (both directions).
    Returns the worse of the two boundary classifications.
    """
    low_health = classify(value, confidence, correcting, band.low_thresholds)
    high_health = classify(value, confidence, correcting, band.high_thresholds)
    # Return the worse one
    if SEVERITY.get(high_health, 0) > SEVERITY.get(low_health, 0):
        return high_health
    return low_health


@dataclass
class VitalSign:
    """Definition of a clinical vital sign with normal ranges."""
    name: str
    display_name: str
    band: BandThresholds
    category: str = "vital"  # "vital", "lab", "derived"


# -----------------------------------------------------------------------
# Standard adult resting vital signs
# WHO / AHA / standard clinical references
# -----------------------------------------------------------------------

VITAL_SIGNS: dict[str, VitalSign] = {
    "hr": VitalSign(
        name="hr",
        display_name="Heart Rate",
        band=BandThresholds(
            normal_low=60, normal_high=100,
            critical_low=40, critical_high=150,
            baseline=72, unit="bpm",
        ),
    ),
    "sbp": VitalSign(
        name="sbp",
        display_name="Systolic BP",
        band=BandThresholds(
            normal_low=90, normal_high=120,
            critical_low=70, critical_high=180,
            baseline=115, unit="mmHg",
        ),
    ),
    "dbp": VitalSign(
        name="dbp",
        display_name="Diastolic BP",
        band=BandThresholds(
            normal_low=60, normal_high=80,
            critical_low=40, critical_high=120,
            baseline=75, unit="mmHg",
        ),
    ),
    "spo2": VitalSign(
        name="spo2",
        display_name="SpO2",
        band=BandThresholds(
            normal_low=95, normal_high=100,
            critical_low=90, critical_high=100,
            baseline=98, unit="%",
        ),
    ),
    "temp": VitalSign(
        name="temp",
        display_name="Temperature",
        band=BandThresholds(
            normal_low=36.1, normal_high=37.2,
            critical_low=35.0, critical_high=40.0,
            baseline=36.8, unit="°C",
        ),
    ),
    "rr": VitalSign(
        name="rr",
        display_name="Respiratory Rate",
        band=BandThresholds(
            normal_low=12, normal_high=20,
            critical_low=8, critical_high=30,
            baseline=16, unit="/min",
        ),
    ),
    "glucose": VitalSign(
        name="glucose",
        display_name="Blood Glucose",
        band=BandThresholds(
            normal_low=70, normal_high=100,
            critical_low=54, critical_high=250,
            baseline=90, unit="mg/dL",
        ),
        category="lab",
    ),
    "map": VitalSign(
        name="map",
        display_name="Mean Arterial Pressure",
        band=BandThresholds(
            normal_low=70, normal_high=100,
            critical_low=60, critical_high=110,
            baseline=85, unit="mmHg",
        ),
        category="derived",
    ),
}


# -----------------------------------------------------------------------
# Parsing: raw vitals → margin Observations
# -----------------------------------------------------------------------

def _vital_to_observation(
    vital: VitalSign,
    value: float,
    confidence: Confidence = Confidence.MODERATE,
    measured_at: Optional[datetime] = None,
) -> Observation:
    """Convert a single vital reading to a margin Observation."""
    health = classify_band(value, vital.band, confidence)

    # Sigma: use band baseline, polarity depends on which side we're on
    if value < vital.band.baseline:
        higher_is_better = True
    else:
        higher_is_better = False

    return Observation(
        name=vital.name,
        health=health,
        value=value,
        baseline=vital.band.baseline,
        confidence=confidence,
        higher_is_better=higher_is_better,
        measured_at=measured_at,
    )


def parse_vitals(
    readings: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    vital_defs: Optional[dict[str, VitalSign]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse a dict of vital sign readings into margin Observations.

    Args:
        readings: {"hr": 88, "sbp": 135, "spo2": 94, ...}
        confidence: measurement confidence (device-dependent)
        vital_defs: override vital definitions (defaults to VITAL_SIGNS)
        measured_at: when these readings were taken
    """
    defs = vital_defs or VITAL_SIGNS
    observations = {}
    for name, value in readings.items():
        vital = defs.get(name)
        if vital is None:
            continue
        observations[name] = _vital_to_observation(vital, value, confidence, measured_at)
    return observations


def patient_expression(
    readings: dict[str, float],
    patient_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    vital_defs: Optional[dict[str, VitalSign]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """
    Build a complete patient Expression from vital sign readings.

    Returns a margin Expression where each observation is a vital sign
    classified against its clinical range. Net confidence is the weakest
    reading.
    """
    obs_dict = parse_vitals(readings, confidence, vital_defs, measured_at)
    observations = list(obs_dict.values())

    net_conf = min(
        (o.confidence for o in observations),
        default=Confidence.INDETERMINATE,
    )

    return Expression(
        observations=observations,
        confidence=net_conf,
        label=patient_id,
    )
