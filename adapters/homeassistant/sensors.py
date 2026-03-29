"""
Home Assistant sensor entities as margin observations.

Standard thresholds for common HA device classes.
Override per-sensor for custom ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Health, Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence
from adapters.healthcare.vitals import BandThresholds, classify_band


@dataclass
class SensorProfile:
    """Threshold profile for a Home Assistant device class."""
    name: str
    display_name: str
    thresholds: Optional[Thresholds] = None
    band: Optional[BandThresholds] = None  # for sensors with both-direction ranges
    unit: str = ""


SENSOR_PROFILES: dict[str, SensorProfile] = {
    # ── Higher is better ──
    "battery": SensorProfile(
        name="battery", display_name="Battery",
        thresholds=Thresholds(intact=30.0, ablated=10.0, higher_is_better=True),
        unit="%",
    ),
    "signal_strength": SensorProfile(
        name="signal_strength", display_name="Signal Strength",
        thresholds=Thresholds(intact=-70.0, ablated=-90.0, higher_is_better=True),
        unit="dBm",
    ),
    "solar_production": SensorProfile(
        name="solar_production", display_name="Solar Production",
        thresholds=Thresholds(intact=500.0, ablated=100.0, higher_is_better=True),
        unit="W",
    ),

    # ── Lower is better ──
    "power_consumption": SensorProfile(
        name="power_consumption", display_name="Power Consumption",
        thresholds=Thresholds(intact=3000.0, ablated=8000.0, higher_is_better=False),
        unit="W",
    ),
    "humidity_high": SensorProfile(
        name="humidity_high", display_name="Humidity (mold risk)",
        thresholds=Thresholds(intact=60.0, ablated=80.0, higher_is_better=False),
        unit="%",
    ),

    # ── Band (both directions) ──
    "temperature_indoor": SensorProfile(
        name="temperature_indoor", display_name="Indoor Temperature",
        band=BandThresholds(
            normal_low=18.0, normal_high=24.0,
            critical_low=12.0, critical_high=30.0,
            baseline=21.0, unit="°C",
        ),
    ),
    "temperature_outdoor": SensorProfile(
        name="temperature_outdoor", display_name="Outdoor Temperature",
        band=BandThresholds(
            normal_low=-5.0, normal_high=35.0,
            critical_low=-20.0, critical_high=45.0,
            baseline=15.0, unit="°C",
        ),
    ),
    "humidity_indoor": SensorProfile(
        name="humidity_indoor", display_name="Indoor Humidity",
        band=BandThresholds(
            normal_low=30.0, normal_high=60.0,
            critical_low=20.0, critical_high=80.0,
            baseline=45.0, unit="%",
        ),
    ),
}


def _sensor_to_observation(
    profile: SensorProfile,
    value: float,
    entity_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    measured_at: Optional[datetime] = None,
) -> Observation:
    """Classify a single sensor reading."""
    name = entity_id or profile.name

    if profile.band:
        health = classify_band(value, profile.band, confidence)
        higher_is_better = value < profile.band.baseline
        baseline = profile.band.baseline
    elif profile.thresholds:
        from margin.health import classify
        health = classify(value, confidence, thresholds=profile.thresholds)
        higher_is_better = profile.thresholds.higher_is_better
        baseline = (profile.thresholds.intact + profile.thresholds.ablated) / 2
    else:
        return Observation(name=name, health=Health.OOD, value=value, baseline=value,
                           confidence=Confidence.INDETERMINATE, measured_at=measured_at)

    return Observation(
        name=name, health=health, value=value, baseline=baseline,
        confidence=confidence, higher_is_better=higher_is_better,
        measured_at=measured_at,
    )


def parse_sensors(
    readings: dict[str, tuple[str, float]],
    confidence: Confidence = Confidence.MODERATE,
    profiles: Optional[dict[str, SensorProfile]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse HA sensor readings into margin Observations.

    Args:
        readings: {entity_id: (device_class, value)}
            e.g. {"sensor.living_room_temp": ("temperature_indoor", 22.5),
                  "sensor.battery_front_door": ("battery", 45)}
        confidence: measurement confidence
        profiles: override profiles (defaults to SENSOR_PROFILES)
        measured_at: timestamp
    """
    profs = profiles or SENSOR_PROFILES
    observations = {}
    for entity_id, (device_class, value) in readings.items():
        profile = profs.get(device_class)
        if profile is None:
            continue
        observations[entity_id] = _sensor_to_observation(
            profile, value, entity_id, confidence, measured_at,
        )
    return observations


def home_expression(
    readings: dict[str, tuple[str, float]],
    home_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    profiles: Optional[dict[str, SensorProfile]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a home-wide Expression from all sensor readings."""
    obs = parse_sensors(readings, confidence, profiles, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=home_id,
    )
