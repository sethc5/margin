"""
Aquarium water parameters as margin observations.

Standard freshwater tropical ranges. Override for saltwater, planted,
cichlid, or species-specific requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from adapters.healthcare.vitals import BandThresholds, classify_band


@dataclass
class WaterParam:
    name: str
    display_name: str
    band: BandThresholds


WATER_PARAMS: dict[str, WaterParam] = {
    "ph": WaterParam(
        name="ph", display_name="pH",
        band=BandThresholds(
            normal_low=6.5, normal_high=7.5,
            critical_low=5.5, critical_high=8.5,
            baseline=7.0, unit="pH",
        ),
    ),
    "ammonia": WaterParam(
        name="ammonia", display_name="Ammonia (NH₃)",
        band=BandThresholds(
            normal_low=0.0, normal_high=0.02,
            critical_low=0.0, critical_high=0.05,
            baseline=0.0, unit="ppm",
        ),
    ),
    "nitrite": WaterParam(
        name="nitrite", display_name="Nitrite (NO₂)",
        band=BandThresholds(
            normal_low=0.0, normal_high=0.1,
            critical_low=0.0, critical_high=0.5,
            baseline=0.0, unit="ppm",
        ),
    ),
    "nitrate": WaterParam(
        name="nitrate", display_name="Nitrate (NO₃)",
        band=BandThresholds(
            normal_low=0.0, normal_high=40.0,
            critical_low=0.0, critical_high=80.0,
            baseline=10.0, unit="ppm",
        ),
    ),
    "temperature": WaterParam(
        name="temperature", display_name="Water Temperature",
        band=BandThresholds(
            normal_low=24.0, normal_high=28.0,
            critical_low=20.0, critical_high=32.0,
            baseline=26.0, unit="°C",
        ),
    ),
    "kh": WaterParam(
        name="kh", display_name="Carbonate Hardness",
        band=BandThresholds(
            normal_low=3.0, normal_high=8.0,
            critical_low=1.0, critical_high=12.0,
            baseline=5.0, unit="dKH",
        ),
    ),
    "gh": WaterParam(
        name="gh", display_name="General Hardness",
        band=BandThresholds(
            normal_low=4.0, normal_high=12.0,
            critical_low=2.0, critical_high=20.0,
            baseline=8.0, unit="dGH",
        ),
    ),
    "salinity": WaterParam(
        name="salinity", display_name="Salinity",
        band=BandThresholds(
            normal_low=1.020, normal_high=1.025,
            critical_low=1.015, critical_high=1.030,
            baseline=1.023, unit="sg",
        ),
    ),
}


def parse_water(
    readings: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, WaterParam]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    defs = params or WATER_PARAMS
    observations = {}
    for name, value in readings.items():
        param = defs.get(name)
        if param is None:
            continue
        health = classify_band(value, param.band, confidence)
        higher_is_better = value < param.band.baseline
        observations[name] = Observation(
            name=name, health=health, value=value, baseline=param.band.baseline,
            confidence=confidence, higher_is_better=higher_is_better,
            measured_at=measured_at,
        )
    return observations


def tank_expression(
    readings: dict[str, float],
    tank_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, WaterParam]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_water(readings, confidence, params, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=tank_id,
    )
