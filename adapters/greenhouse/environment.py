"""
Greenhouse growing environment as margin observations.

Standard ranges for indoor growing / greenhouse cultivation.
Override for specific crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from adapters.healthcare.vitals import BandThresholds, classify_band


@dataclass
class GrowParam:
    name: str
    display_name: str
    band: BandThresholds


GROW_PARAMS: dict[str, GrowParam] = {
    "air_temp": GrowParam(
        name="air_temp", display_name="Air Temperature",
        band=BandThresholds(
            normal_low=18.0, normal_high=28.0,
            critical_low=10.0, critical_high=38.0,
            baseline=24.0, unit="°C",
        ),
    ),
    "humidity": GrowParam(
        name="humidity", display_name="Relative Humidity",
        band=BandThresholds(
            normal_low=40.0, normal_high=70.0,
            critical_low=25.0, critical_high=85.0,
            baseline=55.0, unit="%",
        ),
    ),
    "soil_moisture": GrowParam(
        name="soil_moisture", display_name="Soil Moisture",
        band=BandThresholds(
            normal_low=30.0, normal_high=70.0,
            critical_low=15.0, critical_high=85.0,
            baseline=50.0, unit="%",
        ),
    ),
    "soil_ph": GrowParam(
        name="soil_ph", display_name="Soil pH",
        band=BandThresholds(
            normal_low=5.5, normal_high=7.0,
            critical_low=4.5, critical_high=8.0,
            baseline=6.2, unit="pH",
        ),
    ),
    "co2": GrowParam(
        name="co2", display_name="CO₂ Concentration",
        band=BandThresholds(
            normal_low=400.0, normal_high=1200.0,
            critical_low=200.0, critical_high=2000.0,
            baseline=800.0, unit="ppm",
        ),
    ),
    "light_dli": GrowParam(
        name="light_dli", display_name="Daily Light Integral",
        band=BandThresholds(
            normal_low=12.0, normal_high=40.0,
            critical_low=6.0, critical_high=60.0,
            baseline=25.0, unit="mol/m²/d",
        ),
    ),
    "ec": GrowParam(
        name="ec", display_name="Electrical Conductivity (nutrients)",
        band=BandThresholds(
            normal_low=1.0, normal_high=2.5,
            critical_low=0.5, critical_high=4.0,
            baseline=1.8, unit="mS/cm",
        ),
    ),
    "vpd": GrowParam(
        name="vpd", display_name="Vapor Pressure Deficit",
        band=BandThresholds(
            normal_low=0.8, normal_high=1.2,
            critical_low=0.4, critical_high=2.0,
            baseline=1.0, unit="kPa",
        ),
    ),
}


def parse_environment(
    readings: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, GrowParam]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    defs = params or GROW_PARAMS
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


def grow_expression(
    readings: dict[str, float],
    zone_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, GrowParam]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    obs = parse_environment(readings, confidence, params, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=zone_id,
    )
