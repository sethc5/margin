"""
Weather conditions as margin observations.

Each weather parameter has bands that depend on context — what's healthy
for agriculture is different from what's healthy for aviation or construction.

Profiles define threshold sets for different use cases. The default profile
is general outdoor activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from adapters.healthcare.vitals import BandThresholds, classify_band


@dataclass
class WeatherParam:
    name: str
    display_name: str
    band: BandThresholds


@dataclass
class WeatherProfile:
    """A named set of weather thresholds for a specific context."""
    name: str
    description: str
    params: dict[str, WeatherParam]


# -----------------------------------------------------------------------
# Standard parameters — general outdoor activity
# -----------------------------------------------------------------------

_GENERAL_PARAMS: dict[str, WeatherParam] = {
    "temperature": WeatherParam(
        name="temperature", display_name="Temperature",
        band=BandThresholds(
            normal_low=10.0, normal_high=32.0,
            critical_low=-5.0, critical_high=42.0,
            baseline=22.0, unit="°C",
        ),
    ),
    "humidity": WeatherParam(
        name="humidity", display_name="Relative Humidity",
        band=BandThresholds(
            normal_low=30.0, normal_high=70.0,
            critical_low=15.0, critical_high=90.0,
            baseline=50.0, unit="%",
        ),
    ),
    "wind_speed": WeatherParam(
        name="wind_speed", display_name="Wind Speed",
        band=BandThresholds(
            normal_low=0.0, normal_high=30.0,
            critical_low=0.0, critical_high=60.0,
            baseline=10.0, unit="km/h",
        ),
    ),
    "pressure": WeatherParam(
        name="pressure", display_name="Barometric Pressure",
        band=BandThresholds(
            normal_low=1005.0, normal_high=1025.0,
            critical_low=980.0, critical_high=1040.0,
            baseline=1013.0, unit="hPa",
        ),
    ),
    "uv_index": WeatherParam(
        name="uv_index", display_name="UV Index",
        band=BandThresholds(
            normal_low=0.0, normal_high=6.0,
            critical_low=0.0, critical_high=11.0,
            baseline=3.0, unit="index",
        ),
    ),
    "aqi": WeatherParam(
        name="aqi", display_name="Air Quality Index",
        band=BandThresholds(
            normal_low=0.0, normal_high=50.0,
            critical_low=0.0, critical_high=150.0,
            baseline=25.0, unit="AQI",
        ),
    ),
    "visibility": WeatherParam(
        name="visibility", display_name="Visibility",
        band=BandThresholds(
            normal_low=5.0, normal_high=50.0,
            critical_low=1.0, critical_high=50.0,
            baseline=15.0, unit="km",
        ),
    ),
    "precipitation_rate": WeatherParam(
        name="precipitation_rate", display_name="Precipitation Rate",
        band=BandThresholds(
            normal_low=0.0, normal_high=5.0,
            critical_low=0.0, critical_high=25.0,
            baseline=0.0, unit="mm/h",
        ),
    ),
    "heat_index": WeatherParam(
        name="heat_index", display_name="Heat Index",
        band=BandThresholds(
            normal_low=10.0, normal_high=32.0,
            critical_low=-10.0, critical_high=41.0,
            baseline=24.0, unit="°C",
        ),
    ),
    "wind_chill": WeatherParam(
        name="wind_chill", display_name="Wind Chill",
        band=BandThresholds(
            normal_low=5.0, normal_high=35.0,
            critical_low=-25.0, critical_high=40.0,
            baseline=18.0, unit="°C",
        ),
    ),
}

# -----------------------------------------------------------------------
# Agriculture profile — crop health focus
# -----------------------------------------------------------------------

_AGRICULTURE_PARAMS: dict[str, WeatherParam] = {
    "temperature": WeatherParam(
        name="temperature", display_name="Temperature",
        band=BandThresholds(
            normal_low=15.0, normal_high=30.0,
            critical_low=0.0, critical_high=40.0,
            baseline=24.0, unit="°C",
        ),
    ),
    "humidity": WeatherParam(
        name="humidity", display_name="Relative Humidity",
        band=BandThresholds(
            normal_low=40.0, normal_high=75.0,
            critical_low=20.0, critical_high=95.0,
            baseline=60.0, unit="%",
        ),
    ),
    "wind_speed": WeatherParam(
        name="wind_speed", display_name="Wind Speed",
        band=BandThresholds(
            normal_low=0.0, normal_high=25.0,
            critical_low=0.0, critical_high=50.0,
            baseline=8.0, unit="km/h",
        ),
    ),
    "soil_temp": WeatherParam(
        name="soil_temp", display_name="Soil Temperature",
        band=BandThresholds(
            normal_low=10.0, normal_high=30.0,
            critical_low=2.0, critical_high=38.0,
            baseline=20.0, unit="°C",
        ),
    ),
    "frost_risk": WeatherParam(
        name="frost_risk", display_name="Frost Risk (min temp forecast)",
        band=BandThresholds(
            normal_low=5.0, normal_high=40.0,
            critical_low=-5.0, critical_high=45.0,
            baseline=12.0, unit="°C",
        ),
    ),
    "precipitation_rate": WeatherParam(
        name="precipitation_rate", display_name="Precipitation Rate",
        band=BandThresholds(
            normal_low=0.0, normal_high=10.0,
            critical_low=0.0, critical_high=40.0,
            baseline=2.0, unit="mm/h",
        ),
    ),
    "evapotranspiration": WeatherParam(
        name="evapotranspiration", display_name="Evapotranspiration",
        band=BandThresholds(
            normal_low=1.0, normal_high=6.0,
            critical_low=0.5, critical_high=10.0,
            baseline=3.5, unit="mm/day",
        ),
    ),
}

# -----------------------------------------------------------------------
# Aviation profile — flight safety
# -----------------------------------------------------------------------

_AVIATION_PARAMS: dict[str, WeatherParam] = {
    "visibility": WeatherParam(
        name="visibility", display_name="Visibility",
        band=BandThresholds(
            normal_low=8.0, normal_high=50.0,
            critical_low=1.5, critical_high=50.0,
            baseline=15.0, unit="km",
        ),
    ),
    "ceiling": WeatherParam(
        name="ceiling", display_name="Cloud Ceiling",
        band=BandThresholds(
            normal_low=1000.0, normal_high=50000.0,
            critical_low=200.0, critical_high=50000.0,
            baseline=5000.0, unit="ft",
        ),
    ),
    "wind_speed": WeatherParam(
        name="wind_speed", display_name="Wind Speed",
        band=BandThresholds(
            normal_low=0.0, normal_high=35.0,
            critical_low=0.0, critical_high=55.0,
            baseline=15.0, unit="km/h",
        ),
    ),
    "crosswind": WeatherParam(
        name="crosswind", display_name="Crosswind Component",
        band=BandThresholds(
            normal_low=0.0, normal_high=20.0,
            critical_low=0.0, critical_high=35.0,
            baseline=5.0, unit="km/h",
        ),
    ),
    "wind_gust": WeatherParam(
        name="wind_gust", display_name="Wind Gusts",
        band=BandThresholds(
            normal_low=0.0, normal_high=45.0,
            critical_low=0.0, critical_high=75.0,
            baseline=10.0, unit="km/h",
        ),
    ),
    "precipitation_rate": WeatherParam(
        name="precipitation_rate", display_name="Precipitation Rate",
        band=BandThresholds(
            normal_low=0.0, normal_high=2.5,
            critical_low=0.0, critical_high=7.5,
            baseline=0.0, unit="mm/h",
        ),
    ),
    "temperature": WeatherParam(
        name="temperature", display_name="Temperature",
        band=BandThresholds(
            normal_low=-20.0, normal_high=40.0,
            critical_low=-40.0, critical_high=50.0,
            baseline=15.0, unit="°C",
        ),
    ),
}

# -----------------------------------------------------------------------
# Construction profile — OSHA-aligned
# -----------------------------------------------------------------------

_CONSTRUCTION_PARAMS: dict[str, WeatherParam] = {
    "temperature": WeatherParam(
        name="temperature", display_name="Temperature",
        band=BandThresholds(
            normal_low=5.0, normal_high=35.0,
            critical_low=-10.0, critical_high=40.0,
            baseline=22.0, unit="°C",
        ),
    ),
    "wind_speed": WeatherParam(
        name="wind_speed", display_name="Wind Speed",
        band=BandThresholds(
            normal_low=0.0, normal_high=40.0,
            critical_low=0.0, critical_high=65.0,
            baseline=10.0, unit="km/h",
        ),
    ),
    "wind_gust": WeatherParam(
        name="wind_gust", display_name="Wind Gusts",
        band=BandThresholds(
            normal_low=0.0, normal_high=50.0,
            critical_low=0.0, critical_high=80.0,
            baseline=15.0, unit="km/h",
        ),
    ),
    "visibility": WeatherParam(
        name="visibility", display_name="Visibility",
        band=BandThresholds(
            normal_low=3.0, normal_high=50.0,
            critical_low=0.5, critical_high=50.0,
            baseline=10.0, unit="km",
        ),
    ),
    "heat_index": WeatherParam(
        name="heat_index", display_name="Heat Index",
        band=BandThresholds(
            normal_low=5.0, normal_high=32.0,
            critical_low=-15.0, critical_high=40.0,
            baseline=24.0, unit="°C",
        ),
    ),
    "precipitation_rate": WeatherParam(
        name="precipitation_rate", display_name="Precipitation Rate",
        band=BandThresholds(
            normal_low=0.0, normal_high=4.0,
            critical_low=0.0, critical_high=15.0,
            baseline=0.0, unit="mm/h",
        ),
    ),
    "lightning_distance": WeatherParam(
        name="lightning_distance", display_name="Lightning Distance",
        band=BandThresholds(
            normal_low=15.0, normal_high=100.0,
            critical_low=5.0, critical_high=100.0,
            baseline=50.0, unit="km",
        ),
    ),
}

# -----------------------------------------------------------------------
# Public health profile — heat/cold/air quality warnings
# -----------------------------------------------------------------------

_PUBLIC_HEALTH_PARAMS: dict[str, WeatherParam] = {
    "heat_index": WeatherParam(
        name="heat_index", display_name="Heat Index",
        band=BandThresholds(
            normal_low=10.0, normal_high=27.0,
            critical_low=-10.0, critical_high=40.0,
            baseline=22.0, unit="°C",
        ),
    ),
    "wind_chill": WeatherParam(
        name="wind_chill", display_name="Wind Chill",
        band=BandThresholds(
            normal_low=0.0, normal_high=30.0,
            critical_low=-30.0, critical_high=40.0,
            baseline=15.0, unit="°C",
        ),
    ),
    "aqi": WeatherParam(
        name="aqi", display_name="Air Quality Index",
        band=BandThresholds(
            normal_low=0.0, normal_high=50.0,
            critical_low=0.0, critical_high=100.0,
            baseline=20.0, unit="AQI",
        ),
    ),
    "uv_index": WeatherParam(
        name="uv_index", display_name="UV Index",
        band=BandThresholds(
            normal_low=0.0, normal_high=5.0,
            critical_low=0.0, critical_high=8.0,
            baseline=3.0, unit="index",
        ),
    ),
    "pollen_index": WeatherParam(
        name="pollen_index", display_name="Pollen Index",
        band=BandThresholds(
            normal_low=0.0, normal_high=5.0,
            critical_low=0.0, critical_high=10.0,
            baseline=2.0, unit="index",
        ),
    ),
    "humidity": WeatherParam(
        name="humidity", display_name="Relative Humidity",
        band=BandThresholds(
            normal_low=30.0, normal_high=60.0,
            critical_low=15.0, critical_high=85.0,
            baseline=45.0, unit="%",
        ),
    ),
}

# -----------------------------------------------------------------------
# Profile registry
# -----------------------------------------------------------------------

WEATHER_PARAMS = _GENERAL_PARAMS

PROFILES: dict[str, WeatherProfile] = {
    "general": WeatherProfile("general", "General outdoor activity", _GENERAL_PARAMS),
    "agriculture": WeatherProfile("agriculture", "Crop health and farming", _AGRICULTURE_PARAMS),
    "aviation": WeatherProfile("aviation", "Flight safety (VFR/IFR)", _AVIATION_PARAMS),
    "construction": WeatherProfile("construction", "Construction site safety (OSHA)", _CONSTRUCTION_PARAMS),
    "public_health": WeatherProfile("public_health", "Heat/cold/air quality warnings", _PUBLIC_HEALTH_PARAMS),
}


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def parse_weather(
    readings: dict[str, float],
    profile: str = "general",
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, WeatherParam]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse weather readings into margin Observations.

    Args:
        readings:   {"temperature": 28, "wind_speed": 35, "aqi": 80, ...}
        profile:    "general", "agriculture", "aviation", "construction", "public_health"
        confidence: measurement confidence
        params:     override parameter definitions
        measured_at: timestamp
    """
    if params is None:
        p = PROFILES.get(profile)
        if p is None:
            p = PROFILES["general"]
        params = p.params

    observations = {}
    for name, value in readings.items():
        param = params.get(name)
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


def weather_expression(
    readings: dict[str, float],
    profile: str = "general",
    location: str = "",
    confidence: Confidence = Confidence.MODERATE,
    params: Optional[dict[str, WeatherParam]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a location-wide weather Expression."""
    obs = parse_weather(readings, profile, confidence, params, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=location,
    )
