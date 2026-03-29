"""
Weather adapter for margin.

Atmospheric conditions as typed health observations.
Context-dependent profiles: agriculture, aviation, construction, outdoor events, public health.
"""
from .conditions import (
    WEATHER_PARAMS, WeatherParam, WeatherProfile,
    PROFILES,
    parse_weather, weather_expression,
)
