"""
Home Assistant adapter for margin.

Maps HA sensor entities into typed health observations.
Each device class (temperature, humidity, battery, etc.) gets
standard thresholds with correct polarity.
"""

from .sensors import (
    SENSOR_PROFILES, SensorProfile,
    parse_sensors, home_expression,
)
