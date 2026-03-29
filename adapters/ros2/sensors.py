"""
Robot sensor profiles as margin observations.

Each robot type has a set of sensors with domain-appropriate thresholds.
Profiles cover common robot categories: mobile, manipulator, drone, AGV.

Sensors use correct polarity:
  - Temperature, current draw, vibration → lower is better
  - Battery voltage, signal strength, uptime → higher is better
  - Joint position, IMU orientation → band thresholds (too high AND too low are bad)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from margin.health import Health, Thresholds, classify


@dataclass
class RobotSensor:
    """Threshold profile for a robot sensor."""
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


@dataclass
class RobotSensorProfile:
    """A named set of sensor thresholds for a robot type."""
    name: str
    description: str
    sensors: dict[str, RobotSensor]


# -----------------------------------------------------------------------
# Common sensors (shared across profiles)
# -----------------------------------------------------------------------

_BATTERY = RobotSensor(
    name="battery_voltage", display_name="Battery Voltage",
    thresholds=Thresholds(intact=22.0, ablated=18.0, higher_is_better=True),
    baseline=24.0, unit="V",
)

_BATTERY_SOC = RobotSensor(
    name="battery_soc", display_name="Battery State of Charge",
    thresholds=Thresholds(intact=30.0, ablated=10.0, higher_is_better=True),
    baseline=80.0, unit="%",
)

_CPU_TEMP = RobotSensor(
    name="cpu_temperature", display_name="CPU Temperature",
    thresholds=Thresholds(intact=70.0, ablated=90.0, higher_is_better=False),
    baseline=45.0, unit="°C",
)

_WIFI_RSSI = RobotSensor(
    name="wifi_rssi", display_name="WiFi Signal Strength",
    thresholds=Thresholds(intact=-60.0, ablated=-80.0, higher_is_better=True),
    baseline=-45.0, unit="dBm",
)

# -----------------------------------------------------------------------
# Mobile robot profile (wheeled/tracked ground robots)
# -----------------------------------------------------------------------

_MOBILE_SENSORS: dict[str, RobotSensor] = {
    "battery_voltage": _BATTERY,
    "battery_soc": _BATTERY_SOC,
    "cpu_temperature": _CPU_TEMP,
    "wifi_rssi": _WIFI_RSSI,
    "motor_temperature": RobotSensor(
        name="motor_temperature", display_name="Motor Temperature",
        thresholds=Thresholds(intact=60.0, ablated=85.0, higher_is_better=False),
        baseline=35.0, unit="°C",
    ),
    "motor_current": RobotSensor(
        name="motor_current", display_name="Motor Current Draw",
        thresholds=Thresholds(intact=5.0, ablated=12.0, higher_is_better=False),
        baseline=2.5, unit="A",
    ),
    "wheel_odometry_drift": RobotSensor(
        name="wheel_odometry_drift", display_name="Odometry Drift",
        thresholds=Thresholds(intact=0.05, ablated=0.20, higher_is_better=False),
        baseline=0.02, unit="m/m",
    ),
    "lidar_signal": RobotSensor(
        name="lidar_signal", display_name="LiDAR Signal Quality",
        thresholds=Thresholds(intact=80.0, ablated=40.0, higher_is_better=True),
        baseline=95.0, unit="%",
    ),
    "bump_sensor": RobotSensor(
        name="bump_sensor", display_name="Bump/Contact Events (per min)",
        thresholds=Thresholds(intact=1.0, ablated=5.0, higher_is_better=False),
        baseline=0.0, unit="events/min",
    ),
    "localization_confidence": RobotSensor(
        name="localization_confidence", display_name="Localization Confidence",
        thresholds=Thresholds(intact=0.8, ablated=0.4, higher_is_better=True),
        baseline=0.95, unit="probability",
    ),
}

# -----------------------------------------------------------------------
# Manipulator profile (robot arms)
# -----------------------------------------------------------------------

_MANIPULATOR_SENSORS: dict[str, RobotSensor] = {
    "battery_voltage": _BATTERY,
    "cpu_temperature": _CPU_TEMP,
    "joint_temperature": RobotSensor(
        name="joint_temperature", display_name="Joint Temperature (max)",
        thresholds=Thresholds(intact=55.0, ablated=80.0, higher_is_better=False),
        baseline=30.0, unit="°C",
    ),
    "joint_torque_ratio": RobotSensor(
        name="joint_torque_ratio", display_name="Joint Torque Ratio (actual/rated)",
        thresholds=Thresholds(intact=0.7, ablated=0.95, higher_is_better=False),
        baseline=0.3, unit="ratio",
    ),
    "position_error": RobotSensor(
        name="position_error", display_name="End-Effector Position Error",
        thresholds=Thresholds(intact=0.5, ablated=2.0, higher_is_better=False),
        baseline=0.1, unit="mm",
    ),
    "vibration": RobotSensor(
        name="vibration", display_name="Vibration (RMS)",
        thresholds=Thresholds(intact=2.0, ablated=8.0, higher_is_better=False),
        baseline=0.5, unit="mm/s",
    ),
    "cycle_time": RobotSensor(
        name="cycle_time", display_name="Pick/Place Cycle Time",
        thresholds=Thresholds(intact=5.0, ablated=12.0, higher_is_better=False),
        baseline=3.0, unit="s",
    ),
    "gripper_force": RobotSensor(
        name="gripper_force", display_name="Gripper Force",
        thresholds=Thresholds(intact=40.0, ablated=15.0, higher_is_better=True),
        baseline=50.0, unit="N",
    ),
    "collision_events": RobotSensor(
        name="collision_events", display_name="Collision Events (per hour)",
        thresholds=Thresholds(intact=0.0, ablated=3.0, higher_is_better=False),
        baseline=0.0, unit="events/h",
    ),
}

# -----------------------------------------------------------------------
# Drone / UAV profile
# -----------------------------------------------------------------------

_DRONE_SENSORS: dict[str, RobotSensor] = {
    "battery_voltage": RobotSensor(
        name="battery_voltage", display_name="Battery Voltage",
        thresholds=Thresholds(intact=14.8, ablated=13.2, higher_is_better=True),
        baseline=16.8, unit="V",
    ),
    "battery_soc": _BATTERY_SOC,
    "cpu_temperature": _CPU_TEMP,
    "motor_rpm_deviation": RobotSensor(
        name="motor_rpm_deviation", display_name="Motor RPM Deviation (%)",
        thresholds=Thresholds(intact=5.0, ablated=15.0, higher_is_better=False),
        baseline=1.0, unit="%",
    ),
    "gps_hdop": RobotSensor(
        name="gps_hdop", display_name="GPS Horizontal Dilution of Precision",
        thresholds=Thresholds(intact=2.0, ablated=5.0, higher_is_better=False),
        baseline=1.0, unit="HDOP",
    ),
    "gps_satellites": RobotSensor(
        name="gps_satellites", display_name="GPS Satellites Visible",
        thresholds=Thresholds(intact=8.0, ablated=4.0, higher_is_better=True),
        baseline=12.0, unit="count",
    ),
    "imu_drift_rate": RobotSensor(
        name="imu_drift_rate", display_name="IMU Drift Rate",
        thresholds=Thresholds(intact=0.5, ablated=2.0, higher_is_better=False),
        baseline=0.1, unit="°/min",
    ),
    "altitude_error": RobotSensor(
        name="altitude_error", display_name="Altitude Hold Error",
        thresholds=Thresholds(intact=1.0, ablated=5.0, higher_is_better=False),
        baseline=0.3, unit="m",
    ),
    "wind_speed": RobotSensor(
        name="wind_speed", display_name="Wind Speed",
        thresholds=Thresholds(intact=8.0, ablated=15.0, higher_is_better=False),
        baseline=3.0, unit="m/s",
    ),
    "rc_signal": RobotSensor(
        name="rc_signal", display_name="RC Signal Strength",
        thresholds=Thresholds(intact=70.0, ablated=30.0, higher_is_better=True),
        baseline=90.0, unit="%",
    ),
}

# -----------------------------------------------------------------------
# AGV / warehouse robot profile
# -----------------------------------------------------------------------

_AGV_SENSORS: dict[str, RobotSensor] = {
    "battery_voltage": _BATTERY,
    "battery_soc": _BATTERY_SOC,
    "cpu_temperature": _CPU_TEMP,
    "wifi_rssi": _WIFI_RSSI,
    "motor_temperature": RobotSensor(
        name="motor_temperature", display_name="Motor Temperature",
        thresholds=Thresholds(intact=55.0, ablated=80.0, higher_is_better=False),
        baseline=35.0, unit="°C",
    ),
    "load_weight": RobotSensor(
        name="load_weight", display_name="Payload Weight",
        thresholds=Thresholds(intact=450.0, ablated=550.0, higher_is_better=False),
        baseline=200.0, unit="kg",
    ),
    "navigation_score": RobotSensor(
        name="navigation_score", display_name="Navigation Confidence",
        thresholds=Thresholds(intact=0.85, ablated=0.5, higher_is_better=True),
        baseline=0.95, unit="score",
    ),
    "obstacle_clearance": RobotSensor(
        name="obstacle_clearance", display_name="Min Obstacle Clearance",
        thresholds=Thresholds(intact=0.3, ablated=0.1, higher_is_better=True),
        baseline=0.8, unit="m",
    ),
    "task_completion_rate": RobotSensor(
        name="task_completion_rate", display_name="Task Completion Rate",
        thresholds=Thresholds(intact=0.95, ablated=0.75, higher_is_better=True),
        baseline=0.99, unit="ratio",
    ),
    "charging_efficiency": RobotSensor(
        name="charging_efficiency", display_name="Charging Efficiency",
        thresholds=Thresholds(intact=0.85, ablated=0.60, higher_is_better=True),
        baseline=0.92, unit="ratio",
    ),
}

# -----------------------------------------------------------------------
# Profile registry
# -----------------------------------------------------------------------

ROBOT_PROFILES: dict[str, RobotSensorProfile] = {
    "mobile": RobotSensorProfile("mobile", "Wheeled/tracked ground robots", _MOBILE_SENSORS),
    "manipulator": RobotSensorProfile("manipulator", "Robot arms and manipulators", _MANIPULATOR_SENSORS),
    "drone": RobotSensorProfile("drone", "UAVs and multirotors", _DRONE_SENSORS),
    "agv": RobotSensorProfile("agv", "Warehouse AGVs and AMRs", _AGV_SENSORS),
}


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def parse_robot(
    readings: dict[str, float],
    profile: str = "mobile",
    confidence: Confidence = Confidence.MODERATE,
    sensors: Optional[dict[str, RobotSensor]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse robot sensor readings into margin Observations.

    Args:
        readings:    {"battery_voltage": 23.1, "motor_temperature": 42.0, ...}
        profile:     "mobile", "manipulator", "drone", "agv"
        confidence:  measurement confidence
        sensors:     override sensor definitions
        measured_at: timestamp
    """
    if sensors is None:
        p = ROBOT_PROFILES.get(profile)
        if p is None:
            p = ROBOT_PROFILES["mobile"]
        sensors = p.sensors

    observations = {}
    for name, value in readings.items():
        sensor = sensors.get(name)
        if sensor is None:
            continue
        health = classify(value, confidence, thresholds=sensor.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value,
            baseline=sensor.baseline,
            confidence=confidence,
            higher_is_better=sensor.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def robot_expression(
    readings: dict[str, float],
    profile: str = "mobile",
    robot_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    sensors: Optional[dict[str, RobotSensor]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a robot-wide health Expression."""
    obs = parse_robot(readings, profile, confidence, sensors, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=robot_id,
    )
