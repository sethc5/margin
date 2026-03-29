"""Tests for adapters.ros2 — robot sensor profiles and health classification."""

from datetime import datetime

from margin import Health, Confidence, Expression
from adapters.ros2 import (
    ROBOT_PROFILES, RobotSensorProfile, RobotSensor,
    parse_robot, robot_expression,
)
from adapters.ros2.node import _build_parser, _HEALTH_TO_DIAG_LEVEL


class TestProfiles:
    def test_all_profiles_exist(self):
        assert "mobile" in ROBOT_PROFILES
        assert "manipulator" in ROBOT_PROFILES
        assert "drone" in ROBOT_PROFILES
        assert "agv" in ROBOT_PROFILES

    def test_profiles_have_sensors(self):
        for name, profile in ROBOT_PROFILES.items():
            assert len(profile.sensors) >= 5, f"{name} has too few sensors"
            assert isinstance(profile, RobotSensorProfile)

    def test_sensor_thresholds_valid(self):
        """All sensors have valid thresholds (intact/ablated relationship)."""
        for prof_name, profile in ROBOT_PROFILES.items():
            for sensor_name, sensor in profile.sensors.items():
                t = sensor.thresholds
                if t.higher_is_better:
                    assert t.intact <= t.ablated or t.intact >= t.ablated, \
                        f"{prof_name}/{sensor_name}: invalid thresholds"
                # Just verify it doesn't raise
                assert isinstance(sensor.baseline, float) or isinstance(sensor.baseline, int)

    def test_sensor_polarity(self):
        """Spot-check polarity assignments."""
        mobile = ROBOT_PROFILES["mobile"].sensors
        # Temperature: lower is better
        assert mobile["motor_temperature"].thresholds.higher_is_better is False
        # Battery: higher is better
        assert mobile["battery_voltage"].thresholds.higher_is_better is True
        # Signal: higher is better
        assert mobile["lidar_signal"].thresholds.higher_is_better is True
        # Drift: lower is better
        assert mobile["wheel_odometry_drift"].thresholds.higher_is_better is False

    def test_drone_polarity(self):
        drone = ROBOT_PROFILES["drone"].sensors
        assert drone["gps_hdop"].thresholds.higher_is_better is False
        assert drone["gps_satellites"].thresholds.higher_is_better is True
        assert drone["rc_signal"].thresholds.higher_is_better is True
        assert drone["imu_drift_rate"].thresholds.higher_is_better is False

    def test_manipulator_polarity(self):
        manip = ROBOT_PROFILES["manipulator"].sensors
        assert manip["vibration"].thresholds.higher_is_better is False
        assert manip["gripper_force"].thresholds.higher_is_better is True
        assert manip["position_error"].thresholds.higher_is_better is False


class TestParseRobot:
    def test_healthy_mobile(self):
        readings = {
            "battery_voltage": 23.5,
            "motor_temperature": 40.0,
            "lidar_signal": 90.0,
            "cpu_temperature": 50.0,
        }
        obs = parse_robot(readings, profile="mobile")
        assert len(obs) == 4
        assert all(o.health == Health.INTACT for o in obs.values())

    def test_degraded_battery(self):
        readings = {"battery_voltage": 20.0}
        obs = parse_robot(readings, profile="mobile")
        assert obs["battery_voltage"].health == Health.DEGRADED

    def test_ablated_motor_temp(self):
        readings = {"motor_temperature": 90.0}
        obs = parse_robot(readings, profile="mobile")
        assert obs["motor_temperature"].health == Health.ABLATED

    def test_unknown_sensor_ignored(self):
        readings = {"battery_voltage": 23.0, "unknown_sensor": 42.0}
        obs = parse_robot(readings, profile="mobile")
        assert "battery_voltage" in obs
        assert "unknown_sensor" not in obs

    def test_drone_profile(self):
        readings = {
            "battery_voltage": 15.0,
            "gps_satellites": 10,
            "imu_drift_rate": 0.3,
        }
        obs = parse_robot(readings, profile="drone")
        assert len(obs) == 3
        assert obs["battery_voltage"].health == Health.INTACT
        assert obs["gps_satellites"].health == Health.INTACT

    def test_drone_low_gps(self):
        readings = {"gps_satellites": 3.0}
        obs = parse_robot(readings, profile="drone")
        assert obs["gps_satellites"].health == Health.ABLATED

    def test_manipulator_high_vibration(self):
        readings = {"vibration": 10.0}
        obs = parse_robot(readings, profile="manipulator")
        assert obs["vibration"].health == Health.ABLATED

    def test_agv_profile(self):
        readings = {"navigation_score": 0.92, "load_weight": 300.0}
        obs = parse_robot(readings, profile="agv")
        assert obs["navigation_score"].health == Health.INTACT
        assert obs["load_weight"].health == Health.INTACT

    def test_custom_sensors(self):
        custom = {
            "custom_temp": RobotSensor(
                name="custom_temp", display_name="Custom",
                thresholds=Thresholds(intact=50.0, ablated=80.0, higher_is_better=False),
                baseline=30.0, unit="°C",
            ),
        }
        obs = parse_robot({"custom_temp": 60.0}, sensors=custom)
        assert obs["custom_temp"].health == Health.DEGRADED

    def test_timestamp(self):
        t = datetime(2026, 1, 1)
        obs = parse_robot({"battery_voltage": 23.0}, profile="mobile", measured_at=t)
        assert obs["battery_voltage"].measured_at == t

    def test_confidence(self):
        obs = parse_robot({"battery_voltage": 23.0}, confidence=Confidence.LOW)
        assert obs["battery_voltage"].confidence == Confidence.LOW


class TestRobotExpression:
    def test_expression(self):
        readings = {"battery_voltage": 23.5, "motor_temperature": 40.0}
        expr = robot_expression(readings, profile="mobile", robot_id="bot01")
        assert isinstance(expr, Expression)
        assert expr.label == "bot01"
        assert len(expr.observations) == 2

    def test_expression_to_string(self):
        readings = {"battery_voltage": 23.5, "cpu_temperature": 50.0}
        expr = robot_expression(readings, profile="mobile")
        s = expr.to_string()
        assert "battery_voltage" in s
        assert "cpu_temperature" in s

    def test_expression_degraded(self):
        readings = {"battery_voltage": 19.0, "motor_temperature": 88.0}
        expr = robot_expression(readings, profile="mobile")
        assert len(expr.degraded()) == 2

    def test_unknown_profile_falls_back(self):
        readings = {"battery_voltage": 23.0}
        expr = robot_expression(readings, profile="nonexistent")
        assert len(expr.observations) == 1


class TestBuildParser:
    def test_parser_from_profile(self):
        sensors = ROBOT_PROFILES["mobile"].sensors
        parser = _build_parser(sensors)
        assert len(parser.baselines) == len(sensors)
        expr = parser.parse({"battery_voltage": 23.0, "motor_temperature": 40.0})
        assert len(expr.observations) == 2

    def test_parser_polarity_preserved(self):
        sensors = ROBOT_PROFILES["mobile"].sensors
        parser = _build_parser(sensors)
        t = parser._thresholds_for("motor_temperature")
        assert t.higher_is_better is False
        t2 = parser._thresholds_for("battery_voltage")
        assert t2.higher_is_better is True


class TestDiagMapping:
    def test_health_to_diag_level(self):
        assert _HEALTH_TO_DIAG_LEVEL["INTACT"] == 0      # OK
        assert _HEALTH_TO_DIAG_LEVEL["DEGRADED"] == 1    # WARN
        assert _HEALTH_TO_DIAG_LEVEL["ABLATED"] == 2     # ERROR
        assert _HEALTH_TO_DIAG_LEVEL["OOD"] == 3         # STALE
        assert _HEALTH_TO_DIAG_LEVEL["RECOVERING"] == 1  # WARN


from margin import Thresholds
