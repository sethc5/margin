"""Tests for adapters.printer3d — 3D printer sensor profiles."""

from datetime import datetime

from margin import Health, Confidence, Expression, Thresholds
from adapters.printer3d import (
    PRINTER_PROFILES, PrinterSensorProfile, PrinterSensor,
    parse_printer, printer_expression,
)


class TestProfiles:
    def test_all_profiles_exist(self):
        assert "fdm" in PRINTER_PROFILES
        assert "resin" in PRINTER_PROFILES
        assert "corexy" in PRINTER_PROFILES

    def test_profiles_have_sensors(self):
        for name, profile in PRINTER_PROFILES.items():
            assert len(profile.sensors) >= 8, f"{name} has too few sensors"

    def test_fdm_polarity(self):
        fdm = PRINTER_PROFILES["fdm"].sensors
        assert fdm["hotend_temp_error"].thresholds.higher_is_better is False
        assert fdm["extruder_current"].thresholds.higher_is_better is False
        assert fdm["filament_flow_rate"].thresholds.higher_is_better is True
        assert fdm["part_cooling_fan"].thresholds.higher_is_better is True
        assert fdm["x_axis_vibration"].thresholds.higher_is_better is False

    def test_resin_polarity(self):
        resin = PRINTER_PROFILES["resin"].sensors
        assert resin["uv_power"].thresholds.higher_is_better is True
        assert resin["resin_level"].thresholds.higher_is_better is True
        assert resin["z_axis_force"].thresholds.higher_is_better is False
        assert resin["fep_tension"].thresholds.higher_is_better is True

    def test_corexy_polarity(self):
        cxy = PRINTER_PROFILES["corexy"].sensors
        assert cxy["input_shaper_accel"].thresholds.higher_is_better is True
        assert cxy["print_speed"].thresholds.higher_is_better is True
        assert cxy["belt_tension_a"].thresholds.higher_is_better is True
        assert cxy["mcu_temperature"].thresholds.higher_is_better is False


class TestParsePrinter:
    def test_healthy_fdm(self):
        readings = {
            "hotend_temp_error": 1.0,
            "bed_temp_error": 0.5,
            "extruder_current": 0.5,
            "filament_flow_rate": 98.0,
        }
        obs = parse_printer(readings, profile="fdm")
        assert len(obs) == 4
        assert all(o.health == Health.INTACT for o in obs.values())

    def test_degraded_hotend(self):
        obs = parse_printer({"hotend_temp_error": 5.0}, profile="fdm")
        assert obs["hotend_temp_error"].health == Health.DEGRADED

    def test_ablated_extruder_jam(self):
        obs = parse_printer({"extruder_current": 1.8}, profile="fdm")
        assert obs["extruder_current"].health == Health.ABLATED

    def test_low_flow_rate(self):
        obs = parse_printer({"filament_flow_rate": 65.0}, profile="fdm")
        assert obs["filament_flow_rate"].health == Health.ABLATED

    def test_resin_profile(self):
        readings = {"uv_power": 95.0, "resin_level": 60.0, "z_axis_force": 10.0}
        obs = parse_printer(readings, profile="resin")
        assert len(obs) == 3
        assert obs["uv_power"].health == Health.INTACT

    def test_resin_low_uv(self):
        obs = parse_printer({"uv_power": 50.0}, profile="resin")
        assert obs["uv_power"].health == Health.ABLATED

    def test_corexy_belt_loose(self):
        obs = parse_printer({"belt_tension_a": 70.0}, profile="corexy")
        assert obs["belt_tension_a"].health == Health.ABLATED

    def test_corexy_high_speed_intact(self):
        obs = parse_printer({"print_speed": 250.0}, profile="corexy")
        assert obs["print_speed"].health == Health.INTACT

    def test_unknown_sensor_ignored(self):
        obs = parse_printer({"hotend_temp_error": 1.0, "magic": 42.0}, profile="fdm")
        assert "hotend_temp_error" in obs
        assert "magic" not in obs

    def test_unknown_profile_falls_back(self):
        obs = parse_printer({"hotend_temp_error": 1.0}, profile="nonexistent")
        assert len(obs) == 1

    def test_custom_sensors(self):
        custom = {
            "nozzle_wear": PrinterSensor(
                name="nozzle_wear", display_name="Nozzle Wear",
                thresholds=Thresholds(intact=0.1, ablated=0.4, higher_is_better=False),
                baseline=0.0, unit="mm",
            ),
        }
        obs = parse_printer({"nozzle_wear": 0.25}, sensors=custom)
        assert obs["nozzle_wear"].health == Health.DEGRADED

    def test_timestamp(self):
        t = datetime(2026, 1, 1)
        obs = parse_printer({"hotend_temp_error": 1.0}, measured_at=t)
        assert obs["hotend_temp_error"].measured_at == t


class TestPrinterExpression:
    def test_expression(self):
        readings = {"hotend_temp_error": 1.0, "extruder_current": 0.5}
        expr = printer_expression(readings, profile="fdm", printer_id="ender3")
        assert isinstance(expr, Expression)
        assert expr.label == "ender3"
        assert len(expr.observations) == 2

    def test_expression_to_string(self):
        readings = {"hotend_temp_error": 1.0, "bed_temp_error": 0.5}
        expr = printer_expression(readings, profile="fdm")
        s = expr.to_string()
        assert "hotend_temp_error" in s

    def test_expression_degraded(self):
        readings = {"hotend_temp_error": 12.0, "extruder_current": 2.0}
        expr = printer_expression(readings, profile="fdm")
        assert len(expr.degraded()) == 2

    def test_with_monitor(self):
        """Verify printer sensors work with streaming Monitor."""
        from margin import Monitor, Parser
        from adapters.printer3d.sensors import _FDM_SENSORS

        baselines = {s.name: s.baseline for s in _FDM_SENSORS.values()}
        component_thresholds = {s.name: s.thresholds for s in _FDM_SENSORS.values()}
        first = next(iter(_FDM_SENSORS.values()))
        parser = Parser(baselines=baselines, thresholds=first.thresholds,
                        component_thresholds=component_thresholds)
        monitor = Monitor(parser, window=20)

        for i in range(10):
            monitor.update({
                "hotend_temp_error": 1.0 + i * 0.3,
                "extruder_current": 0.5,
                "filament_flow_rate": 98.0,
            })
        assert monitor.expression is not None
        assert monitor.step == 10
