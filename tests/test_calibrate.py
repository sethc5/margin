import pytest
from margin.calibrate import calibrate, calibrate_many, parser_from_calibration, CalibrationResult
from margin.health import Thresholds, Health
from margin.confidence import Confidence


class TestCalibrate:
    def test_basic_higher_is_better(self):
        r = calibrate([100.0, 100.0, 100.0])
        assert r.baseline == pytest.approx(100.0)
        assert r.thresholds.higher_is_better is True
        assert r.thresholds.intact == pytest.approx(70.0)
        assert r.thresholds.ablated == pytest.approx(30.0)

    def test_basic_lower_is_better(self):
        r = calibrate([0.01, 0.01, 0.01], higher_is_better=False)
        assert r.baseline == pytest.approx(0.01)
        assert r.thresholds.higher_is_better is False
        # intact = 0.01 * 1.30 = 0.013
        assert r.thresholds.intact == pytest.approx(0.013, abs=0.001)
        # ablated = 0.01 * 1.70 = 0.017
        assert r.thresholds.ablated == pytest.approx(0.017, abs=0.001)

    def test_custom_fractions(self):
        r = calibrate([200.0, 200.0], intact_fraction=0.80, ablated_fraction=0.20)
        assert r.thresholds.intact == pytest.approx(160.0)
        assert r.thresholds.ablated == pytest.approx(40.0)

    def test_std_computed(self):
        r = calibrate([90.0, 100.0, 110.0])
        assert r.std > 0
        assert r.n_samples == 3

    def test_single_value(self):
        r = calibrate([50.0])
        assert r.baseline == 50.0
        assert r.std == 0.0
        assert r.n_samples == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate([])

    def test_to_dict(self):
        r = calibrate([100.0, 100.0])
        d = r.to_dict()
        assert "baseline" in d
        assert "intact" in d
        assert "ablated" in d
        assert "higher_is_better" in d

    def test_threshold_validation_holds(self):
        # Higher is better: ablated < intact
        r = calibrate([100.0], higher_is_better=True)
        assert r.thresholds.ablated <= r.thresholds.intact

        # Lower is better: ablated > intact
        r2 = calibrate([0.01], higher_is_better=False)
        assert r2.thresholds.ablated >= r2.thresholds.intact


class TestCalibrateMany:
    def test_returns_baselines_and_thresholds(self):
        baselines, thresholds = calibrate_many({
            "throughput": [100.0, 105.0, 95.0],
            "error_rate": [0.01, 0.012, 0.008],
        }, polarities={"error_rate": False})

        assert "throughput" in baselines
        assert "error_rate" in baselines
        assert thresholds["throughput"].higher_is_better is True
        assert thresholds["error_rate"].higher_is_better is False

    def test_default_polarity_is_higher(self):
        _, thresholds = calibrate_many({"x": [50.0, 50.0]})
        assert thresholds["x"].higher_is_better is True


class TestParserFromCalibration:
    def test_produces_working_parser(self):
        parser = parser_from_calibration(
            {"throughput": [100.0] * 5, "latency": [50.0] * 5},
            polarities={"latency": False},
        )
        # Healthy readings
        expr = parser.parse({"throughput": 95.0, "latency": 52.0})
        assert expr.health_of("throughput") == Health.INTACT
        assert expr.health_of("latency") == Health.INTACT

    def test_detects_degradation(self):
        parser = parser_from_calibration(
            {"throughput": [100.0] * 5},
        )
        expr = parser.parse({"throughput": 40.0})
        assert expr.health_of("throughput") == Health.DEGRADED

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parser_from_calibration({})
