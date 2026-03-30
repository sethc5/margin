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


class TestCalibrateUseStd:
    def test_higher_is_better_thresholds(self):
        # mean=100, std=10 → intact=85 (100-1.5*10), ablated=70 (100-3*10)
        r = calibrate([90.0, 100.0, 110.0], use_std=True)
        assert r.thresholds.intact == pytest.approx(85.0, abs=0.1)
        assert r.thresholds.ablated == pytest.approx(70.0, abs=0.1)
        assert r.thresholds.higher_is_better is True

    def test_lower_is_better_thresholds(self):
        # mean=0.01, std=0.001 → intact=0.0115 (0.01+1.5*0.001), ablated=0.013 (0.01+3*0.001)
        r = calibrate([0.009, 0.010, 0.011], use_std=True, higher_is_better=False)
        assert r.thresholds.higher_is_better is False
        assert r.thresholds.intact == pytest.approx(0.0115, abs=0.0001)
        assert r.thresholds.ablated == pytest.approx(0.013, abs=0.0001)

    def test_custom_multipliers(self):
        # mean=100, std=10 → intact=100-2*10=80, ablated=100-4*10=60
        r = calibrate([90.0, 100.0, 110.0], use_std=True,
                      intact_std_multiplier=2.0, ablated_std_multiplier=4.0)
        assert r.thresholds.intact == pytest.approx(80.0, abs=0.1)
        assert r.thresholds.ablated == pytest.approx(60.0, abs=0.1)

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError, match="zero-variance"):
            calibrate([100.0, 100.0, 100.0], use_std=True)

    def test_threads_through_calibrate_many(self):
        from margin.calibrate import calibrate_many
        _, thresholds = calibrate_many(
            {"rps": [90.0, 100.0, 110.0]},
            use_std=True,
        )
        # Should not raise; thresholds set from std mode
        assert thresholds["rps"].intact < 100.0

    def test_threads_through_parser_from_calibration(self):
        p = parser_from_calibration(
            {"rps": [90.0, 100.0, 110.0]},
            use_std=True,
        )
        expr = p.parse({"rps": 60.0})
        assert expr.health_of("rps") in (Health.ABLATED, Health.DEGRADED)


class TestNeedsRecalibrationMany:
    from margin.calibrate import needs_recalibration_many as _nrm

    def test_stable_data_no_recalibration(self):
        from margin.calibrate import needs_recalibration_many
        cal = {"cpu": [50.0] * 20, "mem": [70.0] * 20}
        recent = {"cpu": [50.5] * 10, "mem": [70.2] * 10}
        flags = needs_recalibration_many(cal, recent)
        assert flags["cpu"] is False
        assert flags["mem"] is False

    def test_mean_shift_triggers_recalibration(self):
        from margin.calibrate import needs_recalibration_many
        cal = {"cpu": [50.0] * 20}
        recent = {"cpu": [100.0] * 10}  # >20% shift
        flags = needs_recalibration_many(cal, recent)
        assert flags["cpu"] is True

    def test_returns_per_component_dict(self):
        from margin.calibrate import needs_recalibration_many
        cal = {"cpu": [50.0] * 20, "mem": [70.0] * 20, "err": [0.01] * 20}
        recent = {"cpu": [50.0] * 10, "mem": [70.0] * 10, "err": [0.01] * 10}
        flags = needs_recalibration_many(cal, recent)
        assert set(flags.keys()) == {"cpu", "mem", "err"}
        assert all(isinstance(v, bool) for v in flags.values())

    def test_partial_subset_triggers(self):
        from margin.calibrate import needs_recalibration_many
        cal = {"cpu": [50.0] * 20, "mem": [70.0] * 20}
        recent = {"cpu": [100.0] * 10, "mem": [70.1] * 10}
        flags = needs_recalibration_many(cal, recent)
        assert flags["cpu"] is True
        assert flags["mem"] is False


class TestCalibrateManyReturnParser:
    def test_return_parser_gives_parser(self):
        from margin.observation import Parser
        result = calibrate_many(
            {"rps": [90.0, 100.0, 110.0], "latency": [48.0, 50.0, 52.0]},
            polarities={"latency": False},
            return_parser=True,
        )
        assert isinstance(result, Parser)

    def test_return_parser_classifies_correctly(self):
        result = calibrate_many(
            {"rps": [100.0] * 10, "latency": [50.0] * 10},
            polarities={"latency": False},
            return_parser=True,
        )
        from margin.health import Health
        expr = result.parse({"rps": 95.0, "latency": 52.0})
        assert expr.health_of("rps") == Health.INTACT
        assert expr.health_of("latency") == Health.INTACT

    def test_return_parser_false_gives_tuple(self):
        result = calibrate_many({"rps": [100.0] * 5})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_parser_empty_raises(self):
        import pytest
        with pytest.raises(ValueError):
            calibrate_many({}, return_parser=True)
