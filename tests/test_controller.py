"""Tests for Controller and Fingerprint."""
import json
import pytest
from margin.fingerprint import Fingerprint, _percentile, _trimmed_mean
from margin.controller import Controller
from margin import Monitor, Parser, Thresholds, Observation, Health, Confidence


# -----------------------------------------------------------------------
# Fingerprint helpers
# -----------------------------------------------------------------------

class TestPercentile:
    def test_median_even(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_median_odd(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_p0_is_min(self):
        assert _percentile([5.0, 1.0, 3.0], 0) == pytest.approx(1.0)

    def test_p100_is_max(self):
        assert _percentile([5.0, 1.0, 3.0], 100) == pytest.approx(5.0)

    def test_empty(self):
        assert _percentile([], 50) == 0.0


class TestTrimmedMean:
    def test_removes_outliers(self):
        vals = [0.0] + [1.0] * 8 + [100.0]
        # 10% trim removes 1 from each end → trims 0.0 and 100.0
        result = _trimmed_mean(vals, fraction=0.1)
        assert result == pytest.approx(1.0)

    def test_all_same(self):
        assert _trimmed_mean([3.0] * 10, fraction=0.1) == pytest.approx(3.0)

    def test_empty(self):
        assert _trimmed_mean([], fraction=0.1) == 0.0


# -----------------------------------------------------------------------
# Fingerprint class
# -----------------------------------------------------------------------

def _make_fp(values: list[float], name: str = "cpu") -> Fingerprint:
    n = len(values)
    mean = sum(values) / n if n else 0.0
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n >= 2 else 0.0
    stats = {name: {"mean": mean, "std": std, "n": n, "trend": "STABLE"}}
    raw = {name: list(values)}
    return Fingerprint(stats=stats, values=raw)


class TestFingerprintDictCompat:
    def test_getitem(self):
        fp = _make_fp([1.0, 2.0, 3.0])
        assert fp["cpu"]["mean"] == pytest.approx(2.0)

    def test_contains(self):
        fp = _make_fp([1.0, 2.0])
        assert "cpu" in fp
        assert "mem" not in fp

    def test_get(self):
        fp = _make_fp([1.0, 2.0])
        assert fp.get("cpu") is not None
        assert fp.get("missing") is None

    def test_iter(self):
        fp = _make_fp([1.0, 2.0])
        assert list(fp) == ["cpu"]

    def test_len(self):
        fp = _make_fp([1.0, 2.0])
        assert len(fp) == 1

    def test_items(self):
        fp = _make_fp([1.0, 2.0])
        pairs = dict(fp.items())
        assert "cpu" in pairs

    def test_values(self):
        fp = _make_fp([1.0, 2.0])
        vals = list(fp.values())
        assert len(vals) == 1
        assert "mean" in vals[0]


class TestFingerprintRobustTarget:
    def test_median_vs_mean(self):
        # Skewed data: mean >> median
        vals = [0.1, 0.2, 0.3, 0.4, 10.0]
        fp = _make_fp(vals)
        assert fp.robust_target("cpu", "median") == pytest.approx(0.3)
        assert fp.robust_target("cpu", "mean") == pytest.approx(sum(vals) / len(vals))

    def test_trimmed(self):
        vals = [0.0] + [1.0] * 8 + [100.0]
        fp = _make_fp(vals)
        result = fp.robust_target("cpu", "trimmed")
        assert result == pytest.approx(1.0)

    def test_missing_component_falls_back(self):
        fp = _make_fp([1.0, 2.0])
        assert fp.robust_target("missing") == 0.0

    def test_no_raw_values_falls_back_to_mean(self):
        fp = Fingerprint(stats={"cpu": {"mean": 5.0, "std": 1.0, "n": 3, "trend": "STABLE"}})
        assert fp.robust_target("cpu") == pytest.approx(5.0)


class TestFingerprintPercentile:
    def test_p25(self):
        fp = _make_fp([10.0, 20.0, 30.0, 40.0])
        assert fp.percentile("cpu", 25) == pytest.approx(17.5)

    def test_p50_equals_robust_median(self):
        fp = _make_fp([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fp.percentile("cpu", 50) == fp.robust_target("cpu", "median")

    def test_missing_returns_zero(self):
        fp = _make_fp([1.0, 2.0])
        assert fp.percentile("missing", 50) == 0.0


class TestFingerprintN:
    def test_n(self):
        fp = _make_fp([1.0, 2.0, 3.0])
        assert fp.n("cpu") == 3

    def test_n_missing(self):
        fp = _make_fp([1.0])
        assert fp.n("missing") == 0


class TestFingerprintSerialization:
    def test_roundtrip(self):
        fp = _make_fp([1.0, 2.0, 3.0])
        d = fp.to_dict()
        fp2 = Fingerprint.from_dict(d)
        assert fp2["cpu"]["mean"] == pytest.approx(fp["cpu"]["mean"])
        # Raw values are ephemeral — fallback to mean
        assert fp2.robust_target("cpu") == pytest.approx(fp["cpu"]["mean"])

    def test_json_serializable(self):
        # Bug 1: Fingerprint must be directly JSON-serializable (no custom encoder)
        fp = _make_fp([1.0, 2.0, 3.0])
        result = json.dumps(fp)  # must not raise
        parsed = json.loads(result)
        assert parsed["cpu"]["mean"] == pytest.approx(fp["cpu"]["mean"])

    def test_json_in_nested_dict(self):
        # Common pattern: embed fingerprint in a larger metadata dict
        fp = _make_fp([1.0, 2.0, 3.0])
        meta = {"session": "abc", "fingerprint": fp}
        result = json.dumps(meta)  # must not raise
        parsed = json.loads(result)
        assert "cpu" in parsed["fingerprint"]

    def test_isinstance_dict(self):
        fp = _make_fp([1.0, 2.0])
        assert isinstance(fp, dict)


# -----------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------

class TestControllerInit:
    def test_defaults(self):
        ctrl = Controller()
        assert ctrl.strategy == "proportional_asymmetric"
        assert ctrl.kp == 0.3
        assert ctrl.target == 0.5
        assert ctrl.backoff == 0.90
        assert ctrl.alpha_min == 0.0
        assert ctrl.alpha_max == 1.0

    def test_custom_bounds(self):
        ctrl = Controller(kp=0.3, target=2.0, alpha_min=1.0, alpha_max=4.0)
        assert ctrl.alpha_min == 1.0
        assert ctrl.alpha_max == 4.0

    def test_custom(self):
        ctrl = Controller(kp=0.5, target=0.7, backoff=0.85)
        assert ctrl.kp == 0.5
        assert ctrl.target == 0.7
        assert ctrl.backoff == 0.85

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            Controller(strategy="bang_bang")

    def test_repr(self):
        ctrl = Controller(kp=0.3, target=0.5, alpha_min=1.0, alpha_max=4.0)
        assert "proportional_asymmetric" in repr(ctrl)
        assert "1.0" in repr(ctrl)
        assert "4.0" in repr(ctrl)


class TestControllerStep:
    def test_positive_metric_increases_alpha(self):
        ctrl = Controller(kp=0.5, target=0.5)
        alpha_next, reason = ctrl.step(0.5, 0.2)
        assert alpha_next == pytest.approx(0.5 + 0.5 * 0.2)
        assert "P(" in reason

    def test_negative_metric_backoff(self):
        ctrl = Controller(kp=0.5, target=0.5, backoff=0.90)
        alpha_next, reason = ctrl.step(0.5, -0.1)
        assert alpha_next == pytest.approx(0.5 * 0.90)
        assert "backoff" in reason

    def test_zero_metric_no_change(self):
        ctrl = Controller(kp=0.5, target=0.5)
        alpha_next, reason = ctrl.step(0.5, 0.0)
        assert alpha_next == pytest.approx(0.5)

    def test_stored_bounds_used_by_default(self):
        # Bug 2: bounds stored at construction, not required every call
        ctrl = Controller(kp=1.0, target=2.0, alpha_min=1.0, alpha_max=4.0)
        # Positive metric, no bounds passed — should clamp at stored alpha_max=4.0
        alpha_next, _ = ctrl.step(3.9, 1.0)  # 3.9 + 1.0*1.0 = 4.9 → clamped to 4.0
        assert alpha_next == pytest.approx(4.0)

    def test_stored_lower_bound(self):
        ctrl = Controller(kp=0.3, target=2.0, backoff=0.01, alpha_min=1.0, alpha_max=4.0)
        alpha_next, _ = ctrl.step(1.01, -1.0)  # 1.01 * 0.01 = 0.0101 → clamped to 1.0
        assert alpha_next == pytest.approx(1.0)

    def test_per_call_override_beats_stored(self):
        ctrl = Controller(kp=1.0, target=2.0, alpha_min=1.0, alpha_max=4.0)
        # Override alpha_max to 2.0 for this call only
        alpha_next, _ = ctrl.step(1.5, 1.0, alpha_max=2.0)
        assert alpha_next == pytest.approx(2.0)

    def test_clamp_upper_explicit(self):
        ctrl = Controller(kp=2.0, target=0.5)
        alpha_next, _ = ctrl.step(0.9, 1.0, alpha_max=1.0)
        assert alpha_next == pytest.approx(1.0)

    def test_clamp_lower_explicit(self):
        ctrl = Controller(kp=0.3, target=0.5, backoff=0.01)
        alpha_next, _ = ctrl.step(0.01, -1.0, alpha_min=0.05)
        assert alpha_next == pytest.approx(0.05)

    def test_scalar_metric_passthrough(self):
        ctrl = Controller(kp=1.0, target=0.5)
        alpha_next, _ = ctrl.step(0.0, 0.3)
        assert alpha_next == pytest.approx(0.3)

    def test_wide_range_no_silent_clamp(self):
        # Regression: alpha_min=1.0, alpha_max=4.0 — step without explicit bounds
        # must NOT silently clamp to [0, 1]
        ctrl = Controller(kp=0.5, target=2.0, alpha_min=1.0, alpha_max=4.0)
        alpha_next, _ = ctrl.step(2.0, 0.5)  # 2.0 + 0.5*0.5 = 2.25
        assert alpha_next == pytest.approx(2.25)  # not clamped to 1.0


class TestControllerStepFromObservations:
    def _make_obs(self, sigma: float) -> Observation:
        # Construct an Observation with a known sigma.
        # sigma = (value - baseline) / |baseline| for higher_is_better
        baseline = 1.0
        value = baseline + sigma * abs(baseline)
        return Observation(
            name="test", health=Health.INTACT, value=value,
            baseline=baseline, confidence=Confidence.HIGH,
        )

    def test_positive_sigma_increases_alpha(self):
        ctrl = Controller(kp=0.5)
        obs = [self._make_obs(0.4)]
        alpha_next, _ = ctrl.step(0.5, obs)
        assert alpha_next > 0.5

    def test_negative_sigma_backoff(self):
        ctrl = Controller(kp=0.5, backoff=0.90)
        obs = [self._make_obs(-0.3)]
        alpha_next, _ = ctrl.step(0.5, obs)
        assert alpha_next == pytest.approx(0.5 * 0.90)

    def test_mean_sigma_used(self):
        ctrl = Controller(kp=1.0)
        obs = [self._make_obs(0.2), self._make_obs(0.4)]  # mean=0.3
        alpha_next, _ = ctrl.step(0.0, obs)
        assert alpha_next == pytest.approx(0.3)

    def test_empty_observations_noop(self):
        ctrl = Controller(kp=0.5)
        alpha_next, _ = ctrl.step(0.5, [])
        # metric=0.0 → alpha += 0.5*0 → unchanged
        assert alpha_next == pytest.approx(0.5)


class TestControllerFromFingerprint:
    def test_warm_start_uses_median(self):
        # High-std values: median != mean
        vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        fp = _make_fp(vals, name="recovery_ratio")
        ctrl = Controller.from_fingerprint(fp, "recovery_ratio", kp=0.3, cold_target=0.5)
        # median of vals ≈ 0.6, mean would be ≈ 1.4; should use median
        assert ctrl.target == pytest.approx(fp.robust_target("recovery_ratio"))
        assert ctrl.target < 1.0  # not skewed by outlier

    def test_cold_start_when_insufficient_data(self):
        vals = [0.9, 0.8, 0.7]  # n=3 < min_n=10
        fp = _make_fp(vals)
        ctrl = Controller.from_fingerprint(fp, "cpu", kp=0.3, cold_target=0.5)
        assert ctrl.target == pytest.approx(0.5)

    def test_cold_start_missing_component(self):
        fp = _make_fp([0.5] * 20)
        ctrl = Controller.from_fingerprint(fp, "missing", kp=0.3, cold_target=0.42)
        assert ctrl.target == pytest.approx(0.42)

    def test_strategy_forwarded(self):
        fp = _make_fp([0.5] * 15)
        ctrl = Controller.from_fingerprint(
            fp, "cpu", kp=0.3, cold_target=0.5, strategy="proportional_asymmetric"
        )
        assert ctrl.strategy == "proportional_asymmetric"

    def test_plain_dict_fingerprint(self):
        # Backward compat: from_fingerprint accepts plain dict too
        fp = {"recovery_ratio": {"mean": 0.7, "std": 0.1, "n": 20, "trend": "STABLE"}}
        ctrl = Controller.from_fingerprint(fp, "recovery_ratio", kp=0.3, cold_target=0.5)
        assert ctrl.target == pytest.approx(0.7)

    def test_bounds_stored_from_fingerprint(self):
        fp = _make_fp([0.5] * 15)
        ctrl = Controller.from_fingerprint(
            fp, "cpu", kp=0.3, cold_target=0.5,
            alpha_min=1.0, alpha_max=4.0,
        )
        assert ctrl.alpha_min == 1.0
        assert ctrl.alpha_max == 4.0


# -----------------------------------------------------------------------
# Monitor integration: fingerprint returns Fingerprint
# -----------------------------------------------------------------------

def _make_monitor(n_updates: int = 20) -> Monitor:
    parser = Parser(
        baselines={"alpha": 0.5, "beta": 0.3},
        thresholds=Thresholds(intact=0.8, ablated=0.2),
    )
    monitor = Monitor(parser, window=50)
    import random
    rng = random.Random(42)
    for _ in range(n_updates):
        monitor.update({"alpha": rng.uniform(0.4, 0.6), "beta": rng.uniform(0.2, 0.4)})
    return monitor


class TestMonitorFingerprintType:
    def test_returns_fingerprint(self):
        monitor = _make_monitor()
        fp = monitor.fingerprint()
        assert isinstance(fp, Fingerprint)

    def test_dict_access_still_works(self):
        monitor = _make_monitor()
        fp = monitor.fingerprint()
        assert "alpha" in fp
        assert "mean" in fp["alpha"]

    def test_robust_target_available(self):
        monitor = _make_monitor()
        fp = monitor.fingerprint()
        t = fp.robust_target("alpha")
        assert isinstance(t, float)

    def test_percentile_available(self):
        monitor = _make_monitor()
        fp = monitor.fingerprint()
        p = fp.percentile("alpha", 50)
        assert isinstance(p, float)

    def test_with_baselines_still_works(self):
        monitor = _make_monitor()
        fp = monitor.fingerprint()
        new_parser = monitor.parser.with_baselines(fp)
        assert "alpha" in new_parser.baselines


# -----------------------------------------------------------------------
# Monitor.tail
# -----------------------------------------------------------------------

class TestMonitorTail:
    def test_returns_list(self):
        monitor = _make_monitor(10)
        result = monitor.tail(5)
        assert isinstance(result, list)

    def test_at_most_n(self):
        monitor = _make_monitor(10)
        assert len(monitor.tail(3)) <= 3

    def test_empty_before_update(self):
        parser = Parser(
            baselines={"x": 1.0},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )
        monitor = Monitor(parser, window=50)
        assert monitor.tail(10) == []

    def test_observations_are_observation_objects(self):
        monitor = _make_monitor(10)
        result = monitor.tail(5)
        from margin import Observation
        for obs in result:
            assert isinstance(obs, Observation)


# -----------------------------------------------------------------------
# Monitor.suggest_target
# -----------------------------------------------------------------------

class TestMonitorSuggestTarget:
    def test_returns_dict_with_keys(self):
        monitor = _make_monitor(20)
        result = monitor.suggest_target("alpha")
        assert "target" in result
        assert "confidence" in result

    def test_target_is_float(self):
        monitor = _make_monitor(20)
        result = monitor.suggest_target("alpha")
        assert isinstance(result["target"], float)

    def test_high_confidence_at_30(self):
        monitor = _make_monitor(30)
        result = monitor.suggest_target("alpha")
        assert result["confidence"] == "HIGH"

    def test_moderate_confidence_at_10(self):
        monitor = _make_monitor(10)
        result = monitor.suggest_target("alpha")
        assert result["confidence"] == "MODERATE"

    def test_low_confidence_below_10(self):
        monitor = _make_monitor(5)
        result = monitor.suggest_target("alpha")
        assert result["confidence"] == "LOW"

    def test_missing_component(self):
        monitor = _make_monitor(10)
        result = monitor.suggest_target("nonexistent")
        assert result["confidence"] == "NONE"

    def test_target_conservative(self):
        monitor = _make_monitor(30)
        fp = monitor.fingerprint()
        result = monitor.suggest_target("alpha")
        mean = fp["alpha"]["mean"]
        std = fp["alpha"]["std"]
        expected = max(0.0, mean - 0.5 * std)
        assert result["target"] == pytest.approx(expected)
