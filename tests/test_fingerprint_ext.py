"""Tests for Fingerprint extended API: to_tensor, distance, kl_divergence,
similarity, update, merge, metrics, and Monitor.anomaly_score."""
import json
import math
import pytest
from margin.fingerprint import Fingerprint
from margin import Monitor, Parser, Thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(data: dict[str, dict]) -> Fingerprint:
    return Fingerprint(stats=data)


def _uniform_fp(mean: float = 1.0, std: float = 0.5, name: str = "x") -> Fingerprint:
    return _fp({name: {"mean": mean, "std": std, "n": 20, "trend": "STABLE"}})


def _make_monitor(n: int = 20) -> Monitor:
    parser = Parser(
        baselines={"alpha": 0.5, "beta": 0.3},
        thresholds=Thresholds(intact=0.8, ablated=0.2),
    )
    m = Monitor(parser, window=50)
    import random
    rng = random.Random(7)
    for _ in range(n):
        m.update({"alpha": rng.uniform(0.4, 0.6), "beta": rng.uniform(0.2, 0.4)})
    return m


# ---------------------------------------------------------------------------
# Fingerprint.metrics property
# ---------------------------------------------------------------------------

class TestFingerprintMetrics:
    def test_only_nonzero_n(self):
        fp = _fp({
            "a": {"mean": 1.0, "std": 0.1, "n": 5},
            "b": {"mean": 0.0, "std": 0.0, "n": 0},
        })
        assert fp.metrics == ["a"]

    def test_sorted(self):
        fp = _fp({
            "z": {"mean": 1.0, "std": 0.1, "n": 3},
            "a": {"mean": 1.0, "std": 0.1, "n": 5},
        })
        assert fp.metrics == ["a", "z"]

    def test_all_zero_n(self):
        fp = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 0}})
        assert fp.metrics == []

    def test_from_monitor(self):
        monitor = _make_monitor(10)
        fp = monitor.fingerprint()
        assert set(fp.metrics) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Fingerprint.to_tensor
# ---------------------------------------------------------------------------

class TestFingerprintToTensor:
    def test_default_returns_list(self):
        fp = _fp({"rr": {"mean": 0.07, "std": 0.498, "n": 98}})
        v = fp.to_tensor(["rr"])
        assert isinstance(v, list)
        assert len(v) == 2
        assert v[0] == pytest.approx(0.07)
        assert v[1] == pytest.approx(0.498)

    def test_metric_order_preserved(self):
        fp = _fp({
            "a": {"mean": 1.0, "std": 0.1, "n": 5},
            "b": {"mean": 2.0, "std": 0.2, "n": 5},
        })
        v = fp.to_tensor(["b", "a"], ["mean"])
        assert v == pytest.approx([2.0, 1.0])

    def test_custom_stats(self):
        fp = _fp({"x": {"mean": 3.0, "std": 1.0, "n": 10, "median": 2.5}})
        v = fp.to_tensor(["x"], stats=["mean", "std", "median"])
        assert len(v) == 3
        assert v[2] == pytest.approx(2.5)

    def test_none_metrics_uses_sorted_keys(self):
        fp = _fp({
            "b": {"mean": 2.0, "std": 0.2, "n": 5},
            "a": {"mean": 1.0, "std": 0.1, "n": 5},
        })
        v = fp.to_tensor()
        # sorted: a, b → [a_mean, a_std, b_mean, b_std]
        assert v[0] == pytest.approx(1.0)
        assert v[2] == pytest.approx(2.0)

    def test_missing_stat_defaults_to_zero(self):
        fp = _fp({"x": {"mean": 1.0, "n": 5}})
        v = fp.to_tensor(["x"], stats=["mean", "std"])
        assert v[1] == pytest.approx(0.0)

    def test_numpy_format(self):
        pytest.importorskip("numpy")
        import numpy as np
        fp = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 5}})
        v = fp.to_tensor(["x"], format="numpy")
        assert isinstance(v, np.ndarray)
        assert v.dtype == np.float32
        assert v[0] == pytest.approx(1.0)

    def test_torch_format(self):
        pytest.importorskip("torch")
        import torch
        fp = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 5}})
        v = fp.to_tensor(["x"], format="torch")
        assert isinstance(v, torch.Tensor)
        assert v[0].item() == pytest.approx(1.0)

    def test_unknown_format_raises(self):
        fp = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 5}})
        with pytest.raises(ValueError, match="Unknown format"):
            fp.to_tensor(format="jax")

    def test_empty_metrics(self):
        fp = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 5}})
        assert fp.to_tensor([]) == []

    def test_dfp_scenario(self):
        # Simulates the D_fp 64-dim conditioning use case
        fp = _fp({
            "recovery_ratio": {"mean": 0.070, "std": 0.498, "n": 98},
            "improvement":    {"mean": 0.012, "std": 0.045, "n": 98},
        })
        v = fp.to_tensor(["recovery_ratio", "improvement"], ["mean", "std"])
        assert len(v) == 4
        assert v == pytest.approx([0.070, 0.498, 0.012, 0.045])


# ---------------------------------------------------------------------------
# Fingerprint.distance / similarity
# ---------------------------------------------------------------------------

class TestFingerprintDistance:
    def test_self_distance_zero(self):
        fp = _uniform_fp()
        assert fp.distance(fp) == pytest.approx(0.0)

    def test_symmetric(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 10}})
        fp2 = _fp({"x": {"mean": 2.0, "std": 0.3, "n": 10}})
        assert fp1.distance(fp2) == pytest.approx(fp2.distance(fp1))

    def test_known_value(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 1}})
        fp2 = _fp({"x": {"mean": 3.0, "std": 4.0, "n": 1}})
        # L2([0,0], [3,4]) = 5
        assert fp1.distance(fp2) == pytest.approx(5.0)

    def test_metrics_restricts(self):
        fp1 = _fp({
            "a": {"mean": 0.0, "std": 0.0, "n": 1},
            "b": {"mean": 100.0, "std": 100.0, "n": 1},
        })
        fp2 = _fp({
            "a": {"mean": 3.0, "std": 4.0, "n": 1},
            "b": {"mean": 0.0, "std": 0.0, "n": 1},
        })
        # Only compare "a"
        assert fp1.distance(fp2, metrics=["a"]) == pytest.approx(5.0)

    def test_only_shared_keys_by_default(self):
        fp1 = _fp({
            "shared": {"mean": 0.0, "std": 0.0, "n": 1},
            "only_in_1": {"mean": 1000.0, "std": 1000.0, "n": 1},
        })
        fp2 = _fp({"shared": {"mean": 3.0, "std": 4.0, "n": 1}})
        assert fp1.distance(fp2) == pytest.approx(5.0)


class TestFingerprintSimilarity:
    def test_self_similarity_one(self):
        fp = _uniform_fp()
        assert fp.similarity(fp) == pytest.approx(1.0)

    def test_orthogonal_zero(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.0, "n": 1},
                   "y": {"mean": 0.0, "std": 0.0, "n": 1}})
        fp2 = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 1},
                   "y": {"mean": 1.0, "std": 0.0, "n": 1}})
        assert fp1.similarity(fp2) == pytest.approx(0.0)

    def test_opposite_negative(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.5, "n": 1}})
        fp2 = _fp({"x": {"mean": -1.0, "std": -0.5, "n": 1}})
        assert fp1.similarity(fp2) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 1}})
        fp2 = _fp({"x": {"mean": 1.0, "std": 1.0, "n": 1}})
        assert fp1.similarity(fp2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fingerprint.kl_divergence
# ---------------------------------------------------------------------------

class TestFingerprintKLDivergence:
    def test_self_kl_zero(self):
        fp = _uniform_fp()
        assert fp.kl_divergence(fp) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_by_default(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 1.0, "n": 10}})
        fp2 = _fp({"x": {"mean": 1.0, "std": 2.0, "n": 10}})
        assert fp1.kl_divergence(fp2) == pytest.approx(fp2.kl_divergence(fp1))

    def test_asymmetric_flag(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 1.0, "n": 10}})
        fp2 = _fp({"x": {"mean": 1.0, "std": 2.0, "n": 10}})
        forward = fp1.kl_divergence(fp2, symmetric=False)
        backward = fp2.kl_divergence(fp1, symmetric=False)
        assert forward != pytest.approx(backward)

    def test_known_standard_normal_vs_shifted(self):
        # KL(N(0,1) || N(1,1)) = 0.5 (exact closed form)
        fp1 = _fp({"x": {"mean": 0.0, "std": 1.0, "n": 10}})
        fp2 = _fp({"x": {"mean": 1.0, "std": 1.0, "n": 10}})
        kl_forward = fp1.kl_divergence(fp2, symmetric=False)
        assert kl_forward == pytest.approx(0.5, rel=1e-5)

    def test_nonnegative(self):
        fp1 = _fp({"x": {"mean": 0.5, "std": 0.3, "n": 10},
                   "y": {"mean": 1.0, "std": 0.8, "n": 10}})
        fp2 = _fp({"x": {"mean": 0.7, "std": 0.4, "n": 10},
                   "y": {"mean": 1.2, "std": 0.5, "n": 10}})
        assert fp1.kl_divergence(fp2) >= 0.0

    def test_larger_divergence_when_more_different(self):
        fp_ref = _fp({"x": {"mean": 0.0, "std": 1.0, "n": 20}})
        fp_close = _fp({"x": {"mean": 0.1, "std": 1.0, "n": 20}})
        fp_far = _fp({"x": {"mean": 5.0, "std": 1.0, "n": 20}})
        assert fp_ref.kl_divergence(fp_close) < fp_ref.kl_divergence(fp_far)


# ---------------------------------------------------------------------------
# Fingerprint.update (online Welford)
# ---------------------------------------------------------------------------

class TestFingerprintUpdate:
    def test_returns_self(self):
        fp = _fp({"x": {"mean": 1.0, "std": 0.0, "n": 1}})
        assert fp.update("x", 2.0) is fp

    def test_n_increments(self):
        fp = _fp({"x": {"mean": 1.0, "std": 0.0, "n": 5}})
        fp.update("x", 2.0)
        assert fp["x"]["n"] == 6

    def test_mean_updates(self):
        # Start from scratch with known values
        fp = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 0}})
        fp.update("x", 2.0)
        fp.update("x", 4.0)
        assert fp["x"]["mean"] == pytest.approx(3.0)

    def test_std_updates(self):
        fp = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 0}})
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            fp.update("x", v)
        # Population std ≈ 2.0, sample std ≈ 2.138
        assert fp["x"]["std"] == pytest.approx(2.0, rel=0.1)

    def test_new_component(self):
        fp = _fp({})
        fp.update("new_comp", 5.0)
        assert fp["new_comp"]["n"] == 1
        assert fp["new_comp"]["mean"] == pytest.approx(5.0)

    def test_raw_values_appended_if_present(self):
        fp = Fingerprint(
            stats={"x": {"mean": 1.0, "std": 0.0, "n": 1}},
            values={"x": [1.0]},
        )
        fp.update("x", 2.0)
        assert fp._values["x"] == [1.0, 2.0]

    def test_no_crash_when_no_raw_values(self):
        fp = _fp({"x": {"mean": 1.0, "std": 0.0, "n": 1}})
        fp.update("x", 2.0)  # no _values entry — must not crash
        assert fp["x"]["n"] == 2

    def test_welford_matches_batch(self):
        # Online Welford should match batch mean/std for same values
        import statistics
        vals = [1.5, 2.3, 0.8, 3.1, 2.7, 1.9, 0.5, 2.1]
        fp = _fp({})
        for v in vals:
            fp.update("x", v)
        assert fp["x"]["mean"] == pytest.approx(statistics.mean(vals), rel=1e-6)
        assert fp["x"]["std"] == pytest.approx(statistics.stdev(vals), rel=1e-4)


# ---------------------------------------------------------------------------
# Fingerprint.merge
# ---------------------------------------------------------------------------

class TestFingerprintMerge:
    def test_equal_weight_averages_mean(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 10, "trend": "STABLE"}})
        fp2 = _fp({"x": {"mean": 2.0, "std": 0.0, "n": 10, "trend": "STABLE"}})
        merged = fp1.merge(fp2, weight=0.5)
        assert merged["x"]["mean"] == pytest.approx(1.0)

    def test_weight_zero_returns_self(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.1, "n": 10, "trend": "STABLE"}})
        fp2 = _fp({"x": {"mean": 9.0, "std": 0.9, "n": 10, "trend": "STABLE"}})
        merged = fp1.merge(fp2, weight=0.0)
        assert merged["x"]["mean"] == pytest.approx(1.0)

    def test_weight_one_returns_other(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.1, "n": 10, "trend": "STABLE"}})
        fp2 = _fp({"x": {"mean": 9.0, "std": 0.9, "n": 10, "trend": "STABLE"}})
        merged = fp1.merge(fp2, weight=1.0)
        assert merged["x"]["mean"] == pytest.approx(9.0)

    def test_n_summed(self):
        fp1 = _fp({"x": {"mean": 1.0, "std": 0.1, "n": 10, "trend": "STABLE"}})
        fp2 = _fp({"x": {"mean": 2.0, "std": 0.2, "n": 15, "trend": "STABLE"}})
        merged = fp1.merge(fp2)
        assert merged["x"]["n"] == 25

    def test_disjoint_keys_preserved(self):
        fp1 = _fp({"a": {"mean": 1.0, "std": 0.1, "n": 5, "trend": "STABLE"}})
        fp2 = _fp({"b": {"mean": 2.0, "std": 0.2, "n": 5, "trend": "STABLE"}})
        merged = fp1.merge(fp2)
        assert "a" in merged
        assert "b" in merged

    def test_returns_new_fingerprint(self):
        fp1 = _uniform_fp(mean=1.0, name="x")
        fp2 = _uniform_fp(mean=2.0, name="x")
        merged = fp1.merge(fp2)
        assert merged is not fp1
        assert merged is not fp2

    def test_weight_clamped(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 0.0, "n": 1, "trend": "STABLE"}})
        fp2 = _fp({"x": {"mean": 10.0, "std": 0.0, "n": 1, "trend": "STABLE"}})
        merged_hi = fp1.merge(fp2, weight=2.0)  # clamp to 1.0
        assert merged_hi["x"]["mean"] == pytest.approx(10.0)
        merged_lo = fp1.merge(fp2, weight=-1.0)  # clamp to 0.0
        assert merged_lo["x"]["mean"] == pytest.approx(0.0)

    def test_percentiles_interpolated_when_present(self):
        fp1 = _fp({"x": {"mean": 0.0, "std": 1.0, "n": 10,
                          "trend": "STABLE", "median": 0.0, "q25": -1.0, "q75": 1.0}})
        fp2 = _fp({"x": {"mean": 2.0, "std": 1.0, "n": 10,
                          "trend": "STABLE", "median": 2.0, "q25": 1.0, "q75": 3.0}})
        merged = fp1.merge(fp2, weight=0.5)
        assert merged["x"]["median"] == pytest.approx(1.0)
        assert merged["x"]["q25"] == pytest.approx(0.0)
        assert merged["x"]["q75"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Monitor.anomaly_score
# ---------------------------------------------------------------------------

class TestMonitorAnomalyScore:
    def test_returns_float(self):
        monitor = _make_monitor(20)
        ref = monitor.fingerprint()
        score = monitor.anomaly_score(ref)
        assert isinstance(score, float)

    def test_same_session_near_zero(self):
        # Reference = current session → mean z-score should be ~0
        monitor = _make_monitor(20)
        ref = monitor.fingerprint()
        score = monitor.anomaly_score(ref)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_higher_when_diverged(self):
        # Build a reference fingerprint far from current session
        monitor = _make_monitor(20)
        ref_far = Fingerprint(stats={
            "alpha": {"mean": 10.0, "std": 0.1, "n": 20, "trend": "STABLE"},
            "beta":  {"mean": 10.0, "std": 0.1, "n": 20, "trend": "STABLE"},
        })
        score = monitor.anomaly_score(ref_far)
        assert score > 1.0

    def test_metrics_restricts(self):
        monitor = _make_monitor(20)
        ref = Fingerprint(stats={
            "alpha": {"mean": 0.5, "std": 0.05, "n": 20, "trend": "STABLE"},
            "beta":  {"mean": 10.0, "std": 0.1, "n": 20, "trend": "STABLE"},
        })
        score_full = monitor.anomaly_score(ref)
        score_alpha = monitor.anomaly_score(ref, metrics=["alpha"])
        # With only alpha (near-baseline), score should be lower than full
        assert score_alpha < score_full

    def test_zero_ref_std_skipped(self):
        # Components with ref_std=0 are skipped; should not raise
        monitor = _make_monitor(20)
        ref = Fingerprint(stats={
            "alpha": {"mean": 0.5, "std": 0.0, "n": 20},   # std=0 → skipped
            "beta":  {"mean": 0.3, "std": 0.05, "n": 20},
        })
        score = monitor.anomaly_score(ref)
        assert isinstance(score, float)

    def test_no_shared_metrics_returns_zero(self):
        monitor = _make_monitor(20)
        ref = Fingerprint(stats={"nonexistent": {"mean": 1.0, "std": 0.1, "n": 5}})
        score = monitor.anomaly_score(ref)
        assert score == pytest.approx(0.0)

    def test_cross_session_scenario(self):
        # Simulate: Session 1 ref, Session 2 monitor with different distribution
        ref = Fingerprint(stats={
            "alpha": {"mean": 0.50, "std": 0.05, "n": 50, "trend": "STABLE"},
            "beta":  {"mean": 0.30, "std": 0.03, "n": 50, "trend": "STABLE"},
        })
        # Session 2 monitor at a very different operating point
        parser = Parser(
            baselines={"alpha": 0.5, "beta": 0.3},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )
        monitor2 = Monitor(parser, window=50)
        import random
        rng = random.Random(99)
        for _ in range(20):
            monitor2.update({
                "alpha": rng.uniform(0.7, 0.9),  # shifted up
                "beta": rng.uniform(0.5, 0.7),   # shifted up
            })
        score = monitor2.anomaly_score(ref)
        assert score > 2.0  # clearly diverged
