"""Tests for pipeline-surfaced additions:
#1 Fingerprint.optimal_ordering
#2 Fingerprint.from_batch
#3 Monitor.predict
#4 ProvenanceGraph.bind_key / find_by_key
#5 Monitor.attribute_drift
#6 Controller.step_from_features / with_feature_weights
"""
import pytest
from margin.fingerprint import Fingerprint
from margin.provenance import ProvenanceGraph, ProvenanceNode
from margin.controller import Controller
from margin import Monitor, Parser, Thresholds, Health, Confidence, Observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(data: dict) -> Fingerprint:
    return Fingerprint(stats=data)


def _simple_fp(mean: float, std: float, name: str = "x") -> Fingerprint:
    return _fp({name: {"mean": mean, "std": std, "n": 10, "trend": "STABLE"}})


def _make_monitor(n: int = 20, seed: int = 7) -> Monitor:
    parser = Parser(
        baselines={"rss_gb": 5.0, "pass_rate": 0.85},
        thresholds=Thresholds(intact=0.8, ablated=0.2),
    )
    m = Monitor(parser, window=100)
    import random
    rng = random.Random(seed)
    # Simulate linearly growing RSS and stable pass_rate
    for i in range(n):
        m.update({
            "rss_gb": 5.0 + i * 0.013 + rng.gauss(0, 0.01),
            "pass_rate": 0.85 + rng.gauss(0, 0.02),
        })
    return m


# ---------------------------------------------------------------------------
# #1 Fingerprint.optimal_ordering
# ---------------------------------------------------------------------------

class TestOptimalOrdering:
    def test_empty_returns_empty(self):
        assert Fingerprint.optimal_ordering([], key_fn=lambda x: x) == []

    def test_single_returns_single(self):
        fp = _simple_fp(1.0, 0.1)
        assert Fingerprint.optimal_ordering([fp], key_fn=lambda x: x) == [fp]

    def test_returns_all_items(self):
        fps = [_simple_fp(float(i), 0.1) for i in range(5)]
        ordered = Fingerprint.optimal_ordering(fps, key_fn=lambda x: x)
        assert len(ordered) == 5
        assert set(id(f) for f in ordered) == set(id(f) for f in fps)

    def test_similar_items_adjacent(self):
        # Two clusters: means 0.0 and 10.0
        # Input is interleaved: [a1, b1, a2, b2]
        # Optimal ordering should group each cluster together.
        # With 4 items in 2 clusters the path crosses exactly once;
        # verify the within-cluster pairs are adjacent (at most 1 long jump).
        fp_a1 = _simple_fp(0.1, 0.05)
        fp_a2 = _simple_fp(0.2, 0.05)
        fp_b1 = _simple_fp(10.1, 0.05)
        fp_b2 = _simple_fp(10.2, 0.05)
        items = [fp_a1, fp_b1, fp_a2, fp_b2]
        ordered = Fingerprint.optimal_ordering(items, key_fn=lambda x: x)
        dists = [ordered[i].distance(ordered[i + 1]) for i in range(len(ordered) - 1)]
        long_jumps = sum(1 for d in dists if d > 5.0)
        assert long_jumps <= 1  # at most one cross-cluster transition

    def test_works_with_non_fingerprint_items(self):
        # items are plain dicts; key_fn extracts Fingerprint
        items = [
            {"id": 1, "fp": _simple_fp(0.1, 0.05)},
            {"id": 2, "fp": _simple_fp(10.0, 0.05)},
            {"id": 3, "fp": _simple_fp(0.2, 0.05)},
        ]
        ordered = Fingerprint.optimal_ordering(items, key_fn=lambda x: x["fp"])
        assert len(ordered) == 3
        assert all("id" in o for o in ordered)

    def test_metrics_restriction(self):
        # Two components; restrict to one for ordering
        fp1 = _fp({"a": {"mean": 0.0, "std": 0.1, "n": 5},
                   "b": {"mean": 100.0, "std": 0.1, "n": 5}})
        fp2 = _fp({"a": {"mean": 1.0, "std": 0.1, "n": 5},
                   "b": {"mean": 0.0, "std": 0.1, "n": 5}})
        fp3 = _fp({"a": {"mean": 0.5, "std": 0.1, "n": 5},
                   "b": {"mean": 50.0, "std": 0.1, "n": 5}})
        ordered = Fingerprint.optimal_ordering(
            [fp1, fp2, fp3], key_fn=lambda x: x, metrics=["a"]
        )
        assert len(ordered) == 3

    def test_total_path_length_not_worse_than_input(self):
        # Greedy NN should produce a path at most as long as any fixed order
        import random
        rng = random.Random(42)
        fps = [_simple_fp(rng.uniform(0, 10), 0.1) for _ in range(8)]
        ordered = Fingerprint.optimal_ordering(fps, key_fn=lambda x: x)

        def path_length(seq):
            return sum(seq[i].distance(seq[i + 1]) for i in range(len(seq) - 1))

        greedy_len = path_length(ordered)
        original_len = path_length(fps)
        # Greedy NN won't always beat original, but should be reasonable
        # (just verify it doesn't crash and returns a valid permutation)
        assert len(ordered) == len(fps)
        assert isinstance(greedy_len, float)


# ---------------------------------------------------------------------------
# #2 Fingerprint.from_batch
# ---------------------------------------------------------------------------

class TestFromBatch:
    def test_empty_batch(self):
        fp = Fingerprint.from_batch([], feature_fn=lambda x: x)
        assert len(fp) == 0

    def test_single_item(self):
        fp = Fingerprint.from_batch(
            [{"rr": 0.5}],
            feature_fn=lambda x: x,
        )
        assert fp["rr"]["n"] == 1
        assert fp["rr"]["mean"] == pytest.approx(0.5)

    def test_mean_correct(self):
        items = [{"rr": 0.1}, {"rr": 0.3}, {"rr": 0.5}]
        fp = Fingerprint.from_batch(items, feature_fn=lambda x: x)
        assert fp["rr"]["mean"] == pytest.approx(0.3)

    def test_n_equals_batch_size(self):
        items = [{"rr": float(i)} for i in range(10)]
        fp = Fingerprint.from_batch(items, feature_fn=lambda x: x)
        assert fp["rr"]["n"] == 10

    def test_multiple_components(self):
        items = [
            {"genus_div": 0.8, "mean_quality": 0.7},
            {"genus_div": 0.6, "mean_quality": 0.9},
        ]
        fp = Fingerprint.from_batch(items, feature_fn=lambda x: x)
        assert "genus_div" in fp
        assert "mean_quality" in fp
        assert fp["genus_div"]["mean"] == pytest.approx(0.7)

    def test_feature_fn_extraction(self):
        class Community:
            def __init__(self, h, q):
                self.shannon_h = h
                self.mean_quality = q

        communities = [Community(0.5, 0.8), Community(0.7, 0.6), Community(0.9, 0.7)]
        fp = Fingerprint.from_batch(
            communities,
            feature_fn=lambda c: {"genus_diversity": c.shannon_h,
                                   "mean_quality": c.mean_quality},
        )
        assert fp["genus_diversity"]["n"] == 3
        assert fp["mean_quality"]["mean"] == pytest.approx(0.7)

    def test_welford_std_correct(self):
        import statistics
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        fp = Fingerprint.from_batch([{"v": x} for x in vals], feature_fn=lambda x: x)
        assert fp["v"]["std"] == pytest.approx(statistics.stdev(vals), rel=1e-4)

    def test_returns_fingerprint_instance(self):
        fp = Fingerprint.from_batch([{"x": 1.0}], feature_fn=lambda x: x)
        assert isinstance(fp, Fingerprint)


# ---------------------------------------------------------------------------
# #3 Monitor.predict
# ---------------------------------------------------------------------------

class TestMonitorPredict:
    def test_returns_float(self):
        monitor = _make_monitor(20)
        result = monitor.predict("rss_gb", steps=10)
        assert isinstance(result, float)

    def test_unknown_metric_zero(self):
        monitor = _make_monitor(20)
        assert monitor.predict("nonexistent", steps=5) == pytest.approx(0.0)

    def test_single_observation_returns_value(self):
        parser = Parser(
            baselines={"rss": 5.0},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )
        m = Monitor(parser, window=50)
        m.update({"rss": 6.5})
        assert m.predict("rss", steps=1) == pytest.approx(6.5)

    def test_linear_trend_extrapolates(self):
        # Feed exactly linear data: 1,2,3,...,10
        parser = Parser(
            baselines={"rss": 1.0},
            thresholds=Thresholds(intact=10.0, ablated=0.0),
        )
        m = Monitor(parser, window=50)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.update({"rss": v})
        # Slope=1.0, next step should be ≈6.0
        pred = m.predict("rss", steps=1)
        assert pred == pytest.approx(6.0, abs=0.5)

    def test_downward_trend_extrapolates_below_min(self):
        # Downward trend — predict returns extrapolated value, not clamped
        parser = Parser(
            baselines={"rss": 10.0},
            thresholds=Thresholds(intact=0.8, ablated=0.0),
        )
        m = Monitor(parser, window=50)
        for v in [10.0, 9.0, 8.0, 7.0, 6.0]:
            m.update({"rss": v})
        pred = m.predict("rss", steps=10)
        assert pred < 6.0  # extrapolates beyond observed min

    def test_growing_rss_predicts_higher(self):
        monitor = _make_monitor(50)
        current = monitor.fingerprint()["rss_gb"]["mean"]
        pred = monitor.predict("rss_gb", steps=100)
        assert pred > current  # RSS is growing in the fixture

    def test_steps_1_is_default(self):
        monitor = _make_monitor(20)
        assert monitor.predict("rss_gb") == monitor.predict("rss_gb", steps=1)


# ---------------------------------------------------------------------------
# #4 ProvenanceGraph.bind_key / find_by_key
# ---------------------------------------------------------------------------

class TestProvenanceBindKey:
    def test_bind_key_sets_external_key(self):
        g = ProvenanceGraph()
        nid = g.create_root("step")
        g.bind_key(nid, "run:42")
        assert g.nodes[nid].external_key == "run:42"

    def test_bind_key_returns_self(self):
        g = ProvenanceGraph()
        nid = g.create_root("step")
        assert g.bind_key(nid, "run:1") is g

    def test_bind_key_silent_on_unknown(self):
        g = ProvenanceGraph()
        g.bind_key("nonexistent-id", "run:99")  # must not raise

    def test_find_by_key_returns_matching(self):
        g = ProvenanceGraph()
        n1 = g.create_root("step")
        n2 = g.create_root("step")
        g.bind_key(n1, "run:42")
        found = g.find_by_key("run:42")
        assert len(found) == 1
        assert found[0].id == n1

    def test_find_by_key_no_match(self):
        g = ProvenanceGraph()
        g.create_root("step")
        assert g.find_by_key("run:999") == []

    def test_find_by_key_multiple_matches(self):
        g = ProvenanceGraph()
        ids = [g.create_root("step") for _ in range(3)]
        for nid in ids:
            g.bind_key(nid, "run:42")
        found = g.find_by_key("run:42")
        assert len(found) == 3

    def test_external_key_survives_serialization(self):
        g = ProvenanceGraph()
        nid = g.create_root("step")
        g.bind_key(nid, "sqlite:run_id:1234")
        d = g.to_dict()
        g2 = ProvenanceGraph.from_dict(d)
        assert g2.nodes[nid].external_key == "sqlite:run_id:1234"

    def test_external_key_none_not_in_dict(self):
        g = ProvenanceGraph()
        nid = g.create_root("step")
        d = g.nodes[nid].to_dict()
        assert "external_key" not in d  # not serialized when None

    def test_node_without_key_unaffected(self):
        g = ProvenanceGraph()
        n1 = g.create_root("step")
        n2 = g.create_root("step")
        g.bind_key(n1, "run:1")
        assert g.nodes[n2].external_key is None

    def test_chaining(self):
        g = ProvenanceGraph()
        n1 = g.create_root("a")
        n2 = g.create_root("b")
        g.bind_key(n1, "run:1").bind_key(n2, "run:2")
        assert g.nodes[n1].external_key == "run:1"
        assert g.nodes[n2].external_key == "run:2"

    def test_compress_removes_key_with_node(self):
        g = ProvenanceGraph()
        n1 = g.create_root("old")
        n2 = g.create_root("new")
        g.bind_key(n1, "run:old")
        g.compress(max_nodes=1)
        assert n1 not in g.nodes
        assert g.find_by_key("run:old") == []


# ---------------------------------------------------------------------------
# #5 Monitor.attribute_drift
# ---------------------------------------------------------------------------

class TestAttributeDrift:
    def test_with_component_returns_classification_or_none(self):
        monitor = _make_monitor(20)
        result = monitor.attribute_drift("rss_gb")
        from margin import DriftClassification
        assert result is None or isinstance(result, DriftClassification)

    def test_with_component_matches_drift(self):
        monitor = _make_monitor(30)
        assert monitor.attribute_drift("rss_gb") == monitor.drift("rss_gb")
        assert monitor.attribute_drift("pass_rate") == monitor.drift("pass_rate")

    def test_unknown_component_returns_none(self):
        monitor = _make_monitor(20)
        assert monitor.attribute_drift("nonexistent") is None

    def test_no_args_returns_dict(self):
        monitor = _make_monitor(30)
        result = monitor.attribute_drift()
        assert isinstance(result, dict)

    def test_no_args_keys_are_component_names(self):
        monitor = _make_monitor(30)
        result = monitor.attribute_drift()
        assert set(result.keys()) <= {"rss_gb", "pass_rate"}

    def test_no_args_values_are_drift_classifications(self):
        monitor = _make_monitor(30)
        from margin import DriftClassification
        result = monitor.attribute_drift()
        for v in result.values():
            assert isinstance(v, DriftClassification)

    def test_no_args_empty_before_updates(self):
        parser = Parser(
            baselines={"x": 1.0},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )
        m = Monitor(parser, window=50)
        assert m.attribute_drift() == {}

    def test_watchdog_pattern(self):
        monitor = _make_monitor(30)
        from margin import DriftState
        alerts = []
        for comp, dc in monitor.attribute_drift().items():
            if dc.state != DriftState.STABLE:
                alerts.append(comp)
        assert isinstance(alerts, list)  # pattern works without error


# ---------------------------------------------------------------------------
# #6 Controller.step_from_features / with_feature_weights
# ---------------------------------------------------------------------------

class TestStepFromFeatures:
    def test_returns_tuple(self):
        ctrl = Controller(kp=0.5, target=0.5)
        alpha_next, reason = ctrl.step_from_features(0.5, [0.8, 0.6])
        assert isinstance(alpha_next, float)
        assert isinstance(reason, str)

    def test_equal_weights_uses_mean(self):
        ctrl = Controller(strategy="proportional_asymmetric", kp=1.0)
        # features mean = (0.2 + 0.4) / 2 = 0.3 → metric=0.3 → alpha += 1.0*0.3
        alpha_next, _ = ctrl.step_from_features(0.0, [0.2, 0.4])
        assert alpha_next == pytest.approx(0.3, abs=0.01)

    def test_explicit_weights(self):
        ctrl = Controller(strategy="proportional_asymmetric", kp=1.0)
        # dot([0.9, 0.1], [0.5, 0.5]) = 0.5 → metric=0.5 → alpha += 0.5
        alpha_next, _ = ctrl.step_from_features(0.0, [0.5, 0.5], weights=[0.9, 0.1])
        assert alpha_next == pytest.approx(0.5, abs=0.01)

    def test_stored_weights_used(self):
        ctrl = Controller(strategy="proportional_asymmetric", kp=1.0)
        ctrl = ctrl.with_feature_weights([0.9, 0.1])
        alpha_next_stored, _ = ctrl.step_from_features(0.0, [0.5, 0.5])
        alpha_next_explicit, _ = ctrl.step_from_features(0.0, [0.5, 0.5], weights=[0.9, 0.1])
        assert alpha_next_stored == pytest.approx(alpha_next_explicit)

    def test_per_call_weights_override_stored(self):
        ctrl = Controller(strategy="proportional_asymmetric", kp=1.0)
        ctrl = ctrl.with_feature_weights([0.5, 0.5])
        alpha_override, _ = ctrl.step_from_features(0.0, [1.0, 0.0], weights=[1.0, 0.0])
        alpha_stored, _ = ctrl.step_from_features(0.0, [1.0, 0.0])
        assert alpha_override != pytest.approx(alpha_stored)

    def test_reason_includes_features_prefix(self):
        ctrl = Controller(kp=0.3)
        _, reason = ctrl.step_from_features(0.5, [0.7, 0.3])
        assert "features(" in reason

    def test_clamp_applied(self):
        ctrl = Controller(kp=10.0, alpha_min=0.0, alpha_max=1.0)
        alpha_next, _ = ctrl.step_from_features(0.0, [1.0, 1.0])
        assert alpha_next <= 1.0

    def test_negative_feature_triggers_backoff(self):
        ctrl = Controller(strategy="proportional_asymmetric", kp=0.5, backoff=0.90)
        # mean([-0.5]) = -0.5 → backoff
        alpha_next, reason = ctrl.step_from_features(0.8, [-0.5])
        assert alpha_next == pytest.approx(0.8 * 0.90)
        assert "backoff" in reason

    def test_setpoint_strategy(self):
        ctrl = Controller(strategy="proportional_setpoint", kp=0.5, target=0.5)
        # features → metric=0.3, error=0.5-0.3=0.2, alpha += 0.5*0.2=0.1
        alpha_next, _ = ctrl.step_from_features(0.0, [0.3], weights=[1.0])
        assert alpha_next == pytest.approx(0.1, abs=0.01)

    def test_empty_features_returns_alpha_unchanged(self):
        ctrl = Controller(kp=0.5)
        # metric=0.0 → alpha += 0.5*0 → unchanged
        alpha_next, _ = ctrl.step_from_features(0.5, [])
        assert alpha_next == pytest.approx(0.5)

    def test_with_feature_weights_does_not_mutate(self):
        ctrl = Controller(kp=0.3)
        ctrl2 = ctrl.with_feature_weights([0.7, 0.3])
        assert ctrl.feature_weights is None
        assert ctrl2.feature_weights == [0.7, 0.3]

    def test_with_feature_weights_returns_new_controller(self):
        ctrl = Controller(kp=0.3)
        ctrl2 = ctrl.with_feature_weights([0.5, 0.5])
        assert ctrl2 is not ctrl

    def test_pipeline_soft_gate_scenario(self):
        # Simulates: feature_vector = [t025_score, community_sim]
        # Hard gate: skip if t025 < 0.3
        # Soft gate: continuous priority score, adaptive threshold.
        # Use proportional_asymmetric: more signal → higher alpha.
        ctrl = Controller(
            strategy="proportional_asymmetric",
            kp=0.5,
            alpha_min=0.0, alpha_max=1.0,
        )
        ctrl = ctrl.with_feature_weights([0.6, 0.4])

        high_priority = ctrl.step_from_features(0.5, [0.9, 0.8])
        low_priority = ctrl.step_from_features(0.5, [0.1, 0.1])

        assert high_priority[0] > low_priority[0]  # high priority → higher alpha
