import pytest
from margin.health import Health, Thresholds, classify
from margin.confidence import Confidence


class TestThresholds:
    def test_higher_is_better_validates(self):
        Thresholds(intact=80.0, ablated=30.0)  # should not raise

    def test_higher_is_better_rejects_inverted(self):
        with pytest.raises(ValueError):
            Thresholds(intact=30.0, ablated=80.0)

    def test_lower_is_better_validates(self):
        Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)

    def test_lower_is_better_rejects_inverted(self):
        with pytest.raises(ValueError):
            Thresholds(intact=0.10, ablated=0.02, higher_is_better=False)

    def test_equal_thresholds_ok(self):
        Thresholds(intact=5.0, ablated=5.0)
        Thresholds(intact=5.0, ablated=5.0, higher_is_better=False)


class TestClassifyHigherIsBetter:
    t = Thresholds(intact=80.0, ablated=30.0)

    def test_intact(self):
        assert classify(90.0, Confidence.HIGH, thresholds=self.t) == Health.INTACT

    def test_intact_at_boundary(self):
        assert classify(80.0, Confidence.HIGH, thresholds=self.t) == Health.INTACT

    def test_degraded(self):
        assert classify(50.0, Confidence.HIGH, thresholds=self.t) == Health.DEGRADED

    def test_degraded_at_ablated_boundary(self):
        assert classify(30.0, Confidence.HIGH, thresholds=self.t) == Health.DEGRADED

    def test_ablated(self):
        assert classify(10.0, Confidence.HIGH, thresholds=self.t) == Health.ABLATED

    def test_recovering_when_correcting(self):
        assert classify(10.0, Confidence.HIGH, correcting=True, thresholds=self.t) == Health.RECOVERING
        assert classify(50.0, Confidence.HIGH, correcting=True, thresholds=self.t) == Health.RECOVERING

    def test_intact_not_recovering_even_when_correcting(self):
        assert classify(90.0, Confidence.HIGH, correcting=True, thresholds=self.t) == Health.INTACT

    def test_ood_on_indeterminate(self):
        assert classify(50.0, Confidence.INDETERMINATE, thresholds=self.t) == Health.OOD

    def test_none_thresholds_raises(self):
        with pytest.raises(ValueError):
            classify(50.0, Confidence.HIGH)


class TestClassifyLowerIsBetter:
    t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)

    def test_intact(self):
        assert classify(0.005, Confidence.HIGH, thresholds=self.t) == Health.INTACT

    def test_intact_at_boundary(self):
        assert classify(0.02, Confidence.HIGH, thresholds=self.t) == Health.INTACT

    def test_degraded(self):
        assert classify(0.05, Confidence.HIGH, thresholds=self.t) == Health.DEGRADED

    def test_degraded_at_ablated_boundary(self):
        assert classify(0.10, Confidence.HIGH, thresholds=self.t) == Health.DEGRADED

    def test_ablated(self):
        assert classify(0.15, Confidence.HIGH, thresholds=self.t) == Health.ABLATED

    def test_recovering_when_correcting(self):
        assert classify(0.15, Confidence.HIGH, correcting=True, thresholds=self.t) == Health.RECOVERING

    def test_intact_not_recovering(self):
        assert classify(0.005, Confidence.HIGH, correcting=True, thresholds=self.t) == Health.INTACT


class TestClassifyEdgeCases:
    def test_nan_falls_to_degraded(self):
        t = Thresholds(intact=80.0, ablated=30.0)
        assert classify(float('nan'), Confidence.HIGH, thresholds=t) == Health.DEGRADED

    def test_positive_inf_higher_is_better(self):
        t = Thresholds(intact=80.0, ablated=30.0)
        assert classify(float('inf'), Confidence.HIGH, thresholds=t) == Health.INTACT

    def test_positive_inf_lower_is_better(self):
        t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
        assert classify(float('inf'), Confidence.HIGH, thresholds=t) == Health.ABLATED

    def test_negative_inf_lower_is_better(self):
        t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
        assert classify(float('-inf'), Confidence.HIGH, thresholds=t) == Health.INTACT


class TestThresholdsLabels:
    def test_label_for_returns_custom(self):
        t = Thresholds(intact=80.0, ablated=30.0, labels={"ABLATED": "CRITICAL", "INTACT": "OK"})
        assert t.label_for(Health.ABLATED) == "CRITICAL"
        assert t.label_for(Health.INTACT) == "OK"

    def test_label_for_fallback_to_enum(self):
        t = Thresholds(intact=80.0, ablated=30.0, labels={"ABLATED": "CRITICAL"})
        assert t.label_for(Health.DEGRADED) == "DEGRADED"

    def test_no_labels_returns_enum_name(self):
        t = Thresholds(intact=80.0, ablated=30.0)
        assert t.label_for(Health.INTACT) == "INTACT"

    def test_to_dict_includes_labels(self):
        t = Thresholds(intact=80.0, ablated=30.0, labels={"ABLATED": "CRITICAL"})
        d = t.to_dict()
        assert d["labels"] == {"ABLATED": "CRITICAL"}
        assert d["intact"] == 80.0
        assert d["ablated"] == 30.0
        assert d["higher_is_better"] is True
        assert d["active_min"] == pytest.approx(0.05)

    def test_to_dict_omits_labels_when_none(self):
        t = Thresholds(intact=80.0, ablated=30.0)
        d = t.to_dict()
        assert "labels" not in d

    def test_from_dict_roundtrip(self):
        t = Thresholds(intact=0.01, ablated=0.10, higher_is_better=False,
                       active_min=0.1, labels={"INTACT": "GOOD", "DEGRADED": "WARN"})
        t2 = Thresholds.from_dict(t.to_dict())
        assert t2.intact == t.intact
        assert t2.ablated == t.ablated
        assert t2.higher_is_better == t.higher_is_better
        assert t2.active_min == t.active_min
        assert t2.labels == t.labels

    def test_from_dict_no_labels(self):
        t = Thresholds.from_dict({"intact": 80.0, "ablated": 30.0})
        assert t.labels is None
        assert t.higher_is_better is True
        assert t.active_min == pytest.approx(0.05)
