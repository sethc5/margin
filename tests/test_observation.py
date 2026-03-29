import pytest
from margin.observation import (
    Op, Observation, Correction, Expression, Parser,
)
from margin.health import Health, Thresholds
from margin.confidence import Confidence


class TestObservationSigma:
    def test_higher_is_better_above_baseline(self):
        o = Observation("x", Health.INTACT, 120.0, 100.0, Confidence.HIGH, higher_is_better=True)
        assert o.sigma == pytest.approx(0.2)

    def test_higher_is_better_below_baseline(self):
        o = Observation("x", Health.DEGRADED, 50.0, 100.0, Confidence.HIGH, higher_is_better=True)
        assert o.sigma == pytest.approx(-0.5)

    def test_lower_is_better_below_baseline_is_positive(self):
        o = Observation("x", Health.INTACT, 0.005, 0.01, Confidence.HIGH, higher_is_better=False)
        assert o.sigma == pytest.approx(0.5)

    def test_lower_is_better_above_baseline_is_negative(self):
        o = Observation("x", Health.DEGRADED, 0.05, 0.01, Confidence.HIGH, higher_is_better=False)
        assert o.sigma == pytest.approx(-4.0)

    def test_zero_baseline_returns_zero(self):
        o = Observation("x", Health.INTACT, 5.0, 0.0, Confidence.HIGH)
        assert o.sigma == 0.0


class TestObservationAtom:
    def test_intact_atom(self):
        o = Observation("api", Health.INTACT, 95.0, 100.0, Confidence.HIGH)
        assert o.to_atom() == "api:INTACT(-0.05σ)"

    def test_ood_omits_sigma(self):
        o = Observation("api", Health.OOD, 50.0, 100.0, Confidence.INDETERMINATE)
        assert o.to_atom() == "api:OOD"


class TestObservationRoundtrip:
    def test_higher_is_better(self):
        o = Observation("api", Health.DEGRADED, 50.0, 100.0, Confidence.HIGH, True, ["abc"])
        r = Observation.from_dict(o.to_dict())
        assert r.name == "api"
        assert r.health == Health.DEGRADED
        assert r.higher_is_better is True
        assert r.provenance == ["abc"]

    def test_lower_is_better(self):
        o = Observation("err", Health.ABLATED, 0.15, 0.01, Confidence.LOW, False)
        r = Observation.from_dict(o.to_dict())
        assert r.higher_is_better is False
        assert r.health == Health.ABLATED


class TestCorrectionRoundtrip:
    def test_roundtrip(self):
        c = Correction("api", Op.RESTORE, 0.6, 2.0, ["api"], ["xyz"])
        r = Correction.from_dict(c.to_dict())
        assert r.op == Op.RESTORE
        assert r.alpha == 0.6
        assert r.provenance == ["xyz"]

    def test_is_active(self):
        assert Correction("x", Op.RESTORE, 0.5, 1.0).is_active() is True
        assert Correction("x", Op.NOOP, 0.0, 0.0).is_active() is False
        assert Correction("x", Op.RESTORE, 0.0, 1.0).is_active() is False


class TestExpression:
    def test_empty_expression(self):
        e = Expression()
        assert e.to_string() == "[∅]"

    def test_health_of(self):
        e = Expression(observations=[
            Observation("a", Health.INTACT, 90.0, 100.0, Confidence.HIGH),
            Observation("b", Health.DEGRADED, 50.0, 100.0, Confidence.HIGH),
        ])
        assert e.health_of("a") == Health.INTACT
        assert e.health_of("b") == Health.DEGRADED
        assert e.health_of("c") is None

    def test_degraded_includes_recovering(self):
        e = Expression(observations=[
            Observation("a", Health.RECOVERING, 50.0, 100.0, Confidence.HIGH),
            Observation("b", Health.INTACT, 90.0, 100.0, Confidence.HIGH),
        ])
        assert len(e.degraded()) == 1
        assert e.degraded()[0].name == "a"

    def test_intact_excludes_recovering(self):
        e = Expression(observations=[
            Observation("a", Health.RECOVERING, 50.0, 100.0, Confidence.HIGH),
        ])
        assert len(e.intact()) == 0

    def test_roundtrip_json(self):
        e = Expression(
            observations=[Observation("a", Health.INTACT, 90.0, 100.0, Confidence.HIGH)],
            corrections=[Correction("b", Op.RESTORE, 0.5, 1.0)],
            confidence=Confidence.HIGH,
            label="test",
            step=3,
        )
        r = Expression.from_json(e.to_json())
        assert r.to_string() == e.to_string()
        assert r.confidence == Confidence.HIGH
        assert r.label == "test"
        assert r.step == 3

    def test_orphan_correction_renders(self):
        e = Expression(
            observations=[],
            corrections=[Correction("ghost", Op.RESTORE, 0.5, 1.0)],
        )
        assert "?" in e.to_string()


class TestParser:
    def test_basic_parse(self):
        p = Parser(
            baselines={"x": 100.0},
            thresholds=Thresholds(intact=80.0, ablated=30.0),
        )
        e = p.parse({"x": 90.0})
        assert e.health_of("x") == Health.INTACT

    def test_correction_targets_worst(self):
        p = Parser(
            baselines={"a": 100.0, "b": 100.0},
            thresholds=Thresholds(intact=80.0, ablated=30.0),
        )
        e = p.parse({"a": 90.0, "b": 20.0}, correction_magnitude=1.0, alpha=0.5)
        assert e.corrections[0].target == "b"
        assert e.corrections[0].op == Op.RESTORE

    def test_per_component_active_min(self):
        p = Parser(
            baselines={"a": 100.0, "b": 0.01},
            thresholds=Thresholds(intact=80.0, ablated=30.0, active_min=5.0),
            component_thresholds={
                "b": Thresholds(intact=0.02, ablated=0.10, higher_is_better=False, active_min=0.001),
            },
        )
        e = p.parse({"a": 50.0, "b": 0.05}, correction_magnitude=0.01)
        a_obs = [o for o in e.observations if o.name == "a"][0]
        b_obs = [o for o in e.observations if o.name == "b"][0]
        assert a_obs.health == Health.DEGRADED  # global active_min=5.0 not met
        assert b_obs.health == Health.RECOVERING  # component active_min=0.001 met

    def test_empty_values(self):
        p = Parser(baselines={}, thresholds=Thresholds(intact=80.0, ablated=30.0))
        e = p.parse({})
        assert e.to_string() == "[∅]"
        assert e.confidence == Confidence.INDETERMINATE

    def test_net_confidence_is_weakest(self):
        p = Parser(
            baselines={"a": 100.0, "b": 100.0},
            thresholds=Thresholds(intact=80.0, ablated=30.0),
        )
        e = p.parse(
            {"a": 90.0, "b": 50.0},
            confidences={"a": Confidence.HIGH, "b": Confidence.LOW},
        )
        assert e.confidence == Confidence.LOW


class TestParserPolarity:
    def test_lower_is_better_healthy(self):
        p = Parser(
            baselines={"err": 0.01},
            thresholds=Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
        )
        e = p.parse({"err": 0.005})
        assert e.health_of("err") == Health.INTACT

    def test_lower_is_better_ablated(self):
        p = Parser(
            baselines={"err": 0.01},
            thresholds=Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
        )
        e = p.parse({"err": 0.15})
        assert e.health_of("err") == Health.ABLATED

    def test_suppress_only_on_over_performing(self):
        p = Parser(
            baselines={"err": 0.01},
            thresholds=Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
        )
        # Below baseline (better) with correction → SUPPRESS
        e1 = p.parse({"err": 0.001}, correction_magnitude=1.0, alpha=0.5)
        assert e1.corrections[0].op == Op.SUPPRESS

        # Above ablated (degraded) with correction → RESTORE
        e2 = p.parse({"err": 0.15}, correction_magnitude=1.0, alpha=0.5)
        assert e2.corrections[0].op == Op.RESTORE

    def test_mixed_polarity_targets_worst(self):
        p = Parser(
            baselines={"throughput": 100.0, "err": 0.01},
            thresholds=Thresholds(intact=80.0, ablated=30.0),
            component_thresholds={
                "err": Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
            },
        )
        # throughput is fine, error rate is spiking
        e = p.parse({"throughput": 95.0, "err": 0.08}, correction_magnitude=1.0, alpha=0.5)
        assert e.corrections[0].target == "err"
        assert e.corrections[0].op == Op.RESTORE
