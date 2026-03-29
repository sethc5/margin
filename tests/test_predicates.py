import pytest
from margin.predicates import (
    any_health, all_health, count_health, component_health,
    any_degraded, confidence_below, sigma_below, any_correction,
    all_of, any_of, not_, Rule, evaluate_rules,
)
from margin.observation import Observation, Correction, Expression, Op
from margin.health import Health
from margin.confidence import Confidence


def _obs(name, health, value=90.0, baseline=100.0):
    return Observation(name, health, value, baseline, Confidence.HIGH)


def _expr(*observations, corrections=None):
    return Expression(
        observations=list(observations),
        corrections=corrections or [],
        confidence=min((o.confidence for o in observations), default=Confidence.MODERATE),
    )


class TestBasicPredicates:
    def test_any_health_true(self):
        e = _expr(_obs("a", Health.INTACT), _obs("b", Health.ABLATED))
        assert any_health(Health.ABLATED)(e) is True

    def test_any_health_false(self):
        e = _expr(_obs("a", Health.INTACT), _obs("b", Health.INTACT))
        assert any_health(Health.ABLATED)(e) is False

    def test_all_health_true(self):
        e = _expr(_obs("a", Health.INTACT), _obs("b", Health.INTACT))
        assert all_health(Health.INTACT)(e) is True

    def test_all_health_false(self):
        e = _expr(_obs("a", Health.INTACT), _obs("b", Health.DEGRADED))
        assert all_health(Health.INTACT)(e) is False

    def test_all_health_empty(self):
        assert all_health(Health.INTACT)(Expression()) is False

    def test_count_health(self):
        e = _expr(
            _obs("a", Health.DEGRADED),
            _obs("b", Health.DEGRADED),
            _obs("c", Health.INTACT),
        )
        assert count_health(Health.DEGRADED, 2)(e) is True
        assert count_health(Health.DEGRADED, 3)(e) is False

    def test_component_health(self):
        e = _expr(_obs("a", Health.INTACT), _obs("b", Health.ABLATED))
        assert component_health("b", Health.ABLATED)(e) is True
        assert component_health("a", Health.ABLATED)(e) is False

    def test_any_degraded(self):
        e1 = _expr(_obs("a", Health.INTACT))
        assert any_degraded()(e1) is False

        e2 = _expr(_obs("a", Health.DEGRADED))
        assert any_degraded()(e2) is True

        e3 = _expr(_obs("a", Health.RECOVERING))
        assert any_degraded()(e3) is True

    def test_confidence_below(self):
        e = Expression(
            observations=[_obs("a", Health.INTACT)],
            confidence=Confidence.LOW,
        )
        assert confidence_below(Confidence.MODERATE)(e) is True
        assert confidence_below(Confidence.LOW)(e) is False

    def test_sigma_below(self):
        e = _expr(Observation("a", Health.DEGRADED, 50.0, 100.0, Confidence.HIGH))
        # sigma = (50-100)/100 = -0.5
        assert sigma_below("a", -0.3)(e) is True
        assert sigma_below("a", -0.6)(e) is False

    def test_sigma_below_missing_component(self):
        e = _expr(_obs("a", Health.INTACT))
        assert sigma_below("nonexistent", 0.0)(e) is False

    def test_any_correction(self):
        e1 = _expr(_obs("a", Health.INTACT))
        assert any_correction()(e1) is False

        e2 = _expr(
            _obs("a", Health.INTACT),
            corrections=[Correction("a", Op.RESTORE, 0.5, 1.0)],
        )
        assert any_correction()(e2) is True


class TestCombinators:
    def test_all_of(self):
        e = _expr(_obs("a", Health.ABLATED), _obs("b", Health.DEGRADED))
        p = all_of(any_health(Health.ABLATED), any_health(Health.DEGRADED))
        assert p(e) is True

    def test_all_of_fails_if_one_false(self):
        e = _expr(_obs("a", Health.INTACT))
        p = all_of(any_health(Health.INTACT), any_health(Health.ABLATED))
        assert p(e) is False

    def test_any_of(self):
        e = _expr(_obs("a", Health.INTACT))
        p = any_of(any_health(Health.ABLATED), any_health(Health.INTACT))
        assert p(e) is True

    def test_not(self):
        e = _expr(_obs("a", Health.INTACT))
        assert not_(any_health(Health.ABLATED))(e) is True
        assert not_(any_health(Health.INTACT))(e) is False


class TestRules:
    def test_rule_matches(self):
        r = Rule("critical", any_health(Health.ABLATED))
        e = _expr(_obs("a", Health.ABLATED))
        assert r.matches(e) is True

    def test_rule_no_match(self):
        r = Rule("critical", any_health(Health.ABLATED))
        e = _expr(_obs("a", Health.INTACT))
        assert r.matches(e) is False

    def test_evaluate_rules(self):
        rules = [
            Rule("all-clear", all_health(Health.INTACT)),
            Rule("critical", any_health(Health.ABLATED)),
            Rule("any-problem", any_degraded()),
        ]

        e_healthy = _expr(_obs("a", Health.INTACT), _obs("b", Health.INTACT))
        matched = evaluate_rules(rules, e_healthy)
        assert [r.name for r in matched] == ["all-clear"]

        e_ablated = _expr(_obs("a", Health.ABLATED), _obs("b", Health.INTACT))
        matched = evaluate_rules(rules, e_ablated)
        assert "critical" in [r.name for r in matched]
        assert "any-problem" in [r.name for r in matched]
        assert "all-clear" not in [r.name for r in matched]

    def test_repr(self):
        r = Rule("test", any_health(Health.INTACT))
        assert "test" in repr(r)
