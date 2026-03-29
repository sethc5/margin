"""Tests for the policy language."""

import pytest
from margin.policy import (
    EscalationLevel, Escalation, Action, Constraint, PolicyRule, Policy,
)
from margin.observation import Observation, Correction, Expression, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger
from margin.predicates import any_health, all_health, any_degraded, confidence_below


def _obs(name, health, value=50.0, baseline=100.0, conf=Confidence.HIGH):
    return Observation(name, health, value, baseline, conf)


def _expr(*observations, conf=None):
    obs = list(observations)
    c = conf or min((o.confidence for o in obs), default=Confidence.MODERATE)
    return Expression(observations=obs, confidence=c)


def _rec(step, name, health, fired=False, op=Op.NOOP, alpha=0.0):
    return Record(
        step=step, tag=f"s{step}",
        before=_obs(name, health),
        fired=fired, op=op, alpha=alpha,
    )


class TestAction:
    def test_resolve_fixed(self):
        a = Action(target="x", op=Op.RESTORE, alpha=0.6, magnitude=2.0)
        expr = _expr(_obs("x", Health.DEGRADED))
        c = a.resolve(expr)
        assert c.target == "x"
        assert c.op == Op.RESTORE
        assert c.alpha == 0.6
        assert c.magnitude == 2.0

    def test_resolve_wildcard_targets_worst(self):
        a = Action(target="*", op=Op.RESTORE, alpha=0.5)
        expr = _expr(
            _obs("a", Health.INTACT, 90.0),
            _obs("b", Health.DEGRADED, 50.0),
            _obs("c", Health.ABLATED, 10.0),
        )
        c = a.resolve(expr)
        assert c.target == "c"  # worst = ABLATED

    def test_resolve_wildcard_no_degraded(self):
        a = Action(target="*", op=Op.RESTORE)
        expr = _expr(_obs("a", Health.INTACT, 90.0))
        c = a.resolve(expr)
        assert c.target == "a"  # falls back to first observation

    def test_resolve_wildcard_empty(self):
        a = Action(target="*", op=Op.RESTORE)
        expr = Expression()
        c = a.resolve(expr)
        assert c.op == Op.NOOP

    def test_alpha_from_sigma(self):
        a = Action(target="x", op=Op.RESTORE, alpha_from_sigma=True)
        # sigma = (50-100)/100 = -0.5 → alpha = 0.5
        expr = _expr(_obs("x", Health.DEGRADED, 50.0, 100.0))
        c = a.resolve(expr)
        assert c.alpha == pytest.approx(0.5)

    def test_alpha_from_sigma_capped_at_one(self):
        a = Action(target="x", op=Op.RESTORE, alpha_from_sigma=True)
        # sigma = (10-100)/100 = -0.9 → |sigma| = 0.9
        expr = _expr(_obs("x", Health.ABLATED, 10.0, 100.0))
        c = a.resolve(expr)
        assert c.alpha <= 1.0

    def test_roundtrip(self):
        a = Action(target="x", op=Op.SUPPRESS, alpha=0.3, alpha_from_sigma=True)
        a2 = Action.from_dict(a.to_dict())
        assert a2.target == "x"
        assert a2.op == Op.SUPPRESS
        assert a2.alpha == 0.3
        assert a2.alpha_from_sigma is True


class TestConstraint:
    def test_clamps_alpha(self):
        c = Constraint(max_alpha=0.5)
        correction = Correction("x", Op.RESTORE, alpha=0.8, magnitude=1.0)
        result = c.check(correction)
        assert result.alpha == 0.5

    def test_suppresses_below_min_alpha(self):
        c = Constraint(min_alpha=0.3)
        correction = Correction("x", Op.RESTORE, alpha=0.1, magnitude=1.0)
        assert c.check(correction) is None

    def test_cooldown(self):
        c = Constraint(cooldown_steps=3)
        correction = Correction("x", Op.RESTORE, alpha=0.5, magnitude=1.0)

        # Empty ledger — should pass
        assert c.check(correction, Ledger()) is not None

        # Ledger with recent fire for x
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.RECOVERING),
        ])
        assert c.check(correction, ledger) is None

    def test_max_per_window(self):
        c = Constraint(max_per_window=2, window_steps=5)
        correction = Correction("x", Op.RESTORE, alpha=0.5, magnitude=1.0)

        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(2, "x", Health.RECOVERING),
        ])
        assert c.check(correction, ledger) is None

    def test_no_ledger_ignores_cooldown(self):
        c = Constraint(cooldown_steps=3)
        correction = Correction("x", Op.RESTORE, alpha=0.5, magnitude=1.0)
        assert c.check(correction) is not None

    def test_roundtrip(self):
        c = Constraint(max_alpha=0.8, cooldown_steps=5, max_per_window=3, window_steps=20)
        c2 = Constraint.from_dict(c.to_dict())
        assert c2.max_alpha == 0.8
        assert c2.cooldown_steps == 5


class TestPolicyRule:
    def test_match_and_correct(self):
        rule = PolicyRule(
            name="restore-ablated",
            predicate=any_health(Health.ABLATED),
            action=Action(target="*", op=Op.RESTORE, alpha=0.8),
        )
        expr = _expr(_obs("x", Health.ABLATED, 10.0))
        result = rule.evaluate(expr)
        assert isinstance(result, Correction)
        assert result.op == Op.RESTORE
        assert result.alpha == 0.8

    def test_no_match_returns_none(self):
        rule = PolicyRule(
            name="restore-ablated",
            predicate=any_health(Health.ABLATED),
            action=Action(target="*", op=Op.RESTORE),
        )
        expr = _expr(_obs("x", Health.INTACT, 90.0))
        assert rule.evaluate(expr) is None

    def test_confidence_gate_escalates(self):
        rule = PolicyRule(
            name="restore",
            predicate=any_health(Health.DEGRADED),
            action=Action(target="*", op=Op.RESTORE),
            min_confidence=Confidence.HIGH,
        )
        expr = _expr(_obs("x", Health.DEGRADED, 50.0, conf=Confidence.LOW), conf=Confidence.LOW)
        result = rule.evaluate(expr)
        assert isinstance(result, Escalation)
        assert result.level == EscalationLevel.ALERT
        assert "confidence" in result.reason

    def test_escalation_only_rule(self):
        rule = PolicyRule(
            name="halt-on-ood",
            predicate=any_health(Health.OOD),
            action=Action(),
            escalation=Escalation(EscalationLevel.HALT, "OOD detected"),
            min_confidence=Confidence.INDETERMINATE,  # let OOD expressions through
        )
        expr = _expr(_obs("x", Health.OOD, 0.0, conf=Confidence.INDETERMINATE),
                      conf=Confidence.INDETERMINATE)
        result = rule.evaluate(expr)
        assert isinstance(result, Escalation)
        assert result.level == EscalationLevel.HALT

    def test_constraint_suppression(self):
        rule = PolicyRule(
            name="restore",
            predicate=any_health(Health.DEGRADED),
            action=Action(target="*", op=Op.RESTORE, alpha=0.1),
            constraint=Constraint(min_alpha=0.3),
        )
        expr = _expr(_obs("x", Health.DEGRADED, 50.0))
        result = rule.evaluate(expr)
        assert isinstance(result, Escalation)
        assert "suppressed" in result.reason

    def test_roundtrip(self):
        rule = PolicyRule(
            name="test",
            predicate=any_degraded(),
            action=Action(target="x", op=Op.RESTORE, alpha=0.6),
            priority=10,
            constraint=Constraint(max_alpha=0.8),
        )
        d = rule.to_dict()
        rule2 = PolicyRule.from_dict(d, any_degraded())
        assert rule2.name == "test"
        assert rule2.priority == 10
        assert rule2.action.alpha == 0.6
        assert rule2.constraint.max_alpha == 0.8


class TestPolicy:
    def _make_policy(self):
        return Policy(
            name="test-policy",
            rules=[
                PolicyRule(
                    name="critical",
                    predicate=any_health(Health.ABLATED),
                    action=Action(target="*", op=Op.RESTORE, alpha=1.0),
                    priority=50,
                ),
                PolicyRule(
                    name="moderate",
                    predicate=any_health(Health.DEGRADED),
                    action=Action(target="*", op=Op.RESTORE, alpha=0.5),
                    priority=10,
                ),
                PolicyRule(
                    name="all-clear",
                    predicate=all_health(Health.INTACT),
                    action=Action(target="*", op=Op.NOOP),
                    priority=0,
                ),
            ],
            default_escalation=Escalation(EscalationLevel.LOG, "no rules matched"),
        )

    def test_highest_priority_wins(self):
        policy = self._make_policy()
        # ABLATED matches both "critical" and "moderate" (ABLATED is in degraded())
        expr = _expr(_obs("x", Health.ABLATED, 10.0))
        result = policy.evaluate_first(expr)
        assert isinstance(result, Correction)
        assert result.alpha == 1.0  # critical rule won

    def test_moderate_when_not_ablated(self):
        policy = self._make_policy()
        expr = _expr(_obs("x", Health.DEGRADED, 50.0))
        result = policy.evaluate_first(expr)
        assert isinstance(result, Correction)
        assert result.alpha == 0.5

    def test_all_clear(self):
        policy = self._make_policy()
        expr = _expr(_obs("x", Health.INTACT, 90.0))
        result = policy.evaluate_first(expr)
        assert isinstance(result, Correction)
        assert result.op == Op.NOOP

    def test_default_escalation(self):
        policy = self._make_policy()
        # No observations match any rule... actually all_health(INTACT) won't match
        # empty, and no other rule matches. But let's make a weird expression.
        expr = _expr(_obs("x", Health.RECOVERING, 50.0))
        # RECOVERING is not ABLATED, not DEGRADED (for any_health checks), not all INTACT
        # Actually any_health(Health.DEGRADED) checks health == DEGRADED, not "in degraded()"
        result = policy.evaluate_first(expr)
        assert isinstance(result, Escalation)
        assert result.level == EscalationLevel.LOG

    def test_evaluate_returns_all(self):
        policy = self._make_policy()
        expr = _expr(_obs("x", Health.ABLATED, 10.0))
        results = policy.evaluate(expr)
        assert len(results) >= 1

    def test_no_rules_no_default(self):
        policy = Policy(name="empty", rules=[])
        result = policy.evaluate_first(_expr(_obs("x", Health.DEGRADED)))
        assert result is None

    def test_with_ledger_constraint(self):
        policy = Policy(
            name="constrained",
            rules=[PolicyRule(
                name="restore",
                predicate=any_degraded(),
                action=Action(target="*", op=Op.RESTORE, alpha=0.5),
                constraint=Constraint(cooldown_steps=3),
            )],
        )
        expr = _expr(_obs("x", Health.DEGRADED, 50.0))
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
        ])
        result = policy.evaluate_first(expr, ledger)
        assert isinstance(result, Escalation)  # cooldown suppressed

    def test_repr(self):
        policy = self._make_policy()
        assert "test-policy" in repr(policy)
        assert "3 rules" in repr(policy)


class TestBacktest:
    def test_basic_backtest(self):
        policy = Policy(
            name="test",
            rules=[
                PolicyRule(
                    name="restore",
                    predicate=any_health(Health.DEGRADED),
                    action=Action(target="*", op=Op.RESTORE, alpha=0.5),
                ),
            ],
        )
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE, alpha=0.5),
            _rec(1, "x", Health.INTACT, fired=False),
            _rec(2, "x", Health.DEGRADED, fired=True, op=Op.RESTORE, alpha=0.8),
        ])
        results = policy.backtest(ledger)
        assert len(results) == 3
        assert results[0]["match"] is True   # both RESTORE
        assert results[1]["proposed_op"] == "NONE"  # INTACT, no rule matches
        assert results[2]["match"] is True   # both RESTORE
        assert results[2]["proposed_alpha"] == 0.5  # policy says 0.5, actual was 0.8

    def test_backtest_agreement_rate(self):
        policy = Policy(
            name="test",
            rules=[PolicyRule(
                name="r", predicate=any_degraded(),
                action=Action(target="*", op=Op.RESTORE, alpha=0.5),
            )],
        )
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(2, "x", Health.INTACT),
        ])
        results = policy.backtest(ledger)
        agreement = sum(1 for r in results if r["match"]) / len(results)
        assert agreement == pytest.approx(2 / 3)


class TestSerialization:
    def test_escalation_roundtrip(self):
        e = Escalation(EscalationLevel.ALERT, "too hot", "rule-1")
        e2 = Escalation.from_dict(e.to_dict())
        assert e2.level == EscalationLevel.ALERT
        assert e2.reason == "too hot"

    def test_policy_roundtrip(self):
        policy = Policy(
            name="prod",
            rules=[
                PolicyRule(
                    name="critical",
                    predicate=any_health(Health.ABLATED),
                    action=Action(target="*", op=Op.RESTORE, alpha=1.0),
                    priority=50,
                    constraint=Constraint(max_alpha=0.9),
                ),
            ],
            default_escalation=Escalation(EscalationLevel.LOG, "fallback"),
        )
        d = policy.to_dict()
        registry = {"critical": any_health(Health.ABLATED)}
        policy2 = Policy.from_dict(d, registry)
        assert policy2.name == "prod"
        assert len(policy2.rules) == 1
        assert policy2.rules[0].priority == 50
        assert policy2.default_escalation.level == EscalationLevel.LOG

    def test_from_dict_missing_predicate_raises(self):
        d = {"name": "test", "rules": [{"name": "x", "action": {"op": "RESTORE"}}]}
        with pytest.raises(ValueError, match="No predicate"):
            Policy.from_dict(d, {})
