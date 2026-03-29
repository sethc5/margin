import pytest
from margin.policy.core import Policy, PolicyRule, Action, Constraint
from margin.policy.tuning import (
    analyze_backtest, suggest_tuning, apply_tuning, RuleStats,
)
from margin.observation import Observation, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger
from margin.predicates import any_health, any_degraded


def _obs(name, health, value=50.0):
    return Observation(name, health, value, 100.0, Confidence.HIGH)


def _rec(step, name, health, fired=False, op=Op.NOOP, alpha=0.0, beneficial=True):
    before = _obs(name, health)
    after = _obs(name, Health.INTACT if beneficial else Health.ABLATED,
                 90.0 if beneficial else 10.0) if fired else None
    return Record(step=step, tag=f"s{step}", before=before, after=after,
                  fired=fired, op=op, alpha=alpha)


class TestAnalyzeBacktest:
    def test_basic_stats(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="restore", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE, alpha=0.5, beneficial=True),
            _rec(1, "x", Health.DEGRADED, fired=True, op=Op.RESTORE, alpha=0.5, beneficial=True),
            _rec(2, "x", Health.INTACT),
        ])
        bt = policy.backtest(ledger)
        stats = analyze_backtest(policy, bt)
        assert len(stats) == 1
        assert stats[0].name == "restore"
        assert stats[0].times_proposed >= 2

    def test_empty_backtest(self):
        policy = Policy(name="p", rules=[])
        stats = analyze_backtest(policy, [])
        assert len(stats) == 0


class TestSuggestTuning:
    def test_reduce_harmful(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="aggressive", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.9)),
        ])
        # Simulate backtest where corrections are harmful
        stats = [RuleStats(
            name="aggressive",
            times_proposed=10,
            times_matched_actual=8,
            times_actual_beneficial=2,
            times_actual_harmful=6,
            mean_proposed_alpha=0.9,
        )]
        suggestions = suggest_tuning(policy, stats, harmful_threshold=0.3)
        assert len(suggestions) == 1
        assert suggestions[0].suggested_alpha < 0.9
        assert "harmful" in suggestions[0].reason

    def test_boost_beneficial(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="good", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.4)),
        ])
        stats = [RuleStats(
            name="good",
            times_proposed=10,
            times_matched_actual=8,
            times_actual_beneficial=7,
            times_actual_harmful=1,
            mean_proposed_alpha=0.4,
        )]
        suggestions = suggest_tuning(policy, stats)
        assert len(suggestions) == 1
        assert suggestions[0].suggested_alpha > 0.4
        assert "beneficial" in suggestions[0].reason

    def test_no_suggestions_when_neutral(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="neutral", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])
        stats = [RuleStats(
            name="neutral",
            times_proposed=10,
            times_matched_actual=10,
            times_actual_beneficial=6,
            times_actual_harmful=2,  # 20% harmful — below 30% threshold
        )]
        suggestions = suggest_tuning(policy, stats)
        assert len(suggestions) == 0  # not harmful enough to reduce, not beneficial enough to boost

    def test_respects_max_alpha(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="good", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.95)),
        ])
        stats = [RuleStats(name="good", times_proposed=10, times_matched_actual=10,
                           times_actual_beneficial=9, times_actual_harmful=1)]
        suggestions = suggest_tuning(policy, stats, max_alpha=1.0)
        if suggestions:
            assert suggestions[0].suggested_alpha <= 1.0


class TestApplyTuning:
    def test_applies_suggestions(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="r1", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.9), priority=10),
            PolicyRule(name="r2", predicate=any_health(Health.ABLATED),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.5), priority=20),
        ])
        from margin.policy.tuning import TuningResult
        suggestions = [TuningResult(rule_name="r1", current_alpha=0.9, suggested_alpha=0.6, reason="test")]

        tuned = apply_tuning(policy, suggestions)
        assert tuned.name == "p-tuned"
        assert tuned.rules[0].action.alpha == 0.6  # adjusted
        assert tuned.rules[1].action.alpha == 0.5  # unchanged
        # Original not modified
        assert policy.rules[0].action.alpha == 0.9

    def test_preserves_other_fields(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="r1", predicate=any_degraded(),
                       action=Action(target="x", op=Op.SUPPRESS, alpha=0.7, alpha_from_sigma=True),
                       priority=5, constraint=Constraint(max_alpha=0.9)),
        ])
        from margin.policy.tuning import TuningResult
        tuned = apply_tuning(policy, [TuningResult("r1", 0.7, 0.4, "test")])
        r = tuned.rules[0]
        assert r.action.alpha == 0.4
        assert r.action.op == Op.SUPPRESS
        assert r.action.target == "x"
        assert r.action.alpha_from_sigma is True
        assert r.priority == 5
        assert r.constraint.max_alpha == 0.9
