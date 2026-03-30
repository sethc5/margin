import pytest
from margin.policy.core import Policy, PolicyRule, Action, Constraint, Escalation, EscalationLevel
from margin.policy.trace import trace_evaluate, trace_backtest, DecisionTrace
from margin.observation import Observation, Expression, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger
from margin.predicates import any_health, all_health, any_degraded


def _obs(name, health, value=50.0):
    return Observation(name, health, value, 100.0, Confidence.HIGH)


def _expr(*observations):
    return Expression(observations=list(observations), confidence=Confidence.HIGH)


def _rec(step, name, health, fired=False, op=Op.NOOP, alpha=0.0):
    return Record(step=step, tag=f"s{step}", before=_obs(name, health),
                  fired=fired, op=op, alpha=alpha)


class TestTraceEvaluate:
    def _make_policy(self):
        return Policy(
            name="test",
            rules=[
                PolicyRule("critical", any_health(Health.ABLATED),
                           Action(target="*", op=Op.RESTORE, alpha=1.0), priority=50),
                PolicyRule("moderate", any_health(Health.DEGRADED),
                           Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
                PolicyRule("all-clear", all_health(Health.INTACT),
                           Action(target="*", op=Op.NOOP), priority=0),
            ],
        )

    def test_records_all_rules(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.ABLATED, 10.0)))
        assert dt.rules_considered == 3
        assert dt.rules_matched >= 1

    def test_winner_is_highest_priority(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.ABLATED, 10.0)))
        assert dt.winner == "critical"
        assert dt.was_correction

    def test_all_clear(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.INTACT, 90.0)))
        assert dt.winner == "all-clear"

    def test_no_match_returns_none(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.RECOVERING, 50.0)))
        assert dt.winner is None
        assert dt.result is None

    def test_default_escalation(self):
        policy = Policy(
            name="test",
            rules=[PolicyRule("r", all_health(Health.INTACT), Action(target="*", op=Op.NOOP))],
            default_escalation=Escalation(EscalationLevel.LOG, "no match"),
        )
        dt = trace_evaluate(policy, _expr(_obs("x", Health.DEGRADED)))
        assert dt.winner == "default"
        assert dt.was_escalation

    def test_trace_shows_non_matching_rules(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.INTACT, 90.0)))
        non_matching = [e for e in dt.evaluations if not e.matched]
        assert len(non_matching) >= 1  # critical and moderate don't match

    def test_trace_with_causal_context(self):
        context = {"x": {"component": "x", "causes": [{"source": "db", "cause_type": "DEGRADES"}]}}
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.ABLATED)), causal_context=context)
        assert dt.causal_context is not None
        assert "x" in dt.causal_context

    def test_to_string(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.ABLATED, 10.0)))
        s = dt.to_string()
        assert "critical" in s
        assert "winner" in s
        assert "RESTORE" in s

    def test_to_dict(self):
        dt = trace_evaluate(self._make_policy(), _expr(_obs("x", Health.ABLATED, 10.0)))
        d = dt.to_dict()
        assert d["winner"] == "critical"
        assert d["rules_matched"] >= 1
        assert d["result_type"] == "correction"

    def test_constraint_suppression_traced(self):
        policy = Policy(name="test", rules=[
            PolicyRule("r", any_degraded(),
                       Action(target="*", op=Op.RESTORE, alpha=0.1),
                       constraint=Constraint(min_alpha=0.5)),
        ])
        dt = trace_evaluate(policy, _expr(_obs("x", Health.DEGRADED)))
        # Rule matches but constraint suppresses
        r_eval = dt.evaluations[0]
        assert r_eval.matched
        assert r_eval.outcome == "escalation"  # suppressed → escalation
        assert "suppressed" in r_eval.reason


class TestTraceBacktest:
    def test_returns_traces(self):
        policy = Policy(name="test", rules=[
            PolicyRule("r", any_degraded(),
                       Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.INTACT),
            _rec(2, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
        ])
        traces = trace_backtest(policy, ledger)
        assert len(traces) == 3
        assert all(isinstance(t, DecisionTrace) for t in traces)
        assert traces[0].winner == "r"
        assert traces[1].winner is None  # INTACT, no match


class TestBacktestProposedBy:
    def test_backtest_includes_rule_name(self):
        policy = Policy(name="test", rules=[
            PolicyRule("my-rule", any_degraded(),
                       Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE)])
        bt = policy.backtest(ledger)
        assert bt[0]["proposed_by"] == "my-rule"

    def test_backtest_no_match_empty_rule(self):
        policy = Policy(name="test", rules=[])
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        bt = policy.backtest(ledger)
        assert bt[0]["proposed_by"] == ""


class TestMultiRule:
    def _make_multi_policy(self):
        from margin.policy.core import Policy, PolicyRule, Action
        from margin.observation import Op
        from margin.predicates import any_health, any_degraded
        return Policy(
            name="multi",
            multi_rule=True,
            rules=[
                PolicyRule("log", any_degraded(),
                           Action(target="*", op=Op.NOOP, alpha=0.0), priority=5),
                PolicyRule("restore", any_health(Health.DEGRADED),
                           Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
            ],
        )

    def test_multi_rule_all_matching_rules_fire(self):
        policy = self._make_multi_policy()
        dt = trace_evaluate(policy, _expr(_obs("x", Health.DEGRADED)))
        # Both rules match DEGRADED
        assert len(dt.results) == 2

    def test_multi_rule_winner_is_highest_priority(self):
        policy = self._make_multi_policy()
        dt = trace_evaluate(policy, _expr(_obs("x", Health.DEGRADED)))
        assert dt.winner == "restore"
        assert dt.result is not None

    def test_trace_always_has_all_matched_results(self):
        # trace_evaluate always populates dt.results for all matches;
        # multi_rule only gates StepResult.corrections in step()
        from margin.policy.core import Policy, PolicyRule, Action
        from margin.observation import Op
        from margin.predicates import any_degraded
        policy = Policy(
            name="single",
            multi_rule=False,
            rules=[
                PolicyRule("r1", any_degraded(), Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
                PolicyRule("r2", any_degraded(), Action(target="*", op=Op.NOOP), priority=5),
            ],
        )
        dt = trace_evaluate(policy, _expr(_obs("x", Health.DEGRADED)))
        # Both rules match — trace always records all; multi_rule only affects step()
        assert len(dt.results) == 2
        assert dt.winner == "r1"  # highest priority still wins

    def test_multi_rule_to_dict_includes_multi_rule_flag(self):
        policy = self._make_multi_policy()
        d = policy.to_dict()
        assert d["multi_rule"] is True

    def test_multi_rule_default_is_false(self):
        from margin.policy.core import Policy
        policy = Policy(name="p", rules=[])
        assert policy.multi_rule is False
        assert policy.to_dict()["multi_rule"] is False


class TestStepResultCorrections:
    def test_step_corrections_populated_multi_rule(self):
        from margin.loop import step
        from margin.policy.core import Policy, PolicyRule, Action
        from margin.observation import Op
        from margin.predicates import any_degraded

        policy = Policy(
            name="multi",
            multi_rule=True,
            rules=[
                PolicyRule("r1", any_degraded(), Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
                PolicyRule("r2", any_degraded(), Action(target="*", op=Op.NOOP), priority=5),
            ],
        )
        result = step(_expr(_obs("x", Health.DEGRADED)), policy)
        assert len(result.corrections) >= 1

    def test_step_corrections_empty_single_rule(self):
        from margin.loop import step
        from margin.policy.core import Policy, PolicyRule, Action
        from margin.observation import Op
        from margin.predicates import any_degraded

        policy = Policy(
            name="single",
            multi_rule=False,
            rules=[PolicyRule("r", any_degraded(), Action(target="*", op=Op.RESTORE, alpha=0.5))],
        )
        result = step(_expr(_obs("x", Health.DEGRADED)), policy)
        # multi_rule=False → corrections list is empty (only .correction populated)
        assert result.corrections == []
