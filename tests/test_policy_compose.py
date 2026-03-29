import pytest
from margin.policy.core import (
    Policy, PolicyRule, Action, Constraint, Escalation, EscalationLevel,
)
from margin.policy.compose import (
    PolicyChain, CorrectionBundle, bundle_from_policy,
    diff_policies, agreement_rate,
)
from margin.observation import Observation, Expression, Correction, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger
from margin.predicates import any_health, all_health, any_degraded


def _obs(name, health, value=50.0):
    return Observation(name, health, value, 100.0, Confidence.HIGH)


def _expr(*observations):
    obs = list(observations)
    return Expression(observations=obs, confidence=Confidence.HIGH)


def _rec(step, name, health, fired=False, op=Op.NOOP, alpha=0.0):
    return Record(step=step, tag=f"s{step}", before=_obs(name, health), fired=fired, op=op, alpha=alpha)


class TestPolicyChain:
    def _safety_policy(self):
        return Policy(name="safety", rules=[
            PolicyRule(name="halt-ood", predicate=any_health(Health.OOD),
                       action=Action(), escalation=Escalation(EscalationLevel.HALT, "OOD"),
                       min_confidence=Confidence.INDETERMINATE),
        ])

    def _perf_policy(self):
        return Policy(name="perf", rules=[
            PolicyRule(name="restore", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])

    def test_first_mode(self):
        chain = PolicyChain(name="test", policies=[self._safety_policy(), self._perf_policy()], mode="first")
        # No OOD → safety produces nothing → perf produces RESTORE
        expr = _expr(_obs("x", Health.DEGRADED))
        result = chain.evaluate_first(expr)
        assert isinstance(result, Correction)
        assert result.op == Op.RESTORE

    def test_veto_mode_halts(self):
        chain = PolicyChain(name="test", policies=[self._safety_policy(), self._perf_policy()], mode="veto")
        obs = Observation("x", Health.OOD, 0.0, 100.0, Confidence.INDETERMINATE)
        expr = Expression(observations=[obs], confidence=Confidence.INDETERMINATE)
        result = chain.evaluate_first(expr)
        assert isinstance(result, Escalation)
        assert result.level == EscalationLevel.HALT

    def test_veto_mode_passes_through(self):
        chain = PolicyChain(name="test", policies=[self._safety_policy(), self._perf_policy()], mode="veto")
        expr = _expr(_obs("x", Health.DEGRADED))
        result = chain.evaluate_first(expr)
        assert isinstance(result, Correction)

    def test_all_mode_collects(self):
        p1 = Policy(name="p1", rules=[
            PolicyRule(name="r1", predicate=any_degraded(),
                       action=Action(target="a", op=Op.RESTORE, alpha=0.3)),
        ])
        p2 = Policy(name="p2", rules=[
            PolicyRule(name="r2", predicate=any_degraded(),
                       action=Action(target="b", op=Op.SUPPRESS, alpha=0.2)),
        ])
        chain = PolicyChain(name="test", policies=[p1, p2], mode="all")
        expr = _expr(_obs("a", Health.DEGRADED), _obs("b", Health.DEGRADED))
        results = chain.evaluate(expr)
        ops = [r.op for r in results if isinstance(r, Correction)]
        assert Op.RESTORE in ops
        assert Op.SUPPRESS in ops

    def test_repr(self):
        chain = PolicyChain(name="c", policies=[], mode="veto")
        assert "veto" in repr(chain)


class TestCorrectionBundle:
    def test_bundle_from_policy(self):
        policy = Policy(name="multi", rules=[
            PolicyRule(name="r1", predicate=any_health(Health.ABLATED),
                       action=Action(target="a", op=Op.RESTORE, alpha=0.8), priority=10),
            PolicyRule(name="r2", predicate=any_health(Health.DEGRADED),
                       action=Action(target="b", op=Op.RESTORE, alpha=0.3), priority=5),
        ])
        expr = _expr(_obs("a", Health.ABLATED, 10.0), _obs("b", Health.DEGRADED))
        bundle = bundle_from_policy(policy, expr)
        assert bundle.n_active >= 1
        assert len(bundle.targets) >= 1

    def test_empty_bundle(self):
        policy = Policy(name="empty", rules=[])
        bundle = bundle_from_policy(policy, Expression())
        assert bundle.n_active == 0

    def test_repr(self):
        b = CorrectionBundle(
            corrections=[Correction("a", Op.RESTORE, 0.5), Correction("b", Op.SUPPRESS, 0.3)],
            reason="test",
        )
        assert "a" in repr(b) and "b" in repr(b)


class TestDiffPolicies:
    def test_same_policy_agrees(self):
        policy = Policy(name="p", rules=[
            PolicyRule(name="r", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.5)),
        ])
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.INTACT),
        ])
        comps = diff_policies(policy, policy, ledger)
        assert all(c.agree for c in comps)
        assert agreement_rate(comps) == 1.0

    def test_different_policies_disagree(self):
        p1 = Policy(name="conservative", rules=[
            PolicyRule(name="r", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE, alpha=0.3)),
        ])
        p2 = Policy(name="aggressive", rules=[
            PolicyRule(name="r", predicate=any_degraded(),
                       action=Action(target="*", op=Op.SUPPRESS, alpha=0.9)),
        ])
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        comps = diff_policies(p1, p2, ledger)
        assert not comps[0].agree
        assert comps[0].policy_a_op == "RESTORE"
        assert comps[0].policy_b_op == "SUPPRESS"

    def test_empty_ledger(self):
        policy = Policy(name="p", rules=[])
        assert agreement_rate(diff_policies(policy, policy, Ledger())) == 0.0

    def test_to_dict(self):
        p = Policy(name="p", rules=[
            PolicyRule(name="r", predicate=any_degraded(),
                       action=Action(target="*", op=Op.RESTORE)),
        ])
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        comps = diff_policies(p, p, ledger)
        d = comps[0].to_dict()
        assert "policy_a" in d and "policy_b" in d
