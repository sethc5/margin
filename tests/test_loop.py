import pytest
from margin.loop import step, run, StepResult
from margin.observation import Observation, Expression, Op
from margin.health import Health, Thresholds
from margin.confidence import Confidence
from margin.ledger import Ledger, Record
from margin.causal import CausalGraph
from margin.contract import Contract, HealthTarget, SustainHealth, NoHarmful
from margin.policy.core import Policy, PolicyRule, Action, Constraint, Escalation, EscalationLevel
from margin.predicates import any_health, all_health, any_degraded


def _obs(name, health, value=50.0):
    return Observation(name, health, value, 100.0, Confidence.HIGH)


def _expr(*observations, label=""):
    return Expression(observations=list(observations), confidence=Confidence.HIGH, label=label)


class TestStep:
    def _make_policy(self):
        return Policy(name="test", rules=[
            PolicyRule("critical", any_health(Health.ABLATED),
                       Action(target="*", op=Op.RESTORE, alpha=0.9), priority=50),
            PolicyRule("moderate", any_degraded(),
                       Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
            PolicyRule("clear", all_health(Health.INTACT),
                       Action(target="*", op=Op.NOOP), priority=0),
        ])

    def test_basic_step(self):
        result = step(_expr(_obs("x", Health.DEGRADED)), self._make_policy())
        assert isinstance(result, StepResult)
        assert result.acted
        assert result.correction.op == Op.RESTORE

    def test_step_with_causal(self):
        graph = CausalGraph()
        graph.add_degrades("db", "api", 0.9)
        expr = _expr(_obs("db", Health.ABLATED, 10.0), _obs("api", Health.DEGRADED))
        result = step(expr, self._make_policy(), graph=graph)
        assert "api" in result.explanations
        assert result.explanations["api"].has_known_cause
        assert result.decision.causal_context is not None

    def test_step_with_contract(self):
        contract = Contract("sla", terms=[
            HealthTarget("x-intact", "x", Health.INTACT),
        ])
        expr = _expr(_obs("x", Health.DEGRADED))
        result = step(expr, self._make_policy(), contract=contract)
        assert result.contract is not None
        assert result.contract.any_violated

    def test_step_all_three(self):
        graph = CausalGraph()
        graph.add_degrades("db", "api")
        contract = Contract("sla", terms=[
            HealthTarget("api-ok", "api", Health.INTACT),
        ])
        policy = self._make_policy()
        expr = _expr(_obs("db", Health.ABLATED, 10.0), _obs("api", Health.DEGRADED))
        result = step(expr, policy, graph=graph, contract=contract)

        assert result.acted
        assert result.explanations["api"].has_known_cause
        assert result.contract.any_violated
        assert result.decision.winner == "critical"

    def test_step_intact_noop(self):
        result = step(_expr(_obs("x", Health.INTACT, 90.0)), self._make_policy())
        assert not result.acted
        assert result.correction is None or result.correction.op == Op.NOOP

    def test_step_escalation(self):
        policy = Policy(name="strict", rules=[
            PolicyRule("halt", any_health(Health.OOD),
                       Action(), escalation=Escalation(EscalationLevel.HALT, "OOD"),
                       min_confidence=Confidence.INDETERMINATE),
        ])
        expr = Expression(
            observations=[Observation("x", Health.OOD, 0.0, 100.0, Confidence.INDETERMINATE)],
            confidence=Confidence.INDETERMINATE,
        )
        result = step(expr, policy)
        assert result.escalated
        assert not result.acted

    def test_to_string(self):
        graph = CausalGraph()
        graph.add_degrades("db", "api")
        contract = Contract("sla", terms=[HealthTarget("api-ok", "api", Health.INTACT)])
        policy = self._make_policy()
        expr = _expr(_obs("db", Health.ABLATED, 10.0), _obs("api", Health.DEGRADED))
        result = step(expr, policy, graph=graph, contract=contract)
        s = result.to_string()
        assert "Step:" in s
        assert "Why:" in s
        assert "Decision:" in s
        assert "Action:" in s
        assert "Contract:" in s

    def test_to_dict(self):
        result = step(_expr(_obs("x", Health.DEGRADED)), self._make_policy())
        d = result.to_dict()
        assert "expression" in d
        assert "decision" in d
        assert d["acted"] is True

    def test_contract_met(self):
        contract = Contract("sla", terms=[HealthTarget("x-ok", "x", Health.INTACT)])
        result = step(_expr(_obs("x", Health.INTACT, 90.0)), self._make_policy(), contract=contract)
        assert result.contract_met is True

    def test_contract_met_none_without_contract(self):
        result = step(_expr(_obs("x", Health.INTACT, 90.0)), self._make_policy())
        assert result.contract_met is None


class TestRun:
    def _make_policy(self):
        return Policy(name="test", rules=[
            PolicyRule("restore", any_degraded(),
                       Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
            PolicyRule("clear", all_health(Health.INTACT),
                       Action(target="*", op=Op.NOOP), priority=0),
        ])

    def test_basic_run(self):
        expressions = [
            _expr(_obs("x", Health.INTACT, 90.0), label="healthy"),
            _expr(_obs("x", Health.DEGRADED, 50.0), label="spike"),
            _expr(_obs("x", Health.RECOVERING, 70.0), label="recovering"),
        ]
        results, ledger = run(expressions, self._make_policy())
        assert len(results) == 3
        assert len(ledger) == 3
        assert not results[0].acted  # NOOP for intact
        assert results[1].acted      # RESTORE for degraded

    def test_run_builds_ledger(self):
        expressions = [
            _expr(_obs("x", Health.DEGRADED, 50.0)),
            _expr(_obs("x", Health.DEGRADED, 45.0)),
        ]
        results, ledger = run(expressions, self._make_policy())
        assert ledger.n_fired == 2
        assert all(r.op == Op.RESTORE for r in ledger.records)

    def test_run_with_contract(self):
        contract = Contract("sla", terms=[
            HealthTarget("x-ok", "x", Health.INTACT),
        ])
        expressions = [
            _expr(_obs("x", Health.DEGRADED)),
            _expr(_obs("x", Health.INTACT, 90.0)),
        ]
        results, ledger = run(expressions, self._make_policy(), contract=contract)
        assert results[0].contract.any_violated
        assert results[1].contract.all_met

    def test_run_with_causal(self):
        graph = CausalGraph()
        graph.add_degrades("db", "api")
        expressions = [
            _expr(_obs("db", Health.ABLATED, 10.0), _obs("api", Health.DEGRADED)),
        ]
        results, _ = run(expressions, self._make_policy(), graph=graph)
        assert results[0].explanations["api"].has_known_cause

    def test_run_accumulates_into_existing_ledger(self):
        existing = Ledger(label="prior")
        existing.append(Record(step=0, tag="old", before=_obs("x", Health.INTACT, 90.0)))
        expressions = [_expr(_obs("x", Health.DEGRADED))]
        _, ledger = run(expressions, self._make_policy(), ledger=existing)
        assert len(ledger) == 2  # 1 existing + 1 new

    def test_run_empty(self):
        results, ledger = run([], self._make_policy())
        assert len(results) == 0
        assert len(ledger) == 0

    def test_repr(self):
        result = step(_expr(_obs("x", Health.DEGRADED)), self._make_policy())
        assert "CORRECT" in repr(result)
