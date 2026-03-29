"""
The evaluation loop: orchestrates all five layers into one call.

observe → explain → decide → evaluate → record

Takes an Expression, a CausalGraph, a Policy, a Contract, and a Ledger,
runs each layer in order, and returns a typed StepResult that bundles
everything together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from .observation import Expression, Correction
from .ledger import Ledger, Record
from .causal import CausalGraph, Explanation
from .contract import Contract, ContractResult
from .policy.core import Policy, Escalation
from .policy.trace import DecisionTrace, trace_evaluate


@dataclass
class StepResult:
    """
    Complete typed output of one evaluation loop iteration.

    expression:   the observed state
    explanations: causal explanations per component (if graph provided)
    decision:     full decision trace from the policy
    correction:   the correction to apply (None if escalation or no action)
    escalation:   the escalation if the policy escalated (None otherwise)
    contract:     contract evaluation result (if contract provided)
    """
    expression: Expression
    explanations: dict[str, Explanation] = field(default_factory=dict)
    decision: Optional[DecisionTrace] = None
    correction: Optional[Correction] = None
    escalation: Optional[Escalation] = None
    contract: Optional[ContractResult] = None

    @property
    def acted(self) -> bool:
        """True if a real correction was produced (not NOOP, not escalation)."""
        return self.correction is not None and self.correction.is_active()

    @property
    def escalated(self) -> bool:
        return self.escalation is not None

    @property
    def contract_met(self) -> Optional[bool]:
        """True if contract is fully met, False if violated, None if no contract."""
        if self.contract is None:
            return None
        return self.contract.all_met

    def to_dict(self) -> dict:
        d: dict = {
            "expression": self.expression.to_dict(),
            "acted": self.acted,
            "escalated": self.escalated,
        }
        if self.explanations:
            d["explanations"] = {k: v.to_dict() for k, v in self.explanations.items()}
        if self.decision:
            d["decision"] = self.decision.to_dict()
        if self.correction:
            d["correction"] = self.correction.to_dict()
        if self.escalation:
            d["escalation"] = self.escalation.to_dict()
        if self.contract:
            d["contract"] = self.contract.to_dict()
        return d

    def to_string(self) -> str:
        lines = [f"Step: {self.expression.to_string()}"]
        if self.explanations:
            for name, expl in self.explanations.items():
                if expl.has_known_cause:
                    lines.append(f"  Why: {expl.to_string()}")
        if self.decision:
            lines.append(f"  Decision: {self.decision.winner or 'none'} "
                         f"({self.decision.rules_matched}/{self.decision.rules_considered} matched)")
        if self.correction:
            lines.append(f"  Action: {self.correction.op.value}"
                         f"(target={self.correction.target}, α={self.correction.alpha:.2f})")
        if self.escalation:
            lines.append(f"  Escalation: {self.escalation.level.value}: {self.escalation.reason}")
        if self.contract:
            met = len(self.contract.met())
            violated = len(self.contract.violated())
            pending = len(self.contract.pending())
            lines.append(f"  Contract: {met} met, {violated} violated, {pending} pending")
        return "\n".join(lines)

    def __repr__(self) -> str:
        action = "CORRECT" if self.acted else ("ESCALATE" if self.escalated else "NOOP")
        return f"StepResult({action}, contract={'MET' if self.contract_met else 'OPEN'})"


def step(
    expression: Expression,
    policy: Policy,
    ledger: Optional[Ledger] = None,
    graph: Optional[CausalGraph] = None,
    contract: Optional[Contract] = None,
) -> StepResult:
    """
    Run one full evaluation loop iteration.

    1. EXPLAIN: if a CausalGraph is provided, explain every component
    2. DECIDE:  evaluate the policy with full tracing
    3. EVALUATE: if a Contract is provided, score the ledger against it

    Returns a StepResult bundling all outputs. The caller is responsible
    for applying the correction and appending a Record to the ledger.
    """
    # 1. Explain
    explanations = {}
    causal_context = None
    if graph:
        explanations = graph.explain_all(expression)
        causal_context = {k: v.to_dict() for k, v in explanations.items()}

    # 2. Decide
    decision = trace_evaluate(policy, expression, ledger, causal_context)

    correction = None
    escalation = None
    if isinstance(decision.result, Correction):
        correction = decision.result
    elif isinstance(decision.result, Escalation):
        escalation = decision.result

    # 3. Evaluate contract
    contract_result = None
    if contract:
        contract_result = contract.evaluate(ledger or Ledger(), expression)

    return StepResult(
        expression=expression,
        explanations=explanations,
        decision=decision,
        correction=correction,
        escalation=escalation,
        contract=contract_result,
    )


def run(
    expressions: list[Expression],
    policy: Policy,
    graph: Optional[CausalGraph] = None,
    contract: Optional[Contract] = None,
    ledger: Optional[Ledger] = None,
) -> tuple[list[StepResult], Ledger]:
    """
    Run the full loop over a sequence of expressions.

    For each expression, calls step(), builds a Record from the result,
    and appends it to the ledger. Returns the list of StepResults and
    the final ledger.

    This is a simulation — it does NOT apply corrections to produce the
    next expression. Each expression in the input is taken as-is. Use
    this for backtesting or replaying recorded data.
    """
    from .observation import Op

    ledger = ledger or Ledger()
    results = []

    for i, expr in enumerate(expressions):
        result = step(expr, policy, ledger, graph, contract)

        if expr.observations:
            obs = expr.observations[0]
            ledger.append(Record(
                step=i,
                tag=expr.label or f"step-{i}",
                before=obs,
                fired=result.acted,
                op=result.correction.op if result.correction else Op.NOOP,
                alpha=result.correction.alpha if result.correction else 0.0,
                magnitude=result.correction.magnitude if result.correction else 0.0,
            ))

        results.append(result)

    return results, ledger
