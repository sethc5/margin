"""
Decision trace: full audit trail of how a policy reached its decision.

Records which rules were considered, which matched, which won on priority,
whether constraints modified the result, and optionally why (causal context).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ..observation import Expression, Correction
from ..confidence import Confidence
from ..ledger import Ledger
from .core import Policy, PolicyRule, Action, Constraint, Escalation


@dataclass
class RuleEvaluation:
    """What happened when a single rule was evaluated."""
    rule_name: str
    priority: int
    matched: bool
    outcome: Optional[str] = None     # "correction", "escalation", "suppressed", None
    result: Optional[Union[Correction, Escalation]] = None
    reason: str = ""

    def to_dict(self) -> dict:
        d = {
            "rule_name": self.rule_name,
            "priority": self.priority,
            "matched": self.matched,
            "outcome": self.outcome,
            "reason": self.reason,
        }
        if self.result is not None:
            if isinstance(self.result, Correction):
                d["correction"] = self.result.to_dict()
            elif isinstance(self.result, Escalation):
                d["escalation"] = self.result.to_dict()
        return d


@dataclass
class DecisionTrace:
    """
    Complete audit trail of a policy evaluation.

    expression:       the input state
    evaluations:      what happened with each rule (all rules, not just matches)
    winner:           the rule that produced the final result (if any)
    result:           the final Correction or Escalation
    rules_considered: total rules in the policy
    rules_matched:    how many rules matched the expression
    """
    expression: Expression
    evaluations: list[RuleEvaluation] = field(default_factory=list)
    winner: Optional[str] = None
    result: Optional[Union[Correction, Escalation]] = None
    causal_context: Optional[dict] = None  # from causal.explain_all() if available
    results: list = field(default_factory=list)  # all fired corrections/escalations (multi_rule)

    @property
    def rules_considered(self) -> int:
        return len(self.evaluations)

    @property
    def rules_matched(self) -> int:
        return sum(1 for e in self.evaluations if e.matched)

    @property
    def was_escalation(self) -> bool:
        return isinstance(self.result, Escalation)

    @property
    def was_correction(self) -> bool:
        return isinstance(self.result, Correction)

    def to_string(self) -> str:
        lines = [f"DecisionTrace: {self.rules_matched}/{self.rules_considered} rules matched"]
        for e in self.evaluations:
            status = "MATCH" if e.matched else "skip"
            marker = " ← winner" if e.rule_name == self.winner else ""
            lines.append(f"  [{status}] {e.rule_name} (p={e.priority}){marker}")
            if e.reason:
                lines.append(f"         {e.reason}")
        if self.result:
            if isinstance(self.result, Correction):
                lines.append(f"  Result: {self.result.op.value}(target={self.result.target}, α={self.result.alpha:.2f})")
            else:
                lines.append(f"  Result: ESCALATE({self.result.level.value}: {self.result.reason})")
        else:
            lines.append("  Result: no action")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "rules_considered": self.rules_considered,
            "rules_matched": self.rules_matched,
            "winner": self.winner,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }
        if self.result:
            if isinstance(self.result, Correction):
                d["result_type"] = "correction"
                d["result"] = self.result.to_dict()
            else:
                d["result_type"] = "escalation"
                d["result"] = self.result.to_dict()
        else:
            d["result_type"] = None
        if self.causal_context:
            d["causal_context"] = self.causal_context
        if self.results:
            d["results"] = [
                r.to_dict() if hasattr(r, 'to_dict') else str(r)
                for r in self.results
            ]
        return d

    def __repr__(self) -> str:
        w = self.winner or "none"
        return f"DecisionTrace({self.rules_matched}/{self.rules_considered} matched, winner={w})"


def trace_evaluate(
    policy: Policy,
    expr: Expression,
    ledger: Optional[Ledger] = None,
    causal_context: Optional[dict] = None,
) -> DecisionTrace:
    """
    Evaluate a policy with full tracing. Returns a DecisionTrace that
    records what every rule did, not just the winner.

    causal_context: optional output from CausalGraph.explain_all(expr),
    attached to the trace for auditability.
    """
    dt = DecisionTrace(expression=expr, causal_context=causal_context)

    # Evaluate all rules in priority order
    sorted_rules = sorted(policy.rules, key=lambda r: r.priority, reverse=True)

    all_results = []

    for rule in sorted_rules:
        matched = rule.matches(expr)
        ev = RuleEvaluation(
            rule_name=rule.name,
            priority=rule.priority,
            matched=matched,
        )

        if matched:
            result = rule.evaluate(expr, ledger, policy.default_constraint)
            if result is not None:
                if isinstance(result, Correction):
                    ev.outcome = "correction"
                    ev.reason = f"{result.op.value}(target={result.target}, α={result.alpha:.2f})"
                elif isinstance(result, Escalation):
                    ev.outcome = "escalation"
                    ev.reason = f"{result.level.value}: {result.reason}"
                ev.result = result
                all_results.append((rule.name, result))
            else:
                ev.outcome = "suppressed"
                ev.reason = "evaluate returned None"

        dt.evaluations.append(ev)

    if not all_results and policy.default_escalation:
        default_esc = Escalation(
            policy.default_escalation.level,
            policy.default_escalation.reason,
            "default",
        )
        all_results.append(("default", default_esc))

    dt.results = [r for _, r in all_results]
    dt.winner = all_results[0][0] if all_results else None
    dt.result = all_results[0][1] if all_results else None
    return dt


def trace_backtest(
    policy: Policy,
    ledger: Ledger,
) -> list[DecisionTrace]:
    """
    Backtest with full traces. Returns a DecisionTrace per step
    instead of a flat dict.
    """
    traces = []
    for i, rec in enumerate(ledger.records):
        expr = Expression(
            observations=[rec.before],
            confidence=rec.before.confidence,
            step=rec.step,
        )
        history = Ledger(records=list(ledger.records[:i]))
        trace = trace_evaluate(policy, expr, history)
        traces.append(trace)
    return traces
