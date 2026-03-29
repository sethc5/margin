"""
Policy composition: chaining, multi-correction bundles, and policy diffing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ..observation import Expression, Correction
from ..ledger import Ledger
from .core import Policy, PolicyRule, Escalation, EscalationLevel


# -----------------------------------------------------------------------
# PolicyChain — run policies in sequence with veto
# -----------------------------------------------------------------------

@dataclass
class PolicyChain:
    """
    Run multiple policies in order. Earlier policies can veto later ones.

    policies:   ordered list — evaluated left to right
    mode:       "first" = return first non-None result
                "veto"  = if any policy escalates with HALT, stop and return that
                "all"   = collect results from all policies

    In "veto" mode, a HALT escalation from any policy prevents subsequent
    policies from running. ALERT and LOG escalations do not veto.
    """
    name: str
    policies: list[Policy] = field(default_factory=list)
    mode: str = "first"  # "first", "veto", "all"

    def evaluate(
        self,
        expr: Expression,
        ledger: Optional[Ledger] = None,
    ) -> list[Union[Correction, Escalation]]:
        results = []

        for policy in self.policies:
            policy_results = policy.evaluate(expr, ledger)

            if self.mode == "veto":
                for r in policy_results:
                    if isinstance(r, Escalation) and r.level == EscalationLevel.HALT:
                        return [r]
                results.extend(policy_results)

            elif self.mode == "first":
                if policy_results:
                    return policy_results

            else:  # "all"
                results.extend(policy_results)

        return results

    def evaluate_first(
        self,
        expr: Expression,
        ledger: Optional[Ledger] = None,
    ) -> Optional[Union[Correction, Escalation]]:
        results = self.evaluate(expr, ledger)
        return results[0] if results else None

    def __repr__(self) -> str:
        return f"PolicyChain({self.name!r}, {len(self.policies)} policies, mode={self.mode!r})"


# -----------------------------------------------------------------------
# CorrectionBundle — coordinated multi-correction
# -----------------------------------------------------------------------

@dataclass
class CorrectionBundle:
    """
    A set of corrections that should be applied together.

    corrections: the corrections in this bundle
    reason:      why these are grouped
    """
    corrections: list[Correction] = field(default_factory=list)
    reason: str = ""

    @property
    def targets(self) -> list[str]:
        return [c.target for c in self.corrections]

    @property
    def n_active(self) -> int:
        return sum(1 for c in self.corrections if c.is_active())

    def to_dict(self) -> dict:
        return {
            "corrections": [c.to_dict() for c in self.corrections],
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        targets = ", ".join(self.targets)
        return f"CorrectionBundle([{targets}], {self.reason!r})"


def bundle_from_policy(
    policy: Policy,
    expr: Expression,
    ledger: Optional[Ledger] = None,
) -> CorrectionBundle:
    """
    Evaluate a policy and collect ALL corrections (not just the first)
    into a CorrectionBundle for coordinated application.
    """
    results = policy.evaluate(expr, ledger)
    corrections = [r for r in results if isinstance(r, Correction)]
    return CorrectionBundle(corrections=corrections, reason=policy.name)


# -----------------------------------------------------------------------
# Policy diffing — compare two policies over a ledger
# -----------------------------------------------------------------------

@dataclass
class PolicyComparison:
    """Result of comparing two policies over a ledger."""
    step: int
    tag: str
    policy_a_op: str
    policy_a_alpha: float
    policy_b_op: str
    policy_b_alpha: float
    agree: bool

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tag": self.tag,
            "policy_a": {"op": self.policy_a_op, "alpha": self.policy_a_alpha},
            "policy_b": {"op": self.policy_b_op, "alpha": self.policy_b_alpha},
            "agree": self.agree,
        }


def diff_policies(
    policy_a: Policy,
    policy_b: Policy,
    ledger: Ledger,
) -> list[PolicyComparison]:
    """
    Compare what two policies would do at each step of a ledger.
    Returns a list of per-step comparisons.
    """
    results = []
    for i, rec in enumerate(ledger.records):
        expr = Expression(
            observations=[rec.before],
            confidence=rec.before.confidence,
            step=rec.step,
        )
        history = Ledger(records=list(ledger.records[:i]))

        def _extract(result):
            if isinstance(result, Correction):
                return result.op.value, result.alpha
            elif isinstance(result, Escalation):
                return f"ESCALATE:{result.level.value}", 0.0
            return "NONE", 0.0

        a_result = policy_a.evaluate_first(expr, history)
        b_result = policy_b.evaluate_first(expr, history)
        a_op, a_alpha = _extract(a_result)
        b_op, b_alpha = _extract(b_result)

        results.append(PolicyComparison(
            step=rec.step, tag=rec.tag,
            policy_a_op=a_op, policy_a_alpha=round(a_alpha, 4),
            policy_b_op=b_op, policy_b_alpha=round(b_alpha, 4),
            agree=(a_op == b_op),
        ))

    return results


def agreement_rate(comparisons: list[PolicyComparison]) -> float:
    """Fraction of steps where two policies agree."""
    if not comparisons:
        return 0.0
    return sum(1 for c in comparisons if c.agree) / len(comparisons)
