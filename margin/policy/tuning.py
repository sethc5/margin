"""
Policy tuning: learn from backtest results.

Adjusts alphas and priorities based on which rules produced beneficial
outcomes vs harmful ones. Not ML — just typed reasoning over the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..observation import Correction
from ..ledger import Ledger
from .core import Policy, PolicyRule, Action


@dataclass
class RuleStats:
    """Aggregate performance stats for a single rule from a backtest."""
    name: str
    times_proposed: int = 0
    times_matched_actual: int = 0
    times_actual_beneficial: int = 0
    times_actual_harmful: int = 0
    mean_proposed_alpha: float = 0.0

    @property
    def match_rate(self) -> float:
        if self.times_proposed == 0:
            return 0.0
        return self.times_matched_actual / self.times_proposed

    @property
    def beneficial_rate(self) -> float:
        """Of the times the actual system agreed with this rule, how often was the outcome good?"""
        if self.times_matched_actual == 0:
            return 0.0
        return self.times_actual_beneficial / self.times_matched_actual

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "times_proposed": self.times_proposed,
            "times_matched_actual": self.times_matched_actual,
            "beneficial_rate": round(self.beneficial_rate, 4),
            "match_rate": round(self.match_rate, 4),
            "mean_proposed_alpha": round(self.mean_proposed_alpha, 4),
        }


def analyze_backtest(policy: Policy, backtest_results: list[dict]) -> list[RuleStats]:
    """
    Compute per-rule stats from a backtest.

    For each rule, counts how many times it was the proposing rule,
    whether the actual system agreed, and whether the actual outcome
    was beneficial.
    """
    # Re-derive which rule proposed each step
    # We need to re-evaluate to know which rule won at each step
    rule_stats: dict[str, RuleStats] = {r.name: RuleStats(name=r.name) for r in policy.rules}

    for bt in backtest_results:
        proposed_op = bt["proposed_op"]
        if proposed_op == "NONE" or proposed_op.startswith("ESCALATE"):
            continue

        # Find which rule would have proposed this
        # The backtest doesn't track which rule won, so we approximate:
        # the highest-priority rule whose op matches the proposed op
        for rule in sorted(policy.rules, key=lambda r: r.priority, reverse=True):
            if rule.action.op.value == proposed_op:
                stats = rule_stats.get(rule.name)
                if stats is None:
                    continue
                stats.times_proposed += 1
                if bt["match"]:
                    stats.times_matched_actual += 1
                    if bt["actual_beneficial"]:
                        stats.times_actual_beneficial += 1
                    else:
                        stats.times_actual_harmful += 1
                stats.mean_proposed_alpha = (
                    (stats.mean_proposed_alpha * (stats.times_proposed - 1) + bt["proposed_alpha"])
                    / stats.times_proposed
                )
                break

    return [s for s in rule_stats.values() if s.times_proposed > 0]


@dataclass
class TuningResult:
    """Suggested adjustments to a policy based on backtest analysis."""
    rule_name: str
    current_alpha: float
    suggested_alpha: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "current_alpha": self.current_alpha,
            "suggested_alpha": round(self.suggested_alpha, 4),
            "reason": self.reason,
        }


def suggest_tuning(
    policy: Policy,
    rule_stats: list[RuleStats],
    harmful_threshold: float = 0.3,
    alpha_reduction: float = 0.8,
    alpha_boost: float = 1.1,
    max_alpha: float = 1.0,
) -> list[TuningResult]:
    """
    Suggest alpha adjustments based on rule performance.

    Rules where the actual outcome was harmful > harmful_threshold of the time
    get their alpha reduced. Rules where beneficial_rate > 0.8 get a slight boost.

    Args:
        policy: the policy to tune
        rule_stats: output of analyze_backtest()
        harmful_threshold: if harmful rate exceeds this, reduce alpha
        alpha_reduction: multiply alpha by this when reducing (< 1.0)
        alpha_boost: multiply alpha by this when boosting (> 1.0)
        max_alpha: ceiling
    """
    stats_by_name = {s.name: s for s in rule_stats}
    suggestions = []

    for rule in policy.rules:
        stats = stats_by_name.get(rule.name)
        if stats is None or stats.times_matched_actual == 0:
            continue

        harmful_rate = stats.times_actual_harmful / max(stats.times_matched_actual, 1)
        current = rule.action.alpha

        if harmful_rate > harmful_threshold:
            suggested = max(0.05, current * alpha_reduction)
            suggestions.append(TuningResult(
                rule_name=rule.name,
                current_alpha=current,
                suggested_alpha=min(suggested, max_alpha),
                reason=f"harmful rate {harmful_rate:.0%} exceeds {harmful_threshold:.0%}",
            ))
        elif stats.beneficial_rate > 0.8 and stats.times_matched_actual >= 3:
            suggested = current * alpha_boost
            if suggested > current and suggested <= max_alpha:
                suggestions.append(TuningResult(
                    rule_name=rule.name,
                    current_alpha=current,
                    suggested_alpha=min(suggested, max_alpha),
                    reason=f"beneficial rate {stats.beneficial_rate:.0%} with {stats.times_matched_actual} samples",
                ))

    return suggestions


def apply_tuning(policy: Policy, suggestions: list[TuningResult]) -> Policy:
    """
    Return a new Policy with suggested alpha adjustments applied.
    The original policy is not modified.
    """
    suggestion_map = {s.rule_name: s.suggested_alpha for s in suggestions}

    new_rules = []
    for rule in policy.rules:
        if rule.name in suggestion_map:
            new_action = Action(
                target=rule.action.target, op=rule.action.op,
                alpha=suggestion_map[rule.name],
                magnitude=rule.action.magnitude,
                alpha_from_sigma=rule.action.alpha_from_sigma,
                magnitude_from_sigma=rule.action.magnitude_from_sigma,
            )
            new_rules.append(PolicyRule(
                name=rule.name, predicate=rule.predicate,
                action=new_action, priority=rule.priority,
                constraint=rule.constraint, escalation=rule.escalation,
                min_confidence=rule.min_confidence,
            ))
        else:
            new_rules.append(rule)

    return Policy(
        name=f"{policy.name}-tuned",
        rules=new_rules,
        default_constraint=policy.default_constraint,
        default_escalation=policy.default_escalation,
    )
