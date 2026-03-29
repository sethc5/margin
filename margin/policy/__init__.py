"""
Policy language: typed decision rules for correction loops.

core:      Action, Constraint, Escalation, PolicyRule, Policy
temporal:  predicates that look at ledger history
compose:   PolicyChain, CorrectionBundle, policy diffing
tuning:    learn from backtest results
trace:     full decision audit trail
validate:  check well-formedness before running
"""

from .core import (
    EscalationLevel, Escalation, Action, Constraint, PolicyRule, Policy,
)
from .temporal import (
    health_sustained, health_for_at_least,
    sigma_trending_below, fire_rate_above, no_improvement,
)
from .compose import (
    PolicyChain, CorrectionBundle, bundle_from_policy,
    PolicyComparison, diff_policies, agreement_rate,
)
from .tuning import (
    RuleStats, TuningResult,
    analyze_backtest, suggest_tuning, apply_tuning,
)
from .trace import (
    RuleEvaluation, DecisionTrace,
    trace_evaluate, trace_backtest,
)
from .validate import (
    ValidationIssue, ValidationResult, validate,
)

__all__ = [
    # Core
    "EscalationLevel", "Escalation", "Action", "Constraint", "PolicyRule", "Policy",
    # Temporal
    "health_sustained", "health_for_at_least",
    "sigma_trending_below", "fire_rate_above", "no_improvement",
    # Compose
    "PolicyChain", "CorrectionBundle", "bundle_from_policy",
    "PolicyComparison", "diff_policies", "agreement_rate",
    # Tuning
    "RuleStats", "TuningResult",
    "analyze_backtest", "suggest_tuning", "apply_tuning",
    # Trace
    "RuleEvaluation", "DecisionTrace",
    "trace_evaluate", "trace_backtest",
    # Validate
    "ValidationIssue", "ValidationResult", "validate",
]
