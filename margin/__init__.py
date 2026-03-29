"""
Margin: typed uncertainty algebra and health classification.

A framework for measurements that carry uncertainty, temporal validity,
provenance, and typed health states — with an auditable correction ledger.

Structure:
    Foundation:    confidence, validity, provenance, uncertain, algebra,
                   health, observation, ledger
    Observability: bridge, calibrate, composite, diff, events, forecast,
                   predicates, transitions
    Policy:        policy/ (core, temporal, compose, tuning, trace, validate)
    Contract:      contract
    Causal:        causal
"""

# Foundation
from .confidence import Confidence
from .validity import Validity, ValidityMode
from .provenance import new_id, are_correlated, merge
from .uncertain import UncertainValue, Source
from .algebra import add, subtract, multiply, divide, scale, compare, weighted_average
from .health import Health, Thresholds, classify, SEVERITY
from .observation import Op, Observation, Correction, Expression, Parser
from .ledger import Record, Ledger

# Observability
from .bridge import observe, observe_many, delta, to_uncertain
from .calibrate import CalibrationResult, calibrate, calibrate_many, parser_from_calibration
from .composite import CompositeObservation, AggregateStrategy
from .diff import ComponentChange, Diff, diff
from .events import EventBus
from .anomaly import (
    AnomalyState, ANOMALY_SEVERITY, AnomalyClassification,
    DistributionShift, Jump,
    classify_anomaly, classify_anomaly_obs,
    check_distribution, detect_jumps,
    anomaly_from_ledger, anomaly_all_from_ledger,
    distribution_shift_from_ledger,
    anomaly_is, any_anomalous, any_novel, is_novel,
)
from .drift import (
    DriftState, DriftDirection, DriftClassification,
    classify_drift, classify_drift_all,
    observations_from_ledger as drift_observations_from_ledger,
    drift_from_ledger, drift_all_from_ledger,
    drift_is, drift_worsening, any_drifting, any_drift_worsening, drift_accelerating,
    DriftForecast, drift_forecast, drift_forecast_from_ledger,
)
from .forecast import Forecast, forecast
from .predicates import (
    any_health, all_health, count_health, component_health,
    any_degraded, confidence_below, sigma_below, any_correction,
    all_of, any_of, not_, Rule, evaluate_rules,
)
from .transitions import Span, Transition, ComponentHistory, track, track_all

# Policy
from .policy import (
    EscalationLevel, Escalation, Action, Constraint, PolicyRule, Policy,
    health_sustained, health_for_at_least,
    sigma_trending_below, fire_rate_above, no_improvement,
    PolicyChain, CorrectionBundle, bundle_from_policy,
    PolicyComparison, diff_policies, agreement_rate,
    RuleStats, TuningResult,
    analyze_backtest, suggest_tuning, apply_tuning,
    RuleEvaluation, DecisionTrace,
    trace_evaluate, trace_backtest,
    ValidationIssue, ValidationResult, validate,
)

# Contract
from .contract import (
    TermStatus, TermResult, ContractTerm,
    HealthTarget, ReachHealth, SustainHealth,
    RecoveryThreshold, NoHarmful,
    ContractResult, Contract,
)

# Causal
from .causal import (
    CauseType, CausalLink, CausalGraph,
    CauseExplanation, Explanation,
)

# Loop
from .loop import StepResult, step, run

__all__ = [
    # Foundation
    "Confidence",
    "Validity", "ValidityMode",
    "new_id", "are_correlated", "merge",
    "UncertainValue", "Source",
    "add", "subtract", "multiply", "divide", "scale", "compare", "weighted_average",
    "Health", "Thresholds", "classify", "SEVERITY",
    "Op", "Observation", "Correction", "Expression", "Parser",
    "Record", "Ledger",
    # Observability
    "observe", "observe_many", "delta", "to_uncertain",
    "CalibrationResult", "calibrate", "calibrate_many", "parser_from_calibration",
    "CompositeObservation", "AggregateStrategy",
    "ComponentChange", "Diff", "diff",
    "AnomalyState", "ANOMALY_SEVERITY", "AnomalyClassification",
    "DistributionShift", "Jump",
    "classify_anomaly", "classify_anomaly_obs",
    "check_distribution", "detect_jumps",
    "anomaly_from_ledger", "anomaly_all_from_ledger",
    "distribution_shift_from_ledger",
    "anomaly_is", "any_anomalous", "any_novel", "is_novel",
    "DriftState", "DriftDirection", "DriftClassification", "classify_drift", "classify_drift_all",
    "drift_observations_from_ledger", "drift_from_ledger", "drift_all_from_ledger",
    "drift_is", "drift_worsening", "any_drifting", "any_drift_worsening", "drift_accelerating",
    "DriftForecast", "drift_forecast", "drift_forecast_from_ledger",
    "EventBus",
    "Forecast", "forecast",
    "any_health", "all_health", "count_health", "component_health",
    "any_degraded", "confidence_below", "sigma_below", "any_correction",
    "all_of", "any_of", "not_", "Rule", "evaluate_rules",
    "Span", "Transition", "ComponentHistory", "track", "track_all",
    # Policy
    "EscalationLevel", "Escalation", "Action", "Constraint", "PolicyRule", "Policy",
    "health_sustained", "health_for_at_least",
    "sigma_trending_below", "fire_rate_above", "no_improvement",
    "PolicyChain", "CorrectionBundle", "bundle_from_policy",
    "PolicyComparison", "diff_policies", "agreement_rate",
    "RuleStats", "TuningResult",
    "analyze_backtest", "suggest_tuning", "apply_tuning",
    "RuleEvaluation", "DecisionTrace",
    "trace_evaluate", "trace_backtest",
    "ValidationIssue", "ValidationResult", "validate",
    # Contract
    "TermStatus", "TermResult", "ContractTerm",
    "HealthTarget", "ReachHealth", "SustainHealth",
    "RecoveryThreshold", "NoHarmful",
    "ContractResult", "Contract",
    # Causal
    "CauseType", "CausalLink", "CausalGraph",
    "CauseExplanation", "Explanation",
    # Loop
    "StepResult", "step", "run",
]
