"""
Policy validation: check well-formedness before running.

Catches structural problems that would cause silent misbehavior:
duplicate names, shadowed rules, conflicting constraints,
missing coverage for health states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..health import Health
from ..observation import Expression, Observation
from ..confidence import Confidence
from ..contract import Contract
from .core import Policy, PolicyRule, Action, Constraint


@dataclass
class ValidationIssue:
    """A single problem found during validation."""
    severity: str   # "error", "warning", "info"
    rule_name: str  # which rule, or "policy" for policy-level
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "rule_name": self.rule_name, "message": self.message}

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule_name}: {self.message}"


@dataclass
class ValidationResult:
    """Output of validate()."""
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_string(self) -> str:
        if not self.issues:
            return "Policy valid: no issues found"
        return "\n".join(repr(i) for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }

    def __repr__(self) -> str:
        return f"ValidationResult({len(self.errors)} errors, {len(self.warnings)} warnings)"


def validate(
    policy: Policy,
    contract: Optional[Contract] = None,
    components: Optional[list[str]] = None,
) -> ValidationResult:
    """
    Validate a policy for structural problems.

    Checks:
    1. Duplicate rule names
    2. Constraint conflicts (min_alpha > action.alpha, or > max_alpha)
    3. Wildcard-only targets (every rule uses "*")
    4. Priority collisions (same priority, both could match)
    5. Missing health coverage (no rule handles ABLATED, DEGRADED, etc.)
    6. Contract coverage (if a contract is provided, check that the
       policy has rules for every component the contract mentions)
    7. Component coverage (if components list provided)
    8. alpha_from_sigma without a constraint floor
    9. min_confidence too high (Parser.parse() defaults to MODERATE, so
       rules with min_confidence >= HIGH may silently never fire)

    Args:
        policy:     the policy to validate
        contract:   optional contract to check coverage against
        components: optional list of known component names
    """
    issues: list[ValidationIssue] = []

    # 1. Duplicate rule names
    names = [r.name for r in policy.rules]
    seen = set()
    for name in names:
        if name in seen:
            issues.append(ValidationIssue("error", name, "duplicate rule name"))
        seen.add(name)

    # 2. Constraint conflicts
    for rule in policy.rules:
        constraint = rule.constraint or policy.default_constraint
        if constraint:
            if constraint.min_alpha > constraint.max_alpha:
                issues.append(ValidationIssue(
                    "error", rule.name,
                    f"constraint min_alpha ({constraint.min_alpha}) > max_alpha ({constraint.max_alpha})"))
            if not rule.action.alpha_from_sigma and rule.action.alpha > constraint.max_alpha:
                issues.append(ValidationIssue(
                    "warning", rule.name,
                    f"action alpha ({rule.action.alpha}) exceeds constraint max_alpha ({constraint.max_alpha}) — will be clamped"))
            if not rule.action.alpha_from_sigma and rule.action.alpha < constraint.min_alpha:
                issues.append(ValidationIssue(
                    "warning", rule.name,
                    f"action alpha ({rule.action.alpha}) below constraint min_alpha ({constraint.min_alpha}) — will be suppressed"))

    # 3. All wildcards
    all_wildcard = all(r.action.target == "*" for r in policy.rules)
    if all_wildcard and len(policy.rules) > 1:
        issues.append(ValidationIssue(
            "info", "policy",
            "all rules use wildcard target '*' — corrections always target worst degraded"))

    # 4. Priority collisions
    by_priority: dict[int, list[str]] = {}
    for rule in policy.rules:
        by_priority.setdefault(rule.priority, []).append(rule.name)
    for priority, rule_names in by_priority.items():
        if len(rule_names) > 1:
            issues.append(ValidationIssue(
                "warning", "policy",
                f"priority {priority} shared by rules: {', '.join(rule_names)} — order depends on declaration"))

    # 5. Health coverage
    _check_health_coverage(policy, issues)

    # 6. Contract coverage
    if contract:
        _check_contract_coverage(policy, contract, issues)

    # 7. Component coverage (if components list provided)
    if components:
        _check_component_coverage(policy, components, issues)

    # 8. alpha_from_sigma without a constraint floor
    _check_alpha_from_sigma(policy, issues)

    # 9. min_confidence gate too high for default parse behaviour
    _check_min_confidence(policy, issues)

    return ValidationResult(issues=issues)


def _check_health_coverage(policy: Policy, issues: list[ValidationIssue]) -> None:
    """Check if the policy has rules that could match each health state."""
    # We can't run predicates without an expression, so we do a best-effort
    # check: build synthetic expressions for each health state and test.
    health_states = [Health.INTACT, Health.DEGRADED, Health.ABLATED,
                     Health.RECOVERING, Health.OOD]

    for health in health_states:
        obs = Observation("_test", health, 50.0, 100.0, Confidence.HIGH)
        if health == Health.OOD:
            obs = Observation("_test", health, 50.0, 100.0, Confidence.INDETERMINATE)
        expr = Expression(observations=[obs], confidence=obs.confidence)

        matched = any(r.matches(expr) for r in policy.rules)
        if not matched and policy.default_escalation is None:
            issues.append(ValidationIssue(
                "warning", "policy",
                f"no rule matches Health.{health.value} — these expressions get no response"))


def _check_contract_coverage(
    policy: Policy,
    contract: Contract,
    issues: list[ValidationIssue],
) -> None:
    """Check if the policy covers components mentioned in the contract."""
    # Extract component names from contract terms
    contract_components = set()
    for term in contract.terms:
        if hasattr(term, 'component'):
            contract_components.add(term.component)

    # Check if policy rules mention these components
    policy_targets = set()
    for rule in policy.rules:
        if rule.action.target != "*":
            policy_targets.add(rule.action.target)

    # Wildcard rules cover everything, so only warn if NO rule is wildcard
    has_wildcard = any(r.action.target == "*" for r in policy.rules)

    if not has_wildcard:
        uncovered = contract_components - policy_targets
        for comp in uncovered:
            issues.append(ValidationIssue(
                "warning", "policy",
                f"contract requires '{comp}' but no policy rule targets it"))


def _check_alpha_from_sigma(policy: Policy, issues: list[ValidationIssue]) -> None:
    """
    Warn when alpha_from_sigma=True with no constraint floor.

    When a rule derives alpha from sigma at runtime, the resulting alpha can
    be arbitrarily small (e.g. 0.001 in a stable period) and fire useless
    micro-corrections on every step. A Constraint(min_alpha=...) suppresses
    these; without one there is no floor.

    Also warns when both alpha_from_sigma and magnitude_from_sigma are set —
    double sigma scaling amplifies corrections non-linearly.
    """
    for rule in policy.rules:
        if not rule.action.alpha_from_sigma:
            continue

        constraint = rule.constraint or policy.default_constraint

        if constraint is None:
            issues.append(ValidationIssue(
                "warning", rule.name,
                "alpha_from_sigma=True with no constraint — sigma-derived alpha has no "
                "floor; add Constraint(min_alpha=...) to suppress micro-corrections "
                "during stable periods, or pair with a drift predicate so the rule only "
                "fires when sigma is meaningfully elevated",
            ))
        elif constraint.min_alpha == 0.0:
            issues.append(ValidationIssue(
                "info", rule.name,
                "alpha_from_sigma=True and constraint.min_alpha=0.0 — corrections will "
                "fire with alpha≈0 during stable periods; consider raising min_alpha",
            ))

        if rule.action.alpha_from_sigma and rule.action.magnitude_from_sigma:
            issues.append(ValidationIssue(
                "warning", rule.name,
                "both alpha_from_sigma and magnitude_from_sigma are True — correction "
                "strength scales as sigma² (alpha × magnitude); this amplifies strongly "
                "for large deviations and may overcorrect",
            ))


def _check_component_coverage(
    policy: Policy,
    components: list[str],
    issues: list[ValidationIssue],
) -> None:
    """Check if the policy can address each known component."""
    has_wildcard = any(r.action.target == "*" for r in policy.rules)
    if has_wildcard:
        return  # wildcard covers everything

    targeted = {r.action.target for r in policy.rules}
    for comp in components:
        if comp not in targeted:
            issues.append(ValidationIssue(
                "info", "policy",
                f"component '{comp}' has no dedicated rule (no wildcard rules either)"))


def _check_min_confidence(policy: Policy, issues: list[ValidationIssue]) -> None:
    """
    Warn when a rule's min_confidence gate is above MODERATE.

    Parser.parse() assigns MODERATE confidence to every observation by
    default (when no per-component confidences dict is passed). A rule
    with min_confidence >= HIGH will therefore silently never fire unless
    the caller explicitly passes HIGH or CERTAIN confidences on every step.
    """
    for rule in policy.rules:
        if rule.min_confidence >= Confidence.HIGH:
            issues.append(ValidationIssue(
                "warning", rule.name,
                f"min_confidence={rule.min_confidence.value} — Parser.parse() defaults "
                "to MODERATE; this rule will never fire unless the caller explicitly "
                "passes per-component confidences of HIGH or CERTAIN",
            ))
