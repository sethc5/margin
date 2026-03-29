import pytest
from margin.policy.core import Policy, PolicyRule, Action, Constraint, Escalation, EscalationLevel
from margin.policy.validate import validate, ValidationResult
from margin.observation import Observation, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.contract import Contract, HealthTarget
from margin.predicates import any_health, all_health, any_degraded


def _make_rule(name, predicate, target="*", alpha=0.5, priority=0, constraint=None):
    return PolicyRule(name, predicate, Action(target=target, op=Op.RESTORE, alpha=alpha),
                      priority=priority, constraint=constraint)


class TestDuplicateNames:
    def test_detects_duplicates(self):
        policy = Policy(name="p", rules=[
            _make_rule("r", any_degraded()),
            _make_rule("r", any_health(Health.ABLATED)),
        ])
        result = validate(policy)
        assert not result.valid
        assert any("duplicate" in i.message for i in result.errors)

    def test_no_duplicates_passes(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded()),
            _make_rule("r2", any_health(Health.ABLATED)),
        ])
        result = validate(policy)
        assert not any("duplicate" in i.message for i in result.errors)


class TestConstraintConflicts:
    def test_min_above_max(self):
        policy = Policy(name="p", rules=[
            _make_rule("r", any_degraded(), constraint=Constraint(min_alpha=0.8, max_alpha=0.3)),
        ])
        result = validate(policy)
        assert any("min_alpha" in i.message and "max_alpha" in i.message for i in result.errors)

    def test_alpha_exceeds_max(self):
        policy = Policy(name="p", rules=[
            _make_rule("r", any_degraded(), alpha=0.9, constraint=Constraint(max_alpha=0.5)),
        ])
        result = validate(policy)
        assert any("clamped" in i.message for i in result.warnings)

    def test_alpha_below_min(self):
        policy = Policy(name="p", rules=[
            _make_rule("r", any_degraded(), alpha=0.1, constraint=Constraint(min_alpha=0.5)),
        ])
        result = validate(policy)
        assert any("suppressed" in i.message for i in result.warnings)


class TestPriorityCollisions:
    def test_detects_collision(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), priority=10),
            _make_rule("r2", any_health(Health.ABLATED), priority=10),
        ])
        result = validate(policy)
        assert any("priority 10" in i.message for i in result.warnings)

    def test_no_collision(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), priority=10),
            _make_rule("r2", any_health(Health.ABLATED), priority=20),
        ])
        result = validate(policy)
        assert not any("priority" in i.message and "shared" in i.message for i in result.warnings)


class TestHealthCoverage:
    def test_warns_on_missing_coverage(self):
        # Only handles INTACT — nothing for DEGRADED, ABLATED, etc.
        policy = Policy(name="p", rules=[
            PolicyRule("clear", all_health(Health.INTACT), Action(target="*", op=Op.NOOP)),
        ])
        result = validate(policy)
        # Should warn about DEGRADED, ABLATED, RECOVERING, OOD
        coverage_warnings = [i for i in result.warnings if "no rule matches" in i.message]
        assert len(coverage_warnings) >= 3

    def test_full_coverage_no_warning(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), priority=10),
            PolicyRule("clear", all_health(Health.INTACT), Action(target="*", op=Op.NOOP)),
            PolicyRule("ood", any_health(Health.OOD), Action(),
                       escalation=Escalation(EscalationLevel.ALERT, "OOD"),
                       min_confidence=Confidence.INDETERMINATE),
        ])
        result = validate(policy)
        coverage_warnings = [i for i in result.warnings if "no rule matches" in i.message]
        # DEGRADED, ABLATED, RECOVERING all match any_degraded()
        # INTACT matches all_health(INTACT)
        # OOD matches any_health(OOD)
        assert len(coverage_warnings) == 0

    def test_default_escalation_covers(self):
        policy = Policy(name="p", rules=[
            PolicyRule("clear", all_health(Health.INTACT), Action(target="*", op=Op.NOOP)),
        ], default_escalation=Escalation(EscalationLevel.LOG, "fallback"))
        result = validate(policy)
        # Default escalation means unmatched states get a response
        coverage_warnings = [i for i in result.warnings if "no rule matches" in i.message]
        assert len(coverage_warnings) == 0


class TestContractCoverage:
    def test_warns_when_contract_component_not_targeted(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), target="api"),
        ])
        contract = Contract("sla", terms=[
            HealthTarget("api-intact", "api", Health.INTACT),
            HealthTarget("db-intact", "db", Health.INTACT),
        ])
        result = validate(policy, contract=contract)
        assert any("db" in i.message and "no policy rule targets" in i.message for i in result.warnings)

    def test_wildcard_covers_all(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), target="*"),
        ])
        contract = Contract("sla", terms=[
            HealthTarget("api-intact", "api", Health.INTACT),
            HealthTarget("db-intact", "db", Health.INTACT),
        ])
        result = validate(policy, contract=contract)
        assert not any("no policy rule targets" in i.message for i in result.warnings)


class TestComponentCoverage:
    def test_warns_on_uncovered_component(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), target="api"),
        ])
        result = validate(policy, components=["api", "db", "cache"])
        assert any("db" in i.message for i in result.issues)
        assert any("cache" in i.message for i in result.issues)

    def test_wildcard_covers_all(self):
        policy = Policy(name="p", rules=[
            _make_rule("r1", any_degraded(), target="*"),
        ])
        result = validate(policy, components=["api", "db", "cache"])
        comp_issues = [i for i in result.issues if "no dedicated rule" in i.message]
        assert len(comp_issues) == 0


class TestValidationResult:
    def test_valid_when_no_errors(self):
        r = ValidationResult(issues=[])
        assert r.valid

    def test_invalid_when_errors(self):
        from margin.policy.validate import ValidationIssue
        r = ValidationResult(issues=[ValidationIssue("error", "r", "bad")])
        assert not r.valid

    def test_to_string(self):
        r = ValidationResult(issues=[])
        assert "no issues" in r.to_string()

    def test_to_dict(self):
        r = ValidationResult(issues=[])
        d = r.to_dict()
        assert d["valid"] is True
        assert d["n_errors"] == 0
