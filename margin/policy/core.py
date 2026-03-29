"""
Core policy types: Action, Constraint, Escalation, PolicyRule, Policy.

The decision engine: Expression → matching rules → highest priority →
constrained Correction or Escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Union

from ..confidence import Confidence
from ..health import Health, SEVERITY
from ..observation import Expression, Observation, Correction, Op
from ..ledger import Ledger, Record
from ..predicates import PredicateFn
from ..provenance import new_id


# -----------------------------------------------------------------------
# Escalation
# -----------------------------------------------------------------------

class EscalationLevel(Enum):
    LOG = "LOG"
    ALERT = "ALERT"
    HALT = "HALT"


@dataclass
class Escalation:
    """What to do when the policy cannot or should not act."""
    level: EscalationLevel = EscalationLevel.LOG
    reason: str = ""
    rule_name: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level.value, "reason": self.reason, "rule_name": self.rule_name}

    @classmethod
    def from_dict(cls, d: dict) -> Escalation:
        return cls(
            level=EscalationLevel(d["level"]),
            reason=d.get("reason", ""),
            rule_name=d.get("rule_name", ""),
        )

    def __repr__(self) -> str:
        return f"Escalation({self.level.value}: {self.reason})"


# -----------------------------------------------------------------------
# Action
# -----------------------------------------------------------------------

@dataclass
class Action:
    """
    Specification for producing a Correction.

    target:     component name, or "*" for worst degraded
    op:         correction operation
    alpha:      fixed intensity (used when alpha_from_sigma is False)
    magnitude:  fixed magnitude
    alpha_from_sigma:     if True, derive alpha from |sigma| of target
    magnitude_from_sigma: if True, derive magnitude from |sigma| of target
    """
    target: str = "*"
    op: Op = Op.RESTORE
    alpha: float = 0.5
    magnitude: float = 1.0
    alpha_from_sigma: bool = False
    magnitude_from_sigma: bool = False

    def resolve(self, expr: Expression) -> Correction:
        """Produce a concrete Correction from this spec and an Expression."""
        if self.target == "*":
            degraded = sorted(
                expr.degraded(),
                key=lambda o: SEVERITY.get(o.health, 0),
                reverse=True,
            )
            if degraded:
                target_obs = degraded[0]
                target_name = target_obs.name
            elif expr.observations:
                target_obs = expr.observations[0]
                target_name = target_obs.name
            else:
                return Correction(target="", op=Op.NOOP)
        else:
            target_name = self.target
            target_obs = None
            for o in expr.observations:
                if o.name == target_name:
                    target_obs = o
                    break

        alpha = self.alpha
        if self.alpha_from_sigma and target_obs:
            alpha = min(1.0, abs(target_obs.sigma))

        magnitude = self.magnitude
        if self.magnitude_from_sigma and target_obs:
            magnitude = abs(target_obs.sigma)

        return Correction(
            target=target_name, op=self.op,
            alpha=alpha, magnitude=magnitude,
            triggered_by=[o.name for o in expr.degraded()],
            provenance=[new_id()],
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target, "op": self.op.value,
            "alpha": self.alpha, "magnitude": self.magnitude,
            "alpha_from_sigma": self.alpha_from_sigma,
            "magnitude_from_sigma": self.magnitude_from_sigma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Action:
        return cls(
            target=d.get("target", "*"),
            op=Op(d["op"]) if "op" in d else Op.RESTORE,
            alpha=d.get("alpha", 0.5), magnitude=d.get("magnitude", 1.0),
            alpha_from_sigma=d.get("alpha_from_sigma", False),
            magnitude_from_sigma=d.get("magnitude_from_sigma", False),
        )


# -----------------------------------------------------------------------
# Constraint
# -----------------------------------------------------------------------

@dataclass
class Constraint:
    """
    Bounds applied after an Action resolves.

    max_alpha:      ceiling on alpha
    min_alpha:      floor — below this the correction is suppressed
    cooldown_steps: minimum steps between corrections to the same target
    max_per_window: maximum corrections in a window (0 = unlimited)
    window_steps:   window size for max_per_window check
    """
    max_alpha: float = 1.0
    min_alpha: float = 0.0
    cooldown_steps: int = 0
    max_per_window: int = 0
    window_steps: int = 0

    def check(self, correction: Correction, ledger: Optional[Ledger] = None) -> Optional[Correction]:
        alpha = min(correction.alpha, self.max_alpha)
        if alpha < self.min_alpha:
            return None
        if ledger and self.cooldown_steps > 0:
            if ledger.for_component(correction.target).last_n(self.cooldown_steps).n_fired > 0:
                return None
        if ledger and self.max_per_window > 0 and self.window_steps > 0:
            if ledger.for_component(correction.target).last_n(self.window_steps).n_fired >= self.max_per_window:
                return None
        return Correction(
            target=correction.target, op=correction.op, alpha=alpha,
            magnitude=correction.magnitude,
            triggered_by=correction.triggered_by, provenance=correction.provenance,
        )

    def to_dict(self) -> dict:
        return {
            "max_alpha": self.max_alpha, "min_alpha": self.min_alpha,
            "cooldown_steps": self.cooldown_steps,
            "max_per_window": self.max_per_window, "window_steps": self.window_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Constraint:
        return cls(**{k: d[k] for k in ("max_alpha", "min_alpha", "cooldown_steps", "max_per_window", "window_steps") if k in d})


# -----------------------------------------------------------------------
# PolicyRule
# -----------------------------------------------------------------------

@dataclass
class PolicyRule:
    """
    Condition + action + constraints + confidence gating.

    Predicate is not serialized — use predicate_registry for from_dict.
    """
    name: str
    predicate: PredicateFn
    action: Action
    priority: int = 0
    constraint: Optional[Constraint] = None
    escalation: Optional[Escalation] = None
    min_confidence: Confidence = Confidence.LOW

    def matches(self, expr: Expression) -> bool:
        return self.predicate(expr)

    def evaluate(
        self, expr: Expression,
        ledger: Optional[Ledger] = None,
        default_constraint: Optional[Constraint] = None,
    ) -> Optional[Union[Correction, Escalation]]:
        if not self.matches(expr):
            return None
        if expr.confidence < self.min_confidence:
            return Escalation(EscalationLevel.ALERT,
                              f"confidence {expr.confidence.value} below {self.min_confidence.value}",
                              self.name)
        if self.escalation is not None:
            return Escalation(self.escalation.level, self.escalation.reason, self.name)
        correction = self.action.resolve(expr)
        constraint = self.constraint or default_constraint
        if constraint:
            correction = constraint.check(correction, ledger)
            if correction is None:
                return Escalation(EscalationLevel.LOG, "constraint suppressed", self.name)
        return correction

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "action": self.action.to_dict(),
                    "priority": self.priority, "min_confidence": self.min_confidence.value}
        if self.constraint:
            d["constraint"] = self.constraint.to_dict()
        if self.escalation:
            d["escalation"] = self.escalation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict, predicate: PredicateFn) -> PolicyRule:
        return cls(
            name=d["name"], predicate=predicate,
            action=Action.from_dict(d["action"]), priority=d.get("priority", 0),
            constraint=Constraint.from_dict(d["constraint"]) if "constraint" in d else None,
            escalation=Escalation.from_dict(d["escalation"]) if "escalation" in d else None,
            min_confidence=Confidence(d["min_confidence"]) if "min_confidence" in d else Confidence.LOW,
        )

    def __repr__(self) -> str:
        return f"PolicyRule({self.name!r}, priority={self.priority})"


# -----------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------

@dataclass
class Policy:
    """Prioritised set of decision rules with backtest capability."""
    name: str
    rules: list[PolicyRule] = field(default_factory=list)
    default_constraint: Optional[Constraint] = None
    default_escalation: Optional[Escalation] = None

    def evaluate(self, expr: Expression, ledger: Optional[Ledger] = None) -> list[Union[Correction, Escalation]]:
        matching = sorted([r for r in self.rules if r.matches(expr)],
                          key=lambda r: r.priority, reverse=True)
        results = []
        for r in matching:
            result = r.evaluate(expr, ledger, self.default_constraint)
            if result is not None:
                # Tag which rule produced this result
                if isinstance(result, Correction) and not any(
                    p.startswith("rule:") for p in result.provenance):
                    result.provenance.append(f"rule:{r.name}")
                results.append(result)
        if not results and self.default_escalation:
            results.append(Escalation(self.default_escalation.level,
                                      self.default_escalation.reason, "default"))
        return results

    def evaluate_first(self, expr: Expression, ledger: Optional[Ledger] = None) -> Optional[Union[Correction, Escalation]]:
        results = self.evaluate(expr, ledger)
        return results[0] if results else None

    def backtest(self, ledger: Ledger) -> list[dict]:
        results = []
        for i, rec in enumerate(ledger.records):
            expr = Expression(observations=[rec.before], confidence=rec.before.confidence, step=rec.step)
            history = Ledger(records=list(ledger.records[:i]))
            proposed = self.evaluate_first(expr, history)
            if isinstance(proposed, Correction):
                p_op, p_alpha = proposed.op.value, proposed.alpha
                p_rule = next((p[5:] for p in proposed.provenance if p.startswith("rule:")), "")
            elif isinstance(proposed, Escalation):
                p_op, p_alpha = f"ESCALATE:{proposed.level.value}", 0.0
                p_rule = proposed.rule_name
            else:
                p_op, p_alpha, p_rule = "NONE", 0.0, ""
            results.append({
                "step": rec.step, "tag": rec.tag,
                "actual_op": rec.op.value, "actual_alpha": rec.alpha,
                "actual_fired": rec.fired,
                "proposed_op": p_op, "proposed_alpha": round(p_alpha, 4),
                "proposed_by": p_rule,
                "match": p_op == rec.op.value, "actual_beneficial": rec.was_beneficial(),
            })
        return results

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "rules": [r.to_dict() for r in self.rules]}
        if self.default_constraint:
            d["default_constraint"] = self.default_constraint.to_dict()
        if self.default_escalation:
            d["default_escalation"] = self.default_escalation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict, predicate_registry: dict[str, PredicateFn]) -> Policy:
        rules = []
        for rd in d.get("rules", []):
            name = rd["name"]
            if name not in predicate_registry:
                raise ValueError(f"No predicate registered for rule {name!r}")
            rules.append(PolicyRule.from_dict(rd, predicate_registry[name]))
        return cls(
            name=d["name"], rules=rules,
            default_constraint=Constraint.from_dict(d["default_constraint"]) if "default_constraint" in d else None,
            default_escalation=Escalation.from_dict(d["default_escalation"]) if "default_escalation" in d else None,
        )

    def __repr__(self) -> str:
        return f"Policy({self.name!r}, {len(self.rules)} rules)"
