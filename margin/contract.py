"""
Contract language: typed success criteria for correction loops.

A Contract declares what the system is trying to achieve. The Ledger
records what happened. The contract language scores one against the other.

Contracts express:
  - Target health states per component
  - Time bounds (achieve INTACT within N steps)
  - Sustained conditions (stay INTACT for N steps)
  - Aggregate thresholds (mean recovery >= 0.8)
  - Compound requirements (all of the above simultaneously)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .health import Health, SEVERITY
from .confidence import Confidence
from .observation import Expression
from .ledger import Ledger


# -----------------------------------------------------------------------
# Contract terms — individual requirements
# -----------------------------------------------------------------------

class TermStatus(Enum):
    """Whether a contract term is currently met."""
    MET = "MET"
    VIOLATED = "VIOLATED"
    PENDING = "PENDING"     # not enough data yet


@dataclass
class TermResult:
    """Evaluation result for a single contract term."""
    term_name: str
    status: TermStatus
    detail: str = ""

    def to_dict(self) -> dict:
        return {"term_name": self.term_name, "status": self.status.value, "detail": self.detail}

    def __repr__(self) -> str:
        return f"TermResult({self.term_name}: {self.status.value})"


class ContractTerm:
    """Base class for contract terms. Subclasses implement evaluate()."""
    name: str

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        raise NotImplementedError


@dataclass
class HealthTarget(ContractTerm):
    """
    Require a component to be in a specific health state (or better).

    name:       term identifier
    component:  component to check
    target:     required health state
    or_better:  if True, any state with lower severity also satisfies
    """
    name: str
    component: str
    target: Health
    or_better: bool = True

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        if expr is None:
            return TermResult(self.name, TermStatus.PENDING, "no expression provided")
        health = expr.health_of(self.component)
        if health is None:
            return TermResult(self.name, TermStatus.PENDING, f"{self.component} not in expression")
        target_sev = SEVERITY.get(self.target, 0)
        actual_sev = SEVERITY.get(health, 0)
        if self.or_better:
            met = actual_sev <= target_sev
        else:
            met = health == self.target
        status = TermStatus.MET if met else TermStatus.VIOLATED
        return TermResult(self.name, status, f"{self.component}:{health.value} vs target {self.target.value}")

    def to_dict(self) -> dict:
        return {"type": "health_target", "name": self.name, "component": self.component,
                "target": self.target.value, "or_better": self.or_better}

    def __repr__(self) -> str:
        return f"HealthTarget({self.name!r}, {self.component}={self.target.value})"


@dataclass
class ReachHealth(ContractTerm):
    """
    Require a component to reach a health state within N steps.

    Looks backward through the ledger: if the component has been at or
    better than `target` at any point in the last `within_steps`, the
    term is MET. If the window has passed without reaching it, VIOLATED.
    If not enough steps yet, PENDING.
    """
    name: str
    component: str
    target: Health
    within_steps: int

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        if ledger.max_records is not None and ledger.max_records < self.within_steps:
            warnings.warn(
                f"ReachHealth '{self.name}': ledger.max_records={ledger.max_records} < "
                f"within_steps={self.within_steps} — records older than max_records are "
                "dropped; if the target was reached before the cap, this term will show "
                "PENDING rather than MET. Increase max_records or reduce within_steps.",
                stacklevel=2,
            )
        recent = ledger.for_component(self.component).last_n(self.within_steps)
        if len(recent) < 1:
            return TermResult(self.name, TermStatus.PENDING, "no records yet")

        target_sev = SEVERITY.get(self.target, 0)
        for rec in recent.records:
            obs = rec.after if rec.after and rec.after.name == self.component else rec.before
            if obs.name == self.component and SEVERITY.get(obs.health, 99) <= target_sev:
                return TermResult(self.name, TermStatus.MET,
                                  f"reached {self.target.value} at step {rec.step}")

        if len(recent) >= self.within_steps:
            return TermResult(self.name, TermStatus.VIOLATED,
                              f"did not reach {self.target.value} within {self.within_steps} steps")
        return TermResult(self.name, TermStatus.PENDING,
                          f"{len(recent)}/{self.within_steps} steps elapsed")

    def to_dict(self) -> dict:
        return {"type": "reach_health", "name": self.name, "component": self.component,
                "target": self.target.value, "within_steps": self.within_steps}

    def __repr__(self) -> str:
        return f"ReachHealth({self.name!r}, {self.component}={self.target.value} within {self.within_steps})"


@dataclass
class SustainHealth(ContractTerm):
    """
    Require a component to sustain a health state for N consecutive steps.
    """
    name: str
    component: str
    target: Health
    for_steps: int
    or_better: bool = True

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        if ledger.max_records is not None and ledger.max_records < self.for_steps:
            warnings.warn(
                f"SustainHealth '{self.name}': ledger.max_records={ledger.max_records} < "
                f"for_steps={self.for_steps} — this term can never be MET; "
                "increase Ledger(max_records=...) or reduce SustainHealth(for_steps=...).",
                stacklevel=2,
            )
        recent = ledger.for_component(self.component).last_n(self.for_steps)
        if len(recent) < self.for_steps:
            return TermResult(self.name, TermStatus.PENDING,
                              f"{len(recent)}/{self.for_steps} steps")

        target_sev = SEVERITY.get(self.target, 0)
        for rec in recent.records:
            obs = rec.after if rec.after and rec.after.name == self.component else rec.before
            if obs.name != self.component:
                continue
            actual_sev = SEVERITY.get(obs.health, 99)
            if self.or_better and actual_sev > target_sev:
                return TermResult(self.name, TermStatus.VIOLATED,
                                  f"{obs.health.value} at step {rec.step}")
            elif not self.or_better and obs.health != self.target:
                return TermResult(self.name, TermStatus.VIOLATED,
                                  f"{obs.health.value} at step {rec.step}")

        return TermResult(self.name, TermStatus.MET,
                          f"sustained {self.target.value} for {self.for_steps} steps")

    def to_dict(self) -> dict:
        return {"type": "sustain_health", "name": self.name, "component": self.component,
                "target": self.target.value, "for_steps": self.for_steps,
                "or_better": self.or_better}

    def __repr__(self) -> str:
        return f"SustainHealth({self.name!r}, {self.component}={self.target.value} for {self.for_steps})"


@dataclass
class RecoveryThreshold(ContractTerm):
    """
    Require mean recovery ratio to be above a threshold over a window.
    """
    name: str
    min_recovery: float
    over_steps: int

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        recent = ledger.last_n(self.over_steps)
        if len(recent) < 1:
            return TermResult(self.name, TermStatus.PENDING, "no records")
        if recent.n_fired == 0:
            return TermResult(self.name, TermStatus.PENDING, "no corrections fired")
        r = recent.mean_recovery
        if r >= self.min_recovery:
            return TermResult(self.name, TermStatus.MET, f"recovery {r:.4f} >= {self.min_recovery}")
        if len(recent) >= self.over_steps:
            return TermResult(self.name, TermStatus.VIOLATED, f"recovery {r:.4f} < {self.min_recovery}")
        return TermResult(self.name, TermStatus.PENDING,
                          f"recovery {r:.4f}, {len(recent)}/{self.over_steps} steps")

    def to_dict(self) -> dict:
        return {"type": "recovery_threshold", "name": self.name,
                "min_recovery": self.min_recovery, "over_steps": self.over_steps}

    def __repr__(self) -> str:
        return f"RecoveryThreshold({self.name!r}, min={self.min_recovery} over {self.over_steps})"


@dataclass
class NoHarmful(ContractTerm):
    """Require no harmful corrections over a window."""
    name: str
    over_steps: int

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> TermResult:
        recent = ledger.last_n(self.over_steps)
        if len(recent) < 1:
            return TermResult(self.name, TermStatus.PENDING, "no records")
        n_harmful = len(recent.harmful())
        if n_harmful == 0:
            return TermResult(self.name, TermStatus.MET, f"0 harmful in {len(recent)} steps")
        return TermResult(self.name, TermStatus.VIOLATED, f"{n_harmful} harmful corrections")

    def to_dict(self) -> dict:
        return {"type": "no_harmful", "name": self.name, "over_steps": self.over_steps}

    def __repr__(self) -> str:
        return f"NoHarmful({self.name!r}, over {self.over_steps})"


_TERM_TYPES: dict = {
    "health_target": lambda d: HealthTarget(
        name=d["name"], component=d["component"],
        target=Health(d["target"]), or_better=d.get("or_better", True),
    ),
    "reach_health": lambda d: ReachHealth(
        name=d["name"], component=d["component"],
        target=Health(d["target"]), within_steps=d["within_steps"],
    ),
    "sustain_health": lambda d: SustainHealth(
        name=d["name"], component=d["component"],
        target=Health(d["target"]), for_steps=d["for_steps"],
        or_better=d.get("or_better", True),
    ),
    "recovery_threshold": lambda d: RecoveryThreshold(
        name=d["name"], min_recovery=d["min_recovery"], over_steps=d["over_steps"],
    ),
    "no_harmful": lambda d: NoHarmful(
        name=d["name"], over_steps=d["over_steps"],
    ),
}


def contract_term_from_dict(d: dict) -> ContractTerm:
    """Deserialize a ContractTerm subclass from a dict produced by its to_dict()."""
    term_type = d.get("type")
    if term_type not in _TERM_TYPES:
        raise ValueError(f"Unknown contract term type: {term_type!r}. "
                         f"Expected one of: {list(_TERM_TYPES)}")
    return _TERM_TYPES[term_type](d)


# -----------------------------------------------------------------------
# Contract — a set of terms evaluated together
# -----------------------------------------------------------------------

@dataclass
class ContractResult:
    """Evaluation of a full contract."""
    contract_name: str
    term_results: list[TermResult] = field(default_factory=list)

    @property
    def all_met(self) -> bool:
        return all(r.status == TermStatus.MET for r in self.term_results)

    @property
    def any_violated(self) -> bool:
        return any(r.status == TermStatus.VIOLATED for r in self.term_results)

    @property
    def any_pending(self) -> bool:
        return any(r.status == TermStatus.PENDING for r in self.term_results)

    def met(self) -> list[TermResult]:
        return [r for r in self.term_results if r.status == TermStatus.MET]

    def violated(self) -> list[TermResult]:
        return [r for r in self.term_results if r.status == TermStatus.VIOLATED]

    def pending(self) -> list[TermResult]:
        return [r for r in self.term_results if r.status == TermStatus.PENDING]

    def to_dict(self) -> dict:
        return {
            "contract_name": self.contract_name,
            "all_met": self.all_met,
            "any_violated": self.any_violated,
            "n_met": len(self.met()),
            "n_violated": len(self.violated()),
            "n_pending": len(self.pending()),
            "terms": [r.to_dict() for r in self.term_results],
        }

    def to_string(self) -> str:
        lines = [f"Contract({self.contract_name}):"]
        for r in self.term_results:
            icon = {"MET": "+", "VIOLATED": "!", "PENDING": "?"}[r.status.value]
            lines.append(f"  [{icon}] {r.term_name}: {r.detail}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ContractResult({self.contract_name}: {len(self.met())} met, {len(self.violated())} violated, {len(self.pending())} pending)"


@dataclass
class Contract:
    """
    A set of requirements that define what success looks like.

    Evaluate against a Ledger to get a ContractResult showing which
    terms are MET, VIOLATED, or PENDING.
    """
    name: str
    terms: list[ContractTerm] = field(default_factory=list)

    def evaluate(self, ledger: Ledger, expr: Optional[Expression] = None) -> ContractResult:
        return ContractResult(
            contract_name=self.name,
            term_results=[t.evaluate(ledger, expr) for t in self.terms],
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "terms": [t.to_dict() for t in self.terms]}

    def __repr__(self) -> str:
        return f"Contract({self.name!r}, {len(self.terms)} terms)"
