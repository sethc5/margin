"""
Intent layer: can this system still achieve its goal?

Health says WHERE components are. Drift says WHERE THEY'RE HEADED.
Intent says CAN WE STILL MAKE IT — given current state, trajectories,
and a goal with requirements and a deadline.

    intent = Intent(
        goal="deliver package to dock 7",
        requires={"battery_soc": 20.0, "navigation": Health.INTACT},
        deadline_seconds=900,
    )
    result = intent.evaluate(monitor)
    result.feasibility  # FEASIBLE / AT_RISK / INFEASIBLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

from .health import Health, SEVERITY
from .confidence import Confidence
from .observation import Expression, Observation
from .drift import DriftClassification, DriftDirection, DriftState


class Feasibility(Enum):
    """Can the system achieve its goal?"""
    FEASIBLE = "FEASIBLE"         # all requirements met, trajectories OK
    AT_RISK = "AT_RISK"           # requirements met now, but drift threatens deadline
    INFEASIBLE = "INFEASIBLE"     # requirements already violated or will be before deadline
    UNKNOWN = "UNKNOWN"           # insufficient data to evaluate


@dataclass
class Requirement:
    """
    One requirement for an intent.

    component:      which component this applies to
    min_health:     minimum acceptable Health state (default: DEGRADED — anything above ABLATED)
    min_value:      minimum acceptable value (optional — for numeric thresholds like battery > 20%)
    critical:       if True, violation makes intent INFEASIBLE; if False, AT_RISK
    """
    component: str
    min_health: Health = Health.DEGRADED
    min_value: Optional[float] = None
    critical: bool = True

    def to_dict(self) -> dict:
        d: dict = {
            "component": self.component,
            "min_health": self.min_health.value,
            "critical": self.critical,
        }
        if self.min_value is not None:
            d["min_value"] = self.min_value
        return d


@dataclass
class RiskFactor:
    """One reason the intent is at risk or infeasible."""
    component: str
    reason: str
    severity: str  # "critical" or "warning"
    eta_seconds: Optional[float] = None  # time until requirement is violated

    def to_dict(self) -> dict:
        d = {"component": self.component, "reason": self.reason, "severity": self.severity}
        if self.eta_seconds is not None:
            d["eta_seconds"] = round(self.eta_seconds, 1)
        return d

    def __repr__(self) -> str:
        eta = f", ETA {self.eta_seconds:.0f}s" if self.eta_seconds is not None else ""
        return f"RiskFactor({self.component}: {self.reason} [{self.severity}]{eta})"


@dataclass
class IntentResult:
    """
    Result of evaluating an intent against current system state.

    feasibility:    FEASIBLE / AT_RISK / INFEASIBLE / UNKNOWN
    risks:          list of RiskFactors explaining why not FEASIBLE
    met:            requirements that are currently satisfied
    violated:       requirements that are currently violated
    trending_bad:   requirements that are met now but drifting toward violation
    confidence:     how much to trust this assessment
    evaluated_at:   when this was computed
    """
    feasibility: Feasibility
    risks: list[RiskFactor] = field(default_factory=list)
    met: list[str] = field(default_factory=list)
    violated: list[str] = field(default_factory=list)
    trending_bad: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    evaluated_at: datetime = field(default_factory=datetime.now)

    @property
    def feasible(self) -> bool:
        return self.feasibility == Feasibility.FEASIBLE

    @property
    def at_risk(self) -> bool:
        return self.feasibility == Feasibility.AT_RISK

    @property
    def infeasible(self) -> bool:
        return self.feasibility == Feasibility.INFEASIBLE

    def summary(self) -> str:
        """Human-readable one-liner."""
        parts = [self.feasibility.value]
        if self.risks:
            top = self.risks[0]
            parts.append(f"{top.component}: {top.reason}")
            if top.eta_seconds is not None:
                parts.append(f"ETA {_fmt_seconds(top.eta_seconds)}")
        return " — ".join(parts)

    def to_dict(self) -> dict:
        return {
            "feasibility": self.feasibility.value,
            "risks": [r.to_dict() for r in self.risks],
            "met": self.met,
            "violated": self.violated,
            "trending_bad": self.trending_bad,
            "confidence": self.confidence.value,
            "summary": self.summary(),
        }

    def __repr__(self) -> str:
        return f"IntentResult({self.summary()})"


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


# -----------------------------------------------------------------------
# Intent
# -----------------------------------------------------------------------

@dataclass
class Intent:
    """
    A system's goal with requirements and a deadline.

    goal:              human-readable description of what the system is trying to do
    requirements:      list of Requirements that must hold
    deadline_seconds:  time budget in seconds (None = no deadline)
    """
    goal: str
    requirements: list[Requirement] = field(default_factory=list)
    deadline_seconds: Optional[float] = None

    def require(
        self,
        component: str,
        min_health: Health = Health.DEGRADED,
        min_value: Optional[float] = None,
        critical: bool = True,
    ) -> 'Intent':
        """Add a requirement. Returns self for chaining."""
        self.requirements.append(Requirement(component, min_health, min_value, critical))
        return self

    def evaluate(
        self,
        expression: Expression,
        drift_by_component: Optional[dict[str, DriftClassification]] = None,
        now: Optional[datetime] = None,
    ) -> IntentResult:
        """
        Evaluate feasibility against current state and trajectories.

        Args:
            expression:          current health Expression
            drift_by_component:  {component: DriftClassification} for trajectory checking
            now:                 current time (for ETA calculations)
        """
        now = now or datetime.now()
        drift_by_component = drift_by_component or {}

        risks: list[RiskFactor] = []
        met: list[str] = []
        violated: list[str] = []
        trending_bad: list[str] = []

        has_critical_violation = False
        has_risk = False

        for req in self.requirements:
            obs = _find_observation(expression, req.component)

            if obs is None:
                # Component not in expression — can't evaluate
                risks.append(RiskFactor(
                    req.component, "component not observed",
                    "critical" if req.critical else "warning",
                ))
                if req.critical:
                    has_critical_violation = True
                else:
                    has_risk = True
                violated.append(req.component)
                continue

            # Check health requirement
            health_ok = SEVERITY[obs.health] <= SEVERITY[req.min_health]

            # Check value requirement
            value_ok = True
            if req.min_value is not None:
                if obs.higher_is_better:
                    value_ok = obs.value >= req.min_value
                else:
                    value_ok = obs.value <= req.min_value

            if not health_ok or not value_ok:
                reason = f"{obs.health.value}"
                if not value_ok and req.min_value is not None:
                    reason += f", value={obs.value:.2f} (need {'≥' if obs.higher_is_better else '≤'}{req.min_value})"
                risks.append(RiskFactor(
                    req.component, reason,
                    "critical" if req.critical else "warning",
                ))
                if req.critical:
                    has_critical_violation = True
                else:
                    has_risk = True
                violated.append(req.component)
                continue

            # Requirement currently met — check drift trajectory
            dc = drift_by_component.get(req.component)
            if dc and dc.direction == DriftDirection.WORSENING:
                # Estimate time to violation
                eta = _estimate_eta_to_violation(obs, dc, req, self.deadline_seconds)
                at_risk_for_deadline = (
                    self.deadline_seconds is not None
                    and eta is not None
                    and eta < self.deadline_seconds
                )
                if at_risk_for_deadline:
                    risks.append(RiskFactor(
                        req.component,
                        f"DRIFTING(WORSENING), will violate before deadline",
                        "critical" if req.critical else "warning",
                        eta_seconds=eta,
                    ))
                    if req.critical:
                        has_risk = True
                    trending_bad.append(req.component)
                else:
                    risks.append(RiskFactor(
                        req.component,
                        f"DRIFTING(WORSENING)",
                        "warning",
                        eta_seconds=eta,
                    ))
                    trending_bad.append(req.component)
                met.append(req.component)
            else:
                met.append(req.component)

        # Determine overall feasibility
        if has_critical_violation:
            feasibility = Feasibility.INFEASIBLE
        elif has_risk or trending_bad:
            feasibility = Feasibility.AT_RISK
        elif not self.requirements:
            feasibility = Feasibility.UNKNOWN
        else:
            feasibility = Feasibility.FEASIBLE

        # Confidence based on data availability
        n_with_drift = sum(1 for r in self.requirements if r.component in drift_by_component)
        if n_with_drift == len(self.requirements) and len(self.requirements) > 0:
            confidence = Confidence.HIGH
        elif n_with_drift > 0:
            confidence = Confidence.MODERATE
        else:
            confidence = Confidence.LOW

        # Sort risks: critical first, then by ETA
        risks.sort(key=lambda r: (0 if r.severity == "critical" else 1, r.eta_seconds or float('inf')))

        return IntentResult(
            feasibility=feasibility,
            risks=risks,
            met=met,
            violated=violated,
            trending_bad=trending_bad,
            confidence=confidence,
            evaluated_at=now,
        )

    def evaluate_monitor(self, monitor) -> IntentResult:
        """
        Convenience: evaluate against a streaming Monitor.

        Extracts expression and drift classifications automatically.
        """
        if monitor.expression is None:
            return IntentResult(feasibility=Feasibility.UNKNOWN)

        drift_map = {}
        for req in self.requirements:
            dc = monitor.drift(req.component)
            if dc is not None:
                drift_map[req.component] = dc

        return self.evaluate(monitor.expression, drift_map)

    def to_dict(self) -> dict:
        d: dict = {
            "goal": self.goal,
            "requirements": [r.to_dict() for r in self.requirements],
        }
        if self.deadline_seconds is not None:
            d["deadline_seconds"] = self.deadline_seconds
        return d

    def __repr__(self) -> str:
        dl = f", deadline={_fmt_seconds(self.deadline_seconds)}" if self.deadline_seconds else ""
        return f"Intent({self.goal!r}, {len(self.requirements)} requirements{dl})"


# -----------------------------------------------------------------------
# ETA estimation
# -----------------------------------------------------------------------

def _find_observation(expr: Expression, name: str) -> Optional[Observation]:
    for o in expr.observations:
        if o.name == name:
            return o
    return None


def _estimate_eta_to_violation(
    obs: Observation,
    dc: DriftClassification,
    req: Requirement,
    deadline: Optional[float],
) -> Optional[float]:
    """
    Estimate seconds until a requirement is violated, given current drift rate.

    Uses the drift rate (units/second) to project when the value will cross
    the requirement threshold.
    """
    if dc.rate == 0:
        return None

    # What value would violate the requirement?
    target: Optional[float] = None

    if req.min_value is not None:
        target = req.min_value
    else:
        # Estimate from health threshold — violation means crossing to ABLATED
        # Use a rough heuristic: current value - some margin
        # Without actual thresholds, we can't compute exactly
        return None

    if target is None:
        return None

    # Rate is polarity-normalised (positive = healthier)
    # For higher-is-better: raw rate = dc.rate (positive = value increasing)
    # For lower-is-better: raw rate = -dc.rate (negative = value increasing)
    raw_rate = dc.rate if obs.higher_is_better else -dc.rate

    # Distance to threshold
    if obs.higher_is_better:
        distance = obs.value - target  # positive when above threshold
        if distance <= 0:
            return 0  # already violated
        if raw_rate >= 0:
            return None  # improving, won't cross
        return abs(distance / raw_rate)
    else:
        distance = target - obs.value  # positive when below threshold
        if distance <= 0:
            return 0  # already violated
        if raw_rate <= 0:
            return None  # improving, won't cross
        return abs(distance / raw_rate)
