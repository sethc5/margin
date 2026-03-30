"""
Observations and corrections: the typed vocabulary for what was measured
and what was done about it.

Observation:  one component measured at one point in time
Correction:   what action was taken (or claimed) on a component
Expression:   composed snapshot — all observations + corrections at one moment
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .confidence import Confidence
from .health import Health, Thresholds, classify


class Op(Enum):
    """
    What corrective operation is being performed.

    RESTORE:  reconstruct a sub-threshold component
    SUPPRESS: silence a spuriously active component
    AMPLIFY:  strengthen a weak-but-present component
    NOOP:     no correction applied
    """
    RESTORE = "RESTORE"
    SUPPRESS = "SUPPRESS"
    AMPLIFY = "AMPLIFY"
    NOOP = "NOOP"


# -----------------------------------------------------------------------
# Observation — one component, one measurement
# -----------------------------------------------------------------------

@dataclass
class Observation:
    """
    A single component's health at one measurement.

    name:       component identifier (e.g. "api-latency", "IOI", "heart-rate")
    health:     typed health predicate
    value:      raw measurement
    baseline:   expected value when healthy (for sigma normalisation)
    confidence: measurement confidence
    higher_is_better: polarity — matches the Thresholds that produced this
    provenance: upstream provenance IDs
    """
    name: str
    health: Health
    value: float
    baseline: float
    confidence: Confidence
    higher_is_better: bool = True
    provenance: list[str] = field(default_factory=list)
    measured_at: Optional[datetime] = None
    health_label: Optional[str] = None

    @property
    def sigma(self) -> float:
        """
        Dimensionless deviation from baseline.
        Always: positive = healthier than baseline, negative = worse.

        For higher_is_better=True:  (value - baseline) / |baseline|
        For higher_is_better=False: (baseline - value) / |baseline|
        """
        if self.baseline == 0.0:
            return 0.0
        raw = (self.value - self.baseline) / abs(self.baseline)
        return raw if self.higher_is_better else -raw

    def age(self, now: Optional[datetime] = None) -> Optional[float]:
        """Seconds since this observation was taken. None if no timestamp."""
        if self.measured_at is None:
            return None
        now = now or datetime.now()
        return (now - self.measured_at).total_seconds()

    def is_fresh(self, max_age_seconds: float = 60.0, now: Optional[datetime] = None) -> bool:
        """True if measured within max_age_seconds. True if no timestamp set."""
        a = self.age(now)
        if a is None:
            return True
        return a <= max_age_seconds

    def to_atom(self) -> str:
        """Compact string: NAME:HEALTH(±σ)"""
        display = self.health_label or self.health.value
        if self.health == Health.OOD:
            return f"{self.name}:{display}"
        sign = "+" if self.sigma >= 0 else ""
        return f"{self.name}:{display}({sign}{self.sigma:.2f}σ)"

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "health": self.health.value,
            "value": self.value,
            "baseline": self.baseline,
            "sigma": self.sigma,
            "confidence": self.confidence.value,
            "higher_is_better": self.higher_is_better,
            "provenance": self.provenance,
        }
        if self.health_label is not None:
            d["health_label"] = self.health_label
        if self.measured_at is not None:
            d["measured_at"] = self.measured_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Observation':
        measured_at = None
        if "measured_at" in d:
            measured_at = datetime.fromisoformat(d["measured_at"])
        return cls(
            name=d["name"],
            health=Health(d["health"]),
            value=d["value"],
            baseline=d["baseline"],
            confidence=Confidence(d["confidence"]),
            higher_is_better=d.get("higher_is_better", True),
            provenance=d.get("provenance", []),
            measured_at=measured_at,
            health_label=d.get("health_label"),
        )


# -----------------------------------------------------------------------
# Correction — what action was taken
# -----------------------------------------------------------------------

@dataclass
class Correction:
    """
    What corrective action is being applied to a component.

    target:     which component
    op:         the operation type
    alpha:      mixing/intensity coefficient (0 = none, 1 = full)
    magnitude:  size of the correction (domain-specific units)
    triggered_by: names of components whose state triggered this
    """
    target: str
    op: Op
    alpha: float = 0.0
    magnitude: float = 0.0
    triggered_by: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.op != Op.NOOP and self.alpha > 0.0

    def to_atom(self) -> str:
        if self.op == Op.NOOP:
            return "NOOP"
        return f"{self.op.value}(α={self.alpha:.2f})"

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "op": self.op.value,
            "alpha": self.alpha,
            "magnitude": self.magnitude,
            "triggered_by": self.triggered_by,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Correction':
        return cls(
            target=d["target"],
            op=Op(d["op"]),
            alpha=d.get("alpha", 0.0),
            magnitude=d.get("magnitude", 0.0),
            triggered_by=d.get("triggered_by", []),
            provenance=d.get("provenance", []),
        )


# -----------------------------------------------------------------------
# Expression — composed snapshot
# -----------------------------------------------------------------------

@dataclass
class Expression:
    """
    Full snapshot of all component observations and corrections at one moment.

    observations: per-component health readings
    corrections:  actions being applied (may be empty)
    confidence:   net confidence = weakest observation
    """
    observations: list[Observation] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    label: str = ""
    step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.observations:
            weakest = min(o.confidence for o in self.observations)
            if self.confidence > weakest:
                warnings.warn(
                    f"Expression.confidence ({self.confidence.value}) exceeds the weakest "
                    f"observation confidence ({weakest.value}). This overclaims certainty "
                    "and will cause min_confidence policy gates to fire incorrectly. "
                    "Use Parser.parse() to construct Expressions with correct net confidence.",
                    stacklevel=2,
                )

    def health_of(self, name: str) -> Optional[Health]:
        for o in self.observations:
            if o.name == name:
                return o.health
        return None

    def correction_for(self, name: str) -> Optional[Correction]:
        for c in self.corrections:
            if c.target == name:
                return c
        return None

    def degraded(self) -> list[Observation]:
        return [o for o in self.observations
                if o.health in (Health.DEGRADED, Health.ABLATED, Health.RECOVERING)]

    def intact(self) -> list[Observation]:
        return [o for o in self.observations if o.health == Health.INTACT]

    def to_string(self) -> str:
        """Compact bracket notation."""
        groups: dict[str, dict] = {}
        for o in self.observations:
            groups.setdefault(o.name, {"obs": [], "correction": None})
            groups[o.name]["obs"].append(o)
        for c in self.corrections:
            if c.target in groups:
                groups[c.target]["correction"] = c

        parts = []
        for name, g in groups.items():
            atoms = ", ".join(o.to_atom() for o in g["obs"])
            c = g["correction"]
            if c and c.is_active():
                parts.append(f"[{atoms} → {c.to_atom()}]")
            else:
                parts.append(f"[{atoms}]")

        for c in self.corrections:
            if c.target not in groups and c.is_active():
                parts.append(f"[? → {c.to_atom()}]")

        return " ".join(parts) if parts else "[∅]"

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence.value,
            "label": self.label,
            "step": self.step,
            "observations": [o.to_dict() for o in self.observations],
            "corrections": [c.to_dict() for c in self.corrections],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> 'Expression':
        return cls(
            observations=[Observation.from_dict(o) for o in d.get("observations", [])],
            corrections=[Correction.from_dict(c) for c in d.get("corrections", [])],
            confidence=Confidence(d["confidence"]) if "confidence" in d else Confidence.MODERATE,
            label=d.get("label", ""),
            step=d.get("step"),
        )

    @classmethod
    def from_json(cls, s: str) -> 'Expression':
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return f"Expression({self.to_string()})"


# -----------------------------------------------------------------------
# Parser — turns raw measurements into typed Expressions
# -----------------------------------------------------------------------

class Parser:
    """
    Converts raw measurements into typed Expressions.

    baselines:   {component_name: expected_healthy_value}
    thresholds:  default Thresholds (can be overridden per-component)
    component_thresholds: optional per-component overrides
    """

    def __init__(
        self,
        baselines: dict[str, float],
        thresholds: Thresholds,
        component_thresholds: Optional[dict[str, Thresholds]] = None,
    ):
        self.baselines = baselines
        self.thresholds = thresholds
        self.component_thresholds = component_thresholds or {}

    def _thresholds_for(self, name: str) -> Thresholds:
        return self.component_thresholds.get(name, self.thresholds)

    def label_for(self, component: str, health: Health) -> str:
        """Return the display label for a component's health state."""
        return self._thresholds_for(component).label_for(health)

    def with_baselines(self, fingerprint: dict[str, dict]) -> 'Parser':
        """
        Return a new Parser with baselines updated from a fingerprint dict.

        Components present in ``fingerprint`` get their baseline replaced with
        ``fingerprint[name]["mean"]``. All other components and all thresholds
        are preserved unchanged.

        Typical use: dispositional calibration from ``Monitor.fingerprint()``
        at a session boundary, without full Parser reconstruction.

            fp = monitor.fingerprint()
            new_parser = monitor.parser.with_baselines(fp)
            # new_parser has empirical session means as baselines
        """
        new_baselines = dict(self.baselines)
        for name, stats in fingerprint.items():
            if name in new_baselines and "mean" in stats:
                new_baselines[name] = stats["mean"]
        return Parser(
            baselines=new_baselines,
            thresholds=self.thresholds,
            component_thresholds=dict(self.component_thresholds),
        )

    def parse(
        self,
        values: dict[str, float],
        correction_magnitude: float = 0.0,
        alpha: float = 0.0,
        confidences: Optional[dict[str, Confidence]] = None,
        label: str = "",
        step: Optional[int] = None,
        provenance: Optional[list[str]] = None,
    ) -> Expression:
        """
        Parse a set of component measurements into a typed Expression.

        values:               {component_name: measurement}
        correction_magnitude: size of the correction being applied
        alpha:                mixing coefficient for the correction
        confidences:          per-component Confidence overrides (defaults to MODERATE)
        label:                tag for this expression (e.g. step identifier)
        step:                 step index (set automatically by Monitor)
        provenance:           provenance tags attached to all observations
        """
        confidences = confidences or {}
        provenance = provenance or []

        observations = []
        for name, val in values.items():
            if name not in self.baselines:
                warnings.warn(
                    f"Parser.parse: component {name!r} has no baseline — "
                    "using the observed value as baseline (sigma=0). "
                    "Add it to Parser.baselines or check for a typo.",
                    stacklevel=2,
                )
            baseline = self.baselines.get(name, val)
            conf = confidences.get(name, Confidence.MODERATE)
            ct = self._thresholds_for(name)
            correcting = correction_magnitude >= ct.active_min
            h = classify(val, conf, correcting, ct)
            observations.append(Observation(
                name=name, health=h, value=val, baseline=baseline,
                confidence=conf, higher_is_better=ct.higher_is_better,
                provenance=list(provenance),
                health_label=ct.label_for(h) if ct.labels else None,
            ))

        corrections = []
        if values:
            target, op = self._classify_op(values, correction_magnitude)
            if target:
                degraded_names = [o.name for o in observations
                                  if o.health in (Health.DEGRADED, Health.ABLATED, Health.RECOVERING)]
                corrections.append(Correction(
                    target=target, op=op,
                    alpha=alpha if op != Op.NOOP else 0.0,
                    magnitude=correction_magnitude,
                    triggered_by=degraded_names,
                    provenance=list(provenance),
                ))

        net_conf = min(
            (o.confidence for o in observations),
            default=Confidence.INDETERMINATE,
        )

        return Expression(
            observations=observations,
            corrections=corrections,
            confidence=net_conf,
            label=label,
            step=step,
        )

    def _classify_op(self, values: dict[str, float], magnitude: float) -> tuple[str, Op]:
        """Infer correction op: target = worst component whose active_min is met."""

        # Rank all components by degradation (most degraded first).
        def degradation(name: str) -> float:
            val = values[name]
            ct = self._thresholds_for(name)
            if ct.higher_is_better:
                return val - ct.intact
            else:
                return ct.intact - val

        ranked = sorted(values, key=degradation)

        # Find the worst component whose active_min is met by the correction.
        # If none meet it, return the overall worst with NOOP.
        worst = ranked[0]
        target = None
        for name in ranked:
            ct = self._thresholds_for(name)
            if magnitude >= ct.active_min:
                target = name
                break

        if target is None:
            return (worst, Op.NOOP)

        worst_t = self._thresholds_for(target)
        worst_val = values[target]
        baseline = self.baselines.get(target, worst_val)

        # Past baseline in the *healthy* direction → SUPPRESS
        # (component is over-performing / spuriously over-active)
        if worst_t.higher_is_better:
            over_performing = worst_val > baseline
        else:
            over_performing = worst_val < baseline
        if over_performing:
            return (target, Op.SUPPRESS)

        # Not at intact threshold → RESTORE
        if not worst_t.is_intact(worst_val):
            return (target, Op.RESTORE)

        return (target, Op.AMPLIFY)
