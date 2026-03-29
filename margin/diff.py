"""
Expression diffing: what changed between two snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .health import Health, SEVERITY
from .observation import Observation, Expression
from .confidence import Confidence


@dataclass
class ComponentChange:
    """
    What changed for a single component between two expressions.

    name:          component identifier
    health_before: Health in the earlier expression (None if component was absent)
    health_after:  Health in the later expression (None if component disappeared)
    sigma_before:  sigma in the earlier expression
    sigma_after:   sigma in the later expression
    """
    name: str
    health_before: Optional[Health]
    health_after: Optional[Health]
    sigma_before: Optional[float]
    sigma_after: Optional[float]

    @property
    def health_changed(self) -> bool:
        return self.health_before != self.health_after

    @property
    def sigma_delta(self) -> Optional[float]:
        """Change in sigma. Positive = healthier. None if either side missing."""
        if self.sigma_before is None or self.sigma_after is None:
            return None
        return self.sigma_after - self.sigma_before

    @property
    def appeared(self) -> bool:
        """Component was not in before but is in after."""
        return self.health_before is None and self.health_after is not None

    @property
    def disappeared(self) -> bool:
        """Component was in before but not in after."""
        return self.health_before is not None and self.health_after is None

    @property
    def worsened(self) -> bool:
        """Health got strictly worse (moved toward ABLATED/OOD)."""
        if not self.health_changed or self.health_before is None or self.health_after is None:
            return False
        return SEVERITY.get(self.health_after, 0) > SEVERITY.get(self.health_before, 0)

    @property
    def improved(self) -> bool:
        """Health got strictly better (moved toward INTACT)."""
        if not self.health_changed or self.health_before is None or self.health_after is None:
            return False
        return SEVERITY.get(self.health_after, 0) < SEVERITY.get(self.health_before, 0)

    def to_string(self) -> str:
        b = self.health_before.value if self.health_before else "absent"
        a = self.health_after.value if self.health_after else "absent"
        if not self.health_changed:
            sd = self.sigma_delta
            if sd is not None and abs(sd) > 0.001:
                sign = "+" if sd >= 0 else ""
                return f"{self.name}: {a} ({sign}{sd:.2f}σ)"
            return f"{self.name}: {a} (unchanged)"
        return f"{self.name}: {b} → {a}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "health_before": self.health_before.value if self.health_before else None,
            "health_after": self.health_after.value if self.health_after else None,
            "sigma_before": round(self.sigma_before, 4) if self.sigma_before is not None else None,
            "sigma_after": round(self.sigma_after, 4) if self.sigma_after is not None else None,
            "sigma_delta": round(self.sigma_delta, 4) if self.sigma_delta is not None else None,
            "health_changed": self.health_changed,
            "worsened": self.worsened,
            "improved": self.improved,
        }



@dataclass
class Diff:
    """
    The difference between two Expressions.

    changes:            per-component changes
    confidence_before:  net confidence of the earlier expression
    confidence_after:   net confidence of the later expression
    """
    changes: list[ComponentChange] = field(default_factory=list)
    confidence_before: Confidence = Confidence.MODERATE
    confidence_after: Confidence = Confidence.MODERATE

    @property
    def any_health_changed(self) -> bool:
        return any(c.health_changed for c in self.changes)

    @property
    def any_worsened(self) -> bool:
        return any(c.worsened for c in self.changes)

    @property
    def any_improved(self) -> bool:
        return any(c.improved for c in self.changes)

    def worsened(self) -> list[ComponentChange]:
        return [c for c in self.changes if c.worsened]

    def improved(self) -> list[ComponentChange]:
        return [c for c in self.changes if c.improved]

    def appeared(self) -> list[ComponentChange]:
        return [c for c in self.changes if c.appeared]

    def disappeared(self) -> list[ComponentChange]:
        return [c for c in self.changes if c.disappeared]

    def to_string(self) -> str:
        if not self.changes:
            return "(no components)"
        return "\n".join(c.to_string() for c in self.changes)

    def to_dict(self) -> dict:
        return {
            "confidence_before": self.confidence_before.value,
            "confidence_after": self.confidence_after.value,
            "changes": [c.to_dict() for c in self.changes],
        }


def diff(before: Expression, after: Expression) -> Diff:
    """
    Compute the difference between two Expressions.

    Reports per-component health changes, sigma deltas, and
    components that appeared or disappeared.
    """
    # For duplicate names, keep the worst observation (most conservative)
    def _worst_by_name(observations: list[Observation]) -> dict[str, Observation]:
        result: dict[str, Observation] = {}
        for o in observations:
            if o.name not in result or SEVERITY.get(o.health, 0) > SEVERITY.get(result[o.name].health, 0):
                result[o.name] = o
        return result

    before_map = _worst_by_name(before.observations)
    after_map = _worst_by_name(after.observations)
    all_names = list(dict.fromkeys(list(before_map) + list(after_map)))

    changes = []
    for name in all_names:
        b = before_map.get(name)
        a = after_map.get(name)
        changes.append(ComponentChange(
            name=name,
            health_before=b.health if b else None,
            health_after=a.health if a else None,
            sigma_before=b.sigma if b else None,
            sigma_after=a.sigma if a else None,
        ))

    return Diff(
        changes=changes,
        confidence_before=before.confidence,
        confidence_after=after.confidence,
    )
