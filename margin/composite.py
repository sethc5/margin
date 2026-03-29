"""
Composite observations: a component with multiple related measurements.

A CompositeObservation holds several sub-measurements (e.g. p50/p95/p99
latency) and derives a single Health from the worst or from a custom
aggregation strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from .confidence import Confidence
from .health import Health, SEVERITY
from .observation import Observation


class AggregateStrategy(Enum):
    """How to combine sub-observations into a single health."""
    WORST = "worst"    # health = worst sub-observation
    BEST = "best"      # health = best sub-observation
    MAJORITY = "majority"  # health = most common sub-observation


def _worst_health(healths: list[Health]) -> Health:
    return max(healths, key=lambda h: SEVERITY.get(h, 0))


def _best_health(healths: list[Health]) -> Health:
    return min(healths, key=lambda h: SEVERITY.get(h, 0))


def _majority_health(healths: list[Health]) -> Health:
    from collections import Counter
    counts = Counter(healths)
    return counts.most_common(1)[0][0]


_STRATEGIES: dict[AggregateStrategy, Callable] = {
    AggregateStrategy.WORST: _worst_health,
    AggregateStrategy.BEST: _best_health,
    AggregateStrategy.MAJORITY: _majority_health,
}


@dataclass
class CompositeObservation:
    """
    A component with multiple related sub-measurements.

    name:          component identifier (e.g. "api-latency")
    sub_observations: individual measurements (e.g. p50, p95, p99)
    strategy:      how to derive overall health
    """
    name: str
    sub_observations: list[Observation] = field(default_factory=list)
    strategy: AggregateStrategy = AggregateStrategy.WORST
    measured_at: Optional[datetime] = None

    @property
    def health(self) -> Health:
        """Aggregate health from sub-observations."""
        if not self.sub_observations:
            return Health.OOD
        healths = [o.health for o in self.sub_observations]
        return _STRATEGIES[self.strategy](healths)

    @property
    def confidence(self) -> Confidence:
        """Net confidence = weakest sub-observation."""
        if not self.sub_observations:
            return Confidence.INDETERMINATE
        return min(o.confidence for o in self.sub_observations)

    @property
    def worst(self) -> Optional[Observation]:
        """The sub-observation with the worst health."""
        if not self.sub_observations:
            return None
        return max(self.sub_observations, key=lambda o: SEVERITY.get(o.health, 0))

    @property
    def best(self) -> Optional[Observation]:
        """The sub-observation with the best health."""
        if not self.sub_observations:
            return None
        return min(self.sub_observations, key=lambda o: SEVERITY.get(o.health, 0))

    def as_observation(self) -> Observation:
        """
        Flatten to a single Observation using the aggregate health.
        Uses the worst sub-observation's value/baseline for sigma.
        """
        w = self.worst
        if w is None:
            return Observation(
                name=self.name, health=Health.OOD, value=0.0, baseline=0.0,
                confidence=Confidence.INDETERMINATE,
                measured_at=self.measured_at,
            )
        return Observation(
            name=self.name,
            health=self.health,
            value=w.value,
            baseline=w.baseline,
            confidence=self.confidence,
            higher_is_better=w.higher_is_better,
            provenance=w.provenance,
            measured_at=self.measured_at,
        )

    def to_atom(self) -> str:
        """Compact string showing aggregate + sub-count."""
        h = self.health.value
        n = len(self.sub_observations)
        worst_sigma = self.worst.sigma if self.worst else 0.0
        sign = "+" if worst_sigma >= 0 else ""
        return f"{self.name}:{h}({sign}{worst_sigma:.2f}σ, {n} sub)"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "health": self.health.value,
            "confidence": self.confidence.value,
            "strategy": self.strategy.value,
            "sub_observations": [o.to_dict() for o in self.sub_observations],
            "measured_at": self.measured_at.isoformat() if self.measured_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CompositeObservation':
        measured_at = datetime.fromisoformat(d["measured_at"]) if d.get("measured_at") else None
        return cls(
            name=d["name"],
            sub_observations=[Observation.from_dict(o) for o in d.get("sub_observations", [])],
            strategy=AggregateStrategy(d.get("strategy", "worst")),
            measured_at=measured_at,
        )

    def __repr__(self) -> str:
        return f"CompositeObservation({self.to_atom()})"
