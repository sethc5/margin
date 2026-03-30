"""
Health classification: typed states for any monitored component.

Maps a scalar measurement against thresholds into a typed health predicate.
Domain-agnostic — works for circuits, services, sensors, vitals, etc.

Supports both polarities:
  higher_is_better=True  (default): throughput, signal strength, logit gap
  higher_is_better=False:           error rate, latency, temperature
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .confidence import Confidence


class Health(Enum):
    """
    Typed health predicate for a monitored component.

    INTACT:     operating normally
    DEGRADED:   impaired but present
    ABLATED:    functionally absent
    RECOVERING: sub-threshold but an active correction is running
    OOD:        measurement confidence is INDETERMINATE — can't classify
    """
    INTACT = "INTACT"
    DEGRADED = "DEGRADED"
    ABLATED = "ABLATED"
    RECOVERING = "RECOVERING"
    OOD = "OOD"


@dataclass
class Thresholds:
    """
    Decision boundaries for health classification.

    intact:           boundary for calling a component healthy
    ablated:          boundary for calling a component absent/failed
    higher_is_better: polarity — True means higher values are healthier
                      (throughput, signal), False means lower values are
                      healthier (error rate, latency)
    active_min:       minimum correction magnitude to count as "active"
    labels:           optional display labels keyed by Health.value strings

    For higher_is_better=True:  intact >= ablated, value >= intact → INTACT
    For higher_is_better=False: intact <= ablated, value <= intact → INTACT
    """
    intact: float
    ablated: float
    higher_is_better: bool = True
    active_min: float = 0.05
    labels: Optional[dict[str, str]] = None

    def label_for(self, health: 'Health') -> str:
        """Return the display label for a health state, or the state name if no label set."""
        if self.labels:
            return self.labels.get(health.value, health.value)
        return health.value

    def __post_init__(self):
        if self.higher_is_better and self.ablated > self.intact:
            raise ValueError(
                f"higher_is_better=True but ablated ({self.ablated}) > intact ({self.intact})")
        if not self.higher_is_better and self.ablated < self.intact:
            raise ValueError(
                f"higher_is_better=False but ablated ({self.ablated}) < intact ({self.intact})")

    def is_intact(self, value: float) -> bool:
        """True if value is in the healthy zone."""
        return value >= self.intact if self.higher_is_better else value <= self.intact

    def is_ablated(self, value: float) -> bool:
        """True if value is in the failed zone."""
        return value < self.ablated if self.higher_is_better else value > self.ablated


# Severity ordering: higher = worse. Used by diff and composite.
SEVERITY = {
    Health.INTACT: 0,
    Health.RECOVERING: 1,
    Health.DEGRADED: 2,
    Health.ABLATED: 3,
    Health.OOD: 4,
}


def classify(
    value: float,
    confidence: Confidence,
    correcting: bool = False,
    thresholds: Optional[Thresholds] = None,
) -> Health:
    """
    Classify a scalar measurement as a Health predicate.

    This is the single source of truth for health classification.
    Polarity is handled by the Thresholds object.
    """
    if confidence == Confidence.INDETERMINATE:
        return Health.OOD
    if thresholds is None:
        raise ValueError("Thresholds required for classification")
    if thresholds.is_intact(value):
        return Health.INTACT
    if thresholds.is_ablated(value):
        return Health.RECOVERING if correcting else Health.ABLATED
    return Health.RECOVERING if correcting else Health.DEGRADED
