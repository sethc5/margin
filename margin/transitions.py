"""
State transition tracking over a Ledger.

Tracks how components move between Health states over time:
durations, transition counts, and transition history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import Counter

from .health import Health
from .ledger import Ledger


@dataclass
class Span:
    """
    A contiguous span of time a component spent in one Health state.

    health:     the Health state
    start_step: step index where this span began
    end_step:   step index where this span ended (None if still active)
    start_time: timestamp of the first record in this span
    end_time:   timestamp of the last record (None if still active)
    """
    health: Health
    start_step: int
    end_step: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def n_steps(self) -> int:
        if self.end_step is None:
            return 1
        return self.end_step - self.start_step + 1

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict:
        return {
            "health": self.health.value,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "n_steps": self.n_steps,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
        }


@dataclass
class Transition:
    """A single health state transition."""
    from_health: Health
    to_health: Health
    at_step: int
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "from": self.from_health.value,
            "to": self.to_health.value,
            "at_step": self.at_step,
        }


@dataclass
class ComponentHistory:
    """
    Full state history for one component across a ledger.

    name:         component identifier
    spans:        ordered list of contiguous health spans
    transitions:  ordered list of state transitions
    """
    name: str
    spans: list[Span] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def transition_counts(self) -> dict[tuple[str, str], int]:
        """Count of each (from, to) transition pair."""
        c: Counter = Counter()
        for t in self.transitions:
            c[(t.from_health.value, t.to_health.value)] += 1
        return dict(c)

    def time_in_state(self) -> dict[str, int]:
        """Number of steps spent in each Health state."""
        counts: dict[str, int] = {}
        for s in self.spans:
            counts[s.health.value] = counts.get(s.health.value, 0) + s.n_steps
        return counts

    def last_health(self) -> Optional[Health]:
        if self.spans:
            return self.spans[-1].health
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_transitions": self.n_transitions,
            "transition_counts": {f"{k[0]}→{k[1]}": v for k, v in self.transition_counts().items()},
            "time_in_state": self.time_in_state(),
            "spans": [s.to_dict() for s in self.spans],
        }


def track(ledger: Ledger, component: str) -> ComponentHistory:
    """
    Extract the state history of one component from a ledger.

    Reads the before/after observations in each Record, builds spans
    of contiguous health states, and records transitions between them.
    """
    history = ComponentHistory(name=component)
    current_span: Optional[Span] = None

    for rec in ledger.records:
        obs = rec.after if rec.after and rec.after.name == component else None
        if obs is None and rec.before.name == component:
            obs = rec.before

        if obs is None or obs.name != component:
            continue

        health = obs.health

        if current_span is None:
            current_span = Span(
                health=health,
                start_step=rec.step,
                start_time=rec.timestamp,
            )
        elif health != current_span.health:
            # Close current span
            current_span.end_step = rec.step - 1
            current_span.end_time = rec.timestamp
            history.spans.append(current_span)

            # Record transition
            history.transitions.append(Transition(
                from_health=current_span.health,
                to_health=health,
                at_step=rec.step,
                timestamp=rec.timestamp,
            ))

            # Start new span
            current_span = Span(
                health=health,
                start_step=rec.step,
                start_time=rec.timestamp,
            )
        else:
            # Same state — extend the span
            current_span.end_step = rec.step
            current_span.end_time = rec.timestamp

    if current_span is not None:
        history.spans.append(current_span)

    return history


def track_all(ledger: Ledger) -> dict[str, ComponentHistory]:
    """
    Extract state histories for all components found in a ledger.
    Returns {component_name: ComponentHistory}.
    """
    names: set[str] = set()
    for rec in ledger.records:
        names.add(rec.before.name)
        if rec.after:
            names.add(rec.after.name)
    return {name: track(ledger, name) for name in sorted(names)}
