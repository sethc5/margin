"""
Correction ledger: auditable before/after trail of interventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .observation import Op, Observation, Correction, Expression
from .provenance import new_id


@dataclass
class Record:
    """
    One correction event: what was the state, what was done, what resulted.

    step:       index in the sequence (0-based)
    tag:        label for this event (e.g. token text, request ID)
    before:     observation before correction
    after:      observation after correction (None if correction skipped)
    fired:      whether the correction was applied
    op:         operation that was performed
    alpha:      mixing coefficient used
    magnitude:  size of the correction
    timestamp:  when this happened
    provenance: unique ID for this record
    """
    step: int
    tag: str
    before: Observation
    after: Optional[Observation] = None
    fired: bool = False
    op: Op = Op.NOOP
    alpha: float = 0.0
    magnitude: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    provenance: str = field(default_factory=new_id)

    @property
    def improvement(self) -> float:
        """How much the value improved. Positive = better, regardless of polarity."""
        if self.after is None:
            return 0.0
        delta = self.after.value - self.before.value
        return delta if self.before.higher_is_better else -delta

    @property
    def recovery_ratio(self) -> float:
        """
        How close to baseline the corrected value is. 1.0 = fully restored.

        Polarity-aware: for higher_is_better, ratio = after/baseline.
        For lower_is_better, ratio = baseline/after (inverted so 1.0 still
        means "at baseline" and >1.0 means "better than baseline").
        Returns 0.0 if baseline or after is zero/None.
        """
        if self.after is None or self.before.baseline == 0.0:
            return 0.0
        if self.before.higher_is_better:
            return self.after.value / self.before.baseline
        else:
            if self.after.value == 0.0:
                return 0.0
            return self.before.baseline / self.after.value

    def was_beneficial(self) -> bool:
        if not self.fired:
            return True
        return self.improvement > 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tag": self.tag,
            "fired": self.fired,
            "op": self.op.value,
            "alpha": round(self.alpha, 4),
            "magnitude": round(self.magnitude, 4),
            "improvement": round(self.improvement, 4),
            "recovery": round(self.recovery_ratio, 4),
            "before": self.before.to_dict(),
            "after": self.after.to_dict() if self.after else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Record':
        return cls(
            step=d["step"],
            tag=d["tag"],
            before=Observation.from_dict(d["before"]),
            after=Observation.from_dict(d["after"]) if d.get("after") else None,
            fired=d.get("fired", False),
            op=Op(d["op"]) if "op" in d else Op.NOOP,
            alpha=d.get("alpha", 0.0),
            magnitude=d.get("magnitude", 0.0),
        )

    def __repr__(self) -> str:
        status = "FIRED" if self.fired else "SKIP"
        return f"Record(step={self.step}, {status}, op={self.op.value}, Δ={self.improvement:+.2f})"


@dataclass
class Ledger:
    """
    Accumulates Records across a session.
    Provides aggregate stats and renders the intervention history.

    max_records: optional bound on records kept. When set, old records
                 are dropped (oldest first) on each append. Use this for
                 long-running production loops to prevent unbounded growth.
                 None (default) = unbounded, suitable for batch/replay use.
    """
    label: str = ""
    records: list[Record] = field(default_factory=list)
    max_records: Optional[int] = None

    def append(self, record: Record) -> None:
        self.records.append(record)
        if self.max_records is not None and len(self.records) > self.max_records:
            self.records.pop(0)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_fired(self) -> int:
        return sum(1 for r in self.records if r.fired)

    @property
    def fire_rate(self) -> float:
        if not self.records:
            return 0.0
        return self.n_fired / len(self.records)

    @property
    def mean_improvement(self) -> float:
        fired = [r for r in self.records if r.fired]
        if not fired:
            return 0.0
        return sum(r.improvement for r in fired) / len(fired)

    @property
    def mean_recovery(self) -> float:
        fired = [r for r in self.records if r.fired]
        if not fired:
            return 0.0
        return sum(r.recovery_ratio for r in fired) / len(fired)

    def harmful(self) -> list[Record]:
        """Records where the correction made things worse."""
        return [r for r in self.records if r.fired and not r.was_beneficial()]

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def window(self, duration: timedelta, now: Optional[datetime] = None) -> 'Ledger':
        """Return a new Ledger containing only records within `duration` of `now`."""
        now = now or datetime.now()
        cutoff = now - duration
        return Ledger(
            label=self.label,
            records=[r for r in self.records if r.timestamp >= cutoff],
        )

    def last_n(self, n: int) -> 'Ledger':
        """Return a new Ledger containing only the last `n` records."""
        if n <= 0:
            return Ledger(label=self.label, records=[])
        return Ledger(label=self.label, records=list(self.records[-n:]))

    def for_component(self, name: str) -> 'Ledger':
        """Return a new Ledger filtered to records involving `name`."""
        return Ledger(
            label=self.label,
            records=[r for r in self.records
                     if r.before.name == name or (r.after and r.after.name == name)],
        )

    def render(self) -> str:
        """Compact multi-line string, one line per step."""
        lines = []
        for r in self.records:
            obs = r.after if r.after else r.before
            expr = Expression(
                observations=[obs],
                corrections=[Correction(
                    target=r.before.name, op=r.op,
                    alpha=r.alpha if r.fired else 0.0,
                    magnitude=r.magnitude,
                )] if r.fired else [],
                confidence=obs.confidence,
                step=r.step,
            )
            lines.append(f"step {r.step:3d}: {expr.to_string()}")
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n_steps": len(self.records),
            "n_fired": self.n_fired,
            "fire_rate": round(self.fire_rate, 4),
            "mean_improvement": round(self.mean_improvement, 4),
            "mean_recovery": round(self.mean_recovery, 4),
            "n_harmful": len(self.harmful()),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "label": self.label,
            "records": [r.to_dict() for r in self.records],
        }, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> 'Ledger':
        return cls(
            label=d.get("label", ""),
            records=[Record.from_dict(r) for r in d.get("records", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> 'Ledger':
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return (f"Ledger({len(self.records)} steps, "
                f"fired={self.n_fired}, mean_Δ={self.mean_improvement:+.3f})")
