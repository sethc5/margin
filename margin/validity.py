"""
Temporal validity for uncertain values.
Tracks when a measurement was taken and how its uncertainty grows over time.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


class ValidityMode:
    STATIC = "static"   # uncertainty does not grow
    DECAY = "decay"     # uncertainty doubles on halflife schedule
    EVENT = "event"     # valid until an invalidating event fires


@dataclass
class Validity:
    """
    When a value was measured and how it ages.

    STATIC:  snapshot — uncertainty stays constant.
    DECAY:   uncertainty doubles every `halflife`.
    EVENT:   valid until `invalidating_event` appears in an event log.
    """
    measured_at: datetime
    mode: str = ValidityMode.STATIC
    halflife: Optional[timedelta] = None
    invalidating_event: Optional[str] = None

    def uncertainty_multiplier(self, at_time: datetime) -> float:
        """Multiplier (>= 1.0) to apply to base uncertainty at `at_time`."""
        if self.mode == ValidityMode.DECAY and self.halflife:
            seconds = self.halflife.total_seconds()
            if seconds > 0:
                elapsed = (at_time - self.measured_at).total_seconds()
                return max(1.0, 2 ** (elapsed / seconds))
        return 1.0

    def is_valid(self, at_time: datetime, events: Optional[list[str]] = None) -> bool:
        """Check if this value is still valid at `at_time`."""
        if self.mode == ValidityMode.EVENT:
            if self.invalidating_event and events:
                return self.invalidating_event not in events
        return True

    def to_dict(self) -> dict:
        d = {
            "measured_at": self.measured_at.isoformat(),
            "mode": self.mode,
        }
        if self.halflife is not None:
            d["halflife_seconds"] = self.halflife.total_seconds()
        if self.invalidating_event is not None:
            d["invalidating_event"] = self.invalidating_event
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Validity':
        halflife = timedelta(seconds=d["halflife_seconds"]) if "halflife_seconds" in d else None
        return cls(
            measured_at=datetime.fromisoformat(d["measured_at"]),
            mode=d["mode"],
            halflife=halflife,
            invalidating_event=d.get("invalidating_event"),
        )

    @classmethod
    def static(cls, measured_at: Optional[datetime] = None) -> 'Validity':
        return cls(measured_at=measured_at or datetime.now())

    @classmethod
    def decaying(cls, halflife: timedelta, measured_at: Optional[datetime] = None) -> 'Validity':
        return cls(measured_at=measured_at or datetime.now(), mode=ValidityMode.DECAY, halflife=halflife)

    @classmethod
    def until_event(cls, event: str, measured_at: Optional[datetime] = None) -> 'Validity':
        return cls(measured_at=measured_at or datetime.now(), mode=ValidityMode.EVENT, invalidating_event=event)
