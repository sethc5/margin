"""
Event bus for validity invalidation.

Standalone utility — not wired into Monitor or the evaluation loop.
Use directly in event-driven systems where values should become stale
when something changes (config reload, model update, deployment).

    bus = EventBus()
    bus.fire("config_reload")

    v = Validity.until_event("config_reload")
    bus.is_valid(v)  # False — invalidated

    bus.on("deploy", lambda evt, ts: print(f"deployed at {ts}"))

When an event fires, all Validity descriptors that reference it via
Validity.until_event() become stale. bus.is_valid() / bus.is_value_valid()
check this. EventBus does NOT integrate with Monitor health transitions;
for that use case, attach a listener to Monitor.update() manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from .validity import Validity, ValidityMode
from .uncertain import UncertainValue


class EventBus:
    """
    Tracks fired events and checks validity against them.

    Usage:
        bus = EventBus()
        bus.fire("prompt_change")
        bus.fire("config_reload")

        v = Validity.until_event("prompt_change")
        bus.is_valid(v)  # False — event already fired

        v2 = Validity.until_event("deploy")
        bus.is_valid(v2)  # True — that event hasn't fired
    """

    def __init__(self):
        self._events: dict[str, datetime] = {}
        self._listeners: dict[str, list[Callable[[str, datetime], None]]] = {}

    def fire(self, event: str, at_time: Optional[datetime] = None) -> None:
        """Record that an event has occurred."""
        at_time = at_time or datetime.now()
        self._events[event] = at_time
        for fn in self._listeners.get(event, []):
            fn(event, at_time)
        for fn in self._listeners.get("*", []):
            fn(event, at_time)

    def has_fired(self, event: str) -> bool:
        """True if the named event has ever fired."""
        return event in self._events

    def fired_at(self, event: str) -> Optional[datetime]:
        """When the event fired, or None."""
        return self._events.get(event)

    def is_valid(self, validity: Validity) -> bool:
        """Check if a Validity descriptor is still valid against fired events."""
        if validity.mode != ValidityMode.EVENT:
            return True
        if validity.invalidating_event is None:
            return True
        return not self.has_fired(validity.invalidating_event)

    def is_value_valid(self, value: UncertainValue) -> bool:
        """Check if an UncertainValue's validity still holds."""
        return self.is_valid(value.validity)

    def on(self, event: str, callback: Callable[[str, datetime], None]) -> None:
        """
        Register a listener. Called when `event` fires.
        Use event="*" to listen to all events.
        """
        self._listeners.setdefault(event, []).append(callback)

    def reset(self, event: Optional[str] = None) -> None:
        """Clear one event or all events."""
        if event:
            self._events.pop(event, None)
        else:
            self._events.clear()

    @property
    def fired_events(self) -> list[str]:
        """All events that have fired, in insertion order."""
        return list(self._events.keys())

    def to_dict(self) -> dict:
        return {
            "events": {k: v.isoformat() for k, v in self._events.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EventBus':
        bus = cls()
        for event, ts in d.get("events", {}).items():
            bus._events[event] = datetime.fromisoformat(ts)
        return bus
