import pytest
from datetime import datetime
from margin.events import EventBus
from margin.validity import Validity, ValidityMode
from margin.uncertain import UncertainValue


class TestEventBus:
    def test_fire_and_check(self):
        bus = EventBus()
        assert not bus.has_fired("reboot")
        bus.fire("reboot")
        assert bus.has_fired("reboot")

    def test_fired_at(self):
        bus = EventBus()
        t = datetime(2026, 1, 1, 12, 0)
        bus.fire("deploy", at_time=t)
        assert bus.fired_at("deploy") == t
        assert bus.fired_at("other") is None

    def test_is_valid_event_mode(self):
        bus = EventBus()
        v = Validity.until_event("config_reload")
        assert bus.is_valid(v) is True
        bus.fire("config_reload")
        assert bus.is_valid(v) is False

    def test_is_valid_static_always_true(self):
        bus = EventBus()
        bus.fire("anything")
        v = Validity.static()
        assert bus.is_valid(v) is True

    def test_is_valid_different_event(self):
        bus = EventBus()
        bus.fire("deploy")
        v = Validity.until_event("reboot")
        assert bus.is_valid(v) is True

    def test_is_value_valid(self):
        bus = EventBus()
        uv = UncertainValue(
            point=5.0, uncertainty=0.1,
            validity=Validity.until_event("prompt_change"),
        )
        assert bus.is_value_valid(uv) is True
        bus.fire("prompt_change")
        assert bus.is_value_valid(uv) is False

    def test_listener_called(self):
        bus = EventBus()
        calls = []
        bus.on("deploy", lambda e, t: calls.append(e))
        bus.fire("deploy")
        assert calls == ["deploy"]

    def test_wildcard_listener(self):
        bus = EventBus()
        calls = []
        bus.on("*", lambda e, t: calls.append(e))
        bus.fire("a")
        bus.fire("b")
        assert calls == ["a", "b"]

    def test_reset_single(self):
        bus = EventBus()
        bus.fire("a")
        bus.fire("b")
        bus.reset("a")
        assert not bus.has_fired("a")
        assert bus.has_fired("b")

    def test_reset_all(self):
        bus = EventBus()
        bus.fire("a")
        bus.fire("b")
        bus.reset()
        assert bus.fired_events == []

    def test_fired_events(self):
        bus = EventBus()
        bus.fire("a")
        bus.fire("b")
        bus.fire("c")
        assert bus.fired_events == ["a", "b", "c"]

    def test_roundtrip(self):
        bus = EventBus()
        t = datetime(2026, 3, 28, 12, 0)
        bus.fire("deploy", at_time=t)
        bus.fire("restart")

        bus2 = EventBus.from_dict(bus.to_dict())
        assert bus2.has_fired("deploy")
        assert bus2.has_fired("restart")
        assert bus2.fired_at("deploy") == t
