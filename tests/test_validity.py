import pytest
from datetime import datetime, timedelta
from margin.validity import Validity, ValidityMode


class TestValidityModes:
    def test_static_multiplier_is_one(self):
        v = Validity.static()
        assert v.uncertainty_multiplier(datetime.now()) == 1.0

    def test_decay_doubles_after_one_halflife(self):
        now = datetime.now()
        v = Validity.decaying(timedelta(hours=1), measured_at=now)
        later = now + timedelta(hours=1)
        assert abs(v.uncertainty_multiplier(later) - 2.0) < 0.01

    def test_decay_quadruples_after_two_halflives(self):
        now = datetime.now()
        v = Validity.decaying(timedelta(hours=1), measured_at=now)
        later = now + timedelta(hours=2)
        assert abs(v.uncertainty_multiplier(later) - 4.0) < 0.01

    def test_future_measurement_clamped_to_one(self):
        future = datetime(2030, 1, 1)
        v = Validity.decaying(timedelta(hours=1), measured_at=future)
        assert v.uncertainty_multiplier(datetime.now()) == 1.0

    def test_event_valid_without_events(self):
        v = Validity.until_event("reboot")
        assert v.is_valid(datetime.now()) is True
        assert v.is_valid(datetime.now(), events=[]) is True

    def test_event_invalid_after_event(self):
        v = Validity.until_event("reboot")
        assert v.is_valid(datetime.now(), events=["reboot"]) is False

    def test_event_valid_if_different_event(self):
        v = Validity.until_event("reboot")
        assert v.is_valid(datetime.now(), events=["deploy"]) is True

    def test_static_always_valid(self):
        v = Validity.static()
        assert v.is_valid(datetime.now(), events=["anything"]) is True


class TestValidityRoundtrip:
    def test_static_roundtrip(self):
        v = Validity.static()
        vr = Validity.from_dict(v.to_dict())
        assert vr.mode == ValidityMode.STATIC
        assert vr.halflife is None

    def test_decay_roundtrip(self):
        v = Validity.decaying(timedelta(minutes=30))
        vr = Validity.from_dict(v.to_dict())
        assert vr.mode == ValidityMode.DECAY
        assert abs(vr.halflife.total_seconds() - 1800) < 0.01

    def test_event_roundtrip(self):
        v = Validity.until_event("prompt_change")
        vr = Validity.from_dict(v.to_dict())
        assert vr.mode == ValidityMode.EVENT
        assert vr.invalidating_event == "prompt_change"
