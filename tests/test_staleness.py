import pytest
from datetime import datetime, timedelta
from margin.observation import Observation
from margin.health import Health
from margin.confidence import Confidence


def _obs(measured_at=None):
    return Observation(
        "x", Health.INTACT, 90.0, 100.0, Confidence.HIGH,
        measured_at=measured_at,
    )


class TestStaleness:
    def test_no_timestamp_always_fresh(self):
        o = _obs()
        assert o.is_fresh() is True
        assert o.age() is None

    def test_recent_is_fresh(self):
        o = _obs(measured_at=datetime.now())
        assert o.is_fresh(max_age_seconds=60.0) is True
        assert o.age() < 1.0

    def test_old_is_stale(self):
        o = _obs(measured_at=datetime.now() - timedelta(minutes=5))
        assert o.is_fresh(max_age_seconds=60.0) is False
        assert o.age() > 290

    def test_custom_max_age(self):
        o = _obs(measured_at=datetime.now() - timedelta(seconds=30))
        assert o.is_fresh(max_age_seconds=60.0) is True
        assert o.is_fresh(max_age_seconds=10.0) is False

    def test_explicit_now(self):
        t = datetime(2026, 1, 1, 12, 0, 0)
        o = _obs(measured_at=datetime(2026, 1, 1, 11, 59, 0))
        assert o.age(now=t) == pytest.approx(60.0)
        assert o.is_fresh(max_age_seconds=120.0, now=t) is True
        assert o.is_fresh(max_age_seconds=30.0, now=t) is False

    def test_measured_at_roundtrips(self):
        t = datetime(2026, 3, 28, 14, 30, 0)
        o = _obs(measured_at=t)
        d = o.to_dict()
        assert "measured_at" in d
        o2 = Observation.from_dict(d)
        assert o2.measured_at == t

    def test_no_measured_at_omitted_from_dict(self):
        o = _obs()
        d = o.to_dict()
        assert "measured_at" not in d
        o2 = Observation.from_dict(d)
        assert o2.measured_at is None
