import pytest
from datetime import datetime, timedelta
from margin.forecast import forecast, Forecast
from margin.observation import Observation
from margin.health import Health, Thresholds
from margin.confidence import Confidence


def _obs(value, seconds_offset, name="x"):
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    return Observation(
        name, Health.DEGRADED, value, 100.0, Confidence.HIGH,
        measured_at=t0 + timedelta(seconds=seconds_offset),
    )


class TestForecast:
    def test_linear_decline(self):
        # Value dropping from 80 to 40 over 40 seconds
        obs = [_obs(80.0, 0), _obs(60.0, 20), _obs(40.0, 40)]
        t = Thresholds(intact=80.0, ablated=30.0)
        now = datetime(2026, 1, 1, 0, 0, 40)
        f = forecast(obs, t, now=now)
        assert f is not None
        assert f.worsening  # declining (higher_is_better, slope negative)
        assert f.time_to_ablated is not None
        assert f.time_to_ablated == pytest.approx(10.0, abs=1.0)

    def test_linear_recovery(self):
        # Value rising from 40 to 80 over 40 seconds
        obs = [_obs(40.0, 0), _obs(60.0, 20), _obs(80.0, 40)]
        t = Thresholds(intact=80.0, ablated=30.0)
        now = datetime(2026, 1, 1, 0, 0, 40)
        f = forecast(obs, t, now=now)
        assert f is not None
        assert f.improving

    def test_stable(self):
        obs = [_obs(50.0, 0), _obs(50.0, 10), _obs(50.0, 20)]
        t = Thresholds(intact=80.0, ablated=30.0)
        now = datetime(2026, 1, 1, 0, 0, 20)
        f = forecast(obs, t, now=now)
        assert f is not None
        assert f.stable
        assert f.time_to_intact is None  # not moving
        assert f.time_to_ablated is None

    def test_insufficient_data(self):
        obs = [_obs(50.0, 0)]
        t = Thresholds(intact=80.0, ablated=30.0)
        assert forecast(obs, t) is None

    def test_no_timestamps(self):
        obs = [
            Observation("x", Health.DEGRADED, 50.0, 100.0, Confidence.HIGH),
            Observation("x", Health.DEGRADED, 40.0, 100.0, Confidence.HIGH),
        ]
        t = Thresholds(intact=80.0, ablated=30.0)
        assert forecast(obs, t) is None

    def test_lower_is_better_improving(self):
        # Error rate dropping from 0.08 to 0.04 over 40 seconds
        t0 = datetime(2026, 1, 1)
        obs = [
            Observation("err", Health.DEGRADED, 0.08, 0.01, Confidence.HIGH,
                        higher_is_better=False, measured_at=t0),
            Observation("err", Health.DEGRADED, 0.06, 0.01, Confidence.HIGH,
                        higher_is_better=False, measured_at=t0 + timedelta(seconds=20)),
            Observation("err", Health.DEGRADED, 0.04, 0.01, Confidence.HIGH,
                        higher_is_better=False, measured_at=t0 + timedelta(seconds=40)),
        ]
        t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
        f = forecast(obs, t, now=t0 + timedelta(seconds=40))
        assert f is not None
        assert f.improving  # dropping error rate = improving

    def test_eta_timedelta(self):
        obs = [_obs(60.0, 0), _obs(40.0, 20)]
        t = Thresholds(intact=80.0, ablated=30.0)
        now = datetime(2026, 1, 1, 0, 0, 20)
        f = forecast(obs, t, now=now)
        assert f.eta_ablated is not None
        assert isinstance(f.eta_ablated, timedelta)

    def test_to_dict(self):
        obs = [_obs(60.0, 0), _obs(40.0, 20)]
        t = Thresholds(intact=80.0, ablated=30.0)
        now = datetime(2026, 1, 1, 0, 0, 20)
        f = forecast(obs, t, now=now)
        d = f.to_dict()
        assert "component" in d
        assert "trend_per_second" in d
        assert "improving" in d
        assert d["n_samples"] == 2

    def test_n_samples(self):
        obs = [_obs(80.0, 0), _obs(60.0, 10), _obs(40.0, 20), _obs(20.0, 30)]
        t = Thresholds(intact=80.0, ablated=30.0)
        f = forecast(obs, t, now=datetime(2026, 1, 1, 0, 0, 30))
        assert f.n_samples == 4
