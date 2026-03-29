import pytest
from datetime import datetime
from margin.bridge import to_uncertain
from margin.observation import Observation
from margin.health import Health
from margin.confidence import Confidence
from margin.uncertain import Source


class TestToUncertain:
    def test_basic_conversion(self):
        obs = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.HIGH)
        uv = to_uncertain(obs)
        assert uv.point == 100.0
        assert uv.uncertainty == pytest.approx(5.0)  # HIGH → 5% of 100
        assert uv.source == Source.MODELED

    def test_confidence_maps_to_uncertainty(self):
        # CERTAIN → 1%
        obs_c = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.CERTAIN)
        assert to_uncertain(obs_c).uncertainty == pytest.approx(1.0)

        # MODERATE → 10%
        obs_m = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.MODERATE)
        assert to_uncertain(obs_m).uncertainty == pytest.approx(10.0)

        # LOW → 25%
        obs_l = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.LOW)
        assert to_uncertain(obs_l).uncertainty == pytest.approx(25.0)

        # INDETERMINATE → 50%
        obs_i = Observation("x", Health.OOD, 100.0, 100.0, Confidence.INDETERMINATE)
        assert to_uncertain(obs_i).uncertainty == pytest.approx(50.0)

    def test_zero_value_uses_fraction_directly(self):
        obs = Observation("x", Health.DEGRADED, 0.0, 100.0, Confidence.HIGH)
        uv = to_uncertain(obs)
        assert uv.uncertainty == 0.05  # fraction itself, not fraction * 0

    def test_measured_at_preserved(self):
        t = datetime(2026, 3, 28, 14, 0)
        obs = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.HIGH, measured_at=t)
        uv = to_uncertain(obs)
        assert uv.validity.measured_at == t

    def test_provenance_preserved(self):
        obs = Observation("x", Health.INTACT, 100.0, 100.0, Confidence.HIGH, provenance=["abc"])
        uv = to_uncertain(obs)
        assert "abc" in uv.provenance

    def test_roundtrip_observe_to_uncertain(self):
        from margin.bridge import observe
        from margin.uncertain import UncertainValue
        from margin.health import Thresholds

        # Start with UncertainValue → observe → to_uncertain → check
        original = UncertainValue(point=90.0, uncertainty=2.0)
        baseline = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)

        obs = observe("x", original, baseline, t)
        recovered = to_uncertain(obs)

        assert recovered.point == original.point
        # Uncertainty won't be identical (it's inferred from confidence tier)
        # but the value should roundtrip
        assert recovered.point == 90.0
