import pytest
import math
from margin.algebra import add, subtract, multiply, divide, scale, compare, weighted_average
from margin.uncertain import UncertainValue, Source
from margin.confidence import Confidence
from margin.provenance import new_id


def _uv(point, unc, prov=None):
    """Shorthand for tests."""
    return UncertainValue(point=point, uncertainty=unc, provenance=prov or [new_id()])


class TestAdd:
    def test_point_values_sum(self):
        r = add(_uv(3.0, 0.1), _uv(5.0, 0.2))
        assert abs(r.point - 8.0) < 0.001

    def test_independent_quadrature(self):
        r = add(_uv(3.0, 0.3), _uv(5.0, 0.4))
        assert abs(r.uncertainty - math.sqrt(0.09 + 0.16)) < 0.001

    def test_correlated_linear(self):
        shared = [new_id()]
        a = UncertainValue(point=3.0, uncertainty=0.3, provenance=shared)
        b = UncertainValue(point=5.0, uncertainty=0.4, provenance=shared)
        r = add(a, b)
        assert abs(r.uncertainty - 0.7) < 0.001

    def test_source_is_propagated(self):
        r = add(_uv(1.0, 0.1), _uv(2.0, 0.1))
        assert r.source == Source.PROPAGATED


class TestSubtract:
    def test_point_values_differ(self):
        r = subtract(_uv(10.0, 0.1), _uv(3.0, 0.1))
        assert abs(r.point - 7.0) < 0.001

    def test_independent_quadrature(self):
        r = subtract(_uv(10.0, 0.3), _uv(3.0, 0.4))
        assert abs(r.uncertainty - math.sqrt(0.09 + 0.16)) < 0.001


class TestMultiply:
    def test_point_values_product(self):
        r = multiply(_uv(3.0, 0.1), _uv(5.0, 0.2))
        assert abs(r.point - 15.0) < 0.001

    def test_result_is_relative(self):
        r = multiply(_uv(3.0, 0.1), _uv(5.0, 0.2))
        assert r.relative is True

    def test_zero_operand_does_not_crash(self):
        r = multiply(_uv(0.0, 0.1), _uv(5.0, 0.2))
        assert r.point == 0.0
        assert r.uncertainty >= 0.0
        assert r.relative is False

    def test_zero_times_zero(self):
        r = multiply(_uv(0.0, 0.1), _uv(0.0, 0.2))
        assert r.point == 0.0
        assert r.uncertainty == 0.0

    def test_zero_preserves_other_uncertainty(self):
        r = multiply(_uv(0.0, 0.1), _uv(5.0, 0.2))
        # |b|*σ_a + |a|*σ_b = 5*0.1 + 0*0.2 = 0.5
        assert abs(r.uncertainty - 0.5) < 0.001


class TestDivide:
    def test_point_values_quotient(self):
        r = divide(_uv(10.0, 0.1), _uv(2.0, 0.1))
        assert abs(r.point - 5.0) < 0.001

    def test_zero_denominator_raises(self):
        with pytest.raises(ValueError, match="zero"):
            divide(_uv(10.0, 0.1), _uv(0.0, 0.1))


class TestScale:
    def test_point_scaled(self):
        r = scale(_uv(5.0, 0.3), 2.0)
        assert abs(r.point - 10.0) < 0.001

    def test_uncertainty_scaled(self):
        r = scale(_uv(5.0, 0.3), 2.0)
        assert abs(r.uncertainty - 0.6) < 0.001

    def test_negative_factor(self):
        r = scale(_uv(5.0, 0.3), -2.0)
        assert abs(r.point - (-10.0)) < 0.001
        assert abs(r.uncertainty - 0.6) < 0.001

    def test_provenance_not_growing(self):
        v = _uv(5.0, 0.3)
        orig_len = len(v.provenance)
        for _ in range(50):
            v = scale(v, 1.0)
        assert len(v.provenance) == orig_len


class TestCompare:
    def test_certain_when_far(self):
        v = _uv(10.0, 0.1)
        assert compare(v, 5.0) == Confidence.CERTAIN

    def test_indeterminate_when_straddling(self):
        v = _uv(5.0, 1.0)
        assert compare(v, 5.0) == Confidence.INDETERMINATE

    def test_zero_uncertainty_at_threshold(self):
        v = _uv(5.0, 0.0)
        assert compare(v, 5.0) == Confidence.CERTAIN


class TestWeightedAverage:
    def test_equal_weights(self):
        r = weighted_average([_uv(4.0, 0.1), _uv(6.0, 0.1)])
        assert abs(r.point - 5.0) < 0.001

    def test_inverse_variance_weighting(self):
        # Tighter value should dominate
        tight = _uv(10.0, 0.1)
        loose = _uv(20.0, 10.0)
        r = weighted_average([tight, loose])
        assert abs(r.point - 10.0) < 0.5  # should be close to 10

    def test_single_value_returned(self):
        v = _uv(5.0, 0.3)
        assert weighted_average([v]) is v

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            weighted_average([])

    def test_zero_uncertainty_equal_weights(self):
        r = weighted_average([_uv(4.0, 0.0), _uv(6.0, 0.0)])
        assert abs(r.point - 5.0) < 0.001
