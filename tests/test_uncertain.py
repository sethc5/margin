import pytest
import math
from margin.uncertain import UncertainValue, Source
from margin.validity import Validity


class TestUncertainValue:
    def test_absolute_uncertainty_default(self):
        v = UncertainValue(point=10.0, uncertainty=0.5)
        assert abs(v.absolute_uncertainty() - 0.5) < 0.01

    def test_relative_uncertainty_computed(self):
        v = UncertainValue(point=10.0, uncertainty=0.1, relative=True)
        assert abs(v.absolute_uncertainty() - 1.0) < 0.01

    def test_interval(self):
        v = UncertainValue(point=5.0, uncertainty=1.0)
        lo, hi = v.interval()
        assert lo == 4.0
        assert hi == 6.0

    def test_interval_2_sigma(self):
        v = UncertainValue(point=5.0, uncertainty=1.0)
        lo, hi = v.interval(n_sigma=2.0)
        assert lo == 3.0
        assert hi == 7.0

    def test_to_absolute_noop_if_already(self):
        v = UncertainValue(point=5.0, uncertainty=0.5)
        assert v.to_absolute() is v

    def test_to_relative_noop_if_already(self):
        v = UncertainValue(point=5.0, uncertainty=0.1, relative=True)
        assert v.to_relative() is v

    def test_to_relative_converts(self):
        v = UncertainValue(point=10.0, uncertainty=2.0)
        vr = v.to_relative()
        assert vr.relative is True
        assert abs(vr.uncertainty - 0.2) < 0.001

    def test_to_absolute_converts(self):
        v = UncertainValue(point=10.0, uncertainty=0.2, relative=True)
        va = v.to_absolute()
        assert va.relative is False
        assert abs(va.uncertainty - 2.0) < 0.001

    def test_to_relative_zero_raises(self):
        v = UncertainValue(point=0.0, uncertainty=0.5)
        with pytest.raises(ValueError, match="zero"):
            v.to_relative()

    def test_provenance_auto_generated(self):
        a = UncertainValue(point=1.0, uncertainty=0.1)
        b = UncertainValue(point=2.0, uncertainty=0.1)
        assert len(a.provenance) == 1
        assert len(b.provenance) == 1
        assert a.provenance != b.provenance

    def test_validity_defaults_to_static(self):
        v = UncertainValue(point=1.0, uncertainty=0.1)
        assert v.validity.mode == "static"


class TestUncertainValueRoundtrip:
    def test_basic_roundtrip(self):
        v = UncertainValue(point=5.0, uncertainty=0.3, source=Source.MEASURED)
        vr = UncertainValue.from_dict(v.to_dict())
        assert vr.point == 5.0
        assert vr.uncertainty == 0.3
        assert vr.source == Source.MEASURED
        assert vr.relative is False

    def test_relative_roundtrip(self):
        v = UncertainValue(point=10.0, uncertainty=0.05, relative=True, source=Source.MODELED)
        vr = UncertainValue.from_dict(v.to_dict())
        assert vr.relative is True
        assert vr.source == Source.MODELED

    def test_provenance_preserved(self):
        v = UncertainValue(point=1.0, uncertainty=0.1)
        vr = UncertainValue.from_dict(v.to_dict())
        assert vr.provenance == v.provenance
