import pytest
from margin.confidence import Confidence


class TestConfidenceOrdering:
    def test_certain_is_highest(self):
        assert Confidence.CERTAIN > Confidence.HIGH
        assert Confidence.CERTAIN > Confidence.INDETERMINATE

    def test_indeterminate_is_lowest(self):
        assert Confidence.INDETERMINATE < Confidence.LOW
        assert Confidence.INDETERMINATE < Confidence.CERTAIN

    def test_full_ordering(self):
        assert Confidence.CERTAIN > Confidence.HIGH > Confidence.MODERATE > Confidence.LOW > Confidence.INDETERMINATE

    def test_equal_is_not_greater(self):
        assert not (Confidence.HIGH > Confidence.HIGH)
        assert Confidence.HIGH >= Confidence.HIGH
        assert Confidence.HIGH <= Confidence.HIGH

    def test_min_gives_weakest(self):
        assert min([Confidence.HIGH, Confidence.LOW, Confidence.CERTAIN]) == Confidence.LOW

    def test_max_gives_strongest(self):
        assert max([Confidence.HIGH, Confidence.LOW, Confidence.CERTAIN]) == Confidence.CERTAIN

    def test_not_implemented_for_other_types(self):
        assert Confidence.HIGH.__ge__(42) is NotImplemented
        assert Confidence.HIGH.__lt__("high") is NotImplemented
