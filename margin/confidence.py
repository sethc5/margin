"""
Confidence tiers for uncertain comparisons.
Replaces binary booleans with graded confidence levels.
"""

from enum import Enum


class Confidence(Enum):
    """
    How much an uncertainty interval overlaps a decision boundary.

    CERTAIN:       interval fully clear of boundary
    HIGH:          < 10% overlap
    MODERATE:      10-40% overlap
    LOW:           > 40% overlap
    INDETERMINATE: boundary inside interval — call cannot be made
    """
    CERTAIN = "certain"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INDETERMINATE = "indeterminate"

    def __ge__(self, other):
        if not isinstance(other, Confidence):
            return NotImplemented
        return _RANK[self] >= _RANK[other]

    def __gt__(self, other):
        if not isinstance(other, Confidence):
            return NotImplemented
        return _RANK[self] > _RANK[other]

    def __le__(self, other):
        if not isinstance(other, Confidence):
            return NotImplemented
        return _RANK[self] <= _RANK[other]

    def __lt__(self, other):
        if not isinstance(other, Confidence):
            return NotImplemented
        return _RANK[self] < _RANK[other]


_RANK = {
    Confidence.CERTAIN: 4,
    Confidence.HIGH: 3,
    Confidence.MODERATE: 2,
    Confidence.LOW: 1,
    Confidence.INDETERMINATE: 0,
}
