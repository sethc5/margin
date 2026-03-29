"""
UncertainValue: a scalar measurement with uncertainty, source, validity, and provenance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .validity import Validity
from .provenance import new_id


class Source:
    """Where this value came from."""
    MEASURED = "measured"       # from a real sensor / measurement pipeline
    MODELED = "modeled"        # derived from a formal model
    ASSERTED = "asserted"      # declared by the caller
    PROPAGATED = "propagated"  # derived by the algebra


@dataclass
class UncertainValue:
    """
    A scalar value with associated uncertainty and epistemic metadata.

    point:       central estimate
    uncertainty: magnitude (absolute or relative, per `relative` flag)
    relative:    if True, uncertainty is a fraction of |point|
    source:      how this value was produced (Source constant)
    validity:    temporal validity descriptor
    provenance:  list of ancestor IDs for correlation detection
    """
    point: float
    uncertainty: float
    relative: bool = False
    source: str = Source.MEASURED
    validity: Validity = None
    provenance: list[str] = None

    def __post_init__(self):
        if self.validity is None:
            self.validity = Validity.static()
        if self.provenance is None:
            self.provenance = [new_id()]

    def absolute_uncertainty(self, at_time: Optional[datetime] = None) -> float:
        """Effective absolute uncertainty, accounting for temporal decay."""
        at_time = at_time or datetime.now()
        base = self.uncertainty * abs(self.point) if self.relative else self.uncertainty
        return base * self.validity.uncertainty_multiplier(at_time)

    def to_absolute(self) -> 'UncertainValue':
        """Return a copy with absolute uncertainty."""
        if not self.relative:
            return self
        return UncertainValue(
            point=self.point,
            uncertainty=self.uncertainty * abs(self.point),
            relative=False,
            source=self.source,
            validity=self.validity,
            provenance=self.provenance,
        )

    def to_relative(self) -> 'UncertainValue':
        """Return a copy with relative uncertainty."""
        if self.relative:
            return self
        if self.point == 0:
            raise ValueError("Cannot convert to relative: zero point estimate")
        return UncertainValue(
            point=self.point,
            uncertainty=self.uncertainty / abs(self.point),
            relative=True,
            source=self.source,
            validity=self.validity,
            provenance=self.provenance,
        )

    def interval(self, at_time: Optional[datetime] = None, n_sigma: float = 1.0) -> tuple[float, float]:
        """Confidence interval at `at_time`."""
        u = self.absolute_uncertainty(at_time)
        return (self.point - n_sigma * u, self.point + n_sigma * u)

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "uncertainty": self.uncertainty,
            "relative": self.relative,
            "source": self.source,
            "validity": self.validity.to_dict(),
            "provenance": list(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'UncertainValue':
        return cls(
            point=d["point"],
            uncertainty=d["uncertainty"],
            relative=d.get("relative", False),
            source=d.get("source", Source.MEASURED),
            validity=Validity.from_dict(d["validity"]) if "validity" in d else None,
            provenance=d.get("provenance"),
        )

    def __repr__(self) -> str:
        mode = "±" if not self.relative else "±rel "
        return f"UncertainValue({self.point:.4g} {mode}{self.uncertainty:.4g})"
