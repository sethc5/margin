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
    COMPUTED = "modeled"       # alias for margin-poc compat (EpistemicSource.COMPUTED)
    INFERRED = "modeled"       # alias

# margin-poc compat aliases
EpistemicSource = Source
UncertaintyMode = type('UncertaintyMode', (), {'ABSOLUTE': 'absolute', 'RELATIVE': 'relative'})()


class UncertainValue:
    """
    A scalar value with associated uncertainty and epistemic metadata.

    point:       central estimate (alias: point_estimate)
    uncertainty: magnitude (absolute or relative, per `relative` flag)
    relative:    if True, uncertainty is a fraction of |point| (alias: mode)
    source:      how this value was produced (Source constant)
    validity:    temporal validity descriptor
    provenance:  list of ancestor IDs for correlation detection (alias: provenance_ids)
    """

    def __init__(
        self,
        point: float = None,
        uncertainty: float = 0.0,
        relative: bool = False,
        source: str = Source.MEASURED,
        validity: Validity = None,
        provenance: list = None,
        # margin-poc compat aliases
        point_estimate: float = None,
        mode: str = None,
        provenance_ids: list = None,
    ):
        self.point = point if point is not None else (point_estimate if point_estimate is not None else 0.0)
        self.uncertainty = uncertainty
        self.relative = relative if mode is None else (mode == "relative")
        self.source = source
        self.validity = validity if validity is not None else Validity.static()
        self.provenance = provenance or provenance_ids or [new_id()]

    @property
    def point_estimate(self) -> float:
        """margin-poc compat alias for point."""
        return self.point

    @property
    def provenance_ids(self) -> list:
        """margin-poc compat alias for provenance."""
        return self.provenance

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
