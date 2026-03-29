"""
Uncertainty propagation through arithmetic operations.

Correlated values (shared provenance) combine linearly (conservative).
Independent values combine in quadrature.
"""

import math
from datetime import datetime
from typing import Optional

from .uncertain import UncertainValue, Source
from .validity import Validity
from .confidence import Confidence
from .provenance import new_id, are_correlated


def _propagated_validity(inputs: list[UncertainValue]) -> Validity:
    """Conservative validity: latest measurement, shortest halflife."""
    if not inputs:
        return Validity.static()

    latest = max(inputs, key=lambda v: v.validity.measured_at)
    halflives = [v.validity.halflife for v in inputs if v.validity.halflife]
    shortest = min(halflives) if halflives else None

    if shortest:
        return Validity.decaying(shortest, latest.validity.measured_at)
    return Validity.static(latest.validity.measured_at)


def add(a: UncertainValue, b: UncertainValue) -> UncertainValue:
    """Add two uncertain values with correct uncertainty propagation."""
    aa, bb = a.to_absolute(), b.to_absolute()
    if are_correlated(a.provenance, b.provenance):
        unc = aa.uncertainty + bb.uncertainty
    else:
        unc = math.sqrt(aa.uncertainty**2 + bb.uncertainty**2)
    return UncertainValue(
        point=aa.point + bb.point,
        uncertainty=unc,
        source=Source.PROPAGATED,
        validity=_propagated_validity([a, b]),
        provenance=list(set(a.provenance + b.provenance + [new_id()])),
    )


def subtract(a: UncertainValue, b: UncertainValue) -> UncertainValue:
    """Subtract two uncertain values."""
    aa, bb = a.to_absolute(), b.to_absolute()
    if are_correlated(a.provenance, b.provenance):
        unc = aa.uncertainty + bb.uncertainty
    else:
        unc = math.sqrt(aa.uncertainty**2 + bb.uncertainty**2)
    return UncertainValue(
        point=aa.point - bb.point,
        uncertainty=unc,
        source=Source.PROPAGATED,
        validity=_propagated_validity([a, b]),
        provenance=list(set(a.provenance + b.provenance + [new_id()])),
    )


def multiply(a: UncertainValue, b: UncertainValue) -> UncertainValue:
    """Multiply two uncertain values (relative uncertainties combine).

    When either operand is zero, relative uncertainty is undefined so we
    fall back to absolute propagation: |b|*σ_a + |a|*σ_b (linear, safe).
    """
    product = a.point * b.point
    prov = list(set(a.provenance + b.provenance + [new_id()]))

    if a.point == 0 or b.point == 0:
        aa, bb = a.to_absolute(), b.to_absolute()
        unc = abs(b.point) * aa.uncertainty + abs(a.point) * bb.uncertainty
        return UncertainValue(
            point=product, uncertainty=unc,
            source=Source.PROPAGATED,
            validity=_propagated_validity([a, b]),
            provenance=prov,
        )

    ar, br = a.to_relative(), b.to_relative()
    if are_correlated(a.provenance, b.provenance):
        unc = ar.uncertainty + br.uncertainty
    else:
        unc = math.sqrt(ar.uncertainty**2 + br.uncertainty**2)
    return UncertainValue(
        point=product, uncertainty=unc, relative=True,
        source=Source.PROPAGATED,
        validity=_propagated_validity([a, b]),
        provenance=prov,
    )


def divide(a: UncertainValue, b: UncertainValue) -> UncertainValue:
    """Divide two uncertain values."""
    if b.point == 0:
        raise ValueError("Division by zero")
    ar, br = a.to_relative(), b.to_relative()
    if are_correlated(a.provenance, b.provenance):
        unc = ar.uncertainty + br.uncertainty
    else:
        unc = math.sqrt(ar.uncertainty**2 + br.uncertainty**2)
    return UncertainValue(
        point=ar.point / br.point,
        uncertainty=unc,
        relative=True,
        source=Source.PROPAGATED,
        validity=_propagated_validity([a, b]),
        provenance=list(set(a.provenance + b.provenance + [new_id()])),
    )


def scale(value: UncertainValue, factor: float) -> UncertainValue:
    """Scale by an exact constant. Preserves provenance without growth."""
    return UncertainValue(
        point=value.point * factor,
        uncertainty=value.uncertainty * abs(factor),
        relative=value.relative,
        source=value.source,
        validity=value.validity,
        provenance=list(value.provenance),
    )


def compare(value: UncertainValue, threshold: float, at_time: Optional[datetime] = None) -> Confidence:
    """
    Compare an uncertain value to a threshold. Returns a Confidence tier
    based on how much the uncertainty interval overlaps the threshold.
    """
    at_time = at_time or datetime.now()
    u = value.absolute_uncertainty(at_time)
    lower = value.point - u
    upper = value.point + u
    width = 2 * u

    if lower < threshold < upper:
        return Confidence.INDETERMINATE

    gap = (lower - threshold) if threshold <= lower else (threshold - upper)
    if width <= 0:
        return Confidence.CERTAIN

    ratio = gap / width
    if ratio >= 0.5:
        return Confidence.CERTAIN
    elif ratio >= 0.1:
        return Confidence.HIGH
    elif ratio >= 0.05:
        return Confidence.MODERATE
    else:
        return Confidence.LOW


def weighted_average(values: list[UncertainValue], weights: Optional[list[float]] = None) -> UncertainValue:
    """
    Weighted average. Defaults to inverse-variance weighting.
    """
    if not values:
        raise ValueError("Empty list")
    if len(values) == 1:
        return values[0]

    if weights is None:
        variances = [v.to_absolute().uncertainty**2 for v in values]
        total = sum(1/v for v in variances if v > 0)
        if total == 0:
            weights = [1.0 / len(values)] * len(values)
        else:
            weights = [(1/v) / total for v in variances]

    result = None
    for v, w in zip(values, weights):
        s = scale(v, w)
        result = add(result, s) if result else s
    return result
