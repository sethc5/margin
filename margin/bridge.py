"""
Bridge between the algebra layer (UncertainValue) and the health layer
(Observation, Expression).

This connects the two halves of the library: uncertain measurements flow
into typed health classifications, and the uncertainty informs the
confidence tier automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .uncertain import UncertainValue, Source
from .validity import Validity
from .confidence import Confidence
from .health import Health, Thresholds, classify
from .observation import Observation, Correction, Expression, Op, Parser
from .algebra import compare, subtract


def observe(
    name: str,
    value: UncertainValue,
    baseline: UncertainValue,
    thresholds: Thresholds,
    correction_magnitude: float = 0.0,
    at_time: Optional[datetime] = None,
) -> Observation:
    """
    Create a typed Observation from an UncertainValue.

    The confidence tier is derived automatically by comparing the value
    against the thresholds using the algebra's `compare()` — the
    uncertainty interval determines whether the call is CERTAIN, HIGH,
    MODERATE, LOW, or INDETERMINATE.

    Args:
        name:        component identifier
        value:       the uncertain measurement
        baseline:    expected healthy value (as UncertainValue)
        thresholds:  classification boundaries
        correction_magnitude: size of active correction (for RECOVERING)
        at_time:     evaluation time (defaults to now)
    """
    at_time = at_time or datetime.now()

    # Derive confidence from how the uncertainty interval relates to the
    # intact threshold — the critical decision boundary
    confidence = compare(value, thresholds.intact, at_time)

    correcting = correction_magnitude >= thresholds.active_min
    health = classify(value.point, confidence, correcting, thresholds)

    return Observation(
        name=name,
        health=health,
        value=value.point,
        baseline=baseline.point,
        confidence=confidence,
        higher_is_better=thresholds.higher_is_better,
        provenance=list(value.provenance),
        measured_at=at_time,
    )


def observe_many(
    values: dict[str, UncertainValue],
    baselines: dict[str, UncertainValue],
    thresholds: Thresholds,
    component_thresholds: Optional[dict[str, Thresholds]] = None,
    correction_magnitude: float = 0.0,
    alpha: float = 0.0,
    label: str = "",
    step: Optional[int] = None,
    at_time: Optional[datetime] = None,
) -> Expression:
    """
    Create a typed Expression from a dict of UncertainValues.

    Like Parser.parse() but takes UncertainValues instead of raw floats,
    so confidence is derived from the uncertainty rather than defaulting
    to MODERATE.

    Args:
        values:      {name: UncertainValue} measurements
        baselines:   {name: UncertainValue} expected healthy values
        thresholds:  default classification boundaries
        component_thresholds: per-component overrides
        correction_magnitude: size of active correction
        alpha:       correction intensity coefficient
        label:       optional label for the expression
        step:        optional sequence index
        at_time:     evaluation time (defaults to now)
    """
    at_time = at_time or datetime.now()
    component_thresholds = component_thresholds or {}

    observations = []
    for name, uv in values.items():
        bl = baselines.get(name, uv)
        ct = component_thresholds.get(name, thresholds)
        obs = observe(name, uv, bl, ct, correction_magnitude, at_time)
        observations.append(obs)

    # Build a temporary Parser to reuse _classify_op logic
    corrections = []
    if observations:
        raw_values = {name: uv.point for name, uv in values.items()}
        raw_baselines = {name: baselines.get(name, uv).point for name, uv in values.items()}
        temp_parser = Parser(
            baselines=raw_baselines,
            thresholds=thresholds,
            component_thresholds=component_thresholds,
        )
        target, op = temp_parser._classify_op(raw_values, correction_magnitude)

        if op != Op.NOOP and target:
            degraded_names = [o.name for o in observations
                              if o.health in (Health.DEGRADED, Health.ABLATED, Health.RECOVERING)]
            prov = []
            for o in observations:
                if o.name == target:
                    prov = list(o.provenance)
                    break
            corrections.append(Correction(
                target=target, op=op,
                alpha=alpha,
                magnitude=correction_magnitude,
                triggered_by=degraded_names,
                provenance=prov,
            ))

    net_conf = min(
        (o.confidence for o in observations),
        default=Confidence.INDETERMINATE,
    )

    return Expression(
        observations=observations,
        corrections=corrections,
        confidence=net_conf,
        label=label,
        step=step,
    )


# Confidence → approximate absolute uncertainty mapping.
# These are fractions of the value used as uncertainty estimates
# when reconstructing an UncertainValue from an Observation.
_CONFIDENCE_TO_UNCERTAINTY = {
    Confidence.CERTAIN: 0.01,
    Confidence.HIGH: 0.05,
    Confidence.MODERATE: 0.10,
    Confidence.LOW: 0.25,
    Confidence.INDETERMINATE: 0.50,
}


def to_uncertain(obs: Observation) -> UncertainValue:
    """
    Reconstruct an UncertainValue from an Observation.

    Since Observations don't carry explicit uncertainty, it is inferred
    from the confidence tier: CERTAIN → 1% of |value|, HIGH → 5%,
    MODERATE → 10%, LOW → 25%, INDETERMINATE → 50%.

    This closes the loop: observe() goes algebra→health, to_uncertain()
    goes health→algebra.
    """
    frac = _CONFIDENCE_TO_UNCERTAINTY.get(obs.confidence, 0.10)
    unc = frac * abs(obs.value) if obs.value != 0 else frac

    validity = Validity.static(obs.measured_at) if obs.measured_at else Validity.static()

    return UncertainValue(
        point=obs.value,
        uncertainty=unc,
        source=Source.MODELED,
        validity=validity,
        provenance=list(obs.provenance) if obs.provenance else None,
    )


def delta(
    name: str,
    before: UncertainValue,
    after: UncertainValue,
    baseline: UncertainValue,
    thresholds: Thresholds,
    at_time: Optional[datetime] = None,
) -> tuple[Observation, Observation, UncertainValue]:
    """
    Compute typed before/after observations and the uncertain difference.

    Returns (obs_before, obs_after, diff) where diff is an UncertainValue
    representing the change with correctly propagated uncertainty.
    Useful for building Records with full algebra backing.
    """
    at_time = at_time or datetime.now()
    obs_before = observe(name, before, baseline, thresholds, at_time=at_time)
    obs_after = observe(name, after, baseline, thresholds, at_time=at_time)
    diff = subtract(after, before)
    return obs_before, obs_after, diff
