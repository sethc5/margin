"""
NumPy array health as margin observations.

Given an array (or batch of arrays), computes statistical health metrics
and classifies them. Optionally compares against a reference array
to detect drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class ArrayMetric:
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


ARRAY_METRICS: dict[str, ArrayMetric] = {
    "nan_rate": ArrayMetric(
        name="nan_rate", display_name="NaN Rate",
        thresholds=Thresholds(intact=0.0, ablated=0.05, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "inf_rate": ArrayMetric(
        name="inf_rate", display_name="Inf Rate",
        thresholds=Thresholds(intact=0.0, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "zero_rate": ArrayMetric(
        name="zero_rate", display_name="Zero Rate",
        thresholds=Thresholds(intact=0.50, ablated=0.95, higher_is_better=False),
        baseline=0.1, unit="ratio",
    ),
    "mean_drift": ArrayMetric(
        name="mean_drift", display_name="Mean Drift from Reference",
        thresholds=Thresholds(intact=0.10, ablated=0.50, higher_is_better=False),
        baseline=0.0, unit="relative",
    ),
    "std_ratio": ArrayMetric(
        name="std_ratio", display_name="Std Dev Ratio to Reference",
        thresholds=Thresholds(intact=0.50, ablated=0.10, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "range_violation_rate": ArrayMetric(
        name="range_violation_rate", display_name="Out-of-Range Rate",
        thresholds=Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "kurtosis_excess": ArrayMetric(
        name="kurtosis_excess", display_name="Excess Kurtosis",
        thresholds=Thresholds(intact=5.0, ablated=20.0, higher_is_better=False),
        baseline=0.0, unit="kurtosis",
    ),
    "skewness_abs": ArrayMetric(
        name="skewness_abs", display_name="Absolute Skewness",
        thresholds=Thresholds(intact=1.0, ablated=3.0, higher_is_better=False),
        baseline=0.0, unit="skewness",
    ),
}


@dataclass
class ArrayProfile:
    """Reference statistics for an array, used for drift detection."""
    mean: float
    std: float
    min_val: float
    max_val: float
    shape: tuple
    n_elements: int

    @classmethod
    def from_array(cls, arr) -> ArrayProfile:
        import numpy as np
        flat = np.asarray(arr).ravel()
        finite = flat[np.isfinite(flat)]
        if len(finite) == 0:
            return cls(mean=0.0, std=0.0, min_val=0.0, max_val=0.0,
                       shape=arr.shape, n_elements=flat.size)
        return cls(
            mean=float(np.mean(finite)),
            std=float(np.std(finite)),
            min_val=float(np.min(finite)),
            max_val=float(np.max(finite)),
            shape=arr.shape,
            n_elements=flat.size,
        )


def array_health(
    arr,
    reference: Optional[ArrayProfile] = None,
    expected_range: Optional[tuple[float, float]] = None,
    label: str = "",
    confidence: Confidence = Confidence.MODERATE,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """
    Compute health metrics for a numpy array.

    Args:
        arr:            numpy array (any shape — will be flattened)
        reference:      ArrayProfile from a known-good array (for drift detection)
        expected_range: (min, max) tuple — values outside are range violations
        label:          expression label
        confidence:     measurement confidence
        measured_at:    timestamp

    Returns:
        Expression with one Observation per computed metric.
    """
    import numpy as np
    from margin.health import classify

    arr = np.asarray(arr)
    flat = arr.ravel()
    n = flat.size
    if n == 0:
        return Expression(label=label, confidence=Confidence.INDETERMINATE)

    metrics: dict[str, float] = {}

    # NaN and Inf contamination
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    metrics["nan_rate"] = n_nan / n
    metrics["inf_rate"] = n_inf / n

    # Work with finite values for remaining stats
    finite = flat[np.isfinite(flat)]
    n_finite = len(finite)

    if n_finite > 0:
        # Zero rate
        metrics["zero_rate"] = float((finite == 0).sum()) / n_finite

        # Distribution shape
        mean_val = float(np.mean(finite))
        std_val = float(np.std(finite))

        if std_val > 0:
            centered = (finite - mean_val) / std_val
            metrics["kurtosis_excess"] = abs(float(np.mean(centered ** 4)) - 3.0)
            metrics["skewness_abs"] = abs(float(np.mean(centered ** 3)))

        # Drift from reference
        if reference is not None:
            if abs(reference.mean) > 1e-10:
                metrics["mean_drift"] = abs(mean_val - reference.mean) / abs(reference.mean)
            else:
                metrics["mean_drift"] = abs(mean_val - reference.mean)

            if reference.std > 1e-10:
                metrics["std_ratio"] = std_val / reference.std

        # Range violations
        if expected_range is not None:
            lo, hi = expected_range
            n_violations = int(((finite < lo) | (finite > hi)).sum())
            metrics["range_violation_rate"] = n_violations / n_finite

    # Build observations
    observations = []
    for name, value in metrics.items():
        m = ARRAY_METRICS.get(name)
        if m is None:
            continue
        health = classify(value, confidence, thresholds=m.thresholds)
        observations.append(Observation(
            name=name, health=health, value=value, baseline=m.baseline,
            confidence=confidence, higher_is_better=m.thresholds.higher_is_better,
            measured_at=measured_at,
        ))

    return Expression(
        observations=observations,
        confidence=min((o.confidence for o in observations), default=Confidence.INDETERMINATE),
        label=label,
    )


def compare_arrays(
    current,
    reference,
    label: str = "",
    confidence: Confidence = Confidence.MODERATE,
) -> Expression:
    """
    Compare a current array against a reference array.

    Shorthand for: build a reference profile, then call array_health
    with drift detection enabled.

    Args:
        current:    the array to evaluate
        reference:  the known-good array to compare against
        label:      expression label
        confidence: measurement confidence
    """
    import numpy as np
    ref_profile = ArrayProfile.from_array(reference)
    ref_arr = np.asarray(reference).ravel()
    ref_finite = ref_arr[np.isfinite(ref_arr)]
    expected_range = None
    if len(ref_finite) > 0:
        expected_range = (float(np.min(ref_finite)), float(np.max(ref_finite)))
    return array_health(current, reference=ref_profile, expected_range=expected_range,
                        label=label, confidence=confidence)
