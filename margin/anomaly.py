"""
Anomaly detection: typed states for statistical outliers.

Health tells you WHERE a value is (INTACT/DEGRADED/ABLATED).
Drift tells you WHERE IT'S HEADED (STABLE/DRIFTING/ACCELERATING).
Anomaly tells you IS THIS NORMAL (EXPECTED/UNUSUAL/ANOMALOUS/NOVEL).

A value can be INTACT and STABLE but at a level never seen before.
Health and drift both miss that — anomaly catches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .confidence import Confidence
from .observation import Expression, Observation


class AnomalyState(Enum):
    """
    Typed anomaly classification.

    EXPECTED:  value is within normal statistical range of reference data
    UNUSUAL:   value is uncommon (beyond unusual_threshold σ) but not extreme
    ANOMALOUS: value is a statistical outlier (beyond anomalous_threshold σ)
    NOVEL:     value is outside the entire historical range — never seen before
    """
    EXPECTED = "EXPECTED"
    UNUSUAL = "UNUSUAL"
    ANOMALOUS = "ANOMALOUS"
    NOVEL = "NOVEL"


# Severity ordering: higher = more abnormal
ANOMALY_SEVERITY = {
    AnomalyState.EXPECTED: 0,
    AnomalyState.UNUSUAL: 1,
    AnomalyState.ANOMALOUS: 2,
    AnomalyState.NOVEL: 3,
}


@dataclass
class AnomalyClassification:
    """
    Full anomaly analysis for one value against a reference distribution.

    component:       component name
    state:           anomaly classification
    z_score:         how many σ from the reference mean
    historical_mean: mean of reference data
    historical_std:  std of reference data
    historical_min:  min of reference data
    historical_max:  max of reference data
    is_novel:        True if value is outside historical range
    confidence:      based on reference sample size
    n_reference:     number of reference samples used
    """
    component: str
    state: AnomalyState
    z_score: float
    historical_mean: float
    historical_std: float
    historical_min: float
    historical_max: float
    is_novel: bool
    confidence: Confidence
    n_reference: int

    @property
    def expected(self) -> bool:
        return self.state == AnomalyState.EXPECTED

    @property
    def anomalous(self) -> bool:
        return self.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def to_atom(self) -> str:
        """Compact string: component:STATE(±z σ)"""
        sign = "+" if self.z_score >= 0 else ""
        return f"{self.component}:{self.state.value}({sign}{self.z_score:.2f}σ)"

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "state": self.state.value,
            "z_score": round(self.z_score, 4),
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "historical_min": self.historical_min,
            "historical_max": self.historical_max,
            "is_novel": self.is_novel,
            "confidence": self.confidence.value,
            "n_reference": self.n_reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnomalyClassification:
        return cls(
            component=d["component"],
            state=AnomalyState(d["state"]),
            z_score=d["z_score"],
            historical_mean=d["historical_mean"],
            historical_std=d["historical_std"],
            historical_min=d["historical_min"],
            historical_max=d["historical_max"],
            is_novel=d["is_novel"],
            confidence=Confidence(d["confidence"]),
            n_reference=d["n_reference"],
        )

    def __repr__(self) -> str:
        return f"AnomalyClassification({self.to_atom()})"


# -----------------------------------------------------------------------
# Distribution shift detection
# -----------------------------------------------------------------------

@dataclass
class DistributionShift:
    """
    Comparison of two sample distributions (recent vs reference).

    mean_shift:     relative change in mean: |recent_mean - ref_mean| / |ref_mean|
    std_ratio:      recent_std / ref_std (>1 = wider spread, <1 = narrower)
    kurtosis_delta: change in excess kurtosis (heavier/lighter tails)
    skew_delta:     change in skewness (asymmetry shift)
    state:          overall anomaly classification
    confidence:     based on sample sizes
    """
    component: str
    mean_shift: float
    std_ratio: float
    kurtosis_delta: float
    skew_delta: float
    state: AnomalyState
    confidence: Confidence
    n_recent: int
    n_reference: int

    @property
    def shifted(self) -> bool:
        return self.state != AnomalyState.EXPECTED

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "mean_shift": round(self.mean_shift, 4),
            "std_ratio": round(self.std_ratio, 4),
            "kurtosis_delta": round(self.kurtosis_delta, 4),
            "skew_delta": round(self.skew_delta, 4),
            "state": self.state.value,
            "confidence": self.confidence.value,
            "n_recent": self.n_recent,
            "n_reference": self.n_reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DistributionShift:
        return cls(
            component=d["component"],
            mean_shift=d["mean_shift"],
            std_ratio=d["std_ratio"],
            kurtosis_delta=d["kurtosis_delta"],
            skew_delta=d["skew_delta"],
            state=AnomalyState(d["state"]),
            confidence=Confidence(d["confidence"]),
            n_recent=d["n_recent"],
            n_reference=d["n_reference"],
        )

    def __repr__(self) -> str:
        return (f"DistributionShift({self.component}: mean_shift={self.mean_shift:.3f}, "
                f"std_ratio={self.std_ratio:.3f}, state={self.state.value})")


# -----------------------------------------------------------------------
# Jump detection
# -----------------------------------------------------------------------

@dataclass
class Jump:
    """
    A sudden discontinuity between consecutive observations.

    Health sees INTACT, drift sees STABLE, but the value just teleported.
    """
    component: str
    magnitude_sigma: float  # size of jump in σ of the series
    value_before: float
    value_after: float
    at_index: int           # index in observation sequence

    @property
    def significant(self) -> bool:
        return abs(self.magnitude_sigma) > 0

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "magnitude_sigma": round(self.magnitude_sigma, 4),
            "value_before": self.value_before,
            "value_after": self.value_after,
            "at_index": self.at_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Jump:
        return cls(
            component=d["component"],
            magnitude_sigma=d["magnitude_sigma"],
            value_before=d["value_before"],
            value_after=d["value_after"],
            at_index=d["at_index"],
        )

    def __repr__(self) -> str:
        sign = "+" if self.magnitude_sigma >= 0 else ""
        return f"Jump({self.component}: {sign}{self.magnitude_sigma:.2f}σ at index {self.at_index})"


# -----------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(variance)


def _kurtosis_excess(xs: list[float], mean: float, std: float) -> float:
    """Excess kurtosis (0 for normal distribution)."""
    if std == 0 or len(xs) < 4:
        return 0.0
    n = len(xs)
    m4 = sum((x - mean) ** 4 for x in xs) / n
    return m4 / (std ** 4) - 3.0


def _skewness(xs: list[float], mean: float, std: float) -> float:
    """Skewness (0 for symmetric distribution)."""
    if std == 0 or len(xs) < 3:
        return 0.0
    n = len(xs)
    m3 = sum((x - mean) ** 3 for x in xs) / n
    return m3 / (std ** 3)


def _confidence_from_n(n: int) -> Confidence:
    """Confidence based on reference sample size."""
    if n >= 30:
        return Confidence.HIGH
    if n >= 10:
        return Confidence.MODERATE
    if n >= 3:
        return Confidence.LOW
    return Confidence.INDETERMINATE


# -----------------------------------------------------------------------
# Point anomaly classification
# -----------------------------------------------------------------------

def classify_anomaly(
    value: float,
    reference: list[float],
    component: str = "",
    unusual_threshold: float = 2.0,
    anomalous_threshold: float = 3.0,
    novel_margin: float = 0.1,
    min_reference: int = 3,
) -> Optional[AnomalyClassification]:
    """
    Classify a value against a reference distribution.

    Args:
        value:               the value to check
        reference:           historical values (the "normal" distribution)
        component:           component name for labeling
        unusual_threshold:   z-score threshold for UNUSUAL (default 2.0)
        anomalous_threshold: z-score threshold for ANOMALOUS (default 3.0)
        novel_margin:        fraction beyond historical range for NOVEL (default 0.1)
        min_reference:       minimum reference samples required (default 3)

    Returns AnomalyClassification, or None if insufficient reference data.
    """
    if len(reference) < min_reference:
        return None

    ref_mean = _mean(reference)
    ref_std = _std(reference, ref_mean)
    ref_min = min(reference)
    ref_max = max(reference)
    n = len(reference)

    # Z-score
    if ref_std > 0:
        z = (value - ref_mean) / ref_std
    else:
        # All reference values identical — any deviation is novel
        z = 0.0 if value == ref_mean else float('inf') if value > ref_mean else float('-inf')

    abs_z = abs(z)

    # Novelty check: beyond historical range with margin
    ref_range = ref_max - ref_min
    margin = max(ref_range * novel_margin, ref_std * 0.1) if ref_range > 0 else ref_std * 0.5
    is_novel = value < (ref_min - margin) or value > (ref_max + margin)

    # Classification
    if is_novel:
        state = AnomalyState.NOVEL
    elif abs_z >= anomalous_threshold:
        state = AnomalyState.ANOMALOUS
    elif abs_z >= unusual_threshold:
        state = AnomalyState.UNUSUAL
    else:
        state = AnomalyState.EXPECTED

    return AnomalyClassification(
        component=component,
        state=state,
        z_score=z,
        historical_mean=ref_mean,
        historical_std=ref_std,
        historical_min=ref_min,
        historical_max=ref_max,
        is_novel=is_novel,
        confidence=_confidence_from_n(n),
        n_reference=n,
    )


def classify_anomaly_obs(
    observation: Observation,
    history: list[Observation],
    **kwargs,
) -> Optional[AnomalyClassification]:
    """Classify anomaly from Observation objects."""
    reference = [o.value for o in history]
    return classify_anomaly(
        value=observation.value,
        reference=reference,
        component=observation.name,
        **kwargs,
    )


# -----------------------------------------------------------------------
# Distribution shift
# -----------------------------------------------------------------------

def check_distribution(
    recent: list[float],
    reference: list[float],
    component: str = "",
    mean_shift_threshold: float = 0.2,
    std_ratio_threshold: float = 2.0,
    shape_threshold: float = 1.0,
    min_samples: int = 5,
) -> Optional[DistributionShift]:
    """
    Compare two sample distributions for significant shifts.

    Args:
        recent:               recent values (the "current" distribution)
        reference:            historical values (the "normal" distribution)
        component:            component name
        mean_shift_threshold: relative mean shift for UNUSUAL (default 0.2 = 20%)
        std_ratio_threshold:  std ratio for UNUSUAL (default 2.0 = doubled spread)
        shape_threshold:      kurtosis/skew delta for UNUSUAL (default 1.0)
        min_samples:          minimum samples in each group (default 5)

    Returns DistributionShift, or None if insufficient data.
    """
    if len(recent) < min_samples or len(reference) < min_samples:
        return None

    ref_mean = _mean(reference)
    ref_std = _std(reference, ref_mean)
    rec_mean = _mean(recent)
    rec_std = _std(recent, rec_mean)

    # Mean shift (relative to reference)
    if abs(ref_mean) > 1e-10:
        mean_shift = abs(rec_mean - ref_mean) / abs(ref_mean)
    elif ref_std > 0:
        mean_shift = abs(rec_mean - ref_mean) / ref_std
    else:
        mean_shift = 0.0 if rec_mean == ref_mean else float('inf')

    # Std ratio
    std_ratio = rec_std / ref_std if ref_std > 0 else (1.0 if rec_std == 0 else float('inf'))

    # Shape metrics
    ref_kurt = _kurtosis_excess(reference, ref_mean, ref_std)
    rec_kurt = _kurtosis_excess(recent, rec_mean, rec_std)
    ref_skew = _skewness(reference, ref_mean, ref_std)
    rec_skew = _skewness(recent, rec_mean, rec_std)

    kurt_delta = rec_kurt - ref_kurt
    skew_delta = rec_skew - ref_skew

    # Classification — worst of the signals
    anomalous = (
        mean_shift > mean_shift_threshold * 2
        or std_ratio > std_ratio_threshold * 1.5
        or std_ratio < 1.0 / (std_ratio_threshold * 1.5)
        or abs(kurt_delta) > shape_threshold * 2
        or abs(skew_delta) > shape_threshold * 2
    )
    unusual = (
        mean_shift > mean_shift_threshold
        or std_ratio > std_ratio_threshold
        or std_ratio < 1.0 / std_ratio_threshold
        or abs(kurt_delta) > shape_threshold
        or abs(skew_delta) > shape_threshold
    )

    if anomalous:
        state = AnomalyState.ANOMALOUS
    elif unusual:
        state = AnomalyState.UNUSUAL
    else:
        state = AnomalyState.EXPECTED

    confidence = min(_confidence_from_n(len(recent)), _confidence_from_n(len(reference)))

    return DistributionShift(
        component=component,
        mean_shift=mean_shift,
        std_ratio=std_ratio,
        kurtosis_delta=kurt_delta,
        skew_delta=skew_delta,
        state=state,
        confidence=confidence,
        n_recent=len(recent),
        n_reference=len(reference),
    )


# -----------------------------------------------------------------------
# Jump detection
# -----------------------------------------------------------------------

def detect_jumps(
    observations: list[Observation],
    jump_threshold: float = 3.0,
) -> list[Jump]:
    """
    Find sudden discontinuities in an observation sequence.

    A jump is when consecutive values differ by more than `jump_threshold`
    standard deviations of the series' step-to-step differences.

    Args:
        observations:    ordered observations for one component
        jump_threshold:  minimum σ of step differences to count as a jump

    Returns list of Jump objects (may be empty).
    """
    if len(observations) < 3:
        return []

    values = [o.value for o in observations]
    component = observations[0].name

    # Step-to-step differences
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

    if not diffs:
        return []

    diff_mean = _mean(diffs)
    diff_std = _std(diffs, diff_mean)

    if diff_std == 0:
        return []

    jumps = []
    for i, d in enumerate(diffs):
        z = (d - diff_mean) / diff_std
        if abs(z) >= jump_threshold:
            jumps.append(Jump(
                component=component,
                magnitude_sigma=z,
                value_before=values[i],
                value_after=values[i + 1],
                at_index=i + 1,
            ))

    return jumps


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

def _observations_from_ledger(ledger, component: str) -> list[Observation]:
    """Extract observation history for one component from a Ledger."""
    obs = []
    for rec in ledger.records:
        if rec.after and rec.after.name == component:
            obs.append(rec.after)
        elif rec.before.name == component:
            obs.append(rec.before)
    return obs


def _all_components(ledger) -> set[str]:
    names: set[str] = set()
    for rec in ledger.records:
        names.add(rec.before.name)
        if rec.after:
            names.add(rec.after.name)
    return names


def anomaly_from_ledger(
    ledger,
    component: str,
    reference_fraction: float = 0.8,
    **kwargs,
) -> Optional[AnomalyClassification]:
    """
    Classify the most recent value against earlier values from a Ledger.

    Uses the first `reference_fraction` of records as reference and
    classifies the last value against them.

    Args:
        ledger:             correction Ledger
        component:          component name
        reference_fraction: fraction of history to use as reference (default 0.8)
        **kwargs:           passed to classify_anomaly
    """
    obs = _observations_from_ledger(ledger, component)
    if len(obs) < 4:
        return None

    split = max(int(len(obs) * reference_fraction), 3)
    if split >= len(obs):
        return None

    reference = [o.value for o in obs[:split]]
    latest = obs[-1]

    return classify_anomaly(
        value=latest.value,
        reference=reference,
        component=component,
        **kwargs,
    )


def anomaly_all_from_ledger(
    ledger,
    **kwargs,
) -> dict[str, AnomalyClassification]:
    """Classify anomaly for every component in a Ledger."""
    results = {}
    for name in sorted(_all_components(ledger)):
        ac = anomaly_from_ledger(ledger, name, **kwargs)
        if ac is not None:
            results[name] = ac
    return results


def distribution_shift_from_ledger(
    ledger,
    component: str,
    recent_fraction: float = 0.3,
    **kwargs,
) -> Optional[DistributionShift]:
    """
    Check for distribution shift: compare recent values vs earlier values.

    Splits the ledger history at `1 - recent_fraction` and compares
    the two halves.
    """
    obs = _observations_from_ledger(ledger, component)
    if len(obs) < 10:
        return None

    split = max(int(len(obs) * (1.0 - recent_fraction)), 5)
    if split >= len(obs) - 4:
        return None

    reference = [o.value for o in obs[:split]]
    recent = [o.value for o in obs[split:]]

    return check_distribution(recent, reference, component=component, **kwargs)


# -----------------------------------------------------------------------
# Predicates — for use in Policy rules
# -----------------------------------------------------------------------

PredicateFn = Callable[[Expression], bool]


def anomaly_is(
    component: str,
    state: AnomalyState,
    ledger,
    **kwargs,
) -> PredicateFn:
    """True if the named component's anomaly classification matches `state`."""
    def check(expr: Expression) -> bool:
        ac = anomaly_from_ledger(ledger, component, **kwargs)
        return ac is not None and ac.state == state
    return check


def any_anomalous(ledger, **kwargs) -> PredicateFn:
    """True if any component is ANOMALOUS or NOVEL."""
    def check(expr: Expression) -> bool:
        all_ac = anomaly_all_from_ledger(ledger, **kwargs)
        return any(ac.anomalous for ac in all_ac.values())
    return check


def any_novel(ledger, **kwargs) -> PredicateFn:
    """True if any component has a NOVEL value (never seen before)."""
    def check(expr: Expression) -> bool:
        all_ac = anomaly_all_from_ledger(ledger, **kwargs)
        return any(ac.state == AnomalyState.NOVEL for ac in all_ac.values())
    return check


def is_novel(
    component: str,
    ledger,
    **kwargs,
) -> PredicateFn:
    """True if the named component's latest value is NOVEL."""
    return anomaly_is(component, AnomalyState.NOVEL, ledger, **kwargs)
