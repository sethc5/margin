"""
Threshold derivation from historical data.

Takes a set of "known healthy" measurements and derives baselines and
thresholds automatically.

Also provides runtime recalibration: detect when a baseline has drifted
and construct a replacement Parser with updated baselines/thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .health import Thresholds
from .observation import Parser


@dataclass
class CalibrationResult:
    """
    Output of calibrate(): derived baseline and thresholds for one component.

    baseline:     mean of healthy measurements
    std:          standard deviation of healthy measurements
    n_samples:    number of measurements used
    thresholds:   derived Thresholds object
    """
    baseline: float
    std: float
    n_samples: int
    thresholds: Thresholds

    def to_dict(self) -> dict:
        return {
            "baseline": round(self.baseline, 6),
            "std": round(self.std, 6),
            "n_samples": self.n_samples,
            "intact": self.thresholds.intact,
            "ablated": self.thresholds.ablated,
            "higher_is_better": self.thresholds.higher_is_better,
        }


def calibrate(
    values: list[float],
    higher_is_better: bool = True,
    intact_fraction: float = 0.70,
    ablated_fraction: float = 0.30,
    active_min: float = 0.05,
) -> CalibrationResult:
    """
    Derive a baseline and thresholds from known-healthy measurements.

    Takes a list of measurements collected when the component was operating
    normally, and derives:
    - baseline: mean of the measurements
    - intact threshold: intact_fraction * baseline
    - ablated threshold: ablated_fraction * baseline

    Args:
        values:           list of healthy-state measurements
        higher_is_better: polarity
        intact_fraction:  fraction of baseline for intact threshold (default 0.70)
        ablated_fraction: fraction of baseline for ablated threshold (default 0.30)
        active_min:       minimum correction magnitude for "active"

    For higher_is_better=True:
        intact  = baseline * intact_fraction  (e.g. 70% of healthy mean)
        ablated = baseline * ablated_fraction  (e.g. 30% of healthy mean)

    For higher_is_better=False:
        intact  = baseline * (1 + (1 - intact_fraction))   (e.g. 130% of healthy mean)
        ablated = baseline * (1 + (1 - ablated_fraction))   (e.g. 170% of healthy mean)
        This puts the "bad" direction above baseline for lower-is-better metrics.
    """
    if not values:
        raise ValueError("Cannot calibrate from empty list")

    n = len(values)
    baseline = sum(values) / n

    if baseline == 0:
        raise ValueError("Cannot calibrate from zero baseline — thresholds would be degenerate")
    if higher_is_better and baseline < 0:
        raise ValueError(
            f"Cannot calibrate higher_is_better=True with negative baseline ({baseline}). "
            "Fraction-based thresholds invert for negative values. "
            "Use higher_is_better=False or shift your measurements to be positive.")

    variance = sum((v - baseline) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(variance)

    if higher_is_better:
        intact = baseline * intact_fraction
        ablated = baseline * ablated_fraction
    else:
        # For lower-is-better, thresholds are above baseline (unhealthy direction).
        # Uses abs(baseline) so the math works for both positive and negative baselines.
        intact = baseline + abs(baseline) * (1 - intact_fraction)
        ablated = baseline + abs(baseline) * (1 - ablated_fraction)

    thresholds = Thresholds(
        intact=round(intact, 6),
        ablated=round(ablated, 6),
        higher_is_better=higher_is_better,
        active_min=active_min,
    )

    return CalibrationResult(
        baseline=baseline,
        std=std,
        n_samples=n,
        thresholds=thresholds,
    )


def calibrate_many(
    component_values: dict[str, list[float]],
    polarities: Optional[dict[str, bool]] = None,
    intact_fraction: float = 0.70,
    ablated_fraction: float = 0.30,
    active_min: float = 0.05,
) -> tuple[dict[str, float], dict[str, Thresholds]]:
    """
    Calibrate multiple components at once. Returns (baselines, thresholds)
    ready to pass directly to Parser().

    Args:
        component_values: {name: [healthy_measurements]}
        polarities:       {name: higher_is_better} (defaults to True)
        intact_fraction:  fraction of baseline for intact threshold
        ablated_fraction: fraction of baseline for ablated threshold
        active_min:       minimum correction magnitude

    Returns:
        (baselines_dict, thresholds_dict) suitable for:
            Parser(baselines=baselines_dict,
                   thresholds=first_threshold,
                   component_thresholds=thresholds_dict)
    """
    polarities = polarities or {}
    baselines = {}
    thresholds = {}

    for name, vals in component_values.items():
        hib = polarities.get(name, True)
        result = calibrate(vals, hib, intact_fraction, ablated_fraction, active_min)
        baselines[name] = result.baseline
        thresholds[name] = result.thresholds

    return baselines, thresholds


def needs_recalibration(
    calibration_samples: list[float],
    recent_samples: list[float],
    mean_shift_threshold: float = 0.2,
    std_ratio_high: float = 2.0,
    std_ratio_low: float = 0.5,
    min_recent: int = 5,
) -> bool:
    """
    Return True if recent values suggest the baseline has drifted.

    Uses distribution comparison between calibration-time healthy data and
    a recent window to detect two signals:

      Signal A — mean shift: the operating centre has moved by more than
        `mean_shift_threshold` (relative fraction, default 20%).

      Signal B — spread change: variance has doubled (`std_ratio > 2.0`)
        or halved (`std_ratio < 0.5`) relative to the calibration period.

    Returns True when either signal fires and there are enough recent samples.
    A third signal (AnomalyState.ANOMALOUS at healthy sigma) must be checked
    externally via AnomalyTracker — see `needs_recalibration` docs.

    Args:
        calibration_samples:  original healthy measurements used to build the Parser
        recent_samples:       recent window of observed values
        mean_shift_threshold: relative mean shift that triggers recalibration (default 0.2)
        std_ratio_high:       std ratio above which spread is considered changed (default 2.0)
        std_ratio_low:        std ratio below which spread is considered changed (default 0.5)
        min_recent:           minimum recent samples required to make a judgment (default 5)
    """
    if len(recent_samples) < min_recent or len(calibration_samples) < 2:
        return False

    from .anomaly import check_distribution
    ds = check_distribution(recent_samples, calibration_samples)

    mean_shifted = abs(ds.mean_shift) > mean_shift_threshold
    spread_changed = ds.std_ratio > std_ratio_high or ds.std_ratio < std_ratio_low

    return mean_shifted or spread_changed


def recalibrate_parser(
    parser: Parser,
    new_healthy_data: dict[str, list[float]],
    polarities: Optional[dict[str, bool]] = None,
    components: Optional[list[str]] = None,
    intact_fraction: float = 0.70,
    ablated_fraction: float = 0.30,
    active_min: float = 0.05,
) -> tuple[Parser, dict[str, CalibrationResult]]:
    """
    Return a new Parser with updated baselines for components whose healthy
    operating range has drifted.

    Does NOT mutate the input parser — returns a replacement. Components not
    in `new_healthy_data` (or not in `components`) keep their existing
    baselines and thresholds unchanged.

    Use after `needs_recalibration()` signals that a component's baseline
    is stale, or when you have a new labeled window of known-healthy data.

    Args:
        parser:           the existing Parser to replace
        new_healthy_data: {component: [new_healthy_measurements]}
        polarities:       {component: higher_is_better} overrides (defaults
                          to existing parser polarity for each component)
        components:       allowlist of components to recalibrate; if None,
                          all components in new_healthy_data are recalibrated
        intact_fraction:  fraction of baseline for intact threshold
        ablated_fraction: fraction of baseline for ablated threshold
        active_min:       minimum correction magnitude

    Returns:
        (new_parser, results) where results is {component: CalibrationResult}
        for each recalibrated component.

    Transition notes:
    - Historical Ledger Observations remain anchored to the baseline that was
      active when they were recorded. Recalibration does not rewrite history.
    - After recalibrating, rebuild Monitor from the new Parser:
        save_monitor(old_monitor, "checkpoint.json")  # optional audit trail
        new_monitor = Monitor(new_parser, drift_window=..., anomaly_window=...,
                               correlation_window=...)
    - Per-tracker windows should match the old Monitor's windows for continuity.
    """
    polarities = polarities or {}
    to_recalibrate = set(components) if components else set(new_healthy_data.keys())

    results: dict[str, CalibrationResult] = {}
    new_baselines: dict[str, float] = {}
    new_component_thresholds: dict[str, Thresholds] = {}

    for name in to_recalibrate:
        if name not in new_healthy_data:
            continue
        vals = new_healthy_data[name]
        # Resolve polarity: explicit override → existing parser polarity → True
        existing_t = parser.component_thresholds.get(name) or parser.thresholds
        hib = polarities.get(name, existing_t.higher_is_better)
        result = calibrate(vals, hib, intact_fraction, ablated_fraction, active_min)
        results[name] = result
        new_baselines[name] = result.baseline
        new_component_thresholds[name] = result.thresholds

    # Merge: new overrides old; unrecalibrated components are preserved verbatim
    merged_baselines = {**parser.baselines, **new_baselines}
    merged_component_thresholds = {**parser.component_thresholds, **new_component_thresholds}

    # Default thresholds: recalibrate if the first component was recalibrated,
    # otherwise keep the existing default.
    first_name = list(parser.baselines.keys())[0] if parser.baselines else None
    if first_name and first_name in new_component_thresholds:
        default_thresholds = new_component_thresholds.pop(first_name)
        merged_component_thresholds.pop(first_name, None)
    else:
        default_thresholds = parser.thresholds
        # Remove first component from component_thresholds if it was there
        if first_name:
            merged_component_thresholds.pop(first_name, None)

    new_parser = Parser(
        baselines=merged_baselines,
        thresholds=default_thresholds,
        component_thresholds=merged_component_thresholds,
    )

    return new_parser, results


def parser_from_calibration(
    component_values: dict[str, list[float]],
    polarities: Optional[dict[str, bool]] = None,
    intact_fraction: float = 0.70,
    ablated_fraction: float = 0.30,
    active_min: float = 0.05,
) -> Parser:
    """
    One-shot: calibrate from historical data and return a ready-to-use Parser.

    Uses the first component's thresholds as the default, with all others
    as component_thresholds overrides.
    """
    baselines, thresh_dict = calibrate_many(
        component_values, polarities, intact_fraction, ablated_fraction, active_min,
    )

    names = list(thresh_dict.keys())
    if not names:
        raise ValueError("No components to calibrate")

    default_thresholds = thresh_dict[names[0]]
    component_thresholds = {n: t for n, t in thresh_dict.items() if n != names[0]}

    return Parser(
        baselines=baselines,
        thresholds=default_thresholds,
        component_thresholds=component_thresholds,
    )
