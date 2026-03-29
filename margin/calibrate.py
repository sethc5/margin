"""
Threshold derivation from historical data.

Takes a set of "known healthy" measurements and derives baselines and
thresholds automatically.
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
