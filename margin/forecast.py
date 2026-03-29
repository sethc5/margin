"""
Forecasting: project when a component will cross a threshold.

Given a sequence of observations, fits a linear trend and extrapolates
to estimate when the value will reach the intact or ablated threshold.
Uncertainty widens over the projection horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .health import Thresholds
from .observation import Observation


@dataclass
class Forecast:
    """
    Projection of when a component will cross a threshold.

    component:         component name
    current_value:     most recent measurement
    trend_per_second:  slope of the linear fit (units/second)
    trend_uncertainty: standard error of the slope
    time_to_intact:    seconds until the value reaches the intact threshold (None if already intact or diverging)
    time_to_ablated:   seconds until the value reaches the ablated threshold (None if already ablated or diverging)
    projected_at:      when this forecast was computed
    n_samples:         number of observations used
    """
    component: str
    current_value: float
    trend_per_second: float
    trend_uncertainty: float
    time_to_intact: Optional[float]
    time_to_ablated: Optional[float]
    projected_at: datetime
    n_samples: int

    @property
    def improving(self) -> bool:
        """True if the trend is moving toward healthier values."""
        return self.trend_per_second > 0

    @property
    def worsening(self) -> bool:
        """True if the trend is moving toward unhealthier values."""
        return self.trend_per_second < 0

    @property
    def stable(self) -> bool:
        """True if the trend is within the uncertainty band of zero."""
        return abs(self.trend_per_second) <= self.trend_uncertainty

    @property
    def eta_intact(self) -> Optional[timedelta]:
        if self.time_to_intact is None or self.time_to_intact < 0:
            return None
        return timedelta(seconds=self.time_to_intact)

    @property
    def eta_ablated(self) -> Optional[timedelta]:
        if self.time_to_ablated is None or self.time_to_ablated < 0:
            return None
        return timedelta(seconds=self.time_to_ablated)

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "current_value": round(self.current_value, 6),
            "trend_per_second": round(self.trend_per_second, 8),
            "trend_uncertainty": round(self.trend_uncertainty, 8),
            "time_to_intact": round(self.time_to_intact, 1) if self.time_to_intact is not None else None,
            "time_to_ablated": round(self.time_to_ablated, 1) if self.time_to_ablated is not None else None,
            "improving": self.improving,
            "stable": self.stable,
            "n_samples": self.n_samples,
        }

    def __repr__(self) -> str:
        sign = "+" if self.trend_per_second >= 0 else ""
        return (f"Forecast({self.component}: {sign}{self.trend_per_second:.4g}/s, "
                f"eta_intact={self.eta_intact}, eta_ablated={self.eta_ablated})")


def forecast(
    observations: list[Observation],
    thresholds: Thresholds,
    now: Optional[datetime] = None,
) -> Optional[Forecast]:
    """
    Fit a linear trend to a sequence of observations and project
    when the value will cross the intact and ablated thresholds.

    Requires at least 2 observations with `measured_at` timestamps.
    Returns None if insufficient data.

    The trend direction is normalised to the polarity: positive trend
    means improving (toward healthy), negative means worsening.
    """
    # Filter to observations with timestamps, sorted by time
    timed = sorted(
        [o for o in observations if o.measured_at is not None],
        key=lambda o: o.measured_at,
    )
    if len(timed) < 2:
        return None

    now = now or datetime.now()
    name = timed[0].name

    # Convert to (seconds_from_first, value) pairs
    t0 = timed[0].measured_at
    xs = [(o.measured_at - t0).total_seconds() for o in timed]
    ys = [o.value for o in timed]
    n = len(xs)

    # Linear regression: y = a + b*x
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if ss_xx == 0:
        return None

    slope = ss_xy / ss_xx  # units per second
    intercept = mean_y - slope * mean_x

    # Standard error of the slope
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_res = sum(r ** 2 for r in residuals)
    se_slope = math.sqrt(ss_res / max(n - 2, 1) / ss_xx) if ss_xx > 0 else 0.0

    # Project from now
    seconds_from_t0_to_now = (now - t0).total_seconds()
    current_projected = intercept + slope * seconds_from_t0_to_now

    # Normalise slope direction for polarity
    # For higher_is_better: positive slope = improving (raw slope is fine)
    # For lower_is_better: negative slope = improving (flip sign)
    normalised_slope = slope if thresholds.higher_is_better else -slope

    # Time to cross thresholds
    def _time_to_cross(target: float) -> Optional[float]:
        if slope == 0:
            return None
        t_cross = (target - current_projected) / slope  # seconds from now
        if t_cross < 0:
            return None  # already past or wrong direction
        return t_cross

    time_to_intact = _time_to_cross(thresholds.intact)
    time_to_ablated = _time_to_cross(thresholds.ablated)

    return Forecast(
        component=name,
        current_value=ys[-1],
        trend_per_second=normalised_slope,
        trend_uncertainty=se_slope,
        time_to_intact=time_to_intact,
        time_to_ablated=time_to_ablated,
        projected_at=now,
        n_samples=n,
    )
