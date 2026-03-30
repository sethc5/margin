"""
Drift classification: typed states for value trajectories.

Margin tells you WHERE a value is (INTACT/DEGRADED/ABLATED).
Drift tells you WHERE IT'S HEADED (STABLE/DRIFTING/ACCELERATING/etc).

Given a sequence of timestamped observations, fits linear and quadratic
models to classify the trajectory shape, direction, and confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional

from .confidence import Confidence
from .observation import Expression, Observation


class DriftState(Enum):
    """
    Typed trajectory classification.

    STABLE:       value not changing meaningfully (slope within noise)
    DRIFTING:     consistent linear trend in one direction
    ACCELERATING: rate of change is increasing (getting worse/better faster)
    DECELERATING: rate of change is decreasing (approaching a plateau)
    REVERTING:    was drifting away from baseline, now heading back
    OSCILLATING:  periodic fluctuation around a center
    """
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    REVERTING = "REVERTING"
    OSCILLATING = "OSCILLATING"


class DriftDirection(Enum):
    """Which way the value is moving relative to health."""
    IMPROVING = "IMPROVING"
    WORSENING = "WORSENING"
    NEUTRAL = "NEUTRAL"


@dataclass
class DriftClassification:
    """
    Full drift analysis for one component.

    component:      component name
    state:          trajectory shape
    direction:      improving / worsening / neutral (polarity-aware)
    rate:           slope in units/second (polarity-normalised: positive = healthier)
    acceleration:   second derivative (units/second^2, polarity-normalised)
    r_squared:      goodness of fit for the best model [0, 1]
    confidence:     how much to trust this classification
    n_samples:      observations used
    window_seconds: time span covered
    """
    component: str
    state: DriftState
    direction: DriftDirection
    rate: float
    acceleration: float
    r_squared: float
    confidence: Confidence
    n_samples: int
    window_seconds: float

    @property
    def step_count(self) -> int:
        """Number of observations that contributed to this classification.
        A 3-step OSCILLATING is noise; a 20-step OSCILLATING is real."""
        return self.n_samples

    @property
    def stable(self) -> bool:
        return self.state == DriftState.STABLE

    @property
    def worsening(self) -> bool:
        return self.direction == DriftDirection.WORSENING

    @property
    def improving(self) -> bool:
        return self.direction == DriftDirection.IMPROVING

    def to_atom(self) -> str:
        """Compact string: component:STATE(direction, rate/s)"""
        sign = "+" if self.rate >= 0 else ""
        return f"{self.component}:{self.state.value}({self.direction.value}, {sign}{self.rate:.4g}/s)"

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "state": self.state.value,
            "direction": self.direction.value,
            "rate": self.rate,
            "acceleration": self.acceleration,
            "r_squared": round(self.r_squared, 4),
            "confidence": self.confidence.value,
            "n_samples": self.n_samples,
            "window_seconds": round(self.window_seconds, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DriftClassification:
        return cls(
            component=d["component"],
            state=DriftState(d["state"]),
            direction=DriftDirection(d["direction"]),
            rate=d["rate"],
            acceleration=d["acceleration"],
            r_squared=d["r_squared"],
            confidence=Confidence(d["confidence"]),
            n_samples=d["n_samples"],
            window_seconds=d["window_seconds"],
        )

    def __repr__(self) -> str:
        return f"DriftClassification({self.to_atom()})"


# -----------------------------------------------------------------------
# Linear + quadratic regression helpers
# -----------------------------------------------------------------------

def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Linear regression. Returns (slope, intercept, r_squared, se_slope)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if ss_xx == 0:
        return 0.0, mean_y, 0.0, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1.0 - ss_res / ss_yy if ss_yy > 0 else 0.0
    se = math.sqrt(ss_res / max(n - 2, 1) / ss_xx) if ss_xx > 0 else 0.0

    return slope, intercept, max(r_sq, 0.0), se


def _quadreg(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Quadratic regression y = a*x^2 + b*x + c. Returns (a, b, r_squared, se_a).

    Uses normal equations solved via Cramer's rule.
    """
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0

    # Power sums s[k] = sum(x^k)
    s = [float(n)]
    for k in range(1, 5):
        s.append(sum(x ** k for x in xs))
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum(x ** 2 * y for x, y in zip(xs, ys))

    # M = [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]], rhs = [sy, sxy, sx2y]
    det_m = (s[0] * (s[2] * s[4] - s[3] * s[3])
             - s[1] * (s[1] * s[4] - s[3] * s[2])
             + s[2] * (s[1] * s[3] - s[2] * s[2]))

    if abs(det_m) < 1e-30:
        return 0.0, 0.0, 0.0, 0.0

    # a (x^2 coeff): replace column 2 with rhs
    det_a = (s[0] * (s[2] * sx2y - sxy * s[3])
             - s[1] * (s[1] * sx2y - sxy * s[2])
             + sy * (s[1] * s[3] - s[2] * s[2]))
    a = det_a / det_m

    # b (x coeff): replace column 1 with rhs
    det_b = (s[0] * (sxy * s[4] - s[3] * sx2y)
             - sy * (s[1] * s[4] - s[3] * s[2])
             + s[2] * (s[1] * sx2y - sxy * s[2]))
    b = det_b / det_m

    # c (constant): replace column 0 with rhs
    det_c = (sy * (s[2] * s[4] - s[3] * s[3])
             - s[1] * (sxy * s[4] - s[3] * sx2y)
             + s[2] * (sxy * s[3] - s[2] * sx2y))
    c = det_c / det_m

    # R-squared
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a * x ** 2 + b * x + c)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard error of a
    mse = ss_res / max(n - 3, 1)
    mean_x = s[1] / n
    ss_x4c = sum((x - mean_x) ** 4 for x in xs)
    se_a = math.sqrt(mse / max(ss_x4c, 1e-30))

    return a, b, max(r_sq, 0.0), se_a


def _is_oscillating(
    ys: list[float],
    residuals: list[float],
    min_crossings: int = 2,
    min_rel_amplitude: float = 0.02,
) -> bool:
    """Detect oscillation by counting zero crossings of de-trended residuals.

    An oscillating signal has:
    - Multiple zero crossings of its linear residuals (the wave goes up and down)
    - Significant amplitude relative to the mean (not just noise)
    """
    n = len(ys)
    if n < 5:
        return False

    # Tolerance: ignore residuals within 1% of the max residual magnitude
    max_res = max((abs(r) for r in residuals), default=0.0)
    tol = 0.01 * max_res

    # Count zero crossings of residuals, filtering near-zero values
    crossings = sum(
        1 for i in range(1, len(residuals))
        if residuals[i - 1] * residuals[i] < 0
        and abs(residuals[i - 1]) > tol and abs(residuals[i]) > tol
    )

    # Amplitude check: peak-to-trough vs mean absolute value
    amplitude = max(ys) - min(ys)
    mean_abs = sum(abs(y) for y in ys) / n
    rel_amp = amplitude / max(mean_abs, 1e-10)

    return crossings >= min_crossings and rel_amp > min_rel_amplitude


# -----------------------------------------------------------------------
# Main classification
# -----------------------------------------------------------------------

def classify_drift(
    observations: list[Observation],
    min_samples: int = 3,
    noise_threshold: float = 1.5,
    accel_threshold: float = 0.1,
) -> Optional[DriftClassification]:
    """
    Classify the drift pattern from a sequence of observations.

    Args:
        observations:     timestamped observations for ONE component
        min_samples:      minimum observations required (default 3)
        noise_threshold:  slope must exceed this many standard errors to count
                          as non-stable (default 1.5)
        accel_threshold:  quadratic R^2 must improve by this much over linear
                          to classify as accelerating/decelerating (default 0.1)

    Returns DriftClassification, or None if insufficient data.
    """
    # Filter and sort by time
    timed = sorted(
        [o for o in observations if o.measured_at is not None],
        key=lambda o: o.measured_at,
    )
    if len(timed) < min_samples:
        return None

    name = timed[0].name
    higher_is_better = timed[0].higher_is_better
    baseline = timed[0].baseline

    # Convert to (seconds, value) from first observation
    t0 = timed[0].measured_at
    xs = [(o.measured_at - t0).total_seconds() for o in timed]
    ys = [o.value for o in timed]
    n = len(xs)
    window = xs[-1] - xs[0]

    if window == 0:
        return None

    # Fit linear
    slope, intercept, r2_lin, se_slope = _linreg(xs, ys)

    # Fit quadratic (need at least 3 points)
    a_quad, b_quad, r2_quad, se_a = _quadreg(xs, ys) if n >= 3 else (0.0, 0.0, 0.0, 0.0)

    # Residuals from linear fit (for oscillation detection)
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]

    # Normalise slope for polarity: positive = healthier
    norm_slope = slope if higher_is_better else -slope
    norm_accel = a_quad if higher_is_better else -a_quad

    # --- Classification logic ---

    # 1. Is the slope significant?
    #    se_slope == 0 with nonzero slope means perfect fit (infinitely significant)
    if se_slope == 0:
        slope_significant = abs(slope) > 0
    else:
        slope_significant = abs(slope) > noise_threshold * se_slope

    if not slope_significant:
        # No clear trend — check for oscillation around the mean
        if _is_oscillating(ys, residuals):
            state = DriftState.OSCILLATING
            direction = DriftDirection.NEUTRAL
        else:
            state = DriftState.STABLE
            direction = DriftDirection.NEUTRAL
    else:
        # 2. Direction
        direction = DriftDirection.IMPROVING if norm_slope > 0 else DriftDirection.WORSENING

        # 3. Is the quadratic term significant?
        r2_improvement = r2_quad - r2_lin
        accel_significant = (
            n >= 4
            and r2_improvement > accel_threshold
            and (se_a == 0 and abs(a_quad) > 0 or se_a > 0 and abs(a_quad) > 1.5 * se_a)
        )

        # 4. Reversion check: is the value moving back toward baseline?
        current_val = ys[-1]
        first_val = ys[0]
        if higher_is_better:
            was_unhealthy = first_val < baseline
        else:
            was_unhealthy = first_val > baseline
        now_closer = abs(current_val - baseline) < abs(first_val - baseline)

        # Classify
        if was_unhealthy and now_closer:
            state = DriftState.REVERTING
        elif accel_significant:
            # Same sign = speeding up, opposite = slowing down
            if norm_accel * norm_slope > 0:
                state = DriftState.ACCELERATING
            else:
                state = DriftState.DECELERATING
        else:
            state = DriftState.DRIFTING

    # --- Confidence ---
    # More samples and better fit = higher confidence
    best_r2 = max(r2_lin, r2_quad)
    if n >= 10 and best_r2 > 0.8:
        confidence = Confidence.HIGH
    elif n >= 5 and best_r2 > 0.5:
        confidence = Confidence.MODERATE
    elif n >= min_samples:
        confidence = Confidence.LOW
    else:
        confidence = Confidence.INDETERMINATE

    return DriftClassification(
        component=name,
        state=state,
        direction=direction,
        rate=norm_slope,
        acceleration=norm_accel,
        r_squared=best_r2,
        confidence=confidence,
        n_samples=n,
        window_seconds=window,
    )


def classify_drift_all(
    observations_by_component: dict[str, list[Observation]],
    **kwargs,
) -> dict[str, DriftClassification]:
    """
    Classify drift for multiple components at once.

    Args:
        observations_by_component: {component_name: [observations...]}
        **kwargs: passed to classify_drift

    Returns {component_name: DriftClassification} (omits components with insufficient data).
    """
    results = {}
    for name, obs_list in observations_by_component.items():
        dc = classify_drift(obs_list, **kwargs)
        if dc is not None:
            results[name] = dc
    return results


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

def observations_from_ledger(ledger, component: str) -> list[Observation]:
    """Extract the observation history for one component from a Ledger.

    Uses the 'after' observation if available (post-correction state),
    otherwise falls back to 'before'.
    """
    obs = []
    for rec in ledger.records:
        # Prefer after (corrected state), fall back to before
        if rec.after and rec.after.name == component:
            obs.append(rec.after)
        elif rec.before.name == component:
            obs.append(rec.before)
    return obs


def _all_components(ledger) -> set[str]:
    """Unique component names in a Ledger."""
    names: set[str] = set()
    for rec in ledger.records:
        names.add(rec.before.name)
        if rec.after:
            names.add(rec.after.name)
    return names


def drift_from_ledger(
    ledger,
    component: str,
    **kwargs,
) -> Optional[DriftClassification]:
    """Classify drift for one component from its Ledger history."""
    obs = observations_from_ledger(ledger, component)
    return classify_drift(obs, **kwargs)


def drift_all_from_ledger(
    ledger,
    **kwargs,
) -> dict[str, DriftClassification]:
    """Classify drift for every component found in a Ledger."""
    results = {}
    for name in sorted(_all_components(ledger)):
        dc = drift_from_ledger(ledger, name, **kwargs)
        if dc is not None:
            results[name] = dc
    return results


# -----------------------------------------------------------------------
# Predicates — for use in Policy rules
# -----------------------------------------------------------------------
# Follow the temporal predicate pattern: close over a Ledger,
# return PredicateFn (Expression → bool).

PredicateFn = Callable[[Expression], bool]


def drift_is(
    component: str,
    state: DriftState,
    ledger,
    **kwargs,
) -> PredicateFn:
    """True if the named component's drift classification matches `state`."""
    def check(expr: Expression) -> bool:
        dc = drift_from_ledger(ledger, component, **kwargs)
        return dc is not None and dc.state == state
    return check


def drift_worsening(
    component: str,
    ledger,
    **kwargs,
) -> PredicateFn:
    """True if the named component's drift direction is WORSENING."""
    def check(expr: Expression) -> bool:
        dc = drift_from_ledger(ledger, component, **kwargs)
        return dc is not None and dc.direction == DriftDirection.WORSENING
    return check


def any_drifting(ledger, **kwargs) -> PredicateFn:
    """True if any component in the ledger is in a non-STABLE drift state."""
    def check(expr: Expression) -> bool:
        all_dc = drift_all_from_ledger(ledger, **kwargs)
        return any(dc.state != DriftState.STABLE for dc in all_dc.values())
    return check


def any_drift_worsening(ledger, **kwargs) -> PredicateFn:
    """True if any component's drift direction is WORSENING."""
    def check(expr: Expression) -> bool:
        all_dc = drift_all_from_ledger(ledger, **kwargs)
        return any(dc.direction == DriftDirection.WORSENING for dc in all_dc.values())
    return check


def drift_accelerating(
    component: str,
    ledger,
    **kwargs,
) -> PredicateFn:
    """True if the named component is ACCELERATING."""
    return drift_is(component, DriftState.ACCELERATING, ledger, **kwargs)


# -----------------------------------------------------------------------
# Forecast composition — drift shape + forecast ETA
# -----------------------------------------------------------------------

@dataclass
class DriftForecast:
    """
    Combined view: drift tells you the trajectory SHAPE,
    forecast tells you the ETA to threshold crossings.

    drift:     trajectory classification (STABLE/DRIFTING/ACCELERATING/...)
    forecast:  linear projection with ETA to intact/ablated (None if insufficient data)
    """
    drift: DriftClassification
    forecast: Optional[object] = None  # Forecast (avoid circular import at class level)

    @property
    def component(self) -> str:
        return self.drift.component

    @property
    def summary(self) -> str:
        """Human-readable one-liner."""
        parts = [f"{self.drift.component}: {self.drift.state.value}({self.drift.direction.value})"]

        if self.forecast is not None:
            eta_a = self.forecast.eta_ablated
            eta_i = self.forecast.eta_intact
            if self.drift.worsening and eta_a is not None:
                parts.append(f"ETA ablated: {_fmt_timedelta(eta_a)}")
            elif self.drift.improving and eta_i is not None:
                parts.append(f"ETA intact: {_fmt_timedelta(eta_i)}")

        return ", ".join(parts)

    def to_dict(self) -> dict:
        d = {"drift": self.drift.to_dict()}
        if self.forecast is not None:
            d["forecast"] = self.forecast.to_dict()
        d["summary"] = self.summary
        return d

    def __repr__(self) -> str:
        return f"DriftForecast({self.summary})"


def _fmt_timedelta(td: timedelta) -> str:
    """Format a timedelta as a compact human string."""
    total_s = td.total_seconds()
    if total_s < 60:
        return f"{total_s:.0f}s"
    if total_s < 3600:
        return f"{total_s / 60:.1f}m"
    if total_s < 86400:
        return f"{total_s / 3600:.1f}h"
    return f"{total_s / 86400:.1f}d"


def drift_forecast(
    observations: list[Observation],
    thresholds,
    now: Optional[datetime] = None,
    **kwargs,
) -> Optional[DriftForecast]:
    """
    Classify drift AND project threshold crossings for one component.

    Args:
        observations:  timestamped observations for one component
        thresholds:    Thresholds for forecast projection
        now:           current time (for forecast)
        **kwargs:      passed to classify_drift (min_samples, noise_threshold, etc.)
    """
    from .forecast import forecast as _forecast

    dc = classify_drift(observations, **kwargs)
    if dc is None:
        return None

    fc = _forecast(observations, thresholds, now=now)
    return DriftForecast(drift=dc, forecast=fc)


def drift_forecast_from_ledger(
    ledger,
    component: str,
    thresholds,
    now: Optional[datetime] = None,
    **kwargs,
) -> Optional[DriftForecast]:
    """Classify drift + forecast from a Ledger's history."""
    obs = observations_from_ledger(ledger, component)
    return drift_forecast(obs, thresholds, now=now, **kwargs)
