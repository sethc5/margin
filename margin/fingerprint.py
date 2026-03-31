"""
Fingerprint: session statistics with noise-resistant target queries.

Monitor.fingerprint() returns a Fingerprint instead of a plain dict.
Subclasses dict — fully JSON-serializable, isinstance(fp, dict) is True,
and fp["component"]["mean"] still works unchanged.
"""

from __future__ import annotations

from typing import Optional


def _sorted_vals(values: list[float]) -> list[float]:
    return sorted(values)


def _percentile(values: list[float], p: float) -> float:
    """p-th percentile (0–100) via linear interpolation."""
    if not values:
        return 0.0
    s = _sorted_vals(values)
    n = len(s)
    if n == 1:
        return s[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _trimmed_mean(values: list[float], fraction: float = 0.1) -> float:
    """Mean after removing the top and bottom `fraction` of values."""
    if not values:
        return 0.0
    s = _sorted_vals(values)
    n = len(s)
    k = max(1, int(n * fraction))
    trimmed = s[k:-k] if 2 * k < n else s
    return sum(trimmed) / len(trimmed)


class Fingerprint(dict):
    """
    Session statistics from Monitor.fingerprint().

    Subclasses ``dict`` — fully JSON-serializable without a custom encoder::

        fp["cpu"]["mean"]     # works
        json.dumps(fp)        # works — serializes as {name: {mean, std, n, trend}}
        isinstance(fp, dict)  # True

    Richer noise-resistant queries::

        fp.robust_target("cpu")              # median (default)
        fp.robust_target("cpu", "trimmed")   # 10% trimmed mean
        fp.percentile("cpu", 25)             # 25th percentile

    Cross-session normalization::

        fp.sigma("cpu", value)               # (value − mean) / |mean|
        fp.robust_sigma("cpu", value)        # (value − median) / IQR

    The raw ``values`` dict stores per-component float lists from the drift
    window, enabling true median / percentile without re-scanning trackers.
    Raw values are ephemeral (not included in JSON / to_dict output).
    """

    def __init__(
        self,
        stats: dict[str, dict],
        values: Optional[dict[str, list[float]]] = None,
    ):
        super().__init__(stats)
        self._values: dict[str, list[float]] = values or {}

    # ------------------------------------------------------------------
    # Rich queries (everything else is inherited from dict)
    # ------------------------------------------------------------------

    def robust_target(self, component: str, method: str = "median") -> float:
        """
        Noise-resistant estimate of an achievable target for ``component``.

        ``method="median"``   — median of observed values (default; robust to outliers)
        ``method="trimmed"``  — 10% trimmed mean
        ``method="mean"``     — plain mean (same as ``fp[component]["mean"]``)

        When no raw values are stored, checks for pre-computed ``"median"`` in
        the stats dict (present when fingerprint came from ``Monitor.fingerprint()``),
        then falls back to ``"mean"``.
        """
        vals = self._values.get(component)
        if not vals:
            stats = self.get(component, {})
            if method == "median":
                return stats.get("median", stats.get("mean", 0.0))
            return stats.get("mean", 0.0)
        if method == "median":
            return _percentile(vals, 50)
        if method == "trimmed":
            return _trimmed_mean(vals, fraction=0.1)
        # "mean" or unknown
        return sum(vals) / len(vals)

    def percentile(self, component: str, p: float) -> float:
        """
        Return the p-th percentile (0–100) of observed values for ``component``.

        When no raw values are stored, checks for pre-computed ``"q25"`` / ``"q75"`` /
        ``"median"`` in the stats dict (present when fingerprint came from
        ``Monitor.fingerprint()``), then falls back to ``"mean"``.
        """
        vals = self._values.get(component)
        if not vals:
            stats = self.get(component, {})
            if p == 25 and "q25" in stats:
                return stats["q25"]
            if p == 75 and "q75" in stats:
                return stats["q75"]
            if p == 50 and "median" in stats:
                return stats["median"]
            return stats.get("mean", 0.0)
        return _percentile(vals, p)

    def sigma(self, component: str, value: float) -> float:
        """
        Z-score of ``value`` against the fingerprint's empirical distribution.

        Returns ``(value − mean) / std``  (standard z-score).

        Using ``std`` as denominator is correct for cross-session normalization.
        ``|mean|`` would only be appropriate when the coefficient of variation
        (std/|mean|) is small; for high-CV components (e.g. ``std=0.498``,
        ``mean=0.070``, CV=7.1) it amplifies by 7×, turning a normal ±0.07
        swing into ±1.0 and saturating the controller.

        Falls back to ``(value − mean) / |mean|`` when ``std`` is zero (constant
        window or not yet stored), and returns ``value`` unchanged when both
        ``mean`` and ``std`` are zero.

        Example::

            metric = fp.sigma("recovery_ratio", cq.recovery_ratio)
            alpha, reason = ctrl.step(alpha, metric)
        """
        stats = self.get(component, {})
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        if std != 0.0:
            return (value - mean) / std
        # std=0: constant distribution — fall back to relative deviation
        if mean == 0.0:
            return value
        return (value - mean) / abs(mean)

    def robust_sigma(self, component: str, value: float) -> float:
        """
        Normalize ``value`` against the fingerprint's empirical median, scaled by IQR.

        Returns ``(value − median) / IQR``.

        More stable than :meth:`sigma` when the distribution has high variance
        or outliers (e.g. ``recovery_ratio`` std=0.498 vs mean=0.070 — the
        mean-based sigma would amplify noise).

        Falls back to :meth:`sigma` when IQR is zero (constant window or no
        raw values stored).

        Example::

            metric = fp.robust_sigma("recovery_ratio", cq.recovery_ratio)
            alpha, reason = ctrl.step(alpha, metric)
        """
        median = self.robust_target(component, "median")
        q25 = self.percentile(component, 25)
        q75 = self.percentile(component, 75)
        iqr = q75 - q25
        if iqr == 0.0:
            return self.sigma(component, value)
        return (value - median) / iqr

    def n(self, component: str) -> int:
        """Number of observations for ``component`` in this fingerprint."""
        return self.get(component, {}).get("n", 0)

    def components(self) -> list[str]:
        """Sorted list of component names."""
        return sorted(self.keys())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain dict (stats only; raw values are ephemeral)."""
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        """Deserialize from a plain dict (no raw values; median falls back to mean)."""
        return cls(stats=d)

    def __repr__(self) -> str:
        return f"Fingerprint({len(self)} components, raw_values={bool(self._values)})"
