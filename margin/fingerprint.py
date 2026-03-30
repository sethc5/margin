"""
Fingerprint: session statistics with noise-resistant target queries.

Monitor.fingerprint() returns a Fingerprint instead of a plain dict.
Backward-compatible: fp["component"]["mean"] still works.
Richer: fp.robust_target("component"), fp.percentile("component", 50).
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


class Fingerprint:
    """
    Session statistics from Monitor.fingerprint().

    Supports dict-like access for backward compatibility::

        fp["cpu"]["mean"]     # still works
        fp["cpu"]["std"]      # still works

    Richer noise-resistant queries::

        fp.robust_target("cpu")              # median (default)
        fp.robust_target("cpu", "trimmed")   # 10% trimmed mean
        fp.percentile("cpu", 25)             # 25th percentile

    Construction:

        stats  — {component: {mean, std, n, trend}}  (same as old dict return)
        values — {component: [raw float values]}     (from drift window)

    The ``values`` dict enables true median / percentile without re-scanning
    every tracker observation at call time.
    """

    def __init__(
        self,
        stats: dict[str, dict],
        values: Optional[dict[str, list[float]]] = None,
    ):
        self._stats = stats
        self._values: dict[str, list[float]] = values or {}

    # ------------------------------------------------------------------
    # Dict-like interface (backward compat)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> dict:
        return self._stats[key]

    def __contains__(self, key: object) -> bool:
        return key in self._stats

    def __iter__(self):
        return iter(self._stats)

    def __len__(self) -> int:
        return len(self._stats)

    def keys(self):
        return self._stats.keys()

    def items(self):
        return self._stats.items()

    def values(self):
        """Return stat dicts for all components (mirrors dict.values())."""
        return self._stats.values()

    def get(self, key: str, default=None):
        return self._stats.get(key, default)

    # ------------------------------------------------------------------
    # Rich queries
    # ------------------------------------------------------------------

    def robust_target(self, component: str, method: str = "median") -> float:
        """
        Noise-resistant estimate of an achievable target for ``component``.

        ``method="median"``   — median of observed values (default; robust to outliers)
        ``method="trimmed"``  — 10% trimmed mean
        ``method="mean"``     — plain mean (same as ``fp[component]["mean"]``)

        Falls back to ``fp[component]["mean"]`` when no raw values are stored.
        """
        vals = self._values.get(component)
        if not vals:
            return self._stats.get(component, {}).get("mean", 0.0)
        if method == "median":
            return _percentile(vals, 50)
        if method == "trimmed":
            return _trimmed_mean(vals, fraction=0.1)
        # "mean" or unknown
        return sum(vals) / len(vals)

    def percentile(self, component: str, p: float) -> float:
        """
        Return the p-th percentile (0–100) of observed values for ``component``.

        Falls back to the stored mean when no raw values are available.
        """
        vals = self._values.get(component)
        if not vals:
            return self._stats.get(component, {}).get("mean", 0.0)
        return _percentile(vals, p)

    def n(self, component: str) -> int:
        """Number of observations for ``component`` in this fingerprint."""
        return self._stats.get(component, {}).get("n", 0)

    def components(self) -> list[str]:
        """Sorted list of component names."""
        return sorted(self._stats.keys())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict (stats only; raw values are ephemeral)."""
        return dict(self._stats)

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        """Deserialize from a plain dict (no raw values; median falls back to mean)."""
        return cls(stats=d)

    def __repr__(self) -> str:
        return f"Fingerprint({len(self._stats)} components, raw_values={bool(self._values)})"
