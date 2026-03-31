"""
Fingerprint: session statistics with noise-resistant target queries.

Monitor.fingerprint() returns a Fingerprint instead of a plain dict.
Subclasses dict — fully JSON-serializable, isinstance(fp, dict) is True,
and fp["component"]["mean"] still works unchanged.
"""

from __future__ import annotations

import math
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

    Noise-resistant queries::

        fp.robust_target("cpu")              # median (default)
        fp.robust_target("cpu", "trimmed")   # 10% trimmed mean
        fp.percentile("cpu", 25)             # 25th percentile

    Cross-session normalization::

        fp.sigma("cpu", value)               # z-score: (value − mean) / std
        fp.robust_sigma("cpu", value)        # (value − median) / IQR

    Neural conditioning::

        fp.to_tensor(["rr", "improvement"], ["mean", "std"])   # flat list
        fp.to_tensor(..., format="numpy")                       # np.ndarray
        fp.to_tensor(..., format="torch")                       # torch.Tensor

    Session comparison::

        fp.distance(other_fp)               # L2 in flattened (mean,std) space
        fp.kl_divergence(other_fp)          # symmetric KL of Gaussian components

    Online update::

        fp.update("recovery_ratio", 0.12)   # Welford incremental update

    Multi-session aggregation::

        fp.merge(other_fp, weight=0.5)      # weighted average of two fingerprints

    The raw ``_values`` dict stores per-component float lists from the drift
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
    # Inspection
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> list[str]:
        """Sorted list of components with at least one observation (n > 0)."""
        return [k for k in sorted(self.keys()) if self.get(k, {}).get("n", 0) > 0]

    # ------------------------------------------------------------------
    # Noise-resistant queries
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

    # ------------------------------------------------------------------
    # Cross-session normalization
    # ------------------------------------------------------------------

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
        """
        stats = self.get(component, {})
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        if std != 0.0:
            return (value - mean) / std
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
        """
        median = self.robust_target(component, "median")
        q25 = self.percentile(component, 25)
        q75 = self.percentile(component, 75)
        iqr = q75 - q25
        if iqr == 0.0:
            return self.sigma(component, value)
        return (value - median) / iqr

    # ------------------------------------------------------------------
    # Neural conditioning
    # ------------------------------------------------------------------

    def to_tensor(
        self,
        metrics: Optional[list[str]] = None,
        stats: tuple[str, ...] = ("mean", "std"),
        format: str = "list",
    ):
        """
        Export fingerprint as a flat vector for neural conditioning.

        Returns the values in row-major order:
        ``[metric0_stat0, metric0_stat1, metric1_stat0, ...]``

        Parameters
        ----------
        metrics: components to include, in order; ``None`` = ``sorted(self.keys())``
        stats:   which stat fields to include per component (default: mean and std)
        format:  ``"list"`` (default, no dependencies), ``"numpy"``, or ``"torch"``

        Example::

            # Flat 4-element list: [rr_mean, rr_std, imp_mean, imp_std]
            vec = fp.to_tensor(["recovery_ratio", "improvement"], ["mean", "std"])

            # As torch tensor for D_fp conditioning:
            t = fp.to_tensor(["recovery_ratio", "improvement"], format="torch")
        """
        _metrics = metrics if metrics is not None else sorted(self.keys())
        values = []
        for m in _metrics:
            s = self.get(m, {})
            for stat in stats:
                values.append(float(s.get(stat, 0.0)))

        if format == "list":
            return values
        if format == "numpy":
            import numpy as np
            return np.array(values, dtype=np.float32)
        if format == "torch":
            import torch
            return torch.tensor(values, dtype=torch.float32)
        raise ValueError(
            f"Unknown format {format!r}. Supported: 'list', 'numpy', 'torch'"
        )

    # ------------------------------------------------------------------
    # Session comparison
    # ------------------------------------------------------------------

    def distance(
        self,
        other: "Fingerprint",
        metrics: Optional[list[str]] = None,
        stats: tuple[str, ...] = ("mean", "std"),
    ) -> float:
        """
        L2 distance between two fingerprints in the flattened (mean, std) space.

        Uses the same vector layout as :meth:`to_tensor` — the distance is
        directly interpretable as distance in the tensor space used for neural
        conditioning.

        Only components present in both fingerprints are compared (``metrics``
        can restrict further).

        Example::

            # Find the Session 1 sentence fingerprint closest to current state
            best = min(session1_fps, key=lambda fp1: live_fp.distance(fp1))
        """
        _metrics = metrics if metrics is not None else sorted(
            set(self.keys()) & set(other.keys())
        )
        a = self.to_tensor(metrics=_metrics, stats=stats)
        b = other.to_tensor(metrics=_metrics, stats=stats)
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def kl_divergence(
        self,
        other: "Fingerprint",
        metrics: Optional[list[str]] = None,
        symmetric: bool = True,
    ) -> float:
        """
        KL divergence treating each component as an independent Gaussian N(mean, std²).

        When ``symmetric=True`` (default), returns the symmetric KL:
        ``0.5 * (KL(self || other) + KL(other || self))``.

        Components with std=0 are skipped (degenerate Gaussians).

        Example::

            drift = fp_session1.kl_divergence(fp_session2)
            if drift > 2.0:
                print("sessions differ significantly")
        """
        _metrics = metrics if metrics is not None else sorted(
            set(self.keys()) & set(other.keys())
        )

        def _kl_gaussians(p_fp: "Fingerprint", q_fp: "Fingerprint") -> float:
            total = 0.0
            for m in _metrics:
                mu_p = p_fp.get(m, {}).get("mean", 0.0)
                sig_p = max(p_fp.get(m, {}).get("std", 0.0), 1e-8)
                mu_q = q_fp.get(m, {}).get("mean", 0.0)
                sig_q = max(q_fp.get(m, {}).get("std", 0.0), 1e-8)
                # KL(N(μ_p,σ_p²) || N(μ_q,σ_q²))
                total += (
                    math.log(sig_q / sig_p)
                    + (sig_p ** 2 + (mu_p - mu_q) ** 2) / (2.0 * sig_q ** 2)
                    - 0.5
                )
            return total

        if symmetric:
            return 0.5 * (_kl_gaussians(self, other) + _kl_gaussians(other, self))
        return _kl_gaussians(self, other)

    def similarity(
        self,
        other: "Fingerprint",
        metrics: Optional[list[str]] = None,
        stats: tuple[str, ...] = ("mean", "std"),
    ) -> float:
        """
        Cosine similarity between two fingerprints in the flattened (mean, std) space.

        Returns a value in [−1, 1]; 1.0 = identical direction, 0.0 = orthogonal.
        """
        _metrics = metrics if metrics is not None else sorted(
            set(self.keys()) & set(other.keys())
        )
        a = self.to_tensor(metrics=_metrics, stats=stats)
        b = other.to_tensor(metrics=_metrics, stats=stats)
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, component: str, value: float) -> "Fingerprint":
        """
        Incrementally update statistics for ``component`` using Welford's algorithm.

        O(1), numerically stable.  Updates ``mean``, ``std``, and ``n`` in-place.
        Also appends to the raw values list if one exists (for percentile accuracy).

        Returns ``self`` for chaining.

        Example::

            for obs in live_observations:
                fp.update("recovery_ratio", obs.value)
        """
        stats = dict(self.get(component) or {
            "mean": 0.0, "std": 0.0, "n": 0, "trend": "UNKNOWN"
        })
        n_old = stats.get("n", 0)
        n_new = n_old + 1
        mean_old = stats.get("mean", 0.0)
        mean_new = mean_old + (value - mean_old) / n_new
        # Reconstruct M2 from stored std²*(n-1), update, recompute std
        m2_old = stats.get("std", 0.0) ** 2 * max(n_old - 1, 0)
        m2_new = m2_old + (value - mean_old) * (value - mean_new)
        std_new = (m2_new / (n_new - 1)) ** 0.5 if n_new >= 2 else 0.0
        stats["mean"] = mean_new
        stats["std"] = std_new
        stats["n"] = n_new
        self[component] = stats

        if component in self._values:
            self._values[component].append(value)

        return self

    # ------------------------------------------------------------------
    # Multi-session aggregation
    # ------------------------------------------------------------------

    def merge(self, other: "Fingerprint", weight: float = 0.5) -> "Fingerprint":
        """
        Weighted combination of two fingerprints.

        ``weight=0.5``: equal mix; ``weight=0.0``: returns copy of self;
        ``weight=1.0``: returns copy of other.

        Mean and std are linearly interpolated; n is summed.  Components
        present in only one fingerprint are included unchanged.

        Example::

            # Running average of Session 1 and Session 2
            combined = fp_session1.merge(fp_session2, weight=0.5)
        """
        w = max(0.0, min(1.0, weight))
        all_keys = set(self.keys()) | set(other.keys())
        merged: dict[str, dict] = {}
        for k in sorted(all_keys):
            s = self.get(k) or {}
            o = other.get(k) or {}
            if not s:
                merged[k] = dict(o)
            elif not o:
                merged[k] = dict(s)
            else:
                merged[k] = {
                    "mean": s.get("mean", 0.0) * (1 - w) + o.get("mean", 0.0) * w,
                    "std": s.get("std", 0.0) * (1 - w) + o.get("std", 0.0) * w,
                    "n": s.get("n", 0) + o.get("n", 0),
                    "trend": o.get("trend", s.get("trend", "UNKNOWN")),
                }
                # Carry over percentiles if both have them (interpolate)
                for key in ("median", "q25", "q75"):
                    if key in s and key in o:
                        merged[k][key] = s[key] * (1 - w) + o[key] * w
        return Fingerprint(stats=merged)

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    @classmethod
    def from_batch(
        cls,
        items,
        feature_fn,
    ) -> "Fingerprint":
        """
        Build a Fingerprint from a batch of items using Welford's algorithm.

        ``feature_fn`` extracts a ``{component: float}`` dict from each item.
        All components seen across the batch are aggregated.

        Example::

            # Fingerprint a batch of communities by genus composition
            fp = Fingerprint.from_batch(
                communities,
                feature_fn=lambda c: {"genus_diversity": c.shannon_h,
                                      "mean_quality":   c.mean_t025},
            )

        Parameters
        ----------
        items:      iterable of anything
        feature_fn: callable(item) → dict[str, float]
        """
        result = cls(stats={})
        for item in items:
            features = feature_fn(item)
            for component, value in features.items():
                result.update(component, float(value))
        return result

    # ------------------------------------------------------------------
    # Ordering / retrieval
    # ------------------------------------------------------------------

    @classmethod
    def optimal_ordering(
        cls,
        items: list,
        key_fn,
        metrics: Optional[list[str]] = None,
        stats: tuple[str, ...] = ("mean", "std"),
    ) -> list:
        """
        Reorder ``items`` to maximise cache reuse via greedy nearest-neighbor.

        ``key_fn`` extracts a :class:`Fingerprint` from each item.  The
        algorithm starts from the item whose fingerprint is closest to the
        batch centroid, then always picks the unvisited item closest to the
        current one (greedy TSP approximation).

        This is a classmethod so it can be called without an instance::

            ordered = Fingerprint.optimal_ordering(batches, key_fn=lambda b: b.fp)

        Parameters
        ----------
        items:    list of objects to reorder
        key_fn:   callable(item) → Fingerprint
        metrics:  restrict distance calculation to these components;
                  ``None`` = all shared keys between each pair
        stats:    which stat fields to use for distance (default: mean and std)

        Returns
        -------
        Reordered copy of ``items``.

        Example::

            ordered_batches = Fingerprint.optimal_ordering(
                batches,
                key_fn=lambda b: b.fingerprint,
                metrics=["genus_diversity", "mean_quality"],
            )
        """
        if not items:
            return []
        if len(items) == 1:
            return list(items)

        fps = [key_fn(item) for item in items]

        # Compute centroid fingerprint from all stat means
        all_keys: set[str] = set()
        for fp in fps:
            all_keys.update(fp.keys())
        _metrics = sorted(metrics) if metrics is not None else sorted(all_keys)

        centroid_vals: dict[str, dict] = {}
        for m in _metrics:
            vals_m = [fp.get(m, {}).get("mean", 0.0) for fp in fps]
            stds_m = [fp.get(m, {}).get("std", 0.0) for fp in fps]
            centroid_vals[m] = {
                "mean": sum(vals_m) / len(vals_m),
                "std": sum(stds_m) / len(stds_m),
                "n": 1,
            }
        centroid = cls(stats=centroid_vals)

        # Start from the item closest to the centroid
        start = min(range(len(fps)), key=lambda i: centroid.distance(fps[i], metrics=_metrics, stats=stats))

        visited = [False] * len(items)
        order = [start]
        visited[start] = True

        while len(order) < len(items):
            current_fp = fps[order[-1]]
            best_idx, best_dist = -1, float("inf")
            for j, fp_j in enumerate(fps):
                if visited[j]:
                    continue
                d = current_fp.distance(fp_j, metrics=_metrics, stats=stats)
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            order.append(best_idx)
            visited[best_idx] = True

        return [items[i] for i in order]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def n(self, component: str) -> int:
        """Number of observations for ``component`` in this fingerprint."""
        return self.get(component, {}).get("n", 0)

    def components(self) -> list[str]:
        """Sorted list of all component names (including n=0)."""
        return sorted(self.keys())

    def to_dict(self) -> dict:
        """Plain dict (stats only; raw values are ephemeral)."""
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        """Deserialize from a plain dict (no raw values; median falls back to mean)."""
        return cls(stats=d)

    def __repr__(self) -> str:
        return f"Fingerprint({len(self)} components, raw_values={bool(self._values)})"
