"""
Streaming trackers: incremental health, drift, anomaly, and correlation.

Instead of recomputing from a full observation list each time, trackers
maintain a bounded window and update classification on each new value.

    tracker = DriftTracker("cpu")
    tracker.update(observation)
    tracker.state  # DriftState.DRIFTING
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class WindowConfig:
    """
    Per-concern window sizes for Monitor.

    drift:        window for DriftTracker (how many steps to classify trajectory)
    anomaly:      window for AnomalyTracker (reference window for outlier detection)
    correlation:  window for CorrelationTracker (how many steps for pairwise correlation)

    All default to None, meaning "inherit the Monitor's base `window` parameter."
    Rule of thumb: anomaly ≈ 4× drift, correlation ≈ 10× drift.
    """
    drift: Optional[int] = None
    anomaly: Optional[int] = None
    correlation: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize to plain dict; omits None fields."""
        d: dict = {}
        if self.drift is not None:
            d["drift"] = self.drift
        if self.anomaly is not None:
            d["anomaly"] = self.anomaly
        if self.correlation is not None:
            d["correlation"] = self.correlation
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'WindowConfig':
        """Deserialize from a plain dict produced by to_dict()."""
        return cls(
            drift=d.get("drift"),
            anomaly=d.get("anomaly"),
            correlation=d.get("correlation"),
        )

    def __repr__(self) -> str:
        parts = [f"{k}={v}" for k, v in
                 [("drift", self.drift), ("anomaly", self.anomaly), ("correlation", self.correlation)]
                 if v is not None]
        return f"WindowConfig({', '.join(parts)})" if parts else "WindowConfig()"

from .confidence import Confidence
from .health import Health, Thresholds, classify
from .observation import Observation, Expression
from .drift import (
    DriftState, DriftDirection, DriftClassification,
    classify_drift,
)
from .anomaly import (
    AnomalyState, AnomalyClassification,
    classify_anomaly, detect_jumps, Jump,
)
from .correlate import (
    Correlation, CorrelationMatrix,
    correlate as _correlate,
)


# -----------------------------------------------------------------------
# Drift tracker
# -----------------------------------------------------------------------

class DriftTracker:
    """
    Incremental drift classification for one component.

    Maintains a bounded window of observations and reclassifies on each
    update. Window size is O(1) constant, so updates are bounded-time.

    Usage:
        tracker = DriftTracker("cpu", window=50)
        tracker.update(observation)
        tracker.state       # DriftState
        tracker.direction   # DriftDirection
        tracker.classification  # full DriftClassification or None
    """

    def __init__(
        self,
        component: str,
        window: int = 100,
        min_samples: int = 3,
        noise_threshold: float = 1.5,
        accel_threshold: float = 0.1,
    ):
        self.component = component
        self.window = window
        self._observations: deque[Observation] = deque(maxlen=window)
        self._min_samples = min_samples
        self._noise_threshold = noise_threshold
        self._accel_threshold = accel_threshold
        self._classification: Optional[DriftClassification] = None

    def update(self, observation: Observation) -> Optional[DriftClassification]:
        """Add an observation and reclassify. Returns the new classification."""
        self._observations.append(observation)
        self._classification = classify_drift(
            list(self._observations),
            min_samples=self._min_samples,
            noise_threshold=self._noise_threshold,
            accel_threshold=self._accel_threshold,
        )
        return self._classification

    def update_value(
        self,
        value: float,
        baseline: float,
        higher_is_better: bool = True,
        confidence: Confidence = Confidence.HIGH,
        measured_at: Optional[datetime] = None,
    ) -> Optional[DriftClassification]:
        """Convenience: update from a raw value instead of an Observation."""
        obs = Observation(
            name=self.component,
            health=Health.INTACT,  # placeholder — drift doesn't use health state
            value=value,
            baseline=baseline,
            confidence=confidence,
            higher_is_better=higher_is_better,
            measured_at=measured_at or datetime.now(),
        )
        return self.update(obs)

    @property
    def state(self) -> DriftState:
        if self._classification is None:
            return DriftState.STABLE
        return self._classification.state

    @property
    def direction(self) -> DriftDirection:
        if self._classification is None:
            return DriftDirection.NEUTRAL
        return self._classification.direction

    @property
    def classification(self) -> Optional[DriftClassification]:
        return self._classification

    @property
    def n_observations(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        """Clear the window."""
        self._observations.clear()
        self._classification = None

    def __repr__(self) -> str:
        return f"DriftTracker({self.component}: {self.state.value}, n={self.n_observations})"


# -----------------------------------------------------------------------
# Anomaly tracker
# -----------------------------------------------------------------------

class AnomalyTracker:
    """
    Incremental anomaly classification for one component.

    Maintains a bounded reference window. New values are classified
    against the reference, then added to it (sliding reference).

    Usage:
        tracker = AnomalyTracker("cpu", window=100)
        tracker.update(42.0)
        tracker.state       # AnomalyState
        tracker.classification  # full AnomalyClassification or None
    """

    def __init__(
        self,
        component: str,
        window: int = 100,
        unusual_threshold: float = 2.0,
        anomalous_threshold: float = 3.0,
        novel_margin: float = 0.1,
        min_reference: int = 10,
        jump_threshold: float = 3.0,
    ):
        self.component = component
        self.window = window
        self._values: deque[float] = deque(maxlen=window)
        self._observations: deque[Observation] = deque(maxlen=window)
        self._unusual_threshold = unusual_threshold
        self._anomalous_threshold = anomalous_threshold
        self._novel_margin = novel_margin
        self._min_reference = min_reference
        self._jump_threshold = jump_threshold
        self._classification: Optional[AnomalyClassification] = None
        self._last_jump: Optional[Jump] = None

    def update(self, value: float) -> Optional[AnomalyClassification]:
        """Classify value against reference window, then add to window."""
        # Classify against existing reference (excluding this value)
        ref = list(self._values)
        if len(ref) >= self._min_reference:
            self._classification = classify_anomaly(
                value, ref,
                component=self.component,
                unusual_threshold=self._unusual_threshold,
                anomalous_threshold=self._anomalous_threshold,
                novel_margin=self._novel_margin,
            )
        else:
            self._classification = None

        # Jump detection: check last few values + new one
        obs = Observation(
            name=self.component, health=Health.INTACT, value=value,
            baseline=0.0, confidence=Confidence.HIGH,
        )
        self._observations.append(obs)
        self._values.append(value)

        if len(self._observations) >= 3:
            recent = list(self._observations)[-5:]  # last 5 for jump context
            jumps = detect_jumps(recent, jump_threshold=self._jump_threshold)
            self._last_jump = jumps[-1] if jumps else None
        else:
            self._last_jump = None

        return self._classification

    def update_obs(self, observation: Observation) -> Optional[AnomalyClassification]:
        """Update from an Observation object."""
        return self.update(observation.value)

    @property
    def state(self) -> AnomalyState:
        if self._classification is None:
            return AnomalyState.EXPECTED
        return self._classification.state

    @property
    def classification(self) -> Optional[AnomalyClassification]:
        return self._classification

    @property
    def last_jump(self) -> Optional[Jump]:
        return self._last_jump

    @property
    def n_values(self) -> int:
        return len(self._values)

    def reset(self) -> None:
        self._values.clear()
        self._observations.clear()
        self._classification = None
        self._last_jump = None

    def __repr__(self) -> str:
        return f"AnomalyTracker({self.component}: {self.state.value}, n={self.n_values})"


# -----------------------------------------------------------------------
# Correlation tracker
# -----------------------------------------------------------------------

class CorrelationTracker:
    """
    Incremental correlation tracking across multiple components.

    Maintains aligned value windows and recomputes pairwise correlations
    on each update.

    Usage:
        tracker = CorrelationTracker(["cpu", "mem", "disk"], window=50)
        tracker.update({"cpu": 80, "mem": 60, "disk": 45})
        tracker.matrix      # CorrelationMatrix
        tracker.strongest() # top correlations
    """

    def __init__(
        self,
        components: list[str],
        window: int = 100,
        min_correlation: float = 0.7,
        min_samples: int = 5,
    ):
        self.components = sorted(components)
        self.window = window
        self._series: dict[str, deque[float]] = {c: deque(maxlen=window) for c in self.components}
        self._n_updates = 0
        self._min_correlation = min_correlation
        self._min_samples = min_samples
        self._matrix: Optional[CorrelationMatrix] = None

    def update(self, values: dict[str, float]) -> Optional[CorrelationMatrix]:
        """
        Update with new values for all components.

        All components must be present in `values` for the update to count
        (maintains alignment). Missing components are silently skipped.
        """
        if not all(c in values for c in self.components):
            return self._matrix

        for c in self.components:
            self._series[c].append(values[c])
        self._n_updates += 1

        # Recompute if enough data
        min_len = min(len(self._series[c]) for c in self.components)
        if min_len >= self._min_samples:
            series = {c: list(self._series[c]) for c in self.components}
            self._matrix = _correlate(
                series,
                min_correlation=self._min_correlation,
                min_samples=self._min_samples,
            )
        return self._matrix

    @property
    def matrix(self) -> Optional[CorrelationMatrix]:
        return self._matrix

    def strongest(self, n: int = 5) -> list[Correlation]:
        if self._matrix is None:
            return []
        return self._matrix.strongest(n)

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def reset(self) -> None:
        for c in self.components:
            self._series[c].clear()
        self._n_updates = 0
        self._matrix = None

    def __repr__(self) -> str:
        n_corr = len(self._matrix.correlations) if self._matrix else 0
        return f"CorrelationTracker({len(self.components)} components, {n_corr} correlations, n={self._n_updates})"


# -----------------------------------------------------------------------
# Monitor — unified streaming tracker
# -----------------------------------------------------------------------

class Monitor:
    """
    Unified streaming monitor: health + drift + anomaly + correlation.

    Wraps a Parser and streaming trackers. One `update()` call classifies
    health, updates drift trajectories, checks for anomalies, and tracks
    correlations across all components.

    Usage:
        monitor = Monitor(parser, window=50)
        monitor.update({"cpu": 80, "mem": 60, "error_rate": 0.03})
        monitor.expression      # current Expression
        monitor.drift("cpu")    # DriftClassification
        monitor.anomaly("cpu")  # AnomalyClassification
        monitor.correlations    # CorrelationMatrix
        monitor.status()        # full snapshot dict
    """

    def __init__(
        self,
        parser,
        window: int = 100,
        drift_window: Optional[int] = None,
        anomaly_window: Optional[int] = None,
        correlation_window: Optional[int] = None,
        window_config: Optional[WindowConfig] = None,
        min_correlation: float = 0.7,
        anomaly_min_reference: int = 10,
        features: Optional[set] = None,
    ):
        self.parser = parser
        self.window = window

        # Resolve per-tracker windows: window_config > named params > base window
        cfg = window_config or WindowConfig()
        self.drift_window = cfg.drift or drift_window or window
        self.anomaly_window = cfg.anomaly or anomaly_window or window
        self.correlation_window = cfg.correlation or correlation_window or window

        # Resolve feature flags: None means all enabled (backward compatible)
        self.features = frozenset(features) if features is not None else frozenset({"drift", "anomaly", "correlation"})

        if "anomaly" in self.features and anomaly_min_reference > self.anomaly_window:
            warnings.warn(
                f"Monitor: anomaly_min_reference={anomaly_min_reference} > "
                f"anomaly_window={self.anomaly_window} — AnomalyTrackers will never "
                "classify because the reference window can never be filled; raise "
                "anomaly_window or lower anomaly_min_reference.",
                stacklevel=2,
            )

        self._expression: Optional[Expression] = None
        self._step = 0

        # Create per-component trackers for enabled features only
        self._drift_trackers: dict[str, DriftTracker] = {}
        self._anomaly_trackers: dict[str, AnomalyTracker] = {}

        if "drift" in self.features:
            for name in parser.baselines:
                self._drift_trackers[name] = DriftTracker(name, window=self.drift_window)

        if "anomaly" in self.features:
            for name in parser.baselines:
                self._anomaly_trackers[name] = AnomalyTracker(
                    name, window=self.anomaly_window, min_reference=anomaly_min_reference,
                )

        if "correlation" in self.features:
            self._correlation_tracker: Optional[CorrelationTracker] = CorrelationTracker(
                list(parser.baselines.keys()),
                window=self.correlation_window,
                min_correlation=min_correlation,
            )
        else:
            self._correlation_tracker = None

    def update(
        self,
        values: dict[str, float],
        label: str = "",
        now: Optional[datetime] = None,
        confidences=None,
        provenance=None,
    ) -> Expression:
        """
        Process new measurements: classify health, update drift, check anomalies.

        Returns the current Expression.
        """
        now = now or datetime.now()

        # Health classification via Parser
        self._expression = self.parser.parse(
            values, label=label, step=self._step,
            confidences=confidences, provenance=provenance,
        )
        self._step += 1

        # Update per-component trackers
        for obs in self._expression.observations:
            name = obs.name
            baseline = self.parser.baselines.get(name, obs.value)
            thresholds = self.parser._thresholds_for(name)

            # Drift
            if name in self._drift_trackers:
                stamped = Observation(
                    name=obs.name, health=obs.health, value=obs.value,
                    baseline=obs.baseline, confidence=obs.confidence,
                    higher_is_better=obs.higher_is_better, measured_at=now,
                )
                self._drift_trackers[name].update(stamped)

            # Anomaly
            if name in self._anomaly_trackers:
                self._anomaly_trackers[name].update(obs.value)

        # Correlation
        if self._correlation_tracker is not None:
            self._correlation_tracker.update(values)

        return self._expression

    @property
    def expression(self) -> Optional[Expression]:
        return self._expression

    def drift(self, component: str) -> Optional[DriftClassification]:
        t = self._drift_trackers.get(component)
        return t.classification if t else None

    def anomaly(self, component: str) -> Optional[AnomalyClassification]:
        t = self._anomaly_trackers.get(component)
        return t.classification if t else None

    @property
    def correlations(self) -> Optional[CorrelationMatrix]:
        if self._correlation_tracker is None:
            return None
        return self._correlation_tracker.matrix

    @property
    def step(self) -> int:
        return self._step

    def status(self) -> dict:
        """Full snapshot: health + drift + anomaly + correlations.

        Keys are always present; disabled features or no-data trackers
        return empty dicts so callers can index without key-existence checks.
        """
        d: dict = {
            "step": self._step,
            "expression": self._expression.to_dict() if self._expression else None,
            "drift": {},
            "anomaly": {},
            "correlations": {},
        }
        if "drift" in self.features:
            for name, tracker in self._drift_trackers.items():
                if tracker.classification:
                    d["drift"][name] = tracker.classification.to_dict()
        if "anomaly" in self.features:
            for name, tracker in self._anomaly_trackers.items():
                if tracker.classification:
                    d["anomaly"][name] = tracker.classification.to_dict()
        if self._correlation_tracker is not None and self._correlation_tracker.matrix:
            d["correlations"] = self._correlation_tracker.matrix.to_dict()
        return d

    def reset_anomaly_reference(self, components: Optional[list[str]] = None) -> None:
        """
        Clear AnomalyTracker reference windows for the specified components.

        Call after recalibrate_parser() to prevent old-baseline data from
        contaminating anomaly classification during the warm-up period that
        follows recalibration. The trackers will return None classifications
        until they accumulate min_reference new values under the new baseline.

        Args:
            components: names to reset; None resets all components.
        """
        targets = components or list(self._anomaly_trackers.keys())
        for name in targets:
            if name in self._anomaly_trackers:
                self._anomaly_trackers[name].reset()

    def reset(self) -> None:
        self._expression = None
        self._step = 0
        for t in self._drift_trackers.values():
            t.reset()
        for t in self._anomaly_trackers.values():
            t.reset()
        if self._correlation_tracker is not None:
            self._correlation_tracker.reset()

    def __repr__(self) -> str:
        windows_uniform = (self.drift_window == self.anomaly_window == self.correlation_window)
        if windows_uniform:
            w = f"window={self.drift_window}"
        else:
            w = f"drift={self.drift_window}, anomaly={self.anomaly_window}, corr={self.correlation_window}"
        all_features = frozenset({"drift", "anomaly", "correlation"})
        if self.features == all_features:
            feat = ""
        else:
            feat = f", features={{{', '.join(sorted(self.features))}}}"
        return f"Monitor(step={self._step}, {len(self._drift_trackers)} components, {w}{feat})"
