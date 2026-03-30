"""
Proprioceptive feedback infrastructure for the margin-poc Layer 0-7 roadmap.

This module provides margin-native tooling for the proprioceptive
feedback loop: monitoring the health of the process signal itself,
tracking signal drift across inference steps, detecting anomalous
activations, and validating correction quality.

The margin-poc PoC at /home/seth/dev/llm_arch/margin-poc/ implements
the raw mechanism (ProjectionHead → ResidDeltaNet → injection).
This module adds typed health classification ON TOP of that mechanism
so each layer of the roadmap gets monitoring for free.

Layer 0: Signal health (is the projection head producing good signal?)
Layer 1: Per-dimension health (are individual signal dimensions degrading?)
Layer 2: Temporal trajectory health (is the signal window drifting?)
Layer 3: Persistence health (is the stored fingerprint still valid?)
Layer 4: Multi-model signal health (are agent signals compatible?)
Layer 5: Write channel health (are supervised injections having effect?)
Layer 6: Projection evolution health (is calibration improving?)
Layer 7: Self-knowledge health (are epistemic states well-calibrated?)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from margin.health import Health, Thresholds, classify
from margin.confidence import Confidence
from margin.observation import Observation, Expression, Parser
from margin.streaming import Monitor, DriftTracker, AnomalyTracker
from margin.drift import DriftClassification, DriftState, DriftDirection
from margin.anomaly import AnomalyClassification, AnomalyState
from margin.intent import Intent, Feasibility


# -----------------------------------------------------------------------
# Signal dimension definitions (from ProjectionConfig.SIGNAL_NAMES)
# -----------------------------------------------------------------------

SIGNAL_DIMENSIONS = [
    "confidence", "novelty", "coherence", "retrieval_mode",
    "converging", "drifting", "accelerating", "stalling",
    "epistemic_uncertainty", "aleatoric_uncertainty",
    "superposition_density", "boundary_proximity",
    "reserved_0", "reserved_1", "reserved_2", "reserved_3",
]

# Semantic groups
CORE_STATE = SIGNAL_DIMENSIONS[0:4]
TRAJECTORY = SIGNAL_DIMENSIONS[4:8]
UNCERTAINTY_TEXTURE = SIGNAL_DIMENSIONS[8:12]
RESERVED = SIGNAL_DIMENSIONS[12:16]

# Per-dimension thresholds
# All signals are in [-1, 1]. For each, define what "healthy" means.
SIGNAL_THRESHOLDS: dict[str, Thresholds] = {
    # Core state: higher = healthier
    "confidence": Thresholds(intact=0.3, ablated=-0.3, higher_is_better=True),
    "novelty": Thresholds(intact=-0.5, ablated=0.5, higher_is_better=False),  # low novelty = familiar territory
    "coherence": Thresholds(intact=0.3, ablated=-0.3, higher_is_better=True),
    "retrieval_mode": Thresholds(intact=0.2, ablated=-0.5, higher_is_better=True),  # positive = retrieval (grounded)

    # Trajectory: polarity depends on dimension
    "converging": Thresholds(intact=0.2, ablated=-0.3, higher_is_better=True),
    "drifting": Thresholds(intact=-0.2, ablated=0.3, higher_is_better=False),   # less drift = healthier
    "accelerating": Thresholds(intact=0.1, ablated=-0.3, higher_is_better=True),
    "stalling": Thresholds(intact=-0.2, ablated=0.3, higher_is_better=False),   # less stalling = healthier

    # Uncertainty: lower = healthier (less uncertain)
    "epistemic_uncertainty": Thresholds(intact=0.3, ablated=0.7, higher_is_better=False),
    "aleatoric_uncertainty": Thresholds(intact=0.3, ablated=0.7, higher_is_better=False),
    "superposition_density": Thresholds(intact=0.4, ablated=0.8, higher_is_better=False),
    "boundary_proximity": Thresholds(intact=0.3, ablated=0.7, higher_is_better=False),
}


# -----------------------------------------------------------------------
# Signal health monitor
# -----------------------------------------------------------------------

def make_signal_parser(dimensions: Optional[list[str]] = None) -> Parser:
    """
    Build a margin Parser for process signal dimensions.

    Each dimension gets polarity-correct thresholds from SIGNAL_THRESHOLDS.
    Unknown dimensions default to higher-is-better with [-0.3, 0.3] thresholds.
    """
    dims = dimensions or SIGNAL_DIMENSIONS
    baselines = {}
    component_thresholds = {}

    for dim in dims:
        baselines[dim] = 0.0  # baseline = neutral signal
        if dim in SIGNAL_THRESHOLDS:
            component_thresholds[dim] = SIGNAL_THRESHOLDS[dim]
        else:
            component_thresholds[dim] = Thresholds(intact=0.3, ablated=-0.3, higher_is_better=True)

    default_t = Thresholds(intact=0.3, ablated=-0.3, higher_is_better=True)
    return Parser(baselines=baselines, thresholds=default_t,
                  component_thresholds=component_thresholds)


def make_signal_monitor(
    dimensions: Optional[list[str]] = None,
    window: int = 200,
) -> Monitor:
    """Build a streaming Monitor for process signal health."""
    parser = make_signal_parser(dimensions)
    return Monitor(parser, window=window)


# -----------------------------------------------------------------------
# Correction quality monitoring
# -----------------------------------------------------------------------

@dataclass
class CorrectionQuality:
    """
    Health assessment of a ResidDeltaNet correction step.

    gap_before:      logit gap before correction
    gap_after:       logit gap after correction
    alpha:           correction intensity used
    recovery_ratio:  gap_after / gap_intact
    improvement:     gap_after - gap_before (positive = better)
    health:          typed health of the correction outcome
    """
    circuit: str
    gap_before: float
    gap_after: float
    gap_intact: float
    alpha: float
    recovery_ratio: float
    improvement: float
    health: Health

    @property
    def beneficial(self) -> bool:
        return self.improvement > 0

    @property
    def harmful(self) -> bool:
        return self.improvement < 0

    def to_dict(self) -> dict:
        return {
            "circuit": self.circuit,
            "gap_before": round(self.gap_before, 4),
            "gap_after": round(self.gap_after, 4),
            "gap_intact": round(self.gap_intact, 4),
            "alpha": round(self.alpha, 4),
            "recovery_ratio": round(self.recovery_ratio, 4),
            "improvement": round(self.improvement, 4),
            "health": self.health.value,
            "beneficial": self.beneficial,
        }


def assess_correction(
    circuit: str,
    gap_before: float,
    gap_after: float,
    gap_intact: float,
    alpha: float,
    thresholds: Optional[Thresholds] = None,
) -> CorrectionQuality:
    """
    Assess the quality of a ResidDeltaNet correction.

    Uses pythia-6.9b defaults if no thresholds provided.
    """
    t = thresholds or Thresholds(intact=3.5, ablated=1.5)
    recovery = gap_after / gap_intact if gap_intact != 0 else 0.0
    improvement = gap_after - gap_before
    health = classify(gap_after, Confidence.HIGH, thresholds=t)

    return CorrectionQuality(
        circuit=circuit,
        gap_before=gap_before,
        gap_after=gap_after,
        gap_intact=gap_intact,
        alpha=alpha,
        recovery_ratio=recovery,
        improvement=improvement,
        health=health,
    )


# -----------------------------------------------------------------------
# Layer-specific intents
# -----------------------------------------------------------------------

def layer0_intent(deadline_seconds: float = 60.0) -> Intent:
    """
    Layer 0 success criteria: single proprioceptive channel working.

    - Confidence signal above noise floor
    - Epistemic uncertainty below saturation
    - Correction improving gap (not harmful)
    """
    return (Intent(goal="Layer 0: proprioceptive channel operational", deadline_seconds=deadline_seconds)
            .require("confidence", min_value=0.1, min_health=Health.DEGRADED)
            .require("epistemic_uncertainty", min_health=Health.DEGRADED)  # lower is better — health gate handles polarity
            .require("coherence", min_health=Health.DEGRADED, critical=False))


def layer1_intent(deadline_seconds: float = 60.0) -> Intent:
    """
    Layer 1 success criteria: richer signal vocabulary.

    All 4 core state dimensions independently healthy.
    """
    return (Intent(goal="Layer 1: rich signal vocabulary", deadline_seconds=deadline_seconds)
            .require("confidence", min_value=0.1, min_health=Health.DEGRADED)
            .require("novelty", min_value=0.5, min_health=Health.DEGRADED)
            .require("coherence", min_value=0.1, min_health=Health.DEGRADED)
            .require("retrieval_mode", min_health=Health.DEGRADED, critical=False))


def layer2_intent(deadline_seconds: float = 300.0) -> Intent:
    """
    Layer 2 success criteria: temporal trajectory healthy.

    Trajectory dimensions should show converging (not drifting/stalling).
    """
    return (Intent(goal="Layer 2: temporal depth operational", deadline_seconds=deadline_seconds)
            .require("converging", min_value=0.0, min_health=Health.DEGRADED)
            .require("drifting", min_health=Health.DEGRADED)   # lower-is-better; health gate handles polarity
            .require("stalling", min_health=Health.DEGRADED)   # lower-is-better; health gate handles polarity
            .require("accelerating", min_health=Health.DEGRADED, critical=False))


def layer3_intent() -> Intent:
    """
    Layer 3 success criteria: persistent identity intact.

    After session resume, signal should be coherent (not OOD).
    """
    return (Intent(goal="Layer 3: dispositional memory valid")
            .require("confidence", min_health=Health.DEGRADED)
            .require("coherence", min_health=Health.DEGRADED)
            .require("boundary_proximity", min_value=0.7, min_health=Health.DEGRADED))


# -----------------------------------------------------------------------
# Convenience: full proprioception health check
# -----------------------------------------------------------------------

def check_signal_health(
    signal_values: dict[str, float],
    monitor: Optional[Monitor] = None,
    intent: Optional[Intent] = None,
) -> dict:
    """
    One-call health check for a process signal.

    Args:
        signal_values: {dimension_name: value} from ProjectionHead output
        monitor:       optional streaming Monitor (creates one if not provided)
        intent:        optional Intent to evaluate (defaults to layer0_intent)

    Returns dict with expression, drift, anomaly, and intent results.
    """
    if monitor is None:
        monitor = make_signal_monitor()

    expr = monitor.update(signal_values)

    result = {
        "expression": expr.to_string(),
        "step": monitor.step,
        "drift": {},
        "anomaly": {},
    }

    for dim in signal_values:
        dc = monitor.drift(dim)
        if dc and dc.state != DriftState.STABLE:
            result["drift"][dim] = {"state": dc.state.value, "direction": dc.direction.value}
        ac = monitor.anomaly(dim)
        if ac and ac.state != AnomalyState.EXPECTED:
            result["anomaly"][dim] = {"state": ac.state.value, "z_score": round(ac.z_score, 2)}

    if intent is None:
        intent = layer0_intent()

    drift_map = {}
    for dim in signal_values:
        dc = monitor.drift(dim)
        if dc:
            drift_map[dim] = dc
    intent_result = intent.evaluate(expr, drift_map)
    result["intent"] = intent_result.to_dict()

    return result
