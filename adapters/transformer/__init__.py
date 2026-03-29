"""
Transformer circuit adapter for margin.

Typed health classification for mechanistic interpretability of
transformer circuits (IOI, name-mover, induction-head, suppressor).

Includes streaming monitor with drift/anomaly tracking, known causal
structure for IOI circuits, and auto-correlation discovery.
"""

from .signal import ProcessSignal, DEFAULT_SIGNAL_NAMES
from .circuit import CircuitState, gates_to_confidence
from .parsers import make_pythia_parser, make_from_sweep
from .causal_templates import IOI_GRAPH, make_circuit_graph
from .monitor import CircuitMonitor
from .proprioception import (
    SIGNAL_DIMENSIONS, CORE_STATE, TRAJECTORY, UNCERTAINTY_TEXTURE,
    SIGNAL_THRESHOLDS,
    make_signal_parser, make_signal_monitor,
    assess_correction, CorrectionQuality,
    layer0_intent, layer1_intent, layer2_intent, layer3_intent,
    check_signal_health,
)

__all__ = [
    "ProcessSignal", "DEFAULT_SIGNAL_NAMES",
    "CircuitState", "gates_to_confidence",
    "make_pythia_parser", "make_from_sweep",
    "IOI_GRAPH", "make_circuit_graph",
    "CircuitMonitor",
]
