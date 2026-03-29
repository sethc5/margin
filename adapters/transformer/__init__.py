"""
Transformer circuit adapter for margin.

Thin domain layer on top of generic/ for mechanistic interpretability
of transformer circuits (IOI, name-mover, induction-head, suppressor).
"""

from .signal import ProcessSignal, DEFAULT_SIGNAL_NAMES
from .circuit import CircuitState, gates_to_confidence
from .parsers import make_pythia_parser, make_from_sweep

__all__ = [
    "ProcessSignal", "DEFAULT_SIGNAL_NAMES",
    "CircuitState", "gates_to_confidence",
    "make_pythia_parser", "make_from_sweep",
]
