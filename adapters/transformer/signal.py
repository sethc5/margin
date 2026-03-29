"""
ProcessSignal: typed wrapper for a projection head's n_signals output.

Domain-specific to transformer interpretability. Wraps raw activation
vectors into named, bounded, timestamped signals with event-based validity
(invalidated when the prompt changes).

Built on generic.UncertainValue, generic.Validity, generic.EventBus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from margin.uncertain import UncertainValue, Source
from margin.validity import Validity
from margin.provenance import new_id


# -----------------------------------------------------------------------
# Default signal names — mirrors ProjectionConfig.SIGNAL_NAMES
# -----------------------------------------------------------------------

DEFAULT_SIGNAL_NAMES: list[str] = [
    "confidence", "novelty", "coherence", "retrieval_mode",
    "converging", "drifting", "accelerating", "stalling",
    "epistemic_uncertainty", "aleatoric_uncertainty",
    "superposition_density", "boundary_proximity",
    "reserved_0", "reserved_1", "reserved_2", "reserved_3",
]


# -----------------------------------------------------------------------
# ProcessSignal
# -----------------------------------------------------------------------

@dataclass
class ProcessSignal:
    """
    Typed wrapper for the projection head's n_signals-dim output.

    values:       raw signal vector, length n_signals, in [-1, 1]
    signal_names: per-dimension semantic labels
    measured_at:  when the signal was extracted
    prompt_hash:  hash of the prompt (for event-based validity)
    provenance_id: unique ID for the provenance graph
    hook_layer:   which residual-stream layer was observed
    """
    values: list[float]
    signal_names: list[str] = field(default_factory=lambda: list(DEFAULT_SIGNAL_NAMES))
    measured_at: datetime = field(default_factory=datetime.now)
    prompt_hash: Optional[str] = None
    provenance_id: str = field(default_factory=new_id)
    hook_layer: int = 26

    def __getitem__(self, name: str) -> float:
        """Look up a signal dimension by name."""
        try:
            idx = self.signal_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown signal dimension: {name!r}")
        return self.values[idx]

    @property
    def n_signals(self) -> int:
        return len(self.values)

    def named_dict(self) -> dict[str, float]:
        return {n: v for n, v in zip(self.signal_names, self.values)}

    @property
    def core_state(self) -> dict[str, float]:
        """First 4: confidence, novelty, coherence, retrieval."""
        return {self.signal_names[i]: self.values[i] for i in range(min(4, self.n_signals))}

    @property
    def trajectory(self) -> dict[str, float]:
        """Next 4: converging, drifting, accelerating, stalling."""
        return {self.signal_names[i]: self.values[i] for i in range(4, min(8, self.n_signals))}

    @property
    def uncertainty_texture(self) -> dict[str, float]:
        """Next 4: epistemic, aleatoric, superposition, boundary."""
        return {self.signal_names[i]: self.values[i] for i in range(8, min(12, self.n_signals))}

    def validity(self) -> Validity:
        """Event-invalidated: stale when the prompt changes."""
        return Validity.until_event(
            f"prompt_change:{self.prompt_hash}",
            measured_at=self.measured_at,
        )

    def is_fresh(self, max_age_seconds: float = 60.0) -> bool:
        elapsed = (datetime.now() - self.measured_at).total_seconds()
        return elapsed <= max_age_seconds

    def confidence_as_uncertain(self, uncertainty: float = 0.05) -> UncertainValue:
        """
        Extract the 'confidence' dimension as an UncertainValue.
        Default uncertainty 0.05 = projection head compression loss.
        """
        return UncertainValue(
            point=self["confidence"],
            uncertainty=uncertainty,
            source=Source.MEASURED,
            validity=self.validity(),
            provenance=[self.provenance_id],
        )

    @classmethod
    def from_list(
        cls,
        values: list[float],
        signal_names: Optional[list[str]] = None,
        hook_layer: int = 26,
        prompt_hash: Optional[str] = None,
    ) -> ProcessSignal:
        """Wrap a raw list of floats."""
        return cls(
            values=list(values),
            signal_names=signal_names or list(DEFAULT_SIGNAL_NAMES[:len(values)]),
            hook_layer=hook_layer,
            prompt_hash=prompt_hash,
        )

    @classmethod
    def zeros(cls, n_signals: int = 16) -> ProcessSignal:
        """A silent / null signal."""
        return cls(values=[0.0] * n_signals, signal_names=list(DEFAULT_SIGNAL_NAMES[:n_signals]))

    def to_dict(self) -> dict:
        return {
            "values": [round(v, 6) for v in self.values],
            "signal_names": self.signal_names,
            "measured_at": self.measured_at.isoformat(),
            "prompt_hash": self.prompt_hash,
            "provenance_id": self.provenance_id,
            "hook_layer": self.hook_layer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProcessSignal:
        return cls(
            values=d["values"],
            signal_names=d.get("signal_names", list(DEFAULT_SIGNAL_NAMES)),
            measured_at=datetime.fromisoformat(d["measured_at"]) if "measured_at" in d else datetime.now(),
            prompt_hash=d.get("prompt_hash"),
            provenance_id=d.get("provenance_id", new_id()),
            hook_layer=d.get("hook_layer", 26),
        )

    def __repr__(self) -> str:
        top = sorted(self.named_dict().items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:+.2f}" for k, v in top)
        return f"ProcessSignal({top_str}, ...)"
