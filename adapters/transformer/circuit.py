"""
CircuitState: typed packaging of transformer circuit health measurements.

Wraps raw floats from an IOI correction step into generic margin objects.
One CircuitState per circuit per forward pass.

Built on generic.UncertainValue, generic.Observation, generic.Confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from margin.uncertain import UncertainValue, Source
from margin.validity import Validity
from margin.confidence import Confidence
from margin.health import Health, Thresholds, classify
from margin.observation import Observation, Op
from margin.provenance import new_id
from margin.bridge import observe
from margin.algebra import subtract

from .signal import ProcessSignal


def gates_to_confidence(gap_gate: bool, geo_gate: bool, gap_value: float) -> Confidence:
    """
    Map IOICorrector gate booleans to a Confidence tier.

    both gates    → HIGH
    one gate      → MODERATE
    neither + gap < 0 → INDETERMINATE (OOD)
    neither       → LOW
    """
    if gap_gate and geo_gate:
        return Confidence.HIGH
    if gap_gate or geo_gate:
        return Confidence.MODERATE
    if gap_value < 0:
        return Confidence.INDETERMINATE
    return Confidence.LOW


@dataclass
class CircuitState:
    """
    Typed circuit health snapshot at one forward pass.

    Thin wrapper: holds the transformer-specific fields (signal, cluster,
    gate_reason) and delegates health classification and observation
    creation to the generic layer.
    """
    circuit_name: str
    intact_gap: UncertainValue
    current_gap: UncertainValue
    signal: Optional[ProcessSignal] = None
    gate_fired: bool = False
    gate_confidence: Confidence = Confidence.MODERATE
    gate_reason: str = "neither"
    cluster: str = ""
    cluster_score: float = 0.0
    layer: int = 26
    measured_at: datetime = field(default_factory=datetime.now)
    provenance_id: str = field(default_factory=new_id)

    def health(self, thresholds: Optional[Thresholds] = None) -> Health:
        """Classify using generic.classify(). Defaults to pythia-6.9b thresholds."""
        t = thresholds or Thresholds(intact=3.5, ablated=1.5)
        return classify(
            self.current_gap.point,
            self.gate_confidence,
            self.gate_fired,
            t,
        )

    @property
    def gap_delta(self) -> float:
        """current - intact: negative means degradation."""
        return self.current_gap.point - self.intact_gap.point

    @property
    def recovery_ratio(self) -> float:
        """current / intact. 1.0 = fully recovered."""
        if self.intact_gap.point == 0.0:
            return 0.0
        return self.current_gap.point / self.intact_gap.point

    def to_observation(self, thresholds: Optional[Thresholds] = None) -> Observation:
        """Project into a generic Observation."""
        t = thresholds or Thresholds(intact=3.5, ablated=1.5)
        return Observation(
            name=self.circuit_name,
            health=self.health(t),
            value=self.current_gap.point,
            baseline=self.intact_gap.point,
            confidence=self.gate_confidence,
            higher_is_better=True,  # logit gaps are always higher-is-better
            provenance=[self.provenance_id],
            measured_at=self.measured_at,
        )

    def gap_diff(self) -> UncertainValue:
        """Uncertain difference: current - intact, with propagated uncertainty."""
        return subtract(self.current_gap, self.intact_gap)

    @classmethod
    def from_step_result(
        cls,
        circuit_name: str,
        intact_io_gap: float,
        current_io_gap: float,
        gap_gate: bool,
        geo_gate: bool,
        gate_reason: str,
        corrected: bool,
        cluster: str = "",
        cluster_score: float = 0.0,
        signal: Optional[ProcessSignal] = None,
        layer: int = 26,
        gap_uncertainty: float = 0.15,
    ) -> CircuitState:
        """
        Build from IOICorrector's StepResult fields.

        Maps gate booleans → Confidence via gates_to_confidence().
        Wraps gap floats into UncertainValues with the given uncertainty.
        """
        now = datetime.now()
        prov = new_id()
        validity = Validity.static(now)

        return cls(
            circuit_name=circuit_name,
            intact_gap=UncertainValue(
                point=intact_io_gap,
                uncertainty=gap_uncertainty,
                source=Source.MEASURED,
                validity=validity,
                provenance=[prov],
            ),
            current_gap=UncertainValue(
                point=current_io_gap,
                uncertainty=gap_uncertainty,
                source=Source.MEASURED,
                validity=validity,
                provenance=[prov],
            ),
            signal=signal,
            gate_fired=corrected,
            gate_confidence=gates_to_confidence(gap_gate, geo_gate, current_io_gap),
            gate_reason=gate_reason,
            cluster=cluster,
            cluster_score=cluster_score,
            layer=layer,
            measured_at=now,
            provenance_id=prov,
        )

    def to_dict(self) -> dict:
        return {
            "circuit_name": self.circuit_name,
            "health": self.health().value,
            "intact_gap": self.intact_gap.point,
            "current_gap": self.current_gap.point,
            "gap_delta": self.gap_delta,
            "recovery_ratio": self.recovery_ratio,
            "gate_fired": self.gate_fired,
            "gate_confidence": self.gate_confidence.value,
            "gate_reason": self.gate_reason,
            "cluster": self.cluster,
            "cluster_score": self.cluster_score,
            "layer": self.layer,
            "provenance_id": self.provenance_id,
            "signal": self.signal.to_dict() if self.signal else None,
        }

    def __repr__(self) -> str:
        h = self.health().value
        g = self.current_gap.point
        return f"CircuitState({self.circuit_name}:{h}, gap={g:+.2f})"
