"""
EV charge session as margin observations.

Models the competing priorities in smart EV charging:
maximize SoC, minimize grid draw, maximize solar self-consumption,
stay within grid capacity limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.health import Thresholds
from margin.observation import Observation, Expression
from margin.confidence import Confidence


@dataclass
class ChargeCircuit:
    """One charging concern as a margin circuit."""
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


CHARGE_CIRCUITS: dict[str, ChargeCircuit] = {
    "soc": ChargeCircuit(
        name="soc", display_name="State of Charge",
        thresholds=Thresholds(intact=80.0, ablated=20.0, higher_is_better=True),
        baseline=80.0, unit="%",
    ),
    "grid_draw": ChargeCircuit(
        name="grid_draw", display_name="Grid Draw",
        thresholds=Thresholds(intact=2000.0, ablated=5000.0, higher_is_better=False),
        baseline=500.0, unit="W",
    ),
    "solar_surplus": ChargeCircuit(
        name="solar_surplus", display_name="Solar Surplus",
        thresholds=Thresholds(intact=500.0, ablated=0.0, higher_is_better=True),
        baseline=1500.0, unit="W",
    ),
    "charge_rate": ChargeCircuit(
        name="charge_rate", display_name="Charge Rate",
        thresholds=Thresholds(intact=3000.0, ablated=500.0, higher_is_better=True),
        baseline=7000.0, unit="W",
    ),
    "grid_capacity": ChargeCircuit(
        name="grid_capacity", display_name="Grid Capacity Remaining",
        thresholds=Thresholds(intact=3000.0, ablated=500.0, higher_is_better=True),
        baseline=8000.0, unit="W",
    ),
    "session_efficiency": ChargeCircuit(
        name="session_efficiency", display_name="Solar Self-Consumption",
        thresholds=Thresholds(intact=0.70, ablated=0.20, higher_is_better=True),
        baseline=0.85, unit="ratio",
    ),
}


def parse_charge_state(
    state: dict[str, float],
    confidence: Confidence = Confidence.MODERATE,
    circuits: Optional[dict[str, ChargeCircuit]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse charge session state into margin Observations.

    Args:
        state: {"soc": 45, "grid_draw": 3200, "solar_surplus": 800, ...}
        confidence: measurement confidence
        circuits: override circuit definitions
        measured_at: timestamp
    """
    from margin.health import classify

    defs = circuits or CHARGE_CIRCUITS
    observations = {}
    for name, value in state.items():
        circuit = defs.get(name)
        if circuit is None:
            continue
        health = classify(value, confidence, thresholds=circuit.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value, baseline=circuit.baseline,
            confidence=confidence, higher_is_better=circuit.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def charge_expression(
    state: dict[str, float],
    charger_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    circuits: Optional[dict[str, ChargeCircuit]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a charger-wide Expression from session state."""
    obs = parse_charge_state(state, confidence, circuits, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=charger_id,
    )
