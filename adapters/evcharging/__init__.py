"""
EV charging adapter for margin.

Maps charge session state into competing health circuits:
SoC (higher=better), grid draw (lower=better), solar surplus (higher=better),
charge rate (higher=better within limits).
"""

from .session import (
    CHARGE_CIRCUITS, ChargeCircuit,
    parse_charge_state, charge_expression,
)
