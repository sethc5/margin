"""
Aquarium monitoring adapter for margin.

Water chemistry and environmental parameters as typed health observations.
All parameters have bands — both too high and too low are harmful to fish.
"""

from .water import WATER_PARAMS, WaterParam, parse_water, tank_expression
