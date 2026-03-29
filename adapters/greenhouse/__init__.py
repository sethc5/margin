"""
Greenhouse / growing environment adapter for margin.

Climate and soil parameters as typed health observations.
All parameters have bands optimized for plant growth.
"""

from .environment import GROW_PARAMS, GrowParam, parse_environment, grow_expression
