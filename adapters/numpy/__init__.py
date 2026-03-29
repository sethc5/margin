"""
NumPy array health adapter for margin.

Pass an array, get typed health back: mean drift, variance stability,
NaN contamination, range violations, distribution shape.

No pandas dependency. Pure numpy.
"""
from .array_health import array_health, compare_arrays, ArrayProfile, ARRAY_METRICS
