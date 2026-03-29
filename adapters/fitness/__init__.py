"""
Personal fitness / wearable adapter for margin.

Health and activity metrics as typed observations.
Resting HR, HRV, sleep, steps, stress — with correct polarity and bands.
"""

from .metrics import FITNESS_METRICS, FitnessMetric, parse_fitness, daily_expression
