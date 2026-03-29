"""
pytest / test suite health adapter for margin.

Test suite quality over time: pass rate, flake rate, duration trend,
coverage delta, skip rate.
"""
from .suite import TEST_METRICS, parse_suite, suite_expression
