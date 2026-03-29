"""
DataFrame / data pipeline adapter for margin.

Data quality metrics: completeness, null rate, drift, freshness, row count.
Works with any tabular data — no pandas dependency required.
"""
from .quality import DQ_METRICS, parse_quality, pipeline_expression
