"""
Infrastructure monitoring adapter for margin.

Maps server/service metrics into typed health observations.
CPU, memory, disk, latency, error rate, uptime — with correct polarity.
"""

from .metrics import (
    INFRA_METRICS, InfraMetric,
    parse_metrics, service_expression,
)
