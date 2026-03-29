"""
FastAPI / Django middleware adapter for margin.

Request health per endpoint: latency, error rate, throughput, queue depth.
"""
from .endpoints import ENDPOINT_METRICS, parse_endpoint, endpoint_expression
