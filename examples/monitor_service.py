"""
Monitor a web service — mixed polarity metrics on one dashboard.

Throughput (higher=better) and error rate (lower=better) classified
and normalized to the same scale.
"""

from margin import Parser, Thresholds

parser = Parser(
    baselines={"throughput": 1000.0, "error_rate": 0.002, "p99_latency": 50.0},
    thresholds=Thresholds(intact=800.0, ablated=200.0),
    component_thresholds={
        "error_rate": Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        "p99_latency": Thresholds(intact=200.0, ablated=2000.0, higher_is_better=False),
    },
)

# Normal operation
expr = parser.parse({"throughput": 950.0, "error_rate": 0.003, "p99_latency": 45.0})
print("Normal:", expr.to_string())

# Error rate spiking
expr = parser.parse({"throughput": 900.0, "error_rate": 0.08, "p99_latency": 180.0})
print("Spike: ", expr.to_string())

# Everything degraded
expr = parser.parse({"throughput": 150.0, "error_rate": 0.12, "p99_latency": 3000.0})
print("Down:  ", expr.to_string())
