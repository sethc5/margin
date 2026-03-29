"""
Basic health classification — the simplest margin usage.

Give it a number and thresholds. It tells you if it's healthy.
"""

from margin import classify, Health, Thresholds, Confidence

# Higher is better (throughput, signal strength)
t = Thresholds(intact=80.0, ablated=30.0)
print(classify(95.0, Confidence.HIGH, thresholds=t))  # Health.INTACT
print(classify(50.0, Confidence.HIGH, thresholds=t))  # Health.DEGRADED
print(classify(10.0, Confidence.HIGH, thresholds=t))  # Health.ABLATED

# Lower is better (error rate, latency)
t_err = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
print(classify(0.005, Confidence.HIGH, thresholds=t_err))  # Health.INTACT
print(classify(0.05, Confidence.HIGH, thresholds=t_err))   # Health.DEGRADED
print(classify(0.15, Confidence.HIGH, thresholds=t_err))   # Health.ABLATED
