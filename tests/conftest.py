"""
Enable margin's own pytest health plugin for self-monitoring.

Every test run prints a typed health snapshot of this test suite:
  pass_rate, new_failures, skip_rate, duration_seconds, mean_test_duration

Options:
  --margin-fail-below=ABLATED   CI gate
  --margin-per-file             per-file health breakdown
  --margin-slowest=N            N slowest tests with health classification
  --margin-output=path.json     save snapshot for next run's --margin-baseline
  --margin-baseline=path.json   diff against a previous snapshot
"""

pytest_plugins = ["adapters.pytest.plugin"]
