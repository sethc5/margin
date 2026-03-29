# margin

**Typed health classification for systems that measure things.**

Every system with health bars, thresholds, alerts, or status dashboards solves the same problem: take a number, decide if it's healthy, correct it if it isn't, explain what happened. Margin is that pattern, typed once, with the polarity bug fixed.

```python
from margin import Parser, Thresholds

parser = Parser(
    baselines={"throughput": 500.0, "error_rate": 0.002},
    thresholds=Thresholds(intact=400.0, ablated=150.0),
    component_thresholds={
        "error_rate": Thresholds(intact=0.005, ablated=0.05, higher_is_better=False),
    },
)

expr = parser.parse({"throughput": 480.0, "error_rate": 0.03})
print(expr.to_string())
# [throughput:INTACT(-0.04σ)] [error_rate:DEGRADED(-14.00σ)]
```

Throughput and error rate on the same scale. One is higher-is-better, the other is lower-is-better. Both classified correctly. Sigma-normalised so you can compare them.

## Install

```bash
pip install margin
```

Zero dependencies. Pure Python. 3.10+.

## What it does

A number comes in. Margin gives it:

- **Health** — INTACT / DEGRADED / ABLATED / RECOVERING / OOD
- **Polarity** — higher-is-better or lower-is-better, handled correctly everywhere
- **Sigma** — dimensionless deviation from baseline, always positive = healthier
- **Confidence** — how much the uncertainty interval overlaps the threshold
- **Provenance** — where this value came from, for correlation detection
- **Validity** — how the measurement ages (static, decaying, event-invalidated)
- **Drift** — trajectory classification: STABLE / DRIFTING / ACCELERATING / DECELERATING / REVERTING / OSCILLATING
- **Anomaly** — statistical outlier detection: EXPECTED / UNUSUAL / ANOMALOUS / NOVEL

Then the correction loop:

- **Policy** — typed rules that decide what to do (RESTORE / SUPPRESS / AMPLIFY)
- **Constraints** — alpha clamping, cooldown, rate limiting
- **Escalation** — LOG / ALERT / HALT when the policy can't act
- **Contract** — typed success criteria ("reach INTACT within 5 steps")
- **Causal** — dependency graphs ("api is DEGRADED because db is ABLATED")
- **Auto-correlation** — discover which components move together from data, with lag detection
- **Streaming** — incremental trackers: `Monitor.update(values)` updates health + drift + anomaly + correlation in one call
- **Config** — define everything in YAML/JSON: `margin.load_config("margin.yaml")`
- **Ledger** — full audit trail of every correction, serializable, replayable

All in one call:

```python
from margin import step

result = step(expression, policy, ledger, graph, contract)
# result.correction    — what to do
# result.explanations  — why it happened
# result.decision      — which rule matched, full trace
# result.contract      — are we meeting our goals?
```

## The polarity bug

Every health system you've written has this bug. You check `if value >= threshold` and it works for throughput. Then you add error rate monitoring and the same check says 15% error rate is "healthy" because 0.15 >= 0.02.

Margin handles both polarities:

```python
# Higher is better (throughput, signal strength)
Thresholds(intact=80.0, ablated=30.0)

# Lower is better (error rate, latency)
Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
```

One flag. Threads through every comparison, every sigma calculation, every correction decision, every recovery ratio. You never think about it again.

## Auto-calibrate from data

Don't guess thresholds. Derive them from healthy measurements:

```python
from margin import parser_from_calibration

parser = parser_from_calibration(
    {"rps": [490, 510, 505, 495], "latency": [48, 52, 50, 51]},
    polarities={"latency": False},
)
```

## Five layers

| Layer | Question | Key types |
|---|---|---|
| **Foundation** | What was measured? | `Health`, `Observation`, `Expression`, `UncertainValue` |
| **Observability** | What changed? When will it cross? | `diff()`, `forecast()`, `track()`, `calibrate()` |
| **Policy** | What should we do? | `PolicyRule`, `Action`, `Constraint`, `Escalation` |
| **Contract** | Are we meeting our goals? | `HealthTarget`, `SustainHealth`, `RecoveryThreshold` |
| **Causal** | Why did this happen? | `CausalGraph`, `CausalLink`, `Explanation` |

Plus `step()` and `run()` to orchestrate all five in one call.

## Domain adapters

Ready-to-use threshold profiles for specific domains:

| Adapter | What it monitors | Polarity |
|---|---|---|
| [**healthcare**](https://github.com/sethc5/margin/tree/main/adapters/healthcare/) | Vital signs (HR, BP, SpO2, temp, glucose) — WHO/AHA ranges, sepsis screening | bands |
| [**godot**](https://github.com/sethc5/margin-godot) | Game systems (food, morale, stress, stamina) — native GDScript | mixed |
| [**homeassistant**](https://github.com/sethc5/margin/tree/main/adapters/homeassistant/) | Smart home sensors (temp, humidity, battery, solar, power) | mixed |
| [**evcharging**](https://github.com/sethc5/margin/tree/main/adapters/evcharging/) | EV charging (SoC, grid draw, solar surplus, efficiency) | mixed |
| [**infrastructure**](https://github.com/sethc5/margin/tree/main/adapters/infrastructure/) | Server monitoring (CPU, memory, disk, latency, error rate) | mixed |
| [**aquarium**](https://github.com/sethc5/margin/tree/main/adapters/aquarium/) | Water chemistry (pH, ammonia, nitrite, temp, hardness) | bands |
| [**greenhouse**](https://github.com/sethc5/margin/tree/main/adapters/greenhouse/) | Growing environment (soil moisture, CO₂, light, VPD) | bands |
| [**fitness**](https://github.com/sethc5/margin/tree/main/adapters/fitness/) | Wearables (resting HR, HRV, sleep, steps, stress) | mixed |
| [**transformer**](https://github.com/sethc5/margin/tree/main/adapters/transformer/) | ML circuit interpretability (pythia-6.9b) | higher |
| [**fastapi**](https://github.com/sethc5/margin/tree/main/adapters/fastapi/) | Endpoint health (latency, error rate, throughput, queue depth) | mixed |
| [**database**](https://github.com/sethc5/margin/tree/main/adapters/database/) | DB health (pool usage, query latency, replication lag, deadlocks) | mixed |
| [**celery**](https://github.com/sethc5/margin/tree/main/adapters/celery/) | Task queue (queue depth, failure rate, worker utilization, retries) | mixed |
| [**dataframe**](https://github.com/sethc5/margin/tree/main/adapters/dataframe/) | Data quality (completeness, null rate, drift, freshness, schema) | mixed |
| [**pytest**](https://github.com/sethc5/margin/tree/main/adapters/pytest/) | Test suite health (pass rate, flake rate, coverage, duration) | mixed |

## Docs

Full specification: [margin-language.md](https://github.com/sethc5/margin/blob/main/margin/margin-language.md)

## License

MIT
