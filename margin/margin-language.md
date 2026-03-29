# Margin Language

Margin is a typed observation language for any system where you measure
things, classify their health, correct them, and need to explain what
happened. It sits between raw numbers and natural language — every
measurement is tagged with a health predicate, every intervention with
an operation code, and every object carries provenance back to the
values that produced it.

The name comes from the question every monitoring system asks: *how much
margin do we have before this component fails?*

---

## Theory

### What kind of thing is this?

Margin is a **typed event stream protocol** — a formal vocabulary for
expressing what a sensor has observed about a system's internal state at
one point in time. It is not a programming language (you don't compute
with it) and not a natural language (it doesn't have pragmatics). It is
closer to what SNOMED CT is for clinical measurements: a raw float
becomes a typed finding.

### Formal grammar

An **expression** is the unit of discourse — one fully-parsed state of
all monitored components at one moment:

```
Expression     ::=  "∅"
                  | Observation+  Correction*

Observation    ::=  "[" Name ":" Health "(" Sigma "σ" ")"
                     ( "," Name ":" Health "(" Sigma "σ" ")" )*
                     "]"

Correction     ::=  "→" Op "(" "α=" Alpha ")"

Health         ::=  "INTACT" | "DEGRADED" | "RECOVERING" | "ABLATED" | "OOD"

Op             ::=  "RESTORE" | "SUPPRESS" | "AMPLIFY" | "NOOP"

Name           ::=  identifier       -- e.g. "api-latency", "error_rate", "IOI"
Sigma          ::=  signed-decimal   -- polarity-normalised deviation from baseline
Alpha          ::=  decimal-in-0-1   -- mixing / intensity coefficient
```

Comma-grouping inside one bracket `[ ]` means multiple readings of the
*same* named component. Different components always produce separate
brackets. Each bracket carries at most one `→ Op` (the worst component
gets the correction; one correction per expression).

### Semantic model

Three layers:

| Layer | Object | Meaning |
|---|---|---|
| **Measurement** | `Observation` | "Component C had value v, giving health H with confidence Q" |
| **Intervention** | `Correction` | "Operation Op is being applied to component C with intensity α" |
| **Composition** | `Expression` | "The joint state of all components, with net confidence = weakest measurement" |

Each layer is **monotone in confidence**: composing measurements cannot
produce an expression whose `confidence` exceeds the weakest constituent.
The language never overclaims.

### Health as a partial order

```
OOD          (measurement invalid — confidence is INDETERMINATE)
  ↑
ABLATED      (functionally absent — value past the ablated threshold)
  ↑
DEGRADED     (impaired — value between ablated and intact thresholds)
  ↑
RECOVERING   (sub-threshold but active correction is running)
  ↑
INTACT       (operating normally — value at or past the intact threshold)
```

RECOVERING is not between DEGRADED and INTACT — it is **orthogonal**,
overlapping the DEGRADED and ABLATED bands. The same value is ABLATED
without correction and RECOVERING with it. Health is a product of two
dimensions:

- **Value dimension**: ABLATED < DEGRADED < INTACT (continuous)
- **Correction dimension**: passive vs. active (boolean)

OOD is outside both dimensions — the measurement confidence is
INDETERMINATE so the value is not interpretable.

### Polarity

Components can be **higher-is-better** (throughput, signal strength) or
**lower-is-better** (error rate, latency, temperature). Polarity is set
on `Thresholds` via `higher_is_better: bool` and threads through every
comparison, sigma calculation, and correction decision.

For `higher_is_better=True`:
- `value >= intact` → INTACT
- `value < ablated` → ABLATED
- `sigma = (value - baseline) / |baseline|` — positive = healthier

For `higher_is_better=False`:
- `value <= intact` → INTACT
- `value > ablated` → ABLATED
- `sigma = (baseline - value) / |baseline|` — positive = healthier

Sigma is always **polarity-normalised**: positive means healthier than
baseline, negative means worse, regardless of which direction is good.

### The sigma as a normalised measurement

`sigma = polarity_adjusted(value - baseline) / |baseline|` is the key
observable. It is dimensionless, normalised to the component-specific
baseline, and anchored at 0.0 for a perfectly healthy component. This
lets you compare health across components with different scales and
polarities.

- `sigma = 0.0` → exactly at baseline
- `sigma = -1.0` → value has deviated to the worst possible direction
  by one full baseline unit
- `sigma > 0.0` → component is performing better than baseline

### Edge-case semantics

These behaviours are **specified**, not accidental:

| Input | Behaviour | Rationale |
|---|---|---|
| `values = {}` (empty) | `to_string()` → `"[∅]"`, `corrections = []` | No measurement → null expression |
| `value = NaN` | Falls through to DEGRADED (NaN comparisons are False) | Sensor failure produces the least-confident non-OOD state |
| `value = +∞` (higher_is_better) | INTACT | Infinite value → trivially healthy |
| `value = +∞` (lower_is_better) | ABLATED | Infinite value → maximally unhealthy |
| Unknown component (not in baselines) | Uses current value as baseline fallback; sigma = 0.0 | Graceful degradation — add to `baselines` to silence |
| Value exactly at `intact` | INTACT (boundary belongs to the healthy side) | `>=` for higher_is_better, `<=` for lower_is_better |
| Value exactly at `ablated` | DEGRADED (boundary belongs to the safer side) | Strict `<` / `>` for the ablated check |
| Empty ledger `summary()` | Returns zero-valued dict; `fire_rate = 0.0` | Safe default |

### What the language is NOT

- It is not a probabilistic model. `Confidence` is an ordinal tier, not
  a calibrated probability. Sigma is a deterministic ratio, not a z-score.

- It is not an executor. You don't run computations in it. It is the
  **observation format**: the typed output that records what the system
  observed. Computation lives in the algebra and the correction loop.

- It does not interpret *why* a component is degraded. It describes the
  measurement; causation lives in domain-specific layers above it.

- It does not prescribe what to do. `Correction` describes what action
  *was taken* — not what *should* be taken.

---

## Architecture overview

```
Raw measurement (float)
       │
       ▼
  Observation                    ← observation.py
       │
       ├─ .health  → Health      (via classify() in health.py)
       ├─ .sigma   → float       (polarity-normalised deviation)
       │
       └─ assembled into ────────────────────────────┐
                                                      │
  Correction (Op, alpha, magnitude) ─────────────────┤
                                                      ▼
                                              Expression
                                                ← observation.py
                                                      │
                                                      └─ .to_string()
                                                         .to_dict()
                                                         .to_json()

  Record                         ← ledger.py
       │  (before, after, op, alpha, improvement)
       ▼
  Ledger                         ← ledger.py
       │  .render()  → multi-line string
       │  .summary() → aggregate stats dict
       │  .to_json() → full serialisation
       ▼
  "step   0: [throughput:INTACT(-0.05σ)]"
  "step   1: [error_rate:RECOVERING(-7.00σ) → RESTORE(α=0.60)]"
```

See [Structure](#structure) at the end of this document for the full file map.

---

## `confidence.py` — Confidence

Ordinal tiers for how much an uncertainty interval overlaps a decision
boundary. Supports comparison operators (`Confidence.HIGH > Confidence.LOW`).

| Tier | Meaning | Overlap with boundary |
|---|---|---|
| `CERTAIN` | interval fully clear | none |
| `HIGH` | reliable | < 10% of interval width |
| `MODERATE` | inferred or low-batch | 10-40% |
| `LOW` | weak signal | > 40% |
| `INDETERMINATE` | boundary inside interval; forces OOD | 100% |

```python
from margin import Confidence
Confidence.HIGH > Confidence.LOW      # True
min([Confidence.HIGH, Confidence.LOW]) # Confidence.LOW
```

---

## `validity.py` — Validity

Temporal validity for uncertain values. Three modes:

| Mode | Behaviour | Factory |
|---|---|---|
| `STATIC` | Uncertainty stays constant | `Validity.static()` |
| `DECAY` | Uncertainty doubles every `halflife` | `Validity.decaying(timedelta)` |
| `EVENT` | Valid until named event fires | `Validity.until_event(name)` |

`uncertainty_multiplier(at_time)` returns >= 1.0 always (clamped for
future-dated measurements).

---

## `uncertain.py` — UncertainValue

A scalar measurement with uncertainty, epistemic source, temporal
validity, and provenance.

| Field | Type | Meaning |
|---|---|---|
| `point` | float | Central estimate |
| `uncertainty` | float | Magnitude (absolute or relative) |
| `relative` | bool | If True, uncertainty is a fraction of \|point\| |
| `source` | str | `Source.MEASURED`, `MODELED`, `ASSERTED`, or `PROPAGATED` |
| `validity` | Validity | Temporal validity descriptor |
| `provenance` | list[str] | Ancestor IDs for correlation detection |

```python
from margin import UncertainValue
v = UncertainValue(point=5.0, uncertainty=0.3)
v.absolute_uncertainty()   # 0.3 (accounts for decay if validity is DECAY)
v.interval()               # (4.7, 5.3)
v.to_absolute()            # copy with absolute uncertainty
v.to_relative()            # copy with relative uncertainty
```

---

## `algebra.py` — Uncertainty propagation

Arithmetic operations that propagate uncertainty correctly:

- **Independent** values (no shared provenance): uncertainties combine in
  **quadrature** — `sqrt(σ_a² + σ_b²)`
- **Correlated** values (shared provenance): uncertainties combine
  **linearly** — `σ_a + σ_b` (conservative)

| Function | Behaviour |
|---|---|
| `add(a, b)` | Absolute uncertainties combine |
| `subtract(a, b)` | Same as add |
| `multiply(a, b)` | Relative uncertainties combine; falls back to absolute when either operand is zero |
| `divide(a, b)` | Same as multiply; raises on zero denominator |
| `scale(v, factor)` | Exact scaling; preserves provenance without growth |
| `compare(v, threshold)` | Returns a `Confidence` tier |
| `weighted_average(vs)` | Inverse-variance weighting by default |

```python
from margin import UncertainValue, add, compare
a = UncertainValue(point=5.0, uncertainty=0.3)
b = UncertainValue(point=3.0, uncertainty=0.2)
c = add(a, b)              # point=8.0, uncertainty=0.36 (quadrature)
compare(c, 7.5)            # Confidence.HIGH
```

---

## `health.py` — Health classification

### `Health` (Enum)

Five typed health predicates. All threshold comparisons are
polarity-aware.

| Value | Meaning |
|---|---|
| `INTACT` | value at or past the intact threshold (healthy direction) |
| `DEGRADED` | between ablated and intact thresholds |
| `ABLATED` | value past the ablated threshold (unhealthy direction) |
| `RECOVERING` | sub-threshold but active correction is running |
| `OOD` | measurement confidence is INDETERMINATE |

### `Thresholds` (dataclass)

| Field | Type | Meaning |
|---|---|---|
| `intact` | float | boundary for calling healthy |
| `ablated` | float | boundary for calling failed |
| `higher_is_better` | bool | polarity (default True) |
| `active_min` | float | minimum correction magnitude for "active" (default 0.05) |

Validates on construction: `ablated <= intact` when `higher_is_better`,
`ablated >= intact` when not.

```python
from margin import Thresholds

# Throughput: higher is better
Thresholds(intact=80.0, ablated=30.0)

# Error rate: lower is better
Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
```

### `classify()` (module-level function)

Single source of truth for health classification.

```
confidence == INDETERMINATE           → OOD
thresholds.is_intact(value)           → INTACT
thresholds.is_ablated(value)
    correcting                        → RECOVERING
    else                              → ABLATED
else
    correcting                        → RECOVERING
    else                              → DEGRADED
```

---

## `observation.py` — Observations, Corrections, Expressions

### `Observation` (dataclass)

One component's health at one measurement.

| Field | Type | Meaning |
|---|---|---|
| `name` | str | Component identifier |
| `health` | Health | Typed health predicate |
| `value` | float | Raw measurement |
| `baseline` | float | Expected healthy value (for sigma) |
| `confidence` | Confidence | Measurement confidence |
| `higher_is_better` | bool | Polarity (from Thresholds) |
| `provenance` | list[str] | Upstream provenance IDs |

Computed:
- `sigma` — polarity-normalised deviation from baseline (positive = healthier)

String rendering via `to_atom()`:
```
throughput:INTACT(-0.05σ)
error_rate:DEGRADED(-4.00σ)
cpu:OOD
```

### `Op` (Enum)

| Value | Meaning |
|---|---|
| `RESTORE` | fix a sub-threshold component |
| `SUPPRESS` | silence an over-performing component |
| `AMPLIFY` | strengthen a present-but-weak component |
| `NOOP` | no correction applied |

SUPPRESS fires when the worst component is past baseline in the
**healthy** direction (over-performing). RESTORE fires when it's past
baseline in the **unhealthy** direction and below the intact threshold.
AMPLIFY fires when it's intact but between threshold and baseline.

### `Correction` (dataclass)

| Field | Type | Meaning |
|---|---|---|
| `target` | str | Which component |
| `op` | Op | Operation type |
| `alpha` | float | Intensity coefficient (0 = none, 1 = full) |
| `magnitude` | float | Size of the correction |
| `triggered_by` | list[str] | Names of degraded components that triggered this |

`is_active()` returns True when `op != NOOP` and `alpha > 0`.

### `Expression` (dataclass)

Composed snapshot of all observations and corrections at one moment.

| Field | Type | Meaning |
|---|---|---|
| `observations` | list[Observation] | Per-component health readings |
| `corrections` | list[Correction] | Actions being applied (may be empty) |
| `confidence` | Confidence | Net = weakest observation |
| `label` | str | Optional label |
| `step` | Optional[int] | Sequence index |

Query helpers:
```python
expr.health_of("api-latency")     # → Health or None
expr.correction_for("error_rate") # → Correction or None
expr.degraded()                   # → list[Observation] (DEGRADED, ABLATED, or RECOVERING)
expr.intact()                     # → list[Observation] (INTACT only)
```

String rendering via `to_string()`:
```
[throughput:INTACT(-0.05σ)]
[error_rate:DEGRADED(-4.00σ) → RESTORE(α=0.60)]
[throughput:INTACT(-0.05σ)] [error_rate:RECOVERING(-7.00σ) → RESTORE(α=0.60)]
[∅]
[? → RESTORE(α=0.60)]
```

### `Parser` (class)

Converts raw measurements into typed Expressions.

```python
from margin import Parser, Thresholds, Confidence

p = Parser(
    baselines={"throughput": 100.0, "error_rate": 0.01},
    thresholds=Thresholds(intact=80.0, ablated=30.0),
    component_thresholds={
        "error_rate": Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
    },
)

expr = p.parse(
    values={"throughput": 50.0, "error_rate": 0.08},
    correction_magnitude=2.0,
    alpha=0.6,
)
print(expr.to_string())
# [throughput:RECOVERING(-0.50σ) → RESTORE(α=0.60)] [error_rate:RECOVERING(-7.00σ)]
```

`parse()` generates **at most one Correction** per expression, targeting
the worst component **whose `active_min` is met** by the correction
magnitude (polarity-aware degradation scoring). If the most-degraded
component's `active_min` isn't met but another component's is, the
correction targets that component instead. The `correcting` flag for
health classification is evaluated **per-component** using each
component's own `active_min`.

Net confidence is the **weakest** across all observations. If there are
no observations the expression gets INDETERMINATE.

---

## `ledger.py` — Record and Ledger

### `Record` (dataclass)

One correction event: before state, what was done, after state.

| Field | Type | Meaning |
|---|---|---|
| `step` | int | Sequence index (0-based) |
| `tag` | str | Label (e.g. request ID, token text) |
| `before` | Observation | State before correction |
| `after` | Optional[Observation] | State after (None if skipped) |
| `fired` | bool | Whether the correction was applied |
| `op` | Op | Operation performed |
| `alpha` | float | Intensity used |
| `magnitude` | float | Size of the correction |

Computed (all polarity-aware):
- `improvement` — positive = better, regardless of polarity
- `recovery_ratio` — 1.0 = fully restored to baseline, both polarities
- `was_beneficial()` — True if improvement > 0 or gate didn't fire

### `Ledger` (dataclass)

Accumulates Records across a session.

```python
ledger = Ledger(label="my-run")
ledger.append(record)
```

| Property | Returns |
|---|---|
| `n_fired` | count of fired corrections |
| `fire_rate` | `n_fired / len(records)` |
| `mean_improvement` | mean improvement across fired records |
| `mean_recovery` | mean recovery ratio across fired records |
| `harmful()` | list of records where correction made things worse |

```python
print(ledger.render())
# step   0: [throughput:INTACT(-0.05σ)]
# step   1: [error_rate:RECOVERING(-7.00σ) → RESTORE(α=0.60)]

print(ledger.summary())
# {"label": "my-run", "n_steps": 2, "n_fired": 1, ...}
```

---

## Invariants and design constraints

1. **`classify()` is the single source of truth.** No other code
   re-implements the threshold logic.

2. **Polarity is explicit.** Every `Thresholds` declares
   `higher_is_better`. Every `Observation` carries it. Every comparison,
   sigma, improvement, and recovery calculation respects it.

3. **RECOVERING is a transition state, not a terminal one.** It signals
   that a correction is running now. Whether the component reaches
   INTACT is tracked by `improvement` in the subsequent Record.

4. **Sigma is always polarity-normalised.** Positive = healthier than
   baseline, negative = worse, regardless of which direction "better" is.

5. **Net confidence is bottlenecked.** An Expression's confidence is the
   weakest observation. The language never overclaims.

6. **Provenance threads through every object.** UncertainValue,
   Observation, and Correction carry provenance IDs. Shared provenance
   triggers conservative (linear) uncertainty combination.

7. **`active_min` is per-component.** Each component's Thresholds
   controls when correction magnitude counts as "active" for that
   component's RECOVERING classification *and* for correction targeting.
   The correction targets the worst component whose `active_min` is met,
   not the globally worst component.

8. **Boundary ownership.** Value exactly at `intact` → INTACT. Value
   exactly at `ablated` → DEGRADED. Both boundaries belong to the
   safer classification.

9. **Serialization is lossless.** `to_dict()`/`from_dict()` roundtrips
   preserve exact values. No rounding is applied during serialization.

10. **Severity is defined once.** The `SEVERITY` dict in `health.py`
    is the single source of truth for Health ordering. `diff.py` and
    `composite.py` import it rather than defining their own.

---

## Calibrating for a new domain

All domain-specific numbers live in two places: the `baselines` dict and
the `Thresholds` instances passed to `Parser`. Everything else in the
language is domain-agnostic.

### Step 1 — Determine baselines

For each component, measure its value when the system is healthy. This
becomes the `baseline` used for sigma normalisation.

### Step 2 — Set thresholds

Choose `intact` and `ablated` boundaries. Rules of thumb:

- `intact` = 70-80% of baseline (the point where you'd say "this is fine")
- `ablated` = 20-30% of baseline (the point where the component is
  effectively gone)
- For `higher_is_better=False`, these fractions work in reverse: `intact`
  is slightly above baseline, `ablated` is well above.

### Step 3 — Construct a Parser

```python
from margin import Parser, Thresholds

parser = Parser(
    baselines={
        "throughput": 1000.0,
        "p99_latency": 50.0,
        "error_rate": 0.001,
    },
    thresholds=Thresholds(intact=800.0, ablated=300.0),
    component_thresholds={
        "p99_latency": Thresholds(intact=80.0, ablated=200.0, higher_is_better=False),
        "error_rate": Thresholds(intact=0.005, ablated=0.05, higher_is_better=False),
    },
)
```

---

## `diff.py` — Expression diffing

Compares two Expressions and reports what changed per component.

### `diff(before, after) -> Diff`

```python
from margin import diff, Parser, Thresholds

p = Parser(baselines={"x": 100.0}, thresholds=Thresholds(intact=80.0, ablated=30.0))
e1 = p.parse({"x": 90.0})
e2 = p.parse({"x": 40.0})

d = diff(e1, e2)
print(d.to_string())
# x: INTACT → DEGRADED

d.any_worsened      # True
d.any_improved      # False
d.worsened()        # [ComponentChange(x, INTACT→DEGRADED)]
d.appeared()        # components in after but not before
d.disappeared()     # components in before but not after
```

### `ComponentChange` (dataclass)

| Property | Returns |
|---|---|
| `health_changed` | True if health state differs |
| `sigma_delta` | sigma_after - sigma_before (positive = healthier) |
| `worsened` | True if health moved toward ABLATED/OOD |
| `improved` | True if health moved toward INTACT |
| `appeared` | True if component was absent before |
| `disappeared` | True if component is absent after |

Severity ordering: INTACT < RECOVERING < DEGRADED < ABLATED < OOD.

---

## `calibrate.py` — Threshold derivation

Derives baselines and thresholds from historical "known healthy" data.

### `calibrate(values, higher_is_better) -> CalibrationResult`

```python
from margin import calibrate

r = calibrate([98.0, 102.0, 100.0, 101.0, 99.0])
# r.baseline   = 100.0
# r.std        = 1.58
# r.thresholds = Thresholds(intact=70.0, ablated=30.0)

r_err = calibrate([0.01, 0.012, 0.008], higher_is_better=False)
# r_err.baseline   = 0.01
# r_err.thresholds = Thresholds(intact=0.013, ablated=0.017, higher_is_better=False)
```

For `higher_is_better=True`:

- `intact = baseline * intact_fraction` (default 0.70)
- `ablated = baseline * ablated_fraction` (default 0.30)

For `higher_is_better=False`:

- `intact = baseline * (1 + (1 - intact_fraction))` (default: 130% of baseline)
- `ablated = baseline * (1 + (1 - ablated_fraction))` (default: 170% of baseline)

### `calibrate_many(component_values, polarities) -> (baselines, thresholds)`

Calibrate multiple components at once. Returns dicts ready for `Parser()`.

### `parser_from_calibration(component_values, polarities) -> Parser`

One-shot: calibrate and return a ready-to-use Parser.

```python
from margin import parser_from_calibration

parser = parser_from_calibration(
    {"throughput": [100.0] * 10, "latency": [50.0] * 10},
    polarities={"latency": False},
)
expr = parser.parse({"throughput": 95.0, "latency": 52.0})
```

---

## `transitions.py` — State transition tracking

Tracks how components move between Health states over a Ledger's lifetime.

### `track(ledger, component) -> ComponentHistory`

```python
from margin import track, Ledger

history = track(ledger, "error_rate")
history.n_transitions       # number of state changes
history.transition_counts() # {("INTACT", "DEGRADED"): 2, ...}
history.time_in_state()     # {"INTACT": 15, "DEGRADED": 3, ...}
history.last_health()       # Health.INTACT
```

### `track_all(ledger) -> dict[str, ComponentHistory]`

Extract histories for all components found in a ledger.

### `Span` (dataclass)

A contiguous period in one Health state.

| Field | Type | Meaning |
| --- | --- | --- |
| `health` | Health | the state |
| `start_step` | int | step where this span began |
| `end_step` | Optional[int] | step where it ended |
| `n_steps` | int | number of steps in this span |
| `duration` | Optional[timedelta] | wall-clock duration (if timestamps available) |

### `ComponentHistory` (dataclass)

| Property | Returns |
| --- | --- |
| `spans` | ordered list of Spans |
| `transitions` | ordered list of Transitions |
| `n_transitions` | count |
| `transition_counts()` | {(from, to): count} |
| `time_in_state()` | {health_value: n_steps} |
| `last_health()` | most recent Health |

---

## Staleness checking

`Observation` carries an optional `measured_at: datetime` timestamp.

```python
from margin import Observation, Health, Confidence
from datetime import datetime

obs = Observation("x", Health.INTACT, 90.0, 100.0, Confidence.HIGH,
                  measured_at=datetime.now())

obs.age()                          # seconds since measurement (float)
obs.is_fresh(max_age_seconds=60.0) # True if within 60s
obs.is_fresh(max_age_seconds=5.0)  # False if older than 5s
```

If `measured_at` is None, `age()` returns None and `is_fresh()` returns
True (no timestamp = no staleness judgment).

The `measured_at` field roundtrips through `to_dict()`/`from_dict()`.
It is omitted from the dict when None.

---

## Ledger windowing

`Ledger` supports temporal and structural filtering. All window methods
return a new Ledger — the original is not modified.

### `window(duration, now=None) -> Ledger`

Records within `duration` of `now` (defaults to current time).

```python
from datetime import timedelta
recent = ledger.window(timedelta(minutes=5))
recent.fire_rate   # fire rate over the last 5 minutes
```

### `last_n(n) -> Ledger`

The most recent `n` records.

```python
last_10 = ledger.last_n(10)
```

### `for_component(name) -> Ledger`

Records involving a specific component.

```python
err_ledger = ledger.for_component("error_rate")
err_ledger.mean_improvement  # improvement stats for error_rate only
```

All filtered ledgers support the full Ledger API: `summary()`,
`render()`, `to_json()`, `harmful()`, etc.

---

## `events.py` — Event bus

Wires `Validity.until_event()` to actual event dispatch. When an event
fires, all values whose Validity references that event become stale.

```python
from margin import EventBus, Validity, UncertainValue

bus = EventBus()

# Create a value that's valid until the config reloads
v = Validity.until_event("config_reload")
uv = UncertainValue(point=5.0, uncertainty=0.1, validity=v)

bus.is_value_valid(uv)     # True
bus.fire("config_reload")
bus.is_value_valid(uv)     # False — event has fired
```

Supports listeners:

```python
bus.on("deploy", lambda event, time: print(f"{event} at {time}"))
bus.on("*", lambda event, time: log(event))  # wildcard
```

`bus.fired_events` returns all fired event names. `bus.reset()` clears
all or a specific event. Roundtrips via `to_dict()`/`from_dict()`.

---

## `composite.py` — Composite observations

A component with multiple related sub-measurements (e.g. p50/p95/p99
latency). Derives a single Health from the sub-observations.

```python
from margin import CompositeObservation, AggregateStrategy, Observation, Health, Confidence

c = CompositeObservation(
    name="latency",
    sub_observations=[
        Observation("p50", Health.INTACT, 45.0, 50.0, Confidence.HIGH),
        Observation("p95", Health.DEGRADED, 120.0, 80.0, Confidence.HIGH),
        Observation("p99", Health.ABLATED, 500.0, 200.0, Confidence.MODERATE),
    ],
    strategy=AggregateStrategy.WORST,
)
c.health       # Health.ABLATED (worst of the three)
c.confidence   # Confidence.MODERATE (weakest)
c.worst        # the p99 Observation
c.as_observation()  # flatten to a single Observation
```

Strategies: `WORST` (default), `BEST`, `MAJORITY`. Roundtrips via
`to_dict()`/`from_dict()`.

---

## `forecast.py` — Trend projection

Given a sequence of timestamped observations, fits a linear trend and
projects when the value will cross the intact or ablated threshold.

```python
from margin import forecast, Observation, Health, Thresholds, Confidence
from datetime import datetime, timedelta

t0 = datetime(2026, 1, 1)
obs = [
    Observation("rps", Health.INTACT, 450.0, 500.0, Confidence.HIGH,
                measured_at=t0),
    Observation("rps", Health.DEGRADED, 350.0, 500.0, Confidence.HIGH,
                measured_at=t0 + timedelta(minutes=5)),
    Observation("rps", Health.DEGRADED, 250.0, 500.0, Confidence.HIGH,
                measured_at=t0 + timedelta(minutes=10)),
]

f = forecast(obs, Thresholds(intact=400.0, ablated=150.0),
             now=t0 + timedelta(minutes=10))
f.trend_per_second   # negative (worsening)
f.worsening          # True
f.eta_ablated        # timedelta — estimated time to reach ablated threshold
f.stable             # True if trend is within uncertainty of zero
```

Requires at least 2 observations with `measured_at` timestamps.
Polarity-aware: `improving` always means "moving toward healthy."

---

## `drift.py` — Trajectory classification

Health tells you WHERE a value is. Drift tells you WHERE IT'S HEADED.

Given a sequence of timestamped observations, fits linear and quadratic
models to classify the trajectory shape, direction, and confidence.

### Drift states

| State | Meaning |
| ----- | ------- |
| `STABLE` | Value not changing meaningfully (slope within noise) |
| `DRIFTING` | Consistent linear trend in one direction |
| `ACCELERATING` | Rate of change is increasing |
| `DECELERATING` | Rate of change is decreasing (approaching plateau) |
| `REVERTING` | Was unhealthy, now heading back toward baseline |
| `OSCILLATING` | Periodic fluctuation around a center |

Direction is polarity-aware: `IMPROVING` always means "moving toward healthy,"
`WORSENING` always means "moving toward unhealthy."

### Basic usage

```python
from margin import classify_drift, DriftState, DriftDirection
from margin import Observation, Health, Confidence
from datetime import datetime, timedelta

t0 = datetime(2026, 1, 1)
obs = [
    Observation("rps", Health.INTACT, 450.0, 500.0, Confidence.HIGH,
                measured_at=t0),
    Observation("rps", Health.DEGRADED, 400.0, 500.0, Confidence.HIGH,
                measured_at=t0 + timedelta(minutes=5)),
    Observation("rps", Health.DEGRADED, 350.0, 500.0, Confidence.HIGH,
                measured_at=t0 + timedelta(minutes=10)),
]

dc = classify_drift(obs)
dc.state       # DriftState.DRIFTING
dc.direction   # DriftDirection.WORSENING
dc.rate        # negative (units/second, polarity-normalised)
dc.confidence  # Confidence tier based on sample count and R²
dc.to_atom()   # "rps:DRIFTING(WORSENING, -0.5556/s)"
```

### Ledger integration

Extract observation history from a Ledger and classify:

```python
from margin import drift_from_ledger, drift_all_from_ledger

dc = drift_from_ledger(ledger, "rps")         # one component
all_dc = drift_all_from_ledger(ledger)         # all components
```

### Policy predicates

Drift predicates follow the temporal predicate pattern — close over a
Ledger, return `PredicateFn` for use in PolicyRule conditions:

```python
from margin import (
    drift_worsening, any_drifting, any_drift_worsening,
    drift_is, drift_accelerating,
    DriftState, PolicyRule, Action, Op,
)

# Fire if any component's trajectory is worsening
rule = PolicyRule("drift-alert", any_drift_worsening(ledger),
                  Action(target="*", op=Op.RESTORE), priority=20)

# Fire if a specific component is accelerating
rule = PolicyRule("cpu-accel", drift_accelerating("cpu", ledger),
                  Action(target="cpu", op=Op.RESTORE, alpha=1.0), priority=30)
```

### Forecast composition

`DriftForecast` combines trajectory shape (drift) with threshold
crossing ETAs (forecast) in one object:

```python
from margin import drift_forecast, drift_forecast_from_ledger, Thresholds

df = drift_forecast(obs, Thresholds(intact=400.0, ablated=150.0))
df.drift.state       # DriftState.DRIFTING
df.forecast.eta_ablated  # timedelta to ablated threshold
df.summary           # "rps: DRIFTING(WORSENING), ETA ablated: 7.5m"
```

Classification logic:

1. **Oscillation check** — if slope is not significant but residuals show
   zero crossings with significant amplitude → `OSCILLATING`
2. **Slope significance** — if slope is within noise → `STABLE`
3. **Reversion** — value was unhealthy, now moving back toward baseline → `REVERTING`
4. **Acceleration** — quadratic R² significantly improves over linear → `ACCELERATING` or `DECELERATING`
5. **Otherwise** → `DRIFTING`

Confidence is derived from sample count and goodness of fit:

- 10+ samples, R² > 0.8 → `HIGH`
- 5+ samples, R² > 0.5 → `MODERATE`
- Otherwise → `LOW`

Serialisation: `DriftClassification.to_dict()` / `from_dict()` roundtrips cleanly.

---

## `predicates.py` — Expression pattern matching

Declarative rules for evaluating Expressions. Define conditions and
evaluate them against snapshots.

### Basic predicates

```python
from margin import any_health, all_health, count_health, component_health
from margin import any_degraded, confidence_below, sigma_below, any_correction
from margin import Health, Confidence

any_health(Health.ABLATED)           # True if any component is ABLATED
all_health(Health.INTACT)            # True if all components are INTACT
count_health(Health.DEGRADED, 2)     # True if 2+ components are DEGRADED
component_health("api", Health.ABLATED)  # True if "api" is ABLATED
any_degraded()                       # True if any component is non-INTACT
confidence_below(Confidence.MODERATE)    # True if net confidence is below MODERATE
sigma_below("api", -0.5)            # True if api's sigma < -0.5
any_correction()                     # True if any correction is active
```

### Combinators

```python
from margin import all_of, any_of, not_

critical = all_of(
    any_health(Health.ABLATED),
    confidence_below(Confidence.HIGH),
)
all_clear = not_(any_degraded())
```

### Rules

```python
from margin import Rule, evaluate_rules

rules = [
    Rule("all-clear", all_health(Health.INTACT)),
    Rule("critical", any_health(Health.ABLATED)),
    Rule("low-confidence", confidence_below(Confidence.MODERATE)),
]

matched = evaluate_rules(rules, expr)
# [Rule("critical"), Rule("low-confidence")]
```

---

## `bridge.py` — Reverse bridge (`to_uncertain`)

`to_uncertain()` reconstructs an `UncertainValue` from an `Observation`,
closing the algebra-health loop.

```python
from margin import to_uncertain, Observation, Health, Confidence

obs = Observation("rps", Health.DEGRADED, 350.0, 500.0, Confidence.HIGH)
uv = to_uncertain(obs)
# UncertainValue(350.0 ±17.5)  — HIGH confidence → 5% uncertainty
```

Uncertainty is inferred from the confidence tier:

| Confidence | Uncertainty (fraction of \|value\|) |
| --- | --- |
| CERTAIN | 1% |
| HIGH | 5% |
| MODERATE | 10% |
| LOW | 25% |
| INDETERMINATE | 50% |

This enables round-tripping: `observe()` → Observation → `to_uncertain()`
→ UncertainValue → feed back into the algebra.

---

## `bridge.py` — Forward bridge (UncertainValue → Observation)

`observe()` creates a typed Observation from an UncertainValue. The
confidence tier is derived automatically from the uncertainty interval's
relationship to the threshold — not defaulted to MODERATE.

```python
from margin import observe, UncertainValue, Thresholds

value = UncertainValue(point=45.0, uncertainty=8.0)
baseline = UncertainValue(point=100.0, uncertainty=2.0)
thresholds = Thresholds(intact=80.0, ablated=30.0)

obs = observe("throughput", value, baseline, thresholds)
# obs.health = Health.DEGRADED
# obs.confidence = Confidence.CERTAIN (45±8 is clearly below 80)
# obs.measured_at = datetime.now() (set from at_time)
```

`observe_many()` is the multi-component version — like `Parser.parse()`
but with uncertainty-derived confidence. Delegates correction targeting
to the same `_classify_op` logic as Parser.

`delta()` computes typed before/after observations plus the uncertain
difference with propagated uncertainty:

```python
from margin import delta, UncertainValue, Thresholds

before = UncertainValue(point=0.08, uncertainty=0.005)
after = UncertainValue(point=0.03, uncertainty=0.003)
baseline = UncertainValue(point=0.01, uncertainty=0.001)
t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)

obs_before, obs_after, diff = delta("error_rate", before, after, baseline, t)
# diff.point = -0.05, diff.uncertainty = 0.0058 (propagated)
```

---

## Quickstart

Minimum setup — observe, classify, and record in 10 lines:

```python
from margin import Parser, Thresholds, Ledger

# 1. Configure
parser = Parser(
    baselines={"rps": 500.0, "error_rate": 0.002},
    thresholds=Thresholds(intact=400.0, ablated=150.0),
    component_thresholds={
        "error_rate": Thresholds(intact=0.005, ablated=0.05, higher_is_better=False),
    },
)

# 2. Parse a measurement
expr = parser.parse({"rps": 480.0, "error_rate": 0.03})
print(expr.to_string())
# [rps:INTACT(-0.04σ)] [error_rate:DEGRADED(-14.00σ)]

# 3. Or auto-calibrate from healthy data
from margin import parser_from_calibration
parser = parser_from_calibration(
    {"rps": [490, 510, 505, 495, 500], "error_rate": [0.002, 0.001, 0.003]},
    polarities={"error_rate": False},
)
```

Full loop — observe, explain, decide, evaluate, record:

```python
from margin import (
    step, Parser, Thresholds, Policy, PolicyRule, Action, Constraint,
    Contract, HealthTarget, SustainHealth,
    CausalGraph, Ledger,
    Health, Op, any_health, any_degraded, all_health,
)

# Configure all four layers
parser = Parser(
    baselines={"api": 100.0, "db": 100.0},
    thresholds=Thresholds(intact=80.0, ablated=30.0),
)

graph = CausalGraph()
graph.add_degrades("db", "api", 0.9, evidence="db outage kills api")

policy = Policy(name="prod", rules=[
    PolicyRule("critical", any_health(Health.ABLATED),
               Action(target="*", op=Op.RESTORE, alpha=0.9),
               priority=50, constraint=Constraint(cooldown_steps=3)),
    PolicyRule("moderate", any_degraded(),
               Action(target="*", op=Op.RESTORE, alpha_from_sigma=True),
               priority=10),
    PolicyRule("clear", all_health(Health.INTACT),
               Action(target="*", op=Op.NOOP), priority=0),
])

contract = Contract("sla", terms=[
    HealthTarget("api-intact", "api", Health.INTACT),
    SustainHealth("api-stable", "api", Health.INTACT, for_steps=5),
])

# Run one step
ledger = Ledger()
expr = parser.parse({"api": 40.0, "db": 10.0})
result = step(expr, policy, ledger, graph, contract)

print(result.to_string())
# Step: [api:DEGRADED(-0.60σ)] [db:ABLATED(-0.90σ)]
#   Why: api:DEGRADED ← db:ABLATED --DEGRADES(90%)-->
#   Decision: critical (2/3 matched)
#   Action: RESTORE(target=db, α=0.90)
#   Contract: 0 met, 1 violated, 1 pending

# result.correction   — Correction(target="db", op=RESTORE, alpha=0.9)
# result.explanations  — {"api": Explanation(← db:ABLATED), "db": ...}
# result.decision      — DecisionTrace(winner="critical")
# result.contract      — ContractResult(api-intact: VIOLATED)
```

---

## Layer 3: Policy — What should happen next?

The policy language expresses decision logic as typed, prioritised rules.
An Expression goes in, a Correction or Escalation comes out, and the
full reasoning is auditable.

## `policy/core.py`

### Action

How to build a Correction when a rule matches:

- `target="*"` picks the worst degraded component
- `alpha_from_sigma=True` derives intensity from `|sigma|` (proportional response)
- Fixed `alpha` and `magnitude` for deterministic rules

### Constraint

Bounds applied after action resolution: alpha clamping (`max_alpha`,
`min_alpha`), cooldown (`cooldown_steps`), rate limiting
(`max_per_window` / `window_steps`). Returns clamped Correction or
None (suppressed).

### Escalation

What to return when the policy cannot act: `LOG`, `ALERT`, or `HALT`
with a reason string. Escalation is a value, not a side effect — the
caller decides what each level means.

### PolicyRule

Condition → action with priority and gating:

```python
PolicyRule(
    name="critical-restore",
    predicate=any_health(Health.ABLATED),
    action=Action(target="*", op=Op.RESTORE, alpha=0.9),
    priority=50,
    constraint=Constraint(cooldown_steps=3),
    min_confidence=Confidence.MODERATE,
)
```

Evaluation order: predicate → confidence gate → escalation check →
action resolve → constraint check.

### Policy

A named set of prioritised rules:

- `evaluate(expr, ledger)` — all matching results, priority-descending
- `evaluate_first(expr, ledger)` — highest priority result
- `backtest(ledger)` — replay decisions against history, includes
  `proposed_by` (which rule won at each step)

## `policy/temporal.py`

Predicates that look at ledger history:

| Function | True when |
| --- | --- |
| `health_sustained(name, health, n, ledger)` | Component in `health` for `n` consecutive steps |
| `health_for_at_least(name, healths, n, ledger)` | Component in any of `healths` for `n` steps |
| `sigma_trending_below(name, threshold, n, ledger)` | Sigma below threshold for all of last `n` steps |
| `fire_rate_above(name, rate, n, ledger)` | Fire rate exceeds `rate` over `n` steps |
| `no_improvement(name, n, ledger)` | Corrections firing but mean improvement <= 0 |

## `policy/compose.py`

- **PolicyChain**: run policies in sequence (`first` / `veto` / `all` modes)
- **CorrectionBundle**: coordinated multi-correction package
- **diff_policies / agreement_rate**: compare what two policies would do over a ledger

## `policy/tuning.py`

Learn from history:

- `analyze_backtest(policy, results)` — per-rule performance stats
- `suggest_tuning(policy, stats)` — propose alpha adjustments (reduce harmful, boost beneficial)
- `apply_tuning(policy, suggestions)` — return a new policy with adjustments applied

## `policy/trace.py`

Full decision audit trail:

```python
dt = trace_evaluate(policy, expr, ledger)
print(dt.to_string())
# DecisionTrace: 2/3 rules matched
#   [MATCH] critical (p=50) ← winner
#          RESTORE(target=api, α=0.90)
#   [MATCH] moderate (p=10)
#          RESTORE(target=api, α=0.50)
#   [skip] all-clear (p=0)
#   Result: RESTORE(target=api, α=0.90)
```

`trace_backtest(policy, ledger)` returns a `DecisionTrace` per step.
Optionally carries causal context from `CausalGraph.explain_all()`.

## `policy/validate.py`

Check policy well-formedness before running:

- Duplicate rule names → error
- Constraint conflicts (min_alpha > max_alpha, alpha outside bounds) → warning
- Priority collisions → warning
- Missing health coverage (no rule for ABLATED, etc.) → warning
- Contract component coverage (contract requires "db" but no rule targets it) → warning

```python
result = validate(policy, contract=my_contract, components=["api", "db"])
print(result.to_string())
```

---

## Layer 4: Contract — What does success look like?

The contract language declares requirements and scores the ledger
against them. Each term evaluates to MET, VIOLATED, or PENDING.

## `contract/core.py`

### Contract terms

| Term | Requirement |
| --- | --- |
| `HealthTarget(name, component, health)` | Component is at target health (or better) right now |
| `ReachHealth(name, component, health, within_steps)` | Component reaches health within N steps |
| `SustainHealth(name, component, health, for_steps)` | Component sustains health for N consecutive steps |
| `RecoveryThreshold(name, min_recovery, over_steps)` | Mean recovery ratio >= threshold over window |
| `NoHarmful(name, over_steps)` | Zero harmful corrections in window |

### Contract

A named set of terms evaluated together:

```python
contract = Contract("production-sla", terms=[
    HealthTarget("api-intact", "api", Health.INTACT),
    ReachHealth("api-recover", "api", Health.INTACT, within_steps=5),
    SustainHealth("api-stable", "api", Health.INTACT, for_steps=10),
    RecoveryThreshold("good-recovery", min_recovery=0.8, over_steps=20),
    NoHarmful("no-harm", over_steps=50),
])

result = contract.evaluate(ledger, current_expression)
print(result.to_string())
# Contract(production-sla):
#   [+] api-intact: api:INTACT vs target INTACT
#   [?] api-recover: 3/5 steps elapsed
#   [!] api-stable: DEGRADED at step 7
#   [+] good-recovery: recovery 0.8500 >= 0.8
#   [+] no-harm: 0 harmful in 50 steps
```

`result.all_met`, `result.any_violated`, `result.violated()`,
`result.pending()` for programmatic access.

---

## Layer 5: Causal — Why did this happen?

The causal language expresses dependency structure between components
as a typed directed graph.

## `causal/core.py`

### CausalLink

A typed relationship between two components:

| CauseType | Meaning |
| --- | --- |
| `DEGRADES` | A degrading causes B to degrade |
| `BLOCKS` | A failing prevents B from recovering |
| `TRIGGERS` | A's state change triggers B's state change |
| `CORRELATES` | Co-occurring, direction uncertain |
| `MITIGATES` | A's health improvement helps B recover |

Links have a `strength` (0-1) and optional `condition` (only active when
source is in a specific health state).

### CausalGraph

A DAG of causal links with query methods:

```python
graph = CausalGraph()
graph.add_degrades("db", "api", 0.9, evidence="db outage kills api")
graph.add_blocks("cache", "api", 0.5, condition=Health.ABLATED)
graph.add_degrades("api", "frontend", 0.7)

graph.causes_of("api")     # [db→api, cache→api]
graph.effects_of("db")     # [db→api]
graph.upstream("frontend")  # ["api", "db", "cache"]
graph.downstream("db")      # ["api", "frontend"]
graph.roots()               # ["db", "cache"]
```

### Explanation

`graph.explain(component, expr)` traces upstream through active causal
links (respecting conditions against current health) and returns a
typed explanation:

```python
expl = graph.explain("api", current_expression)
# api:DEGRADED ← db:ABLATED --DEGRADES(90%)-->
expl.root_cause.source  # "db"
expl.has_known_cause    # True
```

`graph.explain_all(expr)` returns explanations for every component
in the expression.

---

## The full loop

The five layers form a complete typed loop:

```text
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
  1. OBSERVE (foundation + observability)             │
     Expression: [api:DEGRADED(-0.5σ)]                │
        │                                             │
        ▼                                             │
  2. EXPLAIN (causal)                                 │
     api:DEGRADED ← db:ABLATED --DEGRADES-->          │
        │                                             │
        ▼                                             │
  3. DECIDE (policy)                                  │
     Rule "critical" matched → RESTORE(api, α=0.9)   │
     DecisionTrace: 2/3 rules matched, winner=critical│
        │                                             │
        ▼                                             │
  4. ACT                                              │
     Apply correction → new state                     │
        │                                             │
        ▼                                             │
  5. EVALUATE (contract)                              │
     [+] api-intact: MET                              │
     [?] api-stable: 2/10 steps                       │
        │                                             │
        ▼                                             │
  6. RECORD (ledger)                                  │
     step 5: [api:RECOVERING(-0.2σ) → RESTORE(α=0.9)]│
        │                                             │
        └─────────── next step ───────────────────────┘
```

Each stage has a typed vocabulary. Every decision is traceable. The
ledger accumulates the full history. The contract scores the history
against goals. The policy can be backtested, tuned, and validated
against the contract. The causal graph explains failures.

## `loop.py` — Orchestrator

### `step(expression, policy, ledger, graph, contract) -> StepResult`

Run one full iteration. All arguments except `expression` and `policy`
are optional — the loop degrades gracefully:

| Argument | If omitted |
| --- | --- |
| `ledger` | No constraint checking, no contract history |
| `graph` | No causal explanations |
| `contract` | No goal evaluation |

```python
from margin import step

result = step(expr, policy)                          # minimal
result = step(expr, policy, ledger)                  # with constraints
result = step(expr, policy, ledger, graph)           # with explanations
result = step(expr, policy, ledger, graph, contract) # full loop
```

### `StepResult` (dataclass)

| Property | Type | Meaning |
| --- | --- | --- |
| `expression` | Expression | the input state |
| `explanations` | dict[str, Explanation] | causal explanations per component |
| `decision` | DecisionTrace | full policy audit trail |
| `correction` | Correction or None | what to apply |
| `escalation` | Escalation or None | if policy escalated |
| `contract` | ContractResult or None | goal evaluation |
| `acted` | bool | True if a real correction was produced (not NOOP) |
| `escalated` | bool | True if the policy escalated |
| `contract_met` | bool or None | True/False/None if no contract |

`to_string()` renders a human-readable summary. `to_dict()` serializes
everything.

### `run(expressions, policy, graph, contract, ledger) -> (results, ledger)`

Iterate `step()` over a list of Expressions, building the ledger as
it goes. Each step's observation is recorded as a Record. Returns
the list of StepResults and the final ledger.

```python
from margin import run, Parser, Thresholds, Policy, PolicyRule, Action, Op, any_degraded

parser = Parser(baselines={"x": 100.0}, thresholds=Thresholds(intact=80.0, ablated=30.0))
expressions = [parser.parse({"x": v}) for v in [90, 50, 30, 60, 85]]

policy = Policy(name="p", rules=[
    PolicyRule("restore", any_degraded(), Action(target="*", op=Op.RESTORE, alpha=0.5)),
])

results, ledger = run(expressions, policy)
# len(results) == 5
# ledger has 5 records
# results[0].acted == False (INTACT)
# results[1].acted == True  (DEGRADED → RESTORE)
```

This is for backtesting and replay — it does NOT simulate correction
outcomes. Each Expression is taken as-is.

---

## Structure

```text
margin/
├── margin-language.md        This document
│
│ Foundation
├── confidence.py             Confidence tiers with ordering
├── validity.py               Temporal decay / event invalidation
├── provenance.py             Correlation detection
├── uncertain.py              UncertainValue with epistemic metadata
├── algebra.py                Uncertainty propagation
├── health.py                 Health, Thresholds, classify, SEVERITY
├── observation.py            Observation, Correction, Expression, Parser, Op
├── ledger.py                 Record, Ledger, windowing
│
│ Observability
├── bridge.py                 observe(), observe_many(), delta(), to_uncertain()
├── calibrate.py              calibrate(), parser_from_calibration()
├── composite.py              CompositeObservation
├── diff.py                   Expression diffing
├── events.py                 EventBus
├── drift.py                  Trajectory classification, predicates, forecast composition
├── forecast.py               Trend projection
├── predicates.py             Pattern matching, combinators, Rule
├── transitions.py            State transition tracking
│
│ Contract + Causal
├── contract.py               HealthTarget, ReachHealth, SustainHealth, etc.
├── causal.py                 CausalGraph, CausalLink, Explanation
│
│ Loop
├── loop.py                   step(), run(), StepResult
│
│ Policy
└── policy/
    ├── core.py               Action, Constraint, Escalation, PolicyRule, Policy
    ├── temporal.py           History-aware predicates
    ├── compose.py            PolicyChain, CorrectionBundle, policy diffing
    ├── tuning.py             Backtest analysis, alpha tuning
    ├── trace.py              DecisionTrace, full audit trail
    └── validate.py           Well-formedness checking
```
