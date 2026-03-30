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

```text
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
| --- | --- | --- |
| **Measurement** | `Observation` | "Component C had value v, giving health H with confidence Q" |
| **Intervention** | `Correction` | "Operation Op is being applied to component C with intensity α" |
| **Composition** | `Expression` | "The joint state of all components, with net confidence = weakest measurement" |

Each layer is **monotone in confidence**: composing measurements cannot
produce an expression whose `confidence` exceeds the weakest constituent.
The language never overclaims.

### Health as a partial order

```text
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

> **Known limitation — sigma overloading:** Sigma is used as a health classifier
> input, a correction intensity source (`alpha_from_sigma`), and an improvement
> metric simultaneously. A component at σ=−0.5 drifting slowly is not the same
> situation as one at σ=−0.5 after a sudden drop, but the language treats them
> identically until you add drift. Use `full_step()` which composes sigma with
> drift classification. Do not use `alpha_from_sigma=True` without also checking
> drift direction — a correction sized by sigma alone on an ACCELERATING component
> will frequently be undersized.

### Edge-case semantics

These behaviours are **specified**, not accidental:

| Input | Behaviour | Rationale |
| --- | --- | --- |
| `values = {}` (empty) | `to_string()` → `"[∅]"`, `corrections = []` | No measurement → null expression |
| `value = NaN` | Falls through to DEGRADED (NaN comparisons are False) | Sensor failure produces the least-confident non-OOD state |
| `value = +∞` (higher_is_better) | INTACT | Infinite value → trivially healthy |
| `value = +∞` (lower_is_better) | ABLATED | Infinite value → maximally unhealthy |
| `value = -∞` (higher_is_better) | ABLATED | Negative-infinite value → maximally unhealthy |
| `value = -∞` (lower_is_better) | INTACT | Negative-infinite value → trivially healthy |
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

```text
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

Set `active_min` to ~5–10% of the smallest correction magnitude meaningful
for this component. The default of 0.05 suits normalized (0–1) values; for
raw-unit components, scale proportionally to your correction range.

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

Or auto-calibrate from known-healthy data:

```python
from margin import parser_from_calibration

parser = parser_from_calibration(
    {"throughput": [980, 1010, 1000], "error_rate": [0.001, 0.0008, 0.0012]},
    polarities={"error_rate": False},
)
```

---

## `confidence.py` — Confidence

Ordinal tiers for how much an uncertainty interval overlaps a decision
boundary. Supports comparison operators (`Confidence.HIGH > Confidence.LOW`).

| Tier | Meaning | Overlap with boundary |
| --- | --- | --- |
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
| --- | --- | --- |
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
| --- | --- | --- |
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
| --- | --- |
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
| --- | --- |
| `INTACT` | value at or past the intact threshold (healthy direction) |
| `DEGRADED` | between ablated and intact thresholds |
| `ABLATED` | value past the ablated threshold (unhealthy direction) |
| `RECOVERING` | sub-threshold but active correction is running |
| `OOD` | measurement confidence is INDETERMINATE |

### `Thresholds` (dataclass)

| Field | Type | Meaning |
| --- | --- | --- |
| `intact` | float | boundary for calling healthy |
| `ablated` | float | boundary for calling failed |
| `higher_is_better` | bool | polarity (default True) |
| `active_min` | float | minimum correction magnitude for "active" (default 0.05). Set to ~5–10% of the smallest correction magnitude meaningful for this component. The default suits normalized (0–1) values; for raw-unit components, scale proportionally to your correction range. |

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

```text
confidence == INDETERMINATE           → OOD
thresholds.is_intact(value)           → INTACT
thresholds.is_ablated(value)
    correcting                        → RECOVERING
    else                              → ABLATED
else
    correcting                        → RECOVERING
    else                              → DEGRADED
```

> **Known limitation — RECOVERING is a function of correction input, not component
> state:** `Health.RECOVERING` means "would be ABLATED without the active
> correction." If the correction is withdrawn, the component snaps back to ABLATED
> with no state change. A component that has been RECOVERING for 50 steps with no
> improvement looks identical to one that just entered RECOVERING. Use
> `no_improvement()` from `policy/temporal.py` to detect stall: it fires when
> corrections are running but mean improvement is ≤ 0.

---

## `observation.py` — Observations, Corrections, Expressions

### `Observation` (dataclass)

One component's health at one measurement.

| Field | Type | Meaning |
| --- | --- | --- |
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

```text
throughput:INTACT(-0.05σ)
error_rate:DEGRADED(-4.00σ)
cpu:OOD
```

### `Op` (Enum)

| Value | Meaning |
| --- | --- |
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
| --- | --- | --- |
| `target` | str | Which component |
| `op` | Op | Operation type |
| `alpha` | float | Intensity coefficient (0 = none, 1 = full) |
| `magnitude` | float | Size of the correction |
| `triggered_by` | list[str] | Names of degraded components that triggered this |

`is_active()` returns True when `op != NOOP` and `alpha > 0`.

### `Expression` (dataclass)

Composed snapshot of all observations and corrections at one moment.

| Field | Type | Meaning |
| --- | --- | --- |
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

```text
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
| --- | --- | --- |
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
ledger = Ledger(label="my-run")                   # unbounded
ledger = Ledger(label="my-run", max_records=1000)  # cap at 1000 records (oldest dropped)
ledger.append(record)
```

| Property | Returns |
| --- | --- |
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

> **Known limitation — Ledger grows without bound by default:** The `Ledger` is
> append-only with no default compaction. For long-running loops, use
> `Ledger(max_records=N)` to cap memory. The `window()` and `last_n()` methods
> create windowed views but do not prevent the underlying list from growing.

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

   preserve exact values for all fields. `alpha` and `magnitude` on
   `Correction` are stored without rounding. `sigma` is not stored in
   the serialized dict — it is recomputed from `value` and `baseline`
   on access.

10. **Severity is defined once.** The `SEVERITY` dict in `health.py`

    is the single source of truth for Health ordering. `diff.py` and
    `composite.py` import it rather than defining their own.

### Violation behavior

These invariants are enforced for Expressions constructed through
`Parser.parse()`. Callers who construct `Observation`, `Correction`, or
`Expression` directly (e.g. for testing or deserialization) are
responsible for maintaining them. The table below documents what happens
when each invariant is violated.

| # | Reachable via | Runtime consequence | Enforced in code | Action |
| --- | --- | --- | --- | --- |
| 1 | Direct `Observation` construction with wrong `health` field | Silent misclassification. `INTACT(-0.95σ)` is visible in rendered output but no exception raised. Policy predicates act on the wrong health state. | No | Document-only. A validator would require threading `Thresholds` into `Observation`, breaking `from_dict()`. |
| 2 | Constructing `Observation` with `higher_is_better` mismatched to the `Thresholds` that classified it | Sigma inverts. A WORSENING drift reads as IMPROVING. `improvement` and `recovery_ratio` compute the wrong direction. No exception. | No | Document-only. `higher_is_better` on `Observation` is deliberately redundant provenance for consumers without access to `Thresholds`. |
| 3 | Constructing `Expression` with `health=RECOVERING` on an `Observation` but no active `Correction` targeting it | Component appears in `degraded()`. `no_improvement()` will not fire on it (it checks the ledger for fired corrections, not the Expression health field). No exception. | No | Document-only. Only reachable by bypassing `Parser`. |
| 4 | Only reachable via invariant 2 violation | See invariant 2. `sigma` is derived from `higher_is_better`. | Corollary of 2 | — |
| 5 | Direct `Expression` construction with `confidence` higher than the weakest observation | Silent overclaim. `min_confidence` policy gates fire when they should not. | **Yes — warning** emitted by `Expression.__post_init__`. Not a raise (would break `from_dict()`). | Warning on construction. |
| 6 | Computing derived `UncertainValue` outside `algebra.py` without preserving provenance | `combine_uncertainties()` treats correlated values as independent, uses quadrature instead of linear combination, underestimates uncertainty intervals. No exception. | No | Document-only. Provenance is opt-in metadata. |
| 7 | Constructing an `Expression` with `health=RECOVERING` on a component whose `Correction.magnitude` is below `active_min` | `Correction.is_active()` returns True (it checks `op != NOOP and alpha > 0`, not `active_min`). Policy rules keyed on `any_correction()` fire incorrectly. | No | Document-only. `is_active()` cannot check `active_min` without access to `Thresholds`. Callers needing `active_min`-aware activity must compare `correction.magnitude` against `thresholds.active_min` themselves. |
| 8 | Direct construction of `Observation` with `health=ABLATED` at a value exactly equal to the `ablated` threshold | Silent misclassification at the boundary. Component treated as failed rather than degraded. | No | Document-only. Invariant is enforced by `classify()` via strict inequality (`<` / `>`). Only reachable by bypassing `classify()`. |
| 9 | N/A — lossless by construction | — | Yes — no rounding in `to_dict()` | — |
| 10 | Adding a new module that defines its own `SEVERITY` dict | Health ordering inconsistencies between modules. `diff.py` and `composite.py` would disagree on relative severity. | By convention | Any new file comparing Health severity must import `SEVERITY` from `health.py`. |

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
| --- | --- |
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

## Baseline recalibration

Baselines are set at `Parser` construction and used for all subsequent sigma
calculations. In long-running systems the "healthy operating point" can shift —
a model after fine-tuning, a sensor after recalibration, a biochem process that
has stabilised at a new equilibrium. Sigma calculated against a stale baseline
produces systematically misleading health classifications with no runtime signal
that the baseline is the problem.

### When to recalibrate

Three signals indicate the baseline may no longer be representative:

**Signal A — mean shift:** the component's recent values are centred away from
the original calibration baseline. Use `check_distribution()` from `anomaly.py`
with the original calibration samples as reference:

```python
from margin import check_distribution

ds = check_distribution(recent_window, calibration_samples)
if abs(ds.mean_shift) > 0.2:   # >20% relative shift
    print("baseline drift detected")
```

**Signal B — spread change:** variance has doubled or halved relative to
calibration (e.g. a newly-stabilised process, or one that has become noisier).
Check `ds.std_ratio > 2.0` or `ds.std_ratio < 0.5`.

**Signal C — anomaly-at-healthy-sigma:** `AnomalyTracker.state == ANOMALOUS`
for K consecutive steps despite sigma near 0 or positive. The component looks
healthy by health classification but statistically unusual relative to its
calibration reference. This is the "silent staleness" pattern — check it
externally via your Monitor's anomaly tracker.

Or use the combined helper:

```python
from margin import needs_recalibration, needs_recalibration_many

# Single component
if needs_recalibration(calibration_samples, recent_window):
    # Signal A or B has fired — consider recalibrating

# Multiple components at once — returns {component: bool}
flags = needs_recalibration_many(
    calibration_samples={"cpu": cpu_cal, "mem": mem_cal},
    recent_samples={"cpu": cpu_recent, "mem": mem_recent},
)
stale = [c for c, flag in flags.items() if flag]
```

Both functions check signals A and B. Signal C must be checked
against `monitor.anomaly(component).state`. Components missing from
either dict are omitted from `needs_recalibration_many()` results.

### `recalibrate_parser(parser, new_healthy_data, ...) -> (Parser, results)`

Recalibrate one or more components and return a **new** `Parser`. The input
parser is never mutated. Components not in `new_healthy_data` keep their
existing baselines verbatim — partial recalibration never drops a component.

```python
from margin import recalibrate_parser, needs_recalibration, save_monitor, Monitor

# 1. Detect
if needs_recalibration(original_samples["throughput"], recent_window):

    # 2. Collect new healthy data (caller must identify a known-good window)
    new_data = {"throughput": collect_healthy_window()}

    # 3. Recalibrate — returns new Parser and CalibrationResult per component
    new_parser, results = recalibrate_parser(old_parser, new_data)
    print(results["throughput"].baseline)   # new baseline value

    # 4. Optional: checkpoint old Monitor state for audit trail
    save_monitor(old_monitor, "checkpoint_before_recal.json")

    # 5. Rebuild Monitor with the new Parser, preserving per-tracker windows
    new_monitor = Monitor(
        new_parser,
        drift_window=old_monitor.drift_window,
        anomaly_window=old_monitor.anomaly_window,
        correlation_window=old_monitor.correlation_window,
    )
```

| Arg | Default | Notes |
| --- | --- | --- |
| `parser` | required | The existing Parser to replace |
| `new_healthy_data` | required | `{component: [measurements]}` |
| `polarities` | `{}` | Override polarity per component; defaults to existing parser polarity |
| `components` | `None` | Allowlist — if None, all keys in `new_healthy_data` are recalibrated |
| `intact_fraction` | `0.70` | Same semantics as `calibrate()` |
| `ablated_fraction` | `0.30` | Same semantics as `calibrate()` |
| `use_std` | `False` | If True, use `baseline ± N×std` thresholds instead of fraction-of-mean |
| `active_min` | inherited | Inherited from the existing parser's thresholds; pass explicitly to override |

### Transition semantics

**Historical Ledger data** — `Observation` objects already in a `Ledger` were
classified against the old parser. Their `.sigma` property recomputes from the
embedded `.baseline` field set at parse-time. Recalibration does not rewrite
history. This is correct: the Ledger is an immutable record of what was
observed under which baseline.

**Monitor tracker windows** — after recalibrating, update the Monitor's parser
and flush the anomaly reference windows so old-baseline values do not
contaminate classifications during warm-up:

```python
new_parser, _ = recalibrate_parser(old_parser, new_data)
monitor.parser = new_parser
monitor.reset_anomaly_reference()   # clears AnomalyTracker windows
# AnomalyTrackers return None for anomaly_min_reference steps, then reclassify
# against the new baseline. DriftTracker windows need not be reset — old
# observations are outweighed as new ones accumulate.
```

Use `save_monitor()` / `load_monitor()` to checkpoint the old state before
rebuilding if you want an audit trail of the transition.

**Marking the boundary** — fire an event on the EventBus so downstream
consumers can identify the recalibration point in the audit trail:

```python
bus.fire("recalibrated")
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
| --- | --- |
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

## `anomaly.py` — Statistical outlier detection

Health tells you IS IT GOOD. Drift tells you IS IT CHANGING.
Anomaly tells you IS THIS NORMAL. A value can be INTACT and STABLE
but at a level never seen before — anomaly catches that.

### Anomaly states

| State | Meaning |
| --- | --- |
| `EXPECTED` | Value within normal statistical range of reference data |
| `UNUSUAL` | Uncommon (beyond 2σ) but not extreme |
| `ANOMALOUS` | Statistical outlier (beyond 3σ) |
| `NOVEL` | Outside the entire historical range — never seen before |

### Point anomaly

Classify a single value against a reference distribution:

```python
from margin import classify_anomaly, AnomalyState

ref = [100, 101, 99, 100, 102, 98, 101, 99, 100, 100]

ac = classify_anomaly(100.5, ref, component="cpu")
ac.state       # AnomalyState.EXPECTED
ac.z_score     # how many σ from historical mean
ac.is_novel    # True if outside historical range entirely

ac = classify_anomaly(500.0, ref, component="cpu")
ac.state       # AnomalyState.NOVEL
```

From Observation objects:

```python
from margin import classify_anomaly_obs

ac = classify_anomaly_obs(current_observation, history_observations)
```

### Distribution shift

Compare two sample distributions (recent vs reference) for
mean shift, spread change, or shape change:

```python
from margin import check_distribution

ds = check_distribution(recent_values, reference_values, component="cpu")
ds.mean_shift       # relative change in mean
ds.std_ratio        # recent_std / ref_std
ds.kurtosis_delta   # change in tail weight
ds.skew_delta       # change in asymmetry
ds.state            # EXPECTED, UNUSUAL, or ANOMALOUS
ds.shifted          # True if state != EXPECTED
```

### Jump detection

Find sudden discontinuities — values that teleport rather than drift:

```python
from margin import detect_jumps

jumps = detect_jumps(observations, jump_threshold=3.0)
for j in jumps:
    j.magnitude_sigma   # robust z-score vs local window of surrounding diffs
    j.value_before      # value before the jump
    j.value_after       # value after the jump
    j.at_index          # index in observation sequence
```

### Anomaly from Ledger

```python
from margin import anomaly_from_ledger, anomaly_all_from_ledger
from margin import distribution_shift_from_ledger

ac = anomaly_from_ledger(ledger, "cpu")            # latest value vs history
all_ac = anomaly_all_from_ledger(ledger)            # all components
ds = distribution_shift_from_ledger(ledger, "cpu")  # recent vs earlier
```

### Anomaly predicates

```python
from margin import any_anomalous, any_novel, is_novel, anomaly_is
from margin import PolicyRule, Action, Op, AnomalyState

rule = PolicyRule("anomaly-alert", any_anomalous(ledger),
                  Action(target="*", op=Op.RESTORE), priority=25)

rule = PolicyRule("novel-halt", any_novel(ledger),
                  Action(target="*", op=Op.NOOP), priority=40)
```

Thresholds are configurable: `unusual_threshold` (default 2σ),
`anomalous_threshold` (default 3σ), `novel_margin` (default 10% beyond range).

Serialisation: all types support `to_dict()` / `from_dict()` roundtrip.

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

## `bridge.py` — UncertainValue ↔ Observation

`bridge.py` is the connection point between the uncertainty algebra layer
(`UncertainValue`) and the health classification layer (`Observation`). Use
it when your values already carry explicit uncertainty structure and you want
confidence derived from the uncertainty interval, rather than the fixed
`Confidence.HIGH` default that `Parser.parse()` applies.

Use `Parser.parse()` for the common case — raw floats, one call, done.
Use `bridge.observe()` / `observe_many()` when upstream code already
produces `UncertainValue` objects (e.g., from `algebra.py` operations or
sensor fusion), or when you need `to_uncertain()` to re-enter the algebra
after classification.

### Reverse bridge — `to_uncertain()`

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

> **Known limitation — confidence cardinality leakage:** These fractions are
> author-chosen conventions with no empirical grounding. `Confidence.HIGH` does
> not mean "90% confident" and the 5% fraction is not derived from measurement
> theory. The round-trip `observe() → to_uncertain()` produces numbers that look
> precise but are derived from a discretization boundary. Do not interpret the
> resulting uncertainty intervals as calibrated probabilities in downstream algebra.

This enables round-tripping: `observe()` → Observation → `to_uncertain()`
→ UncertainValue → feed back into the algebra.

---

### Forward bridge — `observe()`

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

> **Known limitation — no prospective cost model:** Rules fire based on
> conditions. High-alpha and low-alpha corrections differ only in intensity —
> there is no representation of what a correction costs or what side effects it
> may have. The `harmful()` ledger query is retrospective; it catches corrections
> that degraded something else after the fact. For domains where corrections have
> real consequences, pair the policy layer with an Intent to constrain which
> corrections are acceptable given the goal.

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

Each link carries:

| Field | Type | Meaning |
| --- | --- | --- |
| `strength` | float 0–1 | How strongly the cause affects the target |
| `condition` | Optional[Health] | Link is only active when source is in this state |
| `evidence` | Optional[str] | Free-text rationale for asserted links |
| `origin` | str | `"asserted"` (manually declared) or `"discovered"` (auto_causal_graph) |

### CausalGraph

A DAG of causal links with query methods:

```python
graph = CausalGraph()
graph.add_degrades("db", "api", 0.9, evidence="db outage kills api")
graph.add_blocks("cache", "api", 0.5, condition=Health.ABLATED)
graph.add_degrades("api", "frontend", 0.7)

graph.causes_of("api")          # [db→api, cache→api]
graph.effects_of("db")          # [db→api]
graph.upstream("frontend")       # ["api", "db", "cache"]
graph.downstream("db")           # ["api", "frontend"]
graph.roots()                    # ["db", "cache"]
graph.asserted_links()           # links added via add_degrades / add_blocks / etc.
graph.discovered_links()         # links produced by auto_causal_graph()
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

### Auto-correlation (`correlate.py`)

Discover which components move together from observation data,
then feed the results into CausalGraph automatically.

```python
from margin import correlate, correlate_from_ledger, auto_causal_graph

# From raw value series
matrix = correlate({"cpu": cpu_values, "mem": mem_values, "disk": disk_values},
                   min_correlation=0.7, max_lag=3)
matrix.strongest(3)             # top 3 correlations
matrix.for_component("cpu")     # all correlations involving cpu
matrix.coefficient("cpu", "mem") # pairwise r value

# From a Ledger
matrix = correlate_from_ledger(ledger, min_correlation=0.7, max_lag=3)

# One-step: discover + build graph
graph = auto_causal_graph(ledger, existing=manual_graph, min_correlation=0.7)
```

Each `Correlation` has:

- `coefficient` — Pearson r (-1 to +1)
- `lag` — positive means A leads B by N steps
- `strength` — |r|
- `to_causal_link()` — converts to a CausalLink (CORRELATES or DEGRADES based on sign/lag)

Lag detection: when `max_lag > 0`, tests all lags from -max_lag to +max_lag
and returns the lag with the strongest absolute correlation. If A leads B,
the causal direction is inferred as A → B.

> **Known limitation — asserted vs. discovered causal links have different
> epistemic status:** `add_degrades("A", "B")` and `auto_causal_graph(ledger)`
> both produce `CausalLink` objects, but with very different reliability.
> Asserted links are authoritative declarations. Discovered links are
> correlational heuristics — lag-based causal direction is plausible but wrong
> when confounders, feedback loops, or coincident cycles exist. All links carry
> an `origin` field (`"asserted"` or `"discovered"`). Use `graph.asserted_links()`
> and `graph.discovered_links()` to query them separately. Do not treat discovered
> DEGRADES links as ground truth in high-stakes domains.

Predicates for policy rules:

```python
from margin import correlated_with, any_new_correlation

# True if cpu and mem are correlated above threshold
rule = PolicyRule("corr-alert", correlated_with("cpu", "mem", ledger),
                  Action(target="*", op=Op.RESTORE), priority=15)

# True if ledger shows correlations not in the baseline graph
rule = PolicyRule("new-corr", any_new_correlation(ledger, baseline_graph),
                  Action(target="*", op=Op.NOOP), priority=5)
```

---

## Layer 6: Intent — Can we still make it?

Health says where components are. Drift says where they're headed.
Intent says whether the system can still achieve its goal.

### Intent definition

```python
from margin import Intent, Health

intent = (Intent(goal="deliver package to dock 7", deadline_seconds=900)
          .require("battery_soc", min_value=20.0)
          .require("navigation", min_health=Health.DEGRADED)
          .require("wifi", min_health=Health.ABLATED, critical=False))
```

Requirements can specify:

- `min_health` — minimum acceptable Health state
- `min_value` — numeric threshold (polarity-aware)
- `critical` — if True, violation → INFEASIBLE; if False → AT_RISK

### Evaluation

```python
result = intent.evaluate(expression, drift_by_component)
result.feasibility   # FEASIBLE / AT_RISK / INFEASIBLE / UNKNOWN
result.risks         # [RiskFactor(battery_soc: DRIFTING(WORSENING), ETA 600s)]
result.met           # ["navigation", "wifi"]
result.violated      # []
result.trending_bad  # ["battery_soc"]
result.summary()     # "AT_RISK — battery_soc: DRIFTING(WORSENING), ETA 10.0m"
```

Or directly from a Monitor:

```python
result = intent.evaluate_monitor(monitor)
```

### Feasibility states

| State | Meaning |
| --- | --- |
| `FEASIBLE` | All requirements met, trajectories OK |
| `AT_RISK` | Requirements met now, but drift threatens deadline |
| `INFEASIBLE` | Requirements already violated or will be before deadline |
| `UNKNOWN` | Insufficient data to evaluate |

### Drift-aware ETA

When a component is DRIFTING(WORSENING) and has a `min_value` requirement,
intent projects when the value will cross the threshold using the drift rate.
If the ETA is before the deadline, the intent is AT_RISK.

This is the layer that turns margin from monitoring into decision support.

---

## The full loop

The correction loop has six stages; `full_step()` adds a seventh —
goal feasibility — via `Intent.evaluate()`:

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
        ▼                                             │

  7. FEASIBILITY (intent)           [optional]        │

     FEASIBLE — ETA to violation: 42 steps            │
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

### `full_step(monitor, values, policy, ...) -> FullStepResult`

Run all six stages in one call:

1. `Monitor.update()` → health + drift + anomaly + correlation
2. `step()` → explain + decide + contract
3. `Intent.evaluate()` → goal feasibility

```python
from margin import full_step, Monitor, Parser, Thresholds, Policy, Intent

result = full_step(monitor, {"cpu": 48, "mem": 65}, policy,
                   graph=graph, contract=contract, intent=intent)

result.expression       # current health
result.drift            # {component: DriftClassification}
result.anomaly          # {component: AnomalyClassification}
result.correlations     # CorrelationMatrix
result.step.correction  # what to do
result.step.explanations  # why (causal)
result.step.contract    # are we meeting goals?
result.intent           # IntentResult: FEASIBLE / AT_RISK / INFEASIBLE
result.feasible         # bool shorthand
result.to_string()      # human-readable summary of all layers
```

---

## `streaming.py` — Incremental trackers

For production monitoring loops where you get one measurement at a time.
Instead of collecting lists and recomputing, trackers maintain bounded
windows and reclassify on each update.

### Per-component trackers

```python
from margin import DriftTracker, AnomalyTracker

# Drift: is this value's trajectory changing?
dt = DriftTracker("cpu", window=100)
dt.update(observation)       # or dt.update_value(80.0, baseline=100.0)
dt.state                     # DriftState.DRIFTING
dt.direction                 # DriftDirection.WORSENING
dt.classification            # full DriftClassification

# Anomaly: is this value statistically unusual?
at = AnomalyTracker("cpu", window=100, min_reference=10)
at.update(80.0)              # classify against reference window, then add
at.state                     # AnomalyState.EXPECTED
at.last_jump                 # Jump if a discontinuity was detected
```

### Correlation tracker

```python
from margin import CorrelationTracker

ct = CorrelationTracker(["cpu", "mem", "disk"], window=50)
ct.update({"cpu": 80, "mem": 60, "disk": 45})
ct.matrix                    # CorrelationMatrix
ct.strongest(3)              # top 3 correlations
```

All components must be present in each update for alignment.

> **Known limitation — Monitor assumes aligned, synchronous updates:**
> `CorrelationTracker` silently skips partial updates — if any component is
> missing from a call to `ct.update()`, that step is not recorded. In
> multi-sensor environments with different update rates, late arrivals, or
> components that go silent, the effective correlation window will be shorter
> than the configured `window`. Staleness checking exists on individual
> Observations via `is_fresh()` but is not integrated into `Monitor.update()`.

### Monitor — unified streaming

Wraps a Parser and all trackers. One call updates everything:

```python
from margin import Monitor, Parser, Thresholds

parser = Parser(
    baselines={"cpu": 50.0, "error_rate": 0.002},
    thresholds=Thresholds(intact=40.0, ablated=10.0),
    component_thresholds={
        "error_rate": Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
    },
)

monitor = Monitor(parser, window=100)

while True:
    readings = get_sensor_data()
    expr = monitor.update(readings)   # health + drift + anomaly + correlation

    monitor.expression                # current Expression
    monitor.drift("cpu")              # DriftClassification
    monitor.anomaly("cpu")            # AnomalyClassification
    monitor.correlations              # CorrelationMatrix
    monitor.status()                  # full snapshot dict
```

All windows are bounded, so memory is constant regardless of how long the loop runs.

### Window configuration

Different concerns operate on different timescales. Drift detection needs a
short window to catch fast degradation; anomaly detection needs a longer
reference to produce stable z-scores; correlation needs even more history to
separate structure from noise.

```python
from margin import Monitor, WindowConfig

# Named parameters — backward-compatible, all default to `window` when omitted
monitor = Monitor(parser, window=100, drift_window=50, anomaly_window=200, correlation_window=500)

# Or via WindowConfig for config-driven setups
monitor = Monitor(parser, window=100, window_config=WindowConfig(drift=50, anomaly=200, correlation=500))
```

Domain guidance — recommended window ratios (`anomaly ≈ 4× drift`, `correlation ≈ 10× drift`):

| Domain | Step interval | `drift_window` | `anomaly_window` | `correlation_window` | Notes |
| --- | --- | --- | --- | --- | --- |
| Ops / infra monitoring | 1–5 s | 50 | 200 | 500 | Drift catches fast degradation; correlation needs structural history |
| ML training | epoch or batch | 20 | 100 | 200 | Training curves are slow; short drift window catches overfitting knee |
| Biochem pipeline | 100 ms – 1 s | 10 | 30 | 100 | Fast reaction timescales; correlation needs multiple reaction cycles |
| Game loop | frame (16 ms) | 100 | 500 | 1000 | Frames are noisy; drift needs seconds of averaging before classifying |
| Robot / actuator | 10–50 ms | 30 | 100 | 300 | Covers multiple task cycles for meaningful correlation |

Note: `anomaly_min_reference` must be less than `anomaly_window`. The default
is 10; if you set `anomaly_window` below 10, lower `anomaly_min_reference`
accordingly or the AnomalyTracker will never fire.

---

## `persist.py` — Save, restore, and replay

### Save and restore Monitor state

```python
from margin import Monitor, save_monitor, load_monitor

monitor = Monitor(parser)
# ... run for a while ...

save_monitor(monitor, "state.json")

# Later, after restart:
monitor = load_monitor("state.json", parser)
# All tracker windows, step count, and drift classifications restored
```

The state file is plain JSON. Drift trackers are reclassified from
restored observations on load. Per-tracker window configuration
(`drift_window`, `anomaly_window`, `correlation_window`) is saved and
restored automatically.

### Batch replay

Feed historical data through a Monitor and get typed analysis:

```python
from margin import replay, replay_csv

# From a list of dicts
data = [{"cpu": 80, "mem": 60}, {"cpu": 75, "mem": 58}, ...]
monitor, snapshots = replay(parser, data)

# From a CSV file
monitor, snapshots = replay_csv(parser, "metrics.csv", timestamp_column="time")
```

Each snapshot is a `monitor.status()` dict. After replay:

- `monitor.drift("cpu")` — trajectory over the full history
- `monitor.anomaly("cpu")` — final anomaly state
- `monitor.correlations` — discovered correlations
- `snapshots` — per-step analysis for retrospective review

---

## CLI — `python -m margin`

Three commands, no Python required:

```bash
# One-shot: classify values from the command line
python -m margin status --config margin.json cpu=48 mem=65

# Stream: read JSON lines from stdin, print health/drift/anomaly
echo '{"cpu": 48, "mem": 65}' | python -m margin monitor --config margin.json

# Replay: analyze a CSV file
python -m margin replay --config margin.json --data metrics.csv --output analysis.json
```

The monitor command also accepts `key=value,key=value` CSV format on stdin.
Drift and anomaly annotations are printed when non-trivial (not STABLE/EXPECTED).

---

## `config.py` — Config-driven setup

Define Parser, Policy, and Contract from a dict, JSON, or YAML file
instead of writing Python.

### Minimal config

```yaml
components:
  cpu:
    baseline: 50
    intact: 80
    ablated: 30
  error_rate:
    baseline: 0.002
    intact: 0.01
    ablated: 0.10
    lower_is_better: true

policy:
  - name: critical
    when: any_ablated
    action: {op: RESTORE, alpha: 1.0}
    priority: 50

contract:
  - name: cpu-healthy
    component: cpu
    health: INTACT
```

### Full config — all supported fields

```yaml
# Top-level name (optional, used as Policy name)
name: my-policy

# Allow all matching rules to fire in one step (default: false)
multi_rule: false

# Fallback thresholds for components without explicit values (optional)
default_thresholds:
  intact: 80
  ablated: 30
  higher_is_better: true

components:
  cpu:
    baseline: 50
    intact: 80
    ablated: 30
    higher_is_better: true   # default; use lower_is_better: true as shorthand

  error_rate:
    baseline: 0.002
    intact: 0.01
    ablated: 0.10
    lower_is_better: true    # shorthand for higher_is_better: false

  latency_ms:
    baseline: 100
    intact: 150
    ablated: 500
    lower_is_better: true
    # Per-health display labels (override enum names in to_atom() / health_label)
    labels:
      INTACT: OK
      DEGRADED: SLOW
      ABLATED: CRITICAL

policy:

  - name: critical
    when: any_ablated
    priority: 50
    min_confidence: moderate   # rule skipped when expression confidence < this
    action:
      op: RESTORE
      target: cpu              # specific component; omit or use "*" for worst-degraded
      alpha: 1.0
      magnitude: 1.0
      alpha_from_sigma: false  # when true, alpha is derived from |sigma| at runtime
      magnitude_from_sigma: false  # when true, magnitude is derived from |sigma|
    constraint:
      cooldown_steps: 3        # steps to wait between firings of this rule
      max_per_window: 5        # max firings within window_steps
      window_steps: 20
      min_alpha: 0.1           # clamp derived alpha from below
      max_alpha: 0.9           # clamp derived alpha from above
    escalation:
      level: ALERT             # LOG | ALERT | CRITICAL | PAGE
      reason: "CPU ablated — human review needed"

  - name: maintain
    when: any_degraded
    priority: 10
    action: {op: RESTORE, alpha: 0.5}

  - name: normal
    when: all_intact
    priority: 0
    action: {op: NOOP}

contract:

  # HealthTarget — component must be at target health right now
  - type: health_target
    name: cpu-healthy
    component: cpu
    target: INTACT             # INTACT | DEGRADED | ABLATED | RECOVERING | OOD

  # ReachHealth — component must reach target health within N steps
  - type: reach_health
    name: error-rate-recovery
    component: error_rate
    target: INTACT
    within_steps: 10

  # SustainHealth — component must stay at target health for N consecutive steps
  - type: sustain_health
    name: latency-sustained
    component: latency_ms
    target: INTACT
    for_steps: 5

  # RecoveryThreshold — component must recover above a raw value threshold
  - type: recovery_threshold
    name: cpu-above-floor
    component: cpu
    threshold: 60.0
    within_steps: 8

  # NoHarmful — component must never drop below a raw value threshold
  - type: no_harmful
    name: latency-ceiling
    component: latency_ms
    threshold: 400.0
```

### Usage

```python
from margin import from_config, load_config

# From a Python dict
cfg = from_config(config_dict)
# cfg["parser"]    → Parser
# cfg["policy"]    → Policy (if "policy" key present)
# cfg["contract"]  → Contract (if "contract" key present)

# From a file (JSON works out of the box, YAML needs pip install pyyaml)
cfg = load_config("margin.yaml")
```

### Component fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `baseline` | float | required | expected healthy value |
| `intact` | float | 80 | intact threshold |
| `ablated` | float | 30 | ablated threshold |
| `higher_is_better` | bool | true | polarity |
| `lower_is_better` | bool | — | shorthand; sets `higher_is_better: false` |
| `labels` | dict | — | per-health display labels (`INTACT`, `DEGRADED`, `ABLATED`) |

### Policy rule fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | required | unique rule identifier |
| `when` | str or dict | required | predicate; see registry below |
| `priority` | int | 0 | higher fires first |
| `min_confidence` | str | `low` | skip rule below this expression confidence |
| `action.op` | str | `RESTORE` | `RESTORE \| SUPPRESS \| AMPLIFY \| NOOP` |
| `action.target` | str | `*` | component name or `*` for worst-degraded |
| `action.alpha` | float | 0.5 | mixing coefficient |
| `action.magnitude` | float | 1.0 | correction size |
| `action.alpha_from_sigma` | bool | false | derive alpha from \|sigma\| at runtime |
| `action.magnitude_from_sigma` | bool | false | derive magnitude from \|sigma\| at runtime |
| `constraint.cooldown_steps` | int | 0 | steps between firings |
| `constraint.max_per_window` | int | 0 | 0 = unlimited |
| `constraint.window_steps` | int | 0 | window for max_per_window |
| `constraint.min_alpha` | float | 0.0 | floor for sigma-derived alpha |
| `constraint.max_alpha` | float | 1.0 | ceiling for alpha |
| `escalation.level` | str | — | `LOG \| ALERT \| CRITICAL \| PAGE` |
| `escalation.reason` | str | `""` | human-readable message |

### Predicate registry

String names map to predicate factories for policy rules:

- `any_intact`, `any_degraded`, `any_ablated`, `any_recovering`, `any_ood`
- `all_intact`, `all_degraded`, `all_ablated`
- `any_correction`

Composable predicates via dict syntax:

```yaml
when: {all_of: [any_degraded, any_correction]}
when: {any_of: [any_ablated, {component_health: cpu, health: ABLATED}]}
when: {not: all_intact}
when: {sigma_below: cpu, threshold: -0.5}
when: {confidence_below: low}
```

### Contract term types

| `type` key | Required fields | Description |
| --- | --- | --- |
| `health_target` | `component`, `target` | must be at target health now |
| `reach_health` | `component`, `target`, `within_steps` | reach target within N steps |
| `sustain_health` | `component`, `target`, `for_steps` | sustain target for N consecutive steps |
| `recovery_threshold` | `component`, `threshold`, `within_steps` | raw value must exceed threshold within N steps |
| `no_harmful` | `component`, `threshold` | raw value must never drop below threshold |

Backward-compatible: contract entries without a `type` key default to `health_target`.

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
├── anomaly.py                Statistical outlier detection, distribution shift, jump detection
├── correlate.py              Auto-correlation discovery, lag detection, causal graph integration
├── drift.py                  Trajectory classification, predicates, forecast composition
├── forecast.py               Trend projection
├── predicates.py             Pattern matching, combinators, Rule
├── transitions.py            State transition tracking
│
│ Contract + Causal
├── contract.py               HealthTarget, ReachHealth, SustainHealth, etc.
├── causal.py                 CausalGraph, CausalLink, Explanation
│
│ Loop + Streaming + Config
├── intent.py                 Intent, Requirement, Feasibility, evaluate_monitor()
├── loop.py                   step(), run(), StepResult
├── streaming.py              DriftTracker, AnomalyTracker, CorrelationTracker, Monitor
├── persist.py                save_monitor(), load_monitor(), replay(), replay_csv()
├── config.py                 from_config(), load_config(), predicate registry
├── __main__.py               CLI: monitor, replay, status commands
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
