# Changelog

All notable changes to margin are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.9.15] — 2026-03-30

### Added

- `Controller` strategy `"proportional_setpoint"`: standard P controller — `alpha_next = alpha + kp * (target - metric)`; `target` is the setpoint the controller actively tracks; warm and cold controllers (different targets) produce different alpha trajectories from the first step; `backoff` is not applied
- `Controller._STRATEGIES`: set of valid strategy names (`"proportional_asymmetric"`, `"proportional_setpoint"`); `ValueError` now lists all valid strategies

### Fixed

- `Controller` with `proportional_asymmetric`: clarified that `target` in this strategy is the *initial alpha seed* (`alpha = ctrl.target`), not a setpoint — it does not enter `step()` math; `proportional_setpoint` is the strategy where `target` actively drives the loop

---

## [0.9.14] — 2026-03-30

### Fixed

- `Fingerprint` JSON serialization: `Fingerprint` now subclasses `dict` — `json.dumps(fp)` works without a custom encoder, `isinstance(fp, dict)` is `True`, and all dict methods are inherited natively; fixes `TypeError: Object of type Fingerprint is not JSON serializable` when embedding fingerprints in serialized session metadata
- `Controller` silent alpha clamp bug: `alpha_min` and `alpha_max` are now stored on the `Controller` at construction (defaulting to `0.0` / `1.0`) and used by `step()` without requiring the caller to pass them every call; callers with non-default ranges (e.g. `alpha_min=1.0, alpha_max=4.0`) no longer get silently clamped to `[0, 1]`; per-call override of either bound is still supported

### Changed

- `Controller.__init__`: added `alpha_min=0.0` and `alpha_max=1.0` parameters
- `Controller.from_fingerprint`: added `alpha_min=0.0` and `alpha_max=1.0` parameters
- `Controller.step`: `alpha_min` / `alpha_max` are now optional overrides (default `None` → use stored bounds)
- `Controller.__repr__`: now shows `alpha=[min, max]`

---

## [0.9.13] — 2026-03-30

### Fixed

- `Fingerprint.values()`: restores standard dict-protocol method (`values_dict()` was a non-standard rename that broke any caller iterating `fp.values()` — a regression from the plain-dict return of prior versions)

---

## [0.9.12] — 2026-03-30

### Added

- `Fingerprint` class (`margin.fingerprint`): noise-resistant session statistics wrapper returned by `Monitor.fingerprint()`; dict-compatible (`fp["component"]["mean"]` still works); adds `fp.robust_target(component, method="median")` (median of raw drift-window values — more robust than mean when std is high), `fp.percentile(component, p)`, `fp.n(component)`, and `fp.to_dict()` / `from_dict()` round-trip; `values` dict stores raw float lists from the drift window
- `Controller` class (`margin.controller`): adaptive scalar P-controller for Layer 2/3 feedback loops; `Controller(strategy="proportional_asymmetric", kp=0.3, target=0.5, backoff=0.90)`; `ctrl.step(alpha, observations, alpha_min, alpha_max)` → `(alpha_next, reason)` — accepts a list of `Observation` objects (metric = mean sigma, polarity-corrected) or a scalar float; `proportional_asymmetric` strategy: `alpha += kp * metric` when `metric ≥ 0`, `alpha *= backoff` when `metric < 0`
- `Controller.from_fingerprint(fp, component, kp, cold_target, strategy)`: warm-start factory that condenses ~30 lines of per-session calibration boilerplate into one call; uses `fp.robust_target(component)` (median) when `n ≥ 10`, falls back to `cold_target` for cold sessions; accepts both `Fingerprint` objects and plain dicts
- `Monitor.fingerprint()` now returns `Fingerprint` (was `dict[str, dict]`); fully backward-compatible — dict indexing unchanged; raw values stored for median/percentile queries
- `Monitor.suggest_target(component)` → `{target: float, confidence: str}`: suggests a conservative calibration target as `max(0, mean - 0.5 * std)`; confidence is `HIGH` (n≥30), `MODERATE` (n≥10), `LOW` (n<10), or `NONE` (component unknown / no data)
- `Monitor.tail(n=10)` → `list[Observation]`: returns the `n` most recent observations across all components from the drift-tracker windows, sorted by `measured_at`

---

## [0.9.11] — 2026-03-30

### Added

- `Monitor.fingerprint()` → `{name: {mean, std, n, trend}}`: session statistics from the current drift window per component — empirical mean, sample std, observation count, and DriftState string; `trend="UNKNOWN"` when fewer than min_samples observations exist; designed for dispositional calibration at session boundaries
- `Parser.with_baselines(fingerprint)` → `Parser`: create a new Parser with baselines shifted to empirical session means from a fingerprint dict; components absent from the fingerprint keep their original baselines; all thresholds preserved; non-mutating — original Parser unchanged
- `load_monitor(path, parser, warm_only=False)`: new `warm_only` flag — when `True`, loads drift observations (warm prior) but starts fresh on anomaly trackers, correlation history, and step count; enables new-session continuation that inherits trajectory without contaminating anomaly reference windows
- `DriftClassification.step_count`: property alias for `n_samples` — number of observations that contributed to the current classification; distinguishes noise (3-step OSCILLATING) from signal (20-step OSCILLATING)
- `ProvenanceGraph.compress(max_nodes=500)` → `self`: prune oldest nodes by insertion order to stay within `max_nodes`; cleans dangling `source_ids` in surviving nodes; returns self for chaining before `save_monitor`

---

## [0.9.10] — 2026-03-30

### Added

- `Monitor(parser, ..., provenance_graph=None)`: optional `ProvenanceGraph` attachment — when provided, each `update()` call records a root node (`"monitor:step:N"`) and threads its ID into every `Observation.provenance` list; enables full lineage queries via `monitor.provenance_graph.trace_lineage()`
- `Monitor.status()`: includes `"provenance_nodes"` count when a graph is attached
- `Monitor.__repr__`: shows `provenance=True` when a graph is attached
- `save_monitor` / `load_monitor`: serialize and restore the provenance graph when attached

### Changed

- `events.py` docstring: explicit that `EventBus` is a standalone utility, not wired into `Monitor`; documents the intended use pattern (event-driven validity invalidation)
- `margin-language.md` Structure table: corrected `provenance.py` description from "Correlation detection" (wrong) to "Value lineage tracking (ProvenanceGraph, provenance IDs)"

---

## [0.9.9] — 2026-03-30

### Added

- `tests/conftest.py`: margin's own test suite now uses the pytest plugin for self-monitoring — every `pytest` run prints a typed health snapshot of `pass_rate`, `new_failures`, `skip_rate`, `duration_seconds`, and `mean_test_duration`; supports `--margin-per-file`, `--margin-slowest=N`, `--margin-baseline`/`--margin-output` diff mode
- `tests/margin-baseline.json`: committed baseline snapshot (all metrics INTACT at v0.9.9); future runs can diff against it with `--margin-baseline=tests/margin-baseline.json`
- CI gate: `--margin-fail-below=ABLATED --margin-slowest=5` added to GitHub Actions test step — build fails if any test-suite metric hits ABLATED; 5 slowest tests are reported with health classification

---

## [0.9.8] — 2026-03-30

### Added

- `validate()` check #9: warns when a rule's `min_confidence` is `HIGH` or `CERTAIN` — `Parser.parse()` defaults observations to `MODERATE`, so such rules silently never fire without explicit per-component confidences

### Changed

- `Monitor.__init__` docstring: documents all parameters including the `features` asymmetry — `None` enables all trackers, `set()` disables all
- `margin-language.md` config section: expanded from a single minimal example to a full reference — annotated YAML with every supported field, field tables for components / policy rules / contract terms, and `constraint`/`escalation`/`min_confidence`/`alpha_from_sigma`/`multi_rule`/`labels` all documented

---

## [0.9.7] — 2026-03-30

### Added

- `Ledger.to_dict()`: `to_json()` now delegates to `to_dict()` — symmetric with existing `Ledger.from_dict()` and `from_json()`
- `Contract.from_dict(d)`: deserialize a `Contract` and all its terms from a dict produced by `Contract.to_dict()` — uses `contract_term_from_dict` internally
- `from_config()`: `alpha_from_sigma` and `magnitude_from_sigma` now supported on action specs; `min_confidence` on rule specs; `multi_rule` at top-level policy config; `labels` on component threshold specs

### Fixed

- `FullStepResult.to_dict()`: `correlations` field was silently omitted — now serialized when present

### Documentation

- `full_step()` docstring: documents `confidences`, `provenance`, and `label` parameters added in v0.9.6
- `Parser.parse()` docstring: documents `label`, `step`, and `provenance` parameters

---

## [0.9.6] — 2026-03-30

### Added

- `full_step(..., confidences, provenance, label)`: per-component confidence levels, provenance lineage, and expression label now thread through to `Monitor.update()` → `Parser.parse()` — previously these were silently dropped when using `full_step()`
- `Monitor.update(..., confidences, provenance)`: same parameters now accepted and forwarded to `Parser.parse()`
- `from_config()` policy rules: `constraint` and `escalation` keys now supported in YAML/JSON config — `cooldown_steps`, `min_alpha`, `max_alpha`, `max_per_window`, `window_steps`, escalation `level` and `reason` all configurable without writing Python
- `from_config()` contract terms: all five contract term types now deserializable from config via `contract_term_from_dict`; old configs without a `"type"` key remain backward-compatible
- `__repr__` on `CalibrationResult`, `WindowConfig`, `Action`, `Constraint`, `HealthTarget`, `ReachHealth`, `SustainHealth`, `RecoveryThreshold`, `NoHarmful`

### Changed

- `Parser.parse()`: emits `warnings.warn` when a component has no entry in `Parser.baselines` — previously the observed value silently became its own baseline (sigma=0, INTACT); now surfaces the typo/misconfiguration
- `Action.resolve()`: emits `warnings.warn` when a named `target` is not found in the current expression — previously produced a `Correction` referencing a nonexistent component with no signal to the caller

---

## [0.9.5] — 2026-03-30

### Added

- `contract_term_from_dict(d)`: factory to deserialize any `ContractTerm` subclass from its `to_dict()` output — `HealthTarget`, `ReachHealth`, `SustainHealth`, `RecoveryThreshold`, `NoHarmful` all round-trip correctly; raises `ValueError` on unknown `"type"` key

### Fixed

- `StepResult.acted`: now returns `True` when `policy.multi_rule=True` and a real correction exists in `StepResult.corrections` even if the highest-priority result was an `Escalation` — previously only checked `StepResult.correction` (the first result), so a suppressed top rule + active lower rule returned `False` incorrectly
- `recalibrate_parser(active_min=...)`: default changed from `0.05` to `None`; `None` inherits `active_min` from the existing parser's thresholds, an explicit value (including `0.05`) now overrides — previously passing `active_min=0.05` explicitly was silently treated as "inherit"

---

## [0.9.4] — 2026-03-30

### Added

- `WindowConfig.to_dict()` / `WindowConfig.from_dict()`: serialize and deserialize per-tracker window configuration; previously no serialization path existed for `WindowConfig`
- `calibrate_many(..., return_parser=True)`: opt-in to receive a ready-to-use `Parser` instead of the raw `(baselines, thresholds_dict)` tuple; backward-compatible default is `False`

### Changed

- `Monitor.status()`: `"drift"`, `"anomaly"`, and `"correlations"` keys are now always present — empty dict when the feature is disabled or no data has been accumulated yet; callers no longer need key-existence checks before indexing

---

## [0.9.3] — 2026-03-30

### Added

- `Thresholds.to_dict()` / `Thresholds.from_dict()`: serialize and deserialize threshold configuration including `labels` — previously no serialization path existed for `Thresholds`
- Tests for all features introduced in 0.9.1–0.9.2: `Monitor.features`, `Monitor.reset_anomaly_reference()`, `Policy.multi_rule`, `Thresholds.labels`, `calibrate(use_std=True)`, `needs_recalibration_many()`, persist round-trip for `features` and `anomaly_min_reference` (822 tests total, up from 775)

---

## [0.9.2] — 2026-03-30

### Added

- `Monitor(features={"health","drift"})`: opt-in feature flags — skip unused trackers (drift, anomaly, correlation); saves compute and suppresses spurious "anomaly reference" warnings when anomaly tracking is off; `"health"` always implied; `None` (default) enables all three; `features` saved/restored by `save_monitor`/`load_monitor`
- `Policy(multi_rule=True)`: allow all matching rules to fire in one step — `DecisionTrace.results` contains every fired `Correction`/`Escalation`; `StepResult.corrections` exposes the full list; backward-compatible (`decision.result` and `step_result.correction` still hold the first)
- `Thresholds(labels={"ABLATED":"CRITICAL","DEGRADED":"WARNING","INTACT":"OK"})`: per-health display labels — flow through `Parser.parse()` into `Observation.health_label`; `to_atom()` uses the label instead of the enum name; `Parser.label_for(component, health)` convenience accessor

---

## [0.9.1] — 2026-03-30

### Added

- `needs_recalibration_many(calibration_samples, recent_samples, ...)`: per-component recalibration check — returns `dict[str, bool]`; eliminates manual iteration over components
- `calibrate()`, `calibrate_many()`, `recalibrate_parser()`, `parser_from_calibration()`: `use_std=False` opt-in — derive thresholds as `baseline ± N×std` instead of fraction-of-mean; adapts to calibration variance; `intact_std_multiplier` (default 1.5) and `ablated_std_multiplier` (default 3.0) control placement
- `Monitor.reset_anomaly_reference(components=None)`: flush `AnomalyTracker` reference windows after `recalibrate_parser()` to prevent old-baseline data contaminating anomaly classification during warm-up
- `CalibrationResult.to_dict()`: now includes `active_min` field

### Changed

- `recalibrate_parser()`: inherits `active_min` from the existing parser's thresholds rather than silently defaulting to `0.05`; pass `active_min` explicitly to override
- `save_monitor()` / `load_monitor()`: round-trip `anomaly_min_reference` — previously lost on reload, silently reverting to default of 10
- `validate()` check #8: warns when `alpha_from_sigma=True` with no `Constraint` (no floor on sigma-derived alpha), `Constraint(min_alpha=0.0)` (floor present but zero), or both `alpha_from_sigma` and `magnitude_from_sigma` set (sigma² amplification)
- `SustainHealth.evaluate()`: warns when `ledger.max_records < for_steps` — term can never be MET under this configuration
- `ReachHealth.evaluate()`: warns when `ledger.max_records < within_steps` — records older than cap are silently dropped, masking past target achievement

---

## [0.9.0] — 2026-03-30

### Added

- `WindowConfig`: dataclass for per-concern window configuration (`drift`, `anomaly`, `correlation`)
- `Monitor`: `drift_window`, `anomaly_window`, `correlation_window`, `window_config` parameters — all backward-compatible (default to base `window`); `__repr__` shows per-tracker windows when they differ
- `save_monitor` / `load_monitor` / `replay` / `replay_csv`: save and restore per-tracker window configuration
- `recalibrate_parser(parser, new_healthy_data, ...)`: returns a new `Parser` with updated baselines for drifted components; non-mutating, partial-safe (unrecalibrated components preserved verbatim)
- `needs_recalibration(calibration_samples, recent_samples)`: detects baseline staleness via mean shift (>20%) and spread change (>2× or <0.5×) using `check_distribution()`

### Changed

- `Correction.to_dict()`: `alpha` and `magnitude` are now stored without rounding (was `round(..., 4)`); serialization is lossless as invariant 9 always claimed
- `Expression.__post_init__`: emits `warnings.warn` when `confidence` exceeds the weakest observation confidence — catches overclaimed certainty from direct construction

### Documentation

- `margin-language.md` — spec improvements across 9 categories:
  - Layer count corrected: "eight layers" → "six stages" throughout
  - Edge-case table extended: `-∞` behavior specified for both polarities
  - `active_min` guidance added to Thresholds field table and calibration Step 2
  - Calibration guide moved before module-by-module reference
  - Known limitations distributed as inline callouts; standalone block removed
  - `bridge.py` framing section added
  - `Ledger(max_records=N)`, `CausalLink` field table, `CausalGraph.asserted_links()` / `discovered_links()` documented
  - Invariants section gains "Violation behavior" table (10 rows × 5 columns)
  - Window configuration subsection with domain guidance table (ops, ML, biochem, game, robot)
  - Baseline recalibration section: detection signals A/B/C, `recalibrate_parser` docs, transition semantics
- `CHANGELOG.md`: created, covering all versions from 0.1.0

---

## [0.8.2] — 2026-03-30

### Added
- `CausalLink.origin`: `"asserted"` | `"discovered"` — distinguishes manually declared links from auto-discovered correlations; `to_causal_link()` marks all auto-discovered links as `"discovered"`
- `CausalGraph.asserted_links()` / `discovered_links()`: query links by provenance
- `Ledger(max_records=N)`: optional cap for long-running loops; oldest records dropped on append

### Changed
- `margin-language.md`: new "Known limitations" section documenting sigma overloading, RECOVERING boundary semantics, confidence cardinality leakage, asserted vs. discovered causal conflation, no policy cost model, contract-policy behavioral gap, Monitor partial-update assumption, and ledger unbounded growth — each with concrete mitigation

---

## [0.8.1] — 2026-03-29

### Added
- `ProvenanceGraph`: full value lineage tracking — records every transformation from raw input through algebra operations to classified output; queryable by component or observation ID

---

## [0.8.0] — 2026-03-29

### Added
- Proprioception infrastructure: Layer 0–7 architecture scaffolding for self-aware systems (margin-poc roadmap)
- Backward-compatible aliases from `margin-poc` naming; unblocks downstream migration

---

## [0.7.0] — 2026-03-29

### Added
- `Intent`: goal feasibility — FEASIBLE / AT_RISK / INFEASIBLE / UNKNOWN with drift-aware ETA to violation; supports `min_health`, `min_value`, and `critical` requirements
- `Intent.evaluate_monitor()`: evaluate intent directly from a running Monitor
- `full_step()`: runs all six stages (health, drift, anomaly, correlation, policy, intent) in one call; returns `FullStepResult` with `.expression`, `.drift`, `.anomaly`, `.correlations`, `.step`, `.intent`

---

## [0.6.2] — 2026-03-29

### Added
- Transformer / mech-interp adapter: streaming Monitor integration, IOI causal graph for attention circuit health
- Neuroscience adapter: 6 recording modality profiles (EEG, fMRI, ephys, calcium, behavior, EMG)
- 3D printer adapter: FDM, resin, and CoreXY profiles
- Godot 4 addon: drift tracking for game state health variables

---

## [0.6.1] — 2026-03-29

### Added
- ROS2 adapter: 4 robot profiles (mobile, manipulator, drone, AGV), sensor health classification, ROS2 node with diagnostics bridge

---

## [0.6.0] — 2026-03-29

### Added
- `save_monitor()` / `load_monitor()`: persist and restore full Monitor state (tracker windows, step count, drift classifications) as plain JSON
- `replay()` / `replay_csv()`: batch-replay historical data through a Monitor; returns per-step `status()` snapshots
- CLI: `python -m margin status`, `monitor`, `replay` — JSON-lines stdin, CSV file support, config-driven

---

## [0.5.0] — 2026-03-29

### Added
- `DriftTracker`, `AnomalyTracker`: per-component incremental trackers with bounded windows; reclassify on each `update()`
- `CorrelationTracker`: incremental pairwise correlation over a sliding window
- `Monitor`: unified streaming wrapper — one `update()` call returns health + drift + anomaly + correlation; `status()` snapshot dict
- `from_config()` / `load_config()`: build Parser, Policy, and Contract from a dict, JSON file, or YAML file
- Predicate registry for config-driven policy (`any_ablated`, `all_intact`, composable dict syntax)

---

## [0.4.1] — 2026-03-29

### Fixed
- `correlate.coefficient()` returning `None` for `r = 0.0` (exact zero correlation)

---

## [0.4.0] — 2026-03-29

### Added
- `correlate()` / `correlate_from_ledger()`: auto-discover pairwise correlations from value series; lag detection identifies leading/lagging relationships
- `auto_causal_graph()`: one-step correlation discovery → CausalGraph construction
- `CorrelationMatrix`: query by component, rank by strength, extract pairwise coefficients
- `correlated_with()` / `any_new_correlation()`: policy predicates for correlation-driven rules

---

## [0.3.1] — 2026-03-29

### Fixed
- Jump detection: robust z-score against local window; handles edge cases at series boundaries
- Anomaly novel margin: minimum of 1 std to prevent false NOVEL on narrow reference ranges
- `margin-language.md`: drift and anomaly sections updated; generic → margin import paths corrected

---

## [0.3.0] — 2026-03-29

### Added
- `classify_drift()` / `drift_from_ledger()`: trajectory classification — STABLE / DRIFTING / ACCELERATING / DECELERATING / REVERTING / OSCILLATING with polarity-aware direction and R²-based confidence
- `DriftForecast`: combines drift shape with threshold-crossing ETA in one object
- `classify_anomaly()` / `classify_anomaly_obs()`: statistical outlier detection — EXPECTED / UNUSUAL / ANOMALOUS / NOVEL
- `check_distribution()`: compare recent vs. reference distributions for mean shift, spread change, and shape change
- `detect_jumps()`: sudden discontinuity detection via robust z-score against local window
- `anomaly_from_ledger()` / `distribution_shift_from_ledger()`: anomaly analysis from Ledger history
- Drift and anomaly policy predicates: `drift_worsening`, `any_drifting`, `any_novel`, `any_anomalous`, etc.

---

## [0.2.0] — 2026-03-29

### Added
- `classify_drift()`: initial drift classification module — typed trajectory states for value histories

---

## [0.1.1] — 2026-03-28

### Added
- pytest plugin: CI gate, per-file breakdown, diff against baseline JSON (`--margin-fail-below`, `--margin-per-file`, `--margin-baseline`)
- numpy adapter: NaN/Inf contamination, drift detection, distribution shape, range violations
- Weather adapter: 5 profiles (general, agriculture, aviation, construction, public health)
- Examples directory and `CLAUDE.md` for AI coding agents

### Fixed
- PyPI documentation links: use absolute GitHub URLs

---

## [0.1.0] — 2026-03-28

### Added
- Core language: `classify()`, `Health`, `Thresholds`, `Confidence`, `Parser`, `Ledger`, `Observation`, `Expression`, `Correction`, `Op`
- Uncertainty algebra: `UncertainValue`, `algebra.py` (add/subtract/multiply/divide/scale/compare with uncertainty propagation), `bridge.py`, `validity.py`
- Policy layer: `Policy`, `PolicyRule`, `Action`, `Constraint`, `Escalation`; temporal predicates; `PolicyChain`; backtest, tuning, trace, validate
- Contract layer: `HealthTarget`, `ReachHealth`, `SustainHealth`, `RecoveryThreshold`, `NoHarmful`
- Causal layer: `CausalGraph`, `CausalLink` (DEGRADES, BLOCKS, TRIGGERS, CORRELATES, MITIGATES), conditional links, causal explanation
- Supporting modules: `calibrate.py`, `composite.py`, `diff.py`, `drift.py`, `forecast.py`, `predicates.py`, `transitions.py`, `events.py`
- Healthcare adapter: clinical vitals (WHO/AHA ranges)
- Godot 4 addon: typed health classification for game state
- Transformer / mech-interp adapter
- Domain adapters: homeassistant, evcharging, infrastructure, aquarium, greenhouse, fitness
- Python-ecosystem adapters: fastapi, database (SQLAlchemy), celery, dataframe (pandas), pytest
- Observation hooks: `auto-collect from real systems`
- CI: test matrix (Python 3.10–3.13) and PyPI publish via trusted publisher
