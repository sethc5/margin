# Changelog

All notable changes to margin are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
