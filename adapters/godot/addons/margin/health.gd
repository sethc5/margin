## Health classification: typed states for any monitored component.
##
## Maps a scalar measurement against thresholds into a typed health state.
## Supports both polarities: higher_is_better and lower_is_better.
class_name MarginHealth


## Health states — same vocabulary as the Python margin library.
enum State {
	INTACT,      ## Operating normally
	DEGRADED,    ## Impaired but present
	ABLATED,     ## Functionally absent
	RECOVERING,  ## Sub-threshold but correction is active
	OOD,         ## Measurement invalid / out of domain
}

## Severity ordering: higher = worse. Used for worst-of comparisons.
const SEVERITY := {
	State.INTACT: 0,
	State.RECOVERING: 1,
	State.DEGRADED: 2,
	State.ABLATED: 3,
	State.OOD: 4,
}

## Human-readable names.
const STATE_NAMES := {
	State.INTACT: "INTACT",
	State.DEGRADED: "DEGRADED",
	State.ABLATED: "ABLATED",
	State.RECOVERING: "RECOVERING",
	State.OOD: "OOD",
}


## Classify a value against thresholds. Polarity-aware.
static func classify(
	value: float,
	intact: float,
	ablated: float,
	higher_is_better: bool = true,
	correcting: bool = false,
) -> State:
	if higher_is_better:
		if value >= intact:
			return State.INTACT
		if value < ablated:
			return State.RECOVERING if correcting else State.ABLATED
		return State.RECOVERING if correcting else State.DEGRADED
	else:
		if value <= intact:
			return State.INTACT
		if value > ablated:
			return State.RECOVERING if correcting else State.ABLATED
		return State.RECOVERING if correcting else State.DEGRADED


## Classify a value against a band (both too-high and too-low).
## Returns the worse of the two boundary checks.
static func classify_band(
	value: float,
	normal_low: float,
	normal_high: float,
	critical_low: float,
	critical_high: float,
	correcting: bool = false,
) -> State:
	var low_health := classify(value, normal_low, critical_low, true, correcting)
	var high_health := classify(value, normal_high, critical_high, false, correcting)
	if SEVERITY[high_health] > SEVERITY[low_health]:
		return high_health
	return low_health


## Polarity-normalised sigma: positive = healthier, negative = worse.
static func sigma(
	value: float,
	baseline: float,
	higher_is_better: bool = true,
) -> float:
	if absf(baseline) < 0.001:
		return 0.0
	var raw := (value - baseline) / absf(baseline)
	return raw if higher_is_better else -raw


## Return the state name as a string.
static func state_name(state: State) -> String:
	return STATE_NAMES.get(state, "UNKNOWN")


## Compare two states: returns the more severe one.
static func worse(a: State, b: State) -> State:
	return a if SEVERITY[a] >= SEVERITY[b] else b
