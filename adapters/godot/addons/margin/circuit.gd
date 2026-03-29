## Circuit: a named component with baseline, thresholds, and health tracking.
##
## This is the game-facing type. Define circuits for food, health, morale,
## stress, stamina, etc. Each circuit tracks its own health and correction.
class_name MarginCircuit
extends RefCounted


## Circuit definition
var circuit_name: String
var baseline: float
var intact: float
var ablated: float
var higher_is_better: bool
var correction_action: String  ## domain-specific: "eat", "rest", "heal", etc.

## Current state
var current_value: float
var health: MarginHealth.State
var sigma_value: float
var correcting: bool = false

## Drift tracking
var drift_state: MarginDrift.State = MarginDrift.State.STABLE
var drift_direction: MarginDrift.Direction = MarginDrift.Direction.NEUTRAL
var drift_rate: float = 0.0
var is_worsening: bool = false
var is_reverting: bool = false

## History
var _history: Array[Dictionary] = []
var _max_history: int = 100


func _init(
	p_name: String,
	p_baseline: float,
	p_intact_ratio: float = 0.70,
	p_ablated_ratio: float = 0.30,
	p_higher_is_better: bool = true,
	p_correction_action: String = "",
) -> void:
	circuit_name = p_name
	baseline = p_baseline
	higher_is_better = p_higher_is_better
	correction_action = p_correction_action
	current_value = p_baseline

	if p_higher_is_better:
		intact = p_baseline * p_intact_ratio
		ablated = p_baseline * p_ablated_ratio
	else:
		intact = p_baseline * (1.0 + (1.0 - p_intact_ratio))
		ablated = p_baseline * (1.0 + (1.0 - p_ablated_ratio))

	_reclassify()


## Update the circuit with a new measured value.
func update(value: float, is_correcting: bool = false) -> void:
	current_value = value
	correcting = is_correcting
	_reclassify()
	_record()


## Get the compact string representation.
func to_atom() -> String:
	var h := MarginHealth.state_name(health)
	if health == MarginHealth.State.OOD:
		return "%s:%s" % [circuit_name, h]
	var sign := "+" if sigma_value >= 0 else ""
	return "%s:%s(%s%.2fσ)" % [circuit_name, h, sign, sigma_value]


## Dictionary for serialization.
func to_dict() -> Dictionary:
	return {
		"name": circuit_name,
		"health": MarginHealth.state_name(health),
		"value": current_value,
		"baseline": baseline,
		"sigma": sigma_value,
		"higher_is_better": higher_is_better,
		"drift_state": MarginDrift.STATE_NAMES[drift_state],
		"drift_direction": MarginDrift.DIRECTION_NAMES[drift_direction],
		"drift_rate": drift_rate,
		"is_worsening": is_worsening,
	}


## Number of steps this circuit has been in its current health state.
func steps_in_current_state() -> int:
	if _history.is_empty():
		return 0
	var count := 0
	var current := health
	for i in range(_history.size() - 1, -1, -1):
		if _history[i]["health"] == current:
			count += 1
		else:
			break
	return count


## Number of transitions in history.
func transition_count() -> int:
	if _history.size() < 2:
		return 0
	var count := 0
	for i in range(1, _history.size()):
		if _history[i]["health"] != _history[i - 1]["health"]:
			count += 1
	return count


## Time in each state as {state_name: step_count}.
func time_in_state() -> Dictionary:
	var counts := {}
	for entry in _history:
		var h: int = entry["health"]
		var name := MarginHealth.state_name(h)
		counts[name] = counts.get(name, 0) + 1
	return counts


func _reclassify() -> void:
	health = MarginHealth.classify(
		current_value, intact, ablated, higher_is_better, correcting
	)
	sigma_value = MarginHealth.sigma(current_value, baseline, higher_is_better)


func _reclassify_drift() -> void:
	if _history.size() < 3:
		return
	var values: Array = []
	for entry in _history:
		values.append(entry["value"])
	var result := MarginDrift.classify(values, baseline, higher_is_better)
	drift_state = result.state
	drift_direction = result.direction
	drift_rate = result.rate
	is_worsening = drift_direction == MarginDrift.Direction.WORSENING
	is_reverting = drift_state == MarginDrift.State.REVERTING


func _record() -> void:
	_history.append({
		"value": current_value,
		"health": health,
		"sigma": sigma_value,
	})
	if _history.size() > _max_history:
		_history.pop_front()
	_reclassify_drift()
