## CircuitSet: a collection of circuits for one entity (NPC, building, system).
##
## Equivalent to margin's Expression — all circuits for one entity,
## queryable by name, with worst-circuit targeting and compact rendering.
class_name MarginCircuitSet
extends RefCounted


var circuits: Dictionary = {}  ## {name: MarginCircuit}
var label: String = ""


func _init(p_label: String = "") -> void:
	label = p_label


## Add a circuit to the set.
func add(circuit: MarginCircuit) -> void:
	circuits[circuit.circuit_name] = circuit


## Create and add a circuit in one call.
func add_circuit(
	p_name: String,
	p_baseline: float,
	p_intact_ratio: float = 0.70,
	p_ablated_ratio: float = 0.30,
	p_higher_is_better: bool = true,
	p_correction_action: String = "",
) -> MarginCircuit:
	var c := MarginCircuit.new(
		p_name, p_baseline, p_intact_ratio, p_ablated_ratio,
		p_higher_is_better, p_correction_action,
	)
	add(c)
	return c


## Get a circuit by name.
func get_circuit(circuit_name: String) -> MarginCircuit:
	return circuits.get(circuit_name)


## Get the health of a named circuit.
func health_of(circuit_name: String) -> MarginHealth.State:
	var c := get_circuit(circuit_name)
	if c == null:
		return MarginHealth.State.OOD
	return c.health


## All circuits that are DEGRADED, ABLATED, or RECOVERING.
func degraded() -> Array[MarginCircuit]:
	var result: Array[MarginCircuit] = []
	for c: MarginCircuit in circuits.values():
		if c.health in [MarginHealth.State.DEGRADED, MarginHealth.State.ABLATED, MarginHealth.State.RECOVERING]:
			result.append(c)
	return result


## All circuits that are INTACT.
func intact() -> Array[MarginCircuit]:
	var result: Array[MarginCircuit] = []
	for c: MarginCircuit in circuits.values():
		if c.health == MarginHealth.State.INTACT:
			result.append(c)
	return result


## The single worst circuit (highest severity).
func worst() -> MarginCircuit:
	var worst_c: MarginCircuit = null
	var worst_sev := -1
	for c: MarginCircuit in circuits.values():
		var sev: int = MarginHealth.SEVERITY[c.health]
		if sev > worst_sev:
			worst_sev = sev
			worst_c = c
	return worst_c


## The correction action for the worst circuit.
func worst_correction() -> String:
	var w := worst()
	if w == null:
		return ""
	if w.health == MarginHealth.State.INTACT:
		return ""
	return w.correction_action


## Update a circuit by name.
func update(circuit_name: String, value: float, is_correcting: bool = false) -> void:
	var c := get_circuit(circuit_name)
	if c != null:
		c.update(value, is_correcting)


## Compact bracket notation — same format as Python margin.
func to_string_() -> String:
	if circuits.is_empty():
		return "[∅]"
	var parts: PackedStringArray = []
	for c: MarginCircuit in circuits.values():
		parts.append("[%s]" % c.to_atom())
	return " ".join(parts)


## Dictionary for serialization.
func to_dict() -> Dictionary:
	var d := {"label": label, "circuits": {}}
	for name in circuits:
		d["circuits"][name] = circuits[name].to_dict()
	return d


## Summary stats.
func summary() -> Dictionary:
	var n_intact := 0
	var n_degraded := 0
	var n_ablated := 0
	for c: MarginCircuit in circuits.values():
		match c.health:
			MarginHealth.State.INTACT:
				n_intact += 1
			MarginHealth.State.DEGRADED, MarginHealth.State.RECOVERING:
				n_degraded += 1
			MarginHealth.State.ABLATED:
				n_ablated += 1
	return {
		"label": label,
		"total": circuits.size(),
		"intact": n_intact,
		"degraded": n_degraded,
		"ablated": n_ablated,
		"worst": worst().circuit_name if worst() else "",
		"correction": worst_correction(),
	}
