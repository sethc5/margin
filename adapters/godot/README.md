# Margin for Godot

**Typed health classification for game systems.**

A Godot 4 addon that gives every system in your game — food, health, morale, stamina, stress, temperature, anything with a number and a threshold — typed health states, polarity handling, sigma normalization, and correction targeting.

## Install

Copy `addons/margin/` into your project's `addons/` folder. Enable in Project → Project Settings → Plugins.

## Usage

```gdscript
# Define circuits for an NPC
var npc := MarginCircuitSet.new("villager_01")
npc.add_circuit("food", 0.85, 0.70, 0.30, true, "eat")
npc.add_circuit("health", 1.0, 0.70, 0.30, true, "rest")
npc.add_circuit("stress", 0.10, 0.70, 0.30, false, "rest")  # lower is better
npc.add_circuit("morale", 0.60, 0.70, 0.30, true, "socialize")

# Update from simulation
npc.update("food", 0.35)
npc.update("stress", 0.45)

# Read health
print(npc.to_string_())
# [food:DEGRADED(-0.59σ)] [health:INTACT(0.00σ)] [stress:DEGRADED(-3.50σ)] [morale:INTACT(0.00σ)]

# What should the NPC do?
var action = npc.worst_correction()  # "eat" — food is worst
var worst = npc.worst()              # MarginCircuit for food

# Drift: is anything trending badly?
if npc.any_worsening():
    var urgent = npc.most_urgent()   # worst health + worsening drift
    print("%s is %s" % [urgent.circuit_name, "getting worse"])

# Per-circuit drift
print(npc.get_circuit("food").drift_state)  # MarginDrift.State.DRIFTING
print(npc.get_circuit("food").is_worsening) # true

# Summary for UI
print(npc.summary())
# {label: villager_01, total: 4, intact: 2, degraded: 2, ablated: 0,
#  worst: food, correction: eat}
```

## The polarity problem

Every game with a stress system writes this bug:

```gdscript
# "Higher is better" works for food
if food < 0.3:
    state = "critical"

# But stress is "lower is better" — this check is backwards
if stress < 0.3:
    state = "critical"  # Wrong! Low stress is good!
```

Margin handles it:

```gdscript
# Higher is better (food, health, morale)
npc.add_circuit("food", 0.85, 0.70, 0.30, true)

# Lower is better (stress, dissonance, debt)
npc.add_circuit("stress", 0.10, 0.70, 0.30, false)
```

One flag. Classification, sigma, and correction targeting all respect it.

## Core types

### MarginHealth

Static functions for health classification:

```gdscript
MarginHealth.classify(value, intact, ablated, higher_is_better, correcting)
# Returns: MarginHealth.State.INTACT / DEGRADED / ABLATED / RECOVERING / OOD

MarginHealth.sigma(value, baseline, higher_is_better)
# Returns: float — positive = healthier, negative = worse

MarginHealth.classify_band(value, normal_low, normal_high, critical_low, critical_high)
# Returns: worse of the two boundary checks (for temperature, blood pressure, etc.)

MarginHealth.worse(state_a, state_b)
# Returns: the more severe state
```

### MarginCircuit

A named component with baseline, thresholds, health tracking, drift, and history:

```gdscript
var c := MarginCircuit.new("food", 0.85, 0.70, 0.30, true, "eat")
c.update(0.35)

c.health              # MarginHealth.State.DEGRADED
c.sigma_value         # -0.59
c.to_atom()           # "food:DEGRADED(-0.59σ)"
c.drift_state         # MarginDrift.State.DRIFTING
c.drift_direction     # MarginDrift.Direction.WORSENING
c.is_worsening        # true — trajectory is heading toward unhealthy
c.is_reverting        # false
c.steps_in_current_state()  # how long in this state
c.transition_count()        # how many state changes
c.time_in_state()           # {INTACT: 12, DEGRADED: 3}
```

### MarginDrift

Static functions for trajectory classification:

```gdscript
var result := MarginDrift.classify(values_array, baseline, higher_is_better)
result.state      # MarginDrift.State.DRIFTING
result.direction  # MarginDrift.Direction.WORSENING
result.rate       # slope per step (polarity-normalised)
result.r_squared  # goodness of fit
```

States: STABLE, DRIFTING, ACCELERATING, DECELERATING, REVERTING, OSCILLATING.
Computed automatically from circuit history on every `update()`.

### MarginCircuitSet

All circuits for one entity — the equivalent of a margin Expression:

```gdscript
var set := MarginCircuitSet.new("npc_01")
set.add_circuit("food", 0.85)
set.add_circuit("stress", 0.10, 0.70, 0.30, false)

set.health_of("food")       # State
set.worst()                  # MarginCircuit
set.worst_correction()       # "eat"
set.degraded()               # Array[MarginCircuit]
set.worsening()              # Array[MarginCircuit] — drift worsening
set.any_worsening()          # bool
set.most_urgent()            # worst health + worsening drift
set.to_string_()             # bracket notation
set.summary()                # dict for UI
set.to_dict()                # for save/load
```

## Use cases

- **Colony sims** — food, shelter, morale, defense as circuits per settlement
- **Survival games** — hunger, thirst, health, temperature, sanity per character
- **Management games** — customer satisfaction, employee morale, cash flow, equipment condition
- **City builders** — water, power, traffic, happiness, crime per district
- **RPGs** — NPC need systems, faction relations, regional stability

## Format

The bracket notation is the same as the Python margin library:

```
[food:DEGRADED(-0.59σ)] [health:INTACT(0.00σ)] [stress:ABLATED(-3.50σ)]
```

Same output in GDScript and Python. Systems can interoperate.

## License

MIT
