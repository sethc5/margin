"""
Full loop — observe, explain, decide, evaluate in one call.

A water treatment plant with three sensors, a causal graph explaining
dependencies, a policy deciding what to fix, and a contract defining
what "safe water" means.
"""

from margin import (
    step, Parser, Thresholds, Health, Op, Confidence,
    Policy, PolicyRule, Action, Constraint,
    Contract, HealthTarget, SustainHealth,
    CausalGraph,
    any_health, any_degraded, all_health,
    Ledger,
)

# 1. Define the system
parser = Parser(
    baselines={"ph": 7.0, "chlorine": 1.5, "turbidity": 0.5},
    thresholds=Thresholds(intact=6.5, ablated=5.5),  # pH-like default
    component_thresholds={
        "chlorine": Thresholds(intact=1.0, ablated=0.2, higher_is_better=True),
        "turbidity": Thresholds(intact=1.0, ablated=4.0, higher_is_better=False),
    },
)

# 2. Define why things fail
graph = CausalGraph()
graph.add_degrades("turbidity", "chlorine", 0.7, evidence="high turbidity consumes chlorine")
graph.add_degrades("chlorine", "ph", 0.3, evidence="low chlorine allows algae which shifts pH")

# 3. Define what to do
policy = Policy(name="water-treatment", rules=[
    PolicyRule("critical", any_health(Health.ABLATED),
               Action(target="*", op=Op.RESTORE, alpha=1.0), priority=50),
    PolicyRule("maintain", any_degraded(),
               Action(target="*", op=Op.RESTORE, alpha=0.5), priority=10),
    PolicyRule("normal", all_health(Health.INTACT),
               Action(target="*", op=Op.NOOP), priority=0),
])

# 4. Define what success looks like
contract = Contract("safe-water", terms=[
    HealthTarget("ph-safe", "ph", Health.INTACT),
    HealthTarget("chlorine-safe", "chlorine", Health.INTACT),
    HealthTarget("turbidity-safe", "turbidity", Health.INTACT),
])

# 5. Run it
ledger = Ledger()
expr = parser.parse({"ph": 6.8, "chlorine": 0.3, "turbidity": 3.5})
result = step(expr, policy, ledger, graph, contract)

print(result.to_string())
# Step: [ph:INTACT(...)][chlorine:ABLATED(...)][turbidity:DEGRADED(...)]
#   Why: chlorine ← turbidity (high turbidity consumes chlorine)
#   Decision: critical matched
#   Action: RESTORE(target=chlorine, α=1.00)
#   Contract: 1 met, 2 violated, 0 pending
