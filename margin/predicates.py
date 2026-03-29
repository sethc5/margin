"""
Expression predicates: declarative rules for pattern matching over Expressions.

Define conditions like "any component is ABLATED" or "2+ components are
DEGRADED" and evaluate them against Expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .health import Health
from .confidence import Confidence
from .observation import Expression


# Type alias for predicate functions
PredicateFn = Callable[[Expression], bool]


def any_health(health: Health) -> PredicateFn:
    """True if any component has the given health state."""
    def check(expr: Expression) -> bool:
        return any(o.health == health for o in expr.observations)
    return check


def all_health(health: Health) -> PredicateFn:
    """True if all components have the given health state."""
    def check(expr: Expression) -> bool:
        return bool(expr.observations) and all(o.health == health for o in expr.observations)
    return check


def count_health(health: Health, min_count: int) -> PredicateFn:
    """True if at least `min_count` components have the given health state."""
    def check(expr: Expression) -> bool:
        return sum(1 for o in expr.observations if o.health == health) >= min_count
    return check


def component_health(name: str, health: Health) -> PredicateFn:
    """True if the named component has the given health state."""
    def check(expr: Expression) -> bool:
        h = expr.health_of(name)
        return h == health
    return check


def any_degraded() -> PredicateFn:
    """True if any component is DEGRADED, ABLATED, or RECOVERING."""
    def check(expr: Expression) -> bool:
        return bool(expr.degraded())
    return check


def confidence_below(threshold: Confidence) -> PredicateFn:
    """True if the expression's net confidence is below the threshold."""
    def check(expr: Expression) -> bool:
        return expr.confidence < threshold
    return check


def sigma_below(name: str, threshold: float) -> PredicateFn:
    """True if the named component's sigma is below the threshold."""
    def check(expr: Expression) -> bool:
        for o in expr.observations:
            if o.name == name:
                return o.sigma < threshold
        return False
    return check


def any_correction() -> PredicateFn:
    """True if any active correction is being applied."""
    def check(expr: Expression) -> bool:
        return any(c.is_active() for c in expr.corrections)
    return check


# -----------------------------------------------------------------------
# Combinators
# -----------------------------------------------------------------------

def all_of(*predicates: PredicateFn) -> PredicateFn:
    """True if all predicates are true."""
    def check(expr: Expression) -> bool:
        return all(p(expr) for p in predicates)
    return check


def any_of(*predicates: PredicateFn) -> PredicateFn:
    """True if any predicate is true."""
    def check(expr: Expression) -> bool:
        return any(p(expr) for p in predicates)
    return check


def not_(predicate: PredicateFn) -> PredicateFn:
    """Negate a predicate."""
    def check(expr: Expression) -> bool:
        return not predicate(expr)
    return check


# -----------------------------------------------------------------------
# Rule: a named predicate with an action label
# -----------------------------------------------------------------------

@dataclass
class Rule:
    """
    A named predicate that can evaluate Expressions.

    name:       rule identifier (e.g. "critical-alert", "all-clear")
    predicate:  function that takes an Expression and returns bool
    """
    name: str
    predicate: PredicateFn

    def matches(self, expr: Expression) -> bool:
        return self.predicate(expr)

    def __repr__(self) -> str:
        return f"Rule({self.name!r})"


def evaluate_rules(rules: list[Rule], expr: Expression) -> list[Rule]:
    """Return all rules that match the given Expression."""
    return [r for r in rules if r.matches(expr)]
