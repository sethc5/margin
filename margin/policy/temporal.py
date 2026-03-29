"""
Temporal predicates: conditions that look at history, not just the current snapshot.

"DEGRADED for 3+ consecutive steps", "sigma trending below -0.5 over 10 steps",
"fired more than 5 times in last 20 steps".
"""

from __future__ import annotations

from typing import Callable, Optional

from ..health import Health
from ..observation import Expression
from ..ledger import Ledger
from ..predicates import PredicateFn


# Temporal predicates are functions that take (Expression, Ledger) → bool.
# To fit the PredicateFn signature (Expression → bool), we bind the ledger
# at construction time via closure.

def health_sustained(
    component: str,
    health: Health,
    min_steps: int,
    ledger: Ledger,
) -> PredicateFn:
    """
    True if the component has been in the given health state for
    at least `min_steps` consecutive steps in the ledger tail,
    AND the current expression also shows that health.
    """
    def check(expr: Expression) -> bool:
        current = expr.health_of(component)
        if current != health:
            return False
        recent = ledger.for_component(component).last_n(min_steps)
        if len(recent) < min_steps:
            return False
        return all(
            (r.after.health if r.after and r.after.name == component else r.before.health) == health
            for r in recent.records
            if r.before.name == component or (r.after and r.after.name == component)
        )
    return check


def health_for_at_least(
    component: str,
    healths: set[Health],
    min_steps: int,
    ledger: Ledger,
) -> PredicateFn:
    """
    True if the component has been in any of the given health states
    for at least `min_steps` consecutive steps.
    """
    def check(expr: Expression) -> bool:
        current = expr.health_of(component)
        if current not in healths:
            return False
        recent = ledger.for_component(component).last_n(min_steps)
        if len(recent) < min_steps:
            return False
        for r in recent.records:
            obs = r.after if r.after and r.after.name == component else r.before
            if obs.name == component and obs.health not in healths:
                return False
        return True
    return check


def sigma_trending_below(
    component: str,
    threshold: float,
    over_steps: int,
    ledger: Ledger,
) -> PredicateFn:
    """
    True if the component's sigma has been below `threshold` for
    all of the last `over_steps` records AND the current expression.
    """
    def check(expr: Expression) -> bool:
        # Check current
        for o in expr.observations:
            if o.name == component:
                if o.sigma >= threshold:
                    return False
                break
        else:
            return False

        recent = ledger.for_component(component).last_n(over_steps)
        if len(recent) < over_steps:
            return False
        for r in recent.records:
            obs = r.after if r.after and r.after.name == component else r.before
            if obs.name == component and obs.sigma >= threshold:
                return False
        return True
    return check


def fire_rate_above(
    component: str,
    rate: float,
    over_steps: int,
    ledger: Ledger,
) -> PredicateFn:
    """
    True if the correction fire rate for this component over the
    last `over_steps` exceeds `rate`.
    """
    def check(expr: Expression) -> bool:
        recent = ledger.for_component(component).last_n(over_steps)
        if len(recent) == 0:
            return False
        return recent.fire_rate > rate
    return check


def no_improvement(
    component: str,
    over_steps: int,
    ledger: Ledger,
) -> PredicateFn:
    """
    True if the component has had corrections fired but mean improvement
    is <= 0 over the last `over_steps`. Indicates corrections aren't helping.
    """
    def check(expr: Expression) -> bool:
        recent = ledger.for_component(component).last_n(over_steps)
        if recent.n_fired == 0:
            return False
        return recent.mean_improvement <= 0.0
    return check
