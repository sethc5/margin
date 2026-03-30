"""
Config-driven setup: define parsers, policies, and contracts from dicts/YAML.

    margin.from_config({
        "components": {"cpu": {"baseline": 50, "intact": 80, "ablated": 30}},
        "policy": [{"name": "critical", "when": "any_ablated", "action": {"op": "RESTORE"}}],
    })

Or load from YAML/JSON file:

    margin.load_config("margin.yaml")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .health import Thresholds
from .observation import Parser, Op
from .policy.core import Action, PolicyRule, Policy, Constraint, Escalation, EscalationLevel
from .contract import Contract, HealthTarget, Health, contract_term_from_dict
from .predicates import (
    any_health, all_health, any_degraded, any_correction,
    confidence_below, sigma_below, component_health,
    any_of, all_of, not_,
)
from .confidence import Confidence


# -----------------------------------------------------------------------
# Predicate registry — map string names to predicate factories
# -----------------------------------------------------------------------

_PREDICATE_REGISTRY = {
    "any_intact": lambda: any_health(Health.INTACT),
    "any_degraded": lambda: any_degraded(),
    "any_ablated": lambda: any_health(Health.ABLATED),
    "any_recovering": lambda: any_health(Health.RECOVERING),
    "any_ood": lambda: any_health(Health.OOD),
    "all_intact": lambda: all_health(Health.INTACT),
    "all_degraded": lambda: all_health(Health.DEGRADED),
    "all_ablated": lambda: all_health(Health.ABLATED),
    "any_correction": lambda: any_correction(),
}


def _resolve_predicate(spec):
    """Resolve a predicate from a string name or dict spec."""
    if isinstance(spec, str):
        factory = _PREDICATE_REGISTRY.get(spec)
        if factory is None:
            raise ValueError(f"Unknown predicate: {spec!r}. "
                             f"Available: {sorted(_PREDICATE_REGISTRY.keys())}")
        return factory()

    if isinstance(spec, dict):
        if "component_health" in spec:
            return component_health(spec["component_health"], Health(spec["health"]))
        if "sigma_below" in spec:
            return sigma_below(spec["sigma_below"], spec["threshold"])
        if "confidence_below" in spec:
            return confidence_below(Confidence(spec["confidence_below"]))
        if "all_of" in spec:
            return all_of(*[_resolve_predicate(p) for p in spec["all_of"]])
        if "any_of" in spec:
            return any_of(*[_resolve_predicate(p) for p in spec["any_of"]])
        if "not" in spec:
            return not_(_resolve_predicate(spec["not"]))
        raise ValueError(f"Unknown predicate spec: {spec}")

    raise TypeError(f"Predicate must be str or dict, got {type(spec)}")


# -----------------------------------------------------------------------
# Config parsing
# -----------------------------------------------------------------------

def from_config(config: dict) -> dict:
    """
    Build margin objects from a config dict.

    Config format:
        {
            "components": {
                "cpu": {"baseline": 50, "intact": 80, "ablated": 30},
                "error_rate": {"baseline": 0.002, "intact": 0.01, "ablated": 0.10,
                               "lower_is_better": true},
            },
            "default_thresholds": {"intact": 80, "ablated": 30},  # optional
            "policy": [
                {"name": "critical", "when": "any_ablated",
                 "action": {"op": "RESTORE", "alpha": 1.0}, "priority": 50},
            ],
            "contract": [
                {"name": "cpu-healthy", "component": "cpu", "health": "INTACT"},
            ],
        }

    Returns dict with keys: "parser", "policy" (optional), "contract" (optional).
    """
    result = {}

    # --- Components → Parser ---
    components = config.get("components", {})
    if not components:
        raise ValueError("Config must have 'components'")

    defaults = config.get("default_thresholds", {})
    default_intact = defaults.get("intact", 80.0)
    default_ablated = defaults.get("ablated", 30.0)
    default_hib = defaults.get("higher_is_better", True)

    baselines = {}
    component_thresholds = {}

    for name, spec in components.items():
        baselines[name] = spec["baseline"]
        hib = spec.get("higher_is_better", spec.get("lower_is_better") is not True and default_hib)
        # Handle lower_is_better shorthand
        if "lower_is_better" in spec:
            hib = not spec["lower_is_better"]
        component_thresholds[name] = Thresholds(
            intact=spec.get("intact", default_intact),
            ablated=spec.get("ablated", default_ablated),
            higher_is_better=hib,
        )

    # Use first component's thresholds as default (or the explicit default)
    first_name = next(iter(components))
    default_thresh = Thresholds(
        intact=default_intact,
        ablated=default_ablated,
        higher_is_better=default_hib,
    )

    result["parser"] = Parser(
        baselines=baselines,
        thresholds=default_thresh,
        component_thresholds=component_thresholds,
    )

    # --- Policy ---
    policy_specs = config.get("policy", [])
    if policy_specs:
        rules = []
        for spec in policy_specs:
            predicate = _resolve_predicate(spec["when"])
            action_spec = spec.get("action", {})
            action = Action(
                target=action_spec.get("target", "*"),
                op=Op(action_spec.get("op", "RESTORE")),
                alpha=action_spec.get("alpha", 0.5),
                magnitude=action_spec.get("magnitude", 1.0),
            )
            constraint = None
            if "constraint" in spec:
                cs = spec["constraint"]
                constraint = Constraint(
                    max_alpha=cs.get("max_alpha", 1.0),
                    min_alpha=cs.get("min_alpha", 0.0),
                    cooldown_steps=cs.get("cooldown_steps", 0),
                    max_per_window=cs.get("max_per_window", 0),
                    window_steps=cs.get("window_steps", 0),
                )
            escalation = None
            if "escalation" in spec:
                es = spec["escalation"]
                escalation = Escalation(
                    level=EscalationLevel(es.get("level", "LOG")),
                    reason=es.get("reason", ""),
                )
            rules.append(PolicyRule(
                name=spec["name"],
                predicate=predicate,
                action=action,
                priority=spec.get("priority", 0),
                constraint=constraint,
                escalation=escalation,
            ))
        result["policy"] = Policy(name=config.get("name", "config-policy"), rules=rules)

    # --- Contract ---
    contract_specs = config.get("contract", [])
    if contract_specs:
        terms = []
        for spec in contract_specs:
            term_type = spec.get("type", "health_target")
            if term_type == "health_target" and "type" not in spec:
                # backward-compatible: old configs without explicit type field
                terms.append(HealthTarget(
                    name=spec["name"],
                    component=spec["component"],
                    target=Health(spec.get("health", "INTACT")),
                ))
            else:
                # normalise: config may use "health" instead of "target"
                if "health" in spec and "target" not in spec:
                    spec = dict(spec, target=spec["health"])
                terms.append(contract_term_from_dict(spec))
        result["contract"] = Contract(
            name=config.get("contract_name", "config-contract"),
            terms=terms,
        )

    return result


def load_config(path: str) -> dict:
    """
    Load config from a YAML or JSON file and build margin objects.

    YAML requires PyYAML (optional dependency). JSON works out of the box.
    """
    p = Path(path)
    text = p.read_text()

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)

    return from_config(config)
