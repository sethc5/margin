"""Tests for margin.config — config-driven setup."""

import json
import tempfile
from pathlib import Path

from margin import (
    from_config, load_config,
    Health, Op, Parser, Policy, Contract,
)


BASIC_CONFIG = {
    "components": {
        "cpu": {"baseline": 50.0, "intact": 80.0, "ablated": 30.0},
        "mem": {"baseline": 70.0, "intact": 60.0, "ablated": 20.0},
        "error_rate": {"baseline": 0.002, "intact": 0.01, "ablated": 0.10, "lower_is_better": True},
    },
}

FULL_CONFIG = {
    **BASIC_CONFIG,
    "policy": [
        {"name": "critical", "when": "any_ablated", "action": {"op": "RESTORE", "alpha": 1.0}, "priority": 50},
        {"name": "maintain", "when": "any_degraded", "action": {"op": "RESTORE", "alpha": 0.5}, "priority": 10},
        {"name": "normal", "when": "all_intact", "action": {"op": "NOOP"}, "priority": 0},
    ],
    "contract": [
        {"name": "cpu-healthy", "component": "cpu", "health": "INTACT"},
        {"name": "mem-healthy", "component": "mem", "health": "INTACT"},
    ],
}


class TestFromConfig:
    def test_parser_only(self):
        result = from_config(BASIC_CONFIG)
        assert "parser" in result
        assert isinstance(result["parser"], Parser)
        assert "policy" not in result
        assert "contract" not in result

    def test_parser_baselines(self):
        result = from_config(BASIC_CONFIG)
        p = result["parser"]
        assert p.baselines["cpu"] == 50.0
        assert p.baselines["error_rate"] == 0.002

    def test_parser_polarity(self):
        result = from_config(BASIC_CONFIG)
        p = result["parser"]
        # cpu is higher_is_better (default)
        assert p._thresholds_for("cpu").higher_is_better is True
        # error_rate is lower_is_better
        assert p._thresholds_for("error_rate").higher_is_better is False

    def test_parser_thresholds(self):
        result = from_config(BASIC_CONFIG)
        p = result["parser"]
        assert p._thresholds_for("cpu").intact == 80.0
        assert p._thresholds_for("cpu").ablated == 30.0
        assert p._thresholds_for("error_rate").intact == 0.01

    def test_parser_works(self):
        result = from_config(BASIC_CONFIG)
        expr = result["parser"].parse({"cpu": 48.0, "mem": 65.0, "error_rate": 0.003})
        assert expr is not None
        assert len(expr.observations) == 3

    def test_full_config(self):
        result = from_config(FULL_CONFIG)
        assert "parser" in result
        assert "policy" in result
        assert "contract" in result

    def test_policy(self):
        result = from_config(FULL_CONFIG)
        policy = result["policy"]
        assert isinstance(policy, Policy)
        assert len(policy.rules) == 3
        assert policy.rules[0].name == "critical"

    def test_policy_predicates_work(self):
        result = from_config(FULL_CONFIG)
        p = result["parser"]
        policy = result["policy"]
        # Everything intact
        expr = p.parse({"cpu": 90.0, "mem": 80.0, "error_rate": 0.001})
        matched = policy.evaluate(expr)
        provenance = [prov for r in matched for prov in getattr(r, "provenance", [])]
        assert any("normal" in p for p in provenance)

    def test_policy_ablated_fires_critical(self):
        result = from_config(FULL_CONFIG)
        p = result["parser"]
        policy = result["policy"]
        # cpu ablated
        expr = p.parse({"cpu": 5.0, "mem": 80.0, "error_rate": 0.001})
        matched = policy.evaluate(expr)
        provenance = [prov for r in matched for prov in getattr(r, "provenance", [])]
        assert any("critical" in p for p in provenance)

    def test_contract(self):
        result = from_config(FULL_CONFIG)
        contract = result["contract"]
        assert isinstance(contract, Contract)
        assert len(contract.terms) == 2

    def test_empty_components_raises(self):
        try:
            from_config({})
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_unknown_predicate_raises(self):
        try:
            from_config({
                "components": {"x": {"baseline": 1.0, "intact": 2.0, "ablated": 0.5}},
                "policy": [{"name": "bad", "when": "nonexistent_predicate", "action": {"op": "RESTORE"}}],
            })
            assert False, "Should have raised"
        except ValueError as e:
            assert "Unknown predicate" in str(e)

    def test_default_thresholds(self):
        config = {
            "components": {
                "cpu": {"baseline": 50.0},  # no per-component thresholds
            },
            "default_thresholds": {"intact": 90.0, "ablated": 10.0},
        }
        result = from_config(config)
        assert result["parser"]._thresholds_for("cpu").intact == 90.0

    def test_complex_predicates(self):
        config = {
            "components": {"x": {"baseline": 50.0, "intact": 80.0, "ablated": 20.0}},
            "policy": [
                {"name": "combo", "when": {"all_of": ["any_degraded", "any_correction"]},
                 "action": {"op": "RESTORE"}, "priority": 10},
                {"name": "not-intact", "when": {"not": "all_intact"},
                 "action": {"op": "RESTORE"}, "priority": 5},
            ],
        }
        result = from_config(config)
        assert len(result["policy"].rules) == 2

    def test_component_health_predicate(self):
        config = {
            "components": {"cpu": {"baseline": 50.0, "intact": 80.0, "ablated": 20.0}},
            "policy": [
                {"name": "cpu-ablated", "when": {"component_health": "cpu", "health": "ABLATED"},
                 "action": {"op": "RESTORE"}, "priority": 10},
            ],
        }
        result = from_config(config)
        p = result["parser"]
        policy = result["policy"]
        expr = p.parse({"cpu": 5.0})
        matched = policy.evaluate(expr)
        provenance = [prov for r in matched for prov in getattr(r, "provenance", [])]
        assert any("cpu-ablated" in p for p in provenance)


class TestLoadConfig:
    def test_load_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(BASIC_CONFIG, f)
            f.flush()
            result = load_config(f.name)
        assert "parser" in result
        Path(f.name).unlink()

    def test_load_full_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(FULL_CONFIG, f)
            f.flush()
            result = load_config(f.name)
        assert "parser" in result
        assert "policy" in result
        assert "contract" in result
        Path(f.name).unlink()
