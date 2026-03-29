"""Tests for margin.intent — goal feasibility evaluation."""

from datetime import datetime, timedelta

from margin import (
    Health, Confidence, Thresholds, Parser, Monitor,
    Observation, Expression,
    DriftState, DriftDirection, DriftClassification,
    Feasibility, Requirement, RiskFactor, IntentResult, Intent,
)


def _obs(name, health, value, baseline=100.0, hib=True):
    return Observation(
        name=name, health=health, value=value, baseline=baseline,
        confidence=Confidence.HIGH, higher_is_better=hib,
    )


def _expr(*observations):
    return Expression(
        observations=list(observations),
        corrections=[],
        confidence=Confidence.HIGH,
    )


def _drift(component, direction, rate=0.01):
    return DriftClassification(
        component=component,
        state=DriftState.DRIFTING,
        direction=direction,
        rate=rate if direction == DriftDirection.IMPROVING else -rate,
        acceleration=0.0,
        r_squared=0.9,
        confidence=Confidence.HIGH,
        n_samples=20,
        window_seconds=1200.0,
    )


t0 = datetime(2026, 1, 1)


# -----------------------------------------------------------------------
# Basic feasibility
# -----------------------------------------------------------------------

class TestFeasibility:
    def test_all_requirements_met(self):
        intent = Intent(goal="deliver package")
        intent.require("battery", min_health=Health.DEGRADED)
        intent.require("navigation", min_health=Health.DEGRADED)

        expr = _expr(
            _obs("battery", Health.INTACT, 90.0),
            _obs("navigation", Health.INTACT, 0.95),
        )
        result = intent.evaluate(expr)
        assert result.feasibility == Feasibility.FEASIBLE
        assert result.feasible is True
        assert len(result.met) == 2
        assert len(result.violated) == 0

    def test_critical_violation(self):
        intent = Intent(goal="deliver package")
        intent.require("battery", min_health=Health.DEGRADED, critical=True)

        expr = _expr(_obs("battery", Health.ABLATED, 5.0))
        result = intent.evaluate(expr)
        assert result.feasibility == Feasibility.INFEASIBLE
        assert result.infeasible is True
        assert "battery" in result.violated

    def test_non_critical_violation(self):
        intent = Intent(goal="deliver package")
        intent.require("wifi", min_health=Health.DEGRADED, critical=False)

        expr = _expr(_obs("wifi", Health.ABLATED, -85.0))
        result = intent.evaluate(expr)
        assert result.feasibility == Feasibility.AT_RISK
        assert result.at_risk is True

    def test_missing_component(self):
        intent = Intent(goal="deliver")
        intent.require("battery", critical=True)

        expr = _expr()  # no observations
        result = intent.evaluate(expr)
        assert result.feasibility == Feasibility.INFEASIBLE

    def test_no_requirements(self):
        intent = Intent(goal="exist")
        result = intent.evaluate(_expr())
        assert result.feasibility == Feasibility.UNKNOWN

    def test_value_requirement_higher_is_better(self):
        intent = Intent(goal="deliver")
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        assert intent.evaluate(expr).feasible

        expr2 = _expr(_obs("battery_soc", Health.DEGRADED, 15.0))
        result = intent.evaluate(expr2)
        assert result.infeasible

    def test_value_requirement_lower_is_better(self):
        intent = Intent(goal="stable temp")
        intent.require("temperature", min_value=80.0, min_health=Health.ABLATED)

        # lower is better: value must be <= min_value
        expr = _expr(_obs("temperature", Health.INTACT, 60.0, hib=False))
        assert intent.evaluate(expr).feasible

        expr2 = _expr(_obs("temperature", Health.DEGRADED, 90.0, hib=False))
        result = intent.evaluate(expr2)
        assert result.infeasible


# -----------------------------------------------------------------------
# Drift-aware evaluation
# -----------------------------------------------------------------------

class TestDriftAware:
    def test_worsening_drift_at_risk(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        drift = {"battery_soc": _drift("battery_soc", DriftDirection.WORSENING, rate=0.05)}

        result = intent.evaluate(expr, drift)
        assert result.feasibility == Feasibility.AT_RISK
        assert "battery_soc" in result.trending_bad
        assert any(r.eta_seconds is not None for r in result.risks)

    def test_improving_drift_no_risk(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        drift = {"battery_soc": _drift("battery_soc", DriftDirection.IMPROVING, rate=0.05)}

        result = intent.evaluate(expr, drift)
        assert result.feasibility == Feasibility.FEASIBLE

    def test_worsening_but_no_deadline(self):
        intent = Intent(goal="deliver")  # no deadline
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        drift = {"battery_soc": _drift("battery_soc", DriftDirection.WORSENING, rate=0.05)}

        result = intent.evaluate(expr, drift)
        # Still AT_RISK because it's trending bad, just no deadline comparison
        assert result.feasibility == Feasibility.AT_RISK
        assert "battery_soc" in result.trending_bad

    def test_worsening_but_slow(self):
        intent = Intent(goal="deliver", deadline_seconds=60)
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        # Very slow drift — won't reach 20 before deadline
        drift = {"battery_soc": _drift("battery_soc", DriftDirection.WORSENING, rate=0.001)}

        result = intent.evaluate(expr, drift)
        # ETA to violation is 50-20=30 / 0.001 = 30000 seconds >> 60 second deadline
        # Still trending bad though
        assert "battery_soc" in result.trending_bad


# -----------------------------------------------------------------------
# ETA estimation
# -----------------------------------------------------------------------

class TestETA:
    def test_eta_calculation(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.INTACT, 50.0))
        # Rate of -0.05 units/second → 30 units to threshold → 600 seconds
        drift = {"battery_soc": _drift("battery_soc", DriftDirection.WORSENING, rate=0.05)}

        result = intent.evaluate(expr, drift)
        eta_risk = [r for r in result.risks if r.eta_seconds is not None]
        assert len(eta_risk) >= 1
        # ETA should be approximately 600 seconds (30 / 0.05)
        assert 500 < eta_risk[0].eta_seconds < 700

    def test_already_violated_eta_zero(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery_soc", min_value=20.0)

        expr = _expr(_obs("battery_soc", Health.DEGRADED, 15.0))
        result = intent.evaluate(expr)
        assert result.infeasible


# -----------------------------------------------------------------------
# Monitor integration
# -----------------------------------------------------------------------

class TestMonitorIntegration:
    def test_evaluate_monitor(self):
        parser = Parser(
            baselines={"battery_soc": 80.0, "navigation": 0.95},
            thresholds=Thresholds(intact=30.0, ablated=10.0),
            component_thresholds={
                "navigation": Thresholds(intact=0.8, ablated=0.4),
            },
        )
        monitor = Monitor(parser, window=50)

        for i in range(15):
            monitor.update(
                {"battery_soc": 80.0 - i * 3, "navigation": 0.95},
                now=t0 + timedelta(seconds=i * 60),
            )

        intent = Intent(goal="deliver package", deadline_seconds=1800)
        intent.require("battery_soc", min_value=20.0)
        intent.require("navigation", min_health=Health.DEGRADED)

        result = intent.evaluate_monitor(monitor)
        assert result.feasibility in (Feasibility.FEASIBLE, Feasibility.AT_RISK)

    def test_evaluate_monitor_empty(self):
        parser = Parser(baselines={"x": 50.0}, thresholds=Thresholds(intact=40.0, ablated=10.0))
        monitor = Monitor(parser)

        intent = Intent(goal="something")
        intent.require("x")

        result = intent.evaluate_monitor(monitor)
        assert result.feasibility == Feasibility.UNKNOWN


# -----------------------------------------------------------------------
# Chaining and serialization
# -----------------------------------------------------------------------

class TestAPI:
    def test_chaining(self):
        intent = (Intent(goal="deliver", deadline_seconds=900)
                  .require("battery", min_value=20.0)
                  .require("nav", min_health=Health.DEGRADED)
                  .require("wifi", min_health=Health.ABLATED, critical=False))
        assert len(intent.requirements) == 3

    def test_to_dict(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery", min_value=20.0)
        d = intent.to_dict()
        assert d["goal"] == "deliver"
        assert d["deadline_seconds"] == 900
        assert len(d["requirements"]) == 1

    def test_result_to_dict(self):
        intent = Intent(goal="test")
        intent.require("x")
        expr = _expr(_obs("x", Health.INTACT, 50.0))
        result = intent.evaluate(expr)
        d = result.to_dict()
        assert "feasibility" in d
        assert "risks" in d
        assert "summary" in d

    def test_result_summary(self):
        intent = Intent(goal="deliver")
        intent.require("battery", min_health=Health.DEGRADED, critical=True)
        expr = _expr(_obs("battery", Health.ABLATED, 5.0))
        result = intent.evaluate(expr)
        s = result.summary()
        assert "INFEASIBLE" in s
        assert "battery" in s

    def test_repr(self):
        intent = Intent(goal="deliver", deadline_seconds=900)
        intent.require("battery")
        assert "deliver" in repr(intent)
        assert "1 requirements" in repr(intent)

    def test_risk_sorting(self):
        """Critical risks should come before warnings."""
        intent = Intent(goal="deliver")
        intent.require("battery", critical=True)
        intent.require("wifi", critical=False)

        expr = _expr(
            _obs("battery", Health.ABLATED, 5.0),
            _obs("wifi", Health.ABLATED, -90.0),
        )
        result = intent.evaluate(expr)
        assert result.risks[0].severity == "critical"

    def test_confidence_with_drift(self):
        intent = Intent(goal="test")
        intent.require("x")
        intent.require("y")

        expr = _expr(
            _obs("x", Health.INTACT, 50.0),
            _obs("y", Health.INTACT, 50.0),
        )
        # Both have drift data → HIGH confidence
        drift = {
            "x": _drift("x", DriftDirection.IMPROVING),
            "y": _drift("y", DriftDirection.IMPROVING),
        }
        result = intent.evaluate(expr, drift)
        assert result.confidence == Confidence.HIGH

    def test_confidence_without_drift(self):
        intent = Intent(goal="test")
        intent.require("x")

        expr = _expr(_obs("x", Health.INTACT, 50.0))
        result = intent.evaluate(expr)
        assert result.confidence == Confidence.LOW
