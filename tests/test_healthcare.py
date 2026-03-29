"""Tests for the healthcare adapter."""

import pytest
from datetime import datetime
from margin.health import Health
from margin.confidence import Confidence
from adapters.healthcare.vitals import (
    VitalSign, VITAL_SIGNS, BandThresholds,
    classify_band, parse_vitals, patient_expression,
)
from adapters.healthcare.contracts import (
    standard_monitoring_contract, icu_contract, sepsis_screening_contract,
)
from margin.ledger import Ledger, Record
from margin.observation import Observation, Op


class TestBandThresholds:
    def test_normal_value_is_intact(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(72, hr.band) == Health.INTACT

    def test_high_value_is_degraded(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(110, hr.band) == Health.DEGRADED

    def test_low_value_is_degraded(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(55, hr.band) == Health.DEGRADED

    def test_critical_high_is_ablated(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(160, hr.band) == Health.ABLATED

    def test_critical_low_is_ablated(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(35, hr.band) == Health.ABLATED

    def test_at_normal_boundary_is_intact(self):
        hr = VITAL_SIGNS["hr"]
        assert classify_band(60, hr.band) == Health.INTACT
        assert classify_band(100, hr.band) == Health.INTACT

    def test_spo2_normal(self):
        assert classify_band(98, VITAL_SIGNS["spo2"].band) == Health.INTACT

    def test_spo2_low(self):
        assert classify_band(92, VITAL_SIGNS["spo2"].band) == Health.DEGRADED

    def test_spo2_critical(self):
        assert classify_band(88, VITAL_SIGNS["spo2"].band) == Health.ABLATED

    def test_temp_normal(self):
        assert classify_band(36.8, VITAL_SIGNS["temp"].band) == Health.INTACT

    def test_temp_fever(self):
        assert classify_band(38.5, VITAL_SIGNS["temp"].band) == Health.DEGRADED

    def test_temp_critical_fever(self):
        assert classify_band(40.5, VITAL_SIGNS["temp"].band) == Health.ABLATED

    def test_temp_hypothermia(self):
        assert classify_band(34.5, VITAL_SIGNS["temp"].band) == Health.ABLATED

    def test_glucose_normal(self):
        assert classify_band(90, VITAL_SIGNS["glucose"].band) == Health.INTACT

    def test_glucose_hypo(self):
        assert classify_band(50, VITAL_SIGNS["glucose"].band) == Health.ABLATED

    def test_glucose_hyper(self):
        assert classify_band(300, VITAL_SIGNS["glucose"].band) == Health.ABLATED


class TestParseVitals:
    def test_parse_all_normal(self):
        readings = {"hr": 72, "sbp": 115, "spo2": 98, "temp": 36.8, "rr": 16}
        obs = parse_vitals(readings)
        assert all(o.health == Health.INTACT for o in obs.values())

    def test_parse_mixed(self):
        readings = {"hr": 72, "sbp": 145, "spo2": 93}
        obs = parse_vitals(readings)
        assert obs["hr"].health == Health.INTACT
        assert obs["sbp"].health == Health.DEGRADED
        assert obs["spo2"].health == Health.DEGRADED

    def test_ignores_unknown_vitals(self):
        readings = {"hr": 72, "unknown_thing": 42}
        obs = parse_vitals(readings)
        assert "hr" in obs
        assert "unknown_thing" not in obs

    def test_with_timestamp(self):
        t = datetime(2026, 3, 28, 14, 30)
        obs = parse_vitals({"hr": 72}, measured_at=t)
        assert obs["hr"].measured_at == t

    def test_with_confidence(self):
        obs = parse_vitals({"hr": 72}, confidence=Confidence.LOW)
        assert obs["hr"].confidence == Confidence.LOW


class TestPatientExpression:
    def test_basic(self):
        expr = patient_expression(
            {"hr": 72, "sbp": 115, "spo2": 98},
            patient_id="bed-4",
        )
        assert expr.label == "bed-4"
        assert len(expr.observations) == 3
        assert expr.health_of("hr") == Health.INTACT

    def test_net_confidence_weakest(self):
        expr = patient_expression(
            {"hr": 72, "sbp": 115},
            confidence=Confidence.LOW,
        )
        assert expr.confidence == Confidence.LOW

    def test_to_string_readable(self):
        expr = patient_expression({"hr": 88, "spo2": 93})
        s = expr.to_string()
        assert "hr" in s
        assert "spo2" in s

    def test_empty_readings(self):
        expr = patient_expression({})
        assert len(expr.observations) == 0
        assert expr.confidence == Confidence.INDETERMINATE

    def test_sigma_positive_for_healthy(self):
        expr = patient_expression({"hr": 72})
        # 72 is at baseline → sigma ≈ 0
        assert abs(expr.observations[0].sigma) < 0.01

    def test_sigma_negative_for_unhealthy(self):
        # HR 130 is above normal, unhealthy
        expr = patient_expression({"hr": 130})
        assert expr.observations[0].sigma < 0

    def test_sigma_negative_for_low(self):
        # HR 50 is below normal, unhealthy
        expr = patient_expression({"hr": 50})
        assert expr.observations[0].sigma < 0


class TestContracts:
    def _healthy_obs(self, name):
        baseline = VITAL_SIGNS[name].band.baseline
        return Observation(name, Health.INTACT, baseline, baseline, Confidence.HIGH)

    def test_standard_monitoring_all_intact(self):
        expr = patient_expression({"hr": 72, "sbp": 115, "dbp": 75, "spo2": 98, "temp": 36.8, "rr": 16})
        contract = standard_monitoring_contract()
        result = contract.evaluate(Ledger(), expr)
        assert result.all_met

    def test_standard_monitoring_one_ablated(self):
        expr = patient_expression({"hr": 35, "sbp": 115, "dbp": 75, "spo2": 98, "temp": 36.8, "rr": 16})
        contract = standard_monitoring_contract()
        result = contract.evaluate(Ledger(), expr)
        assert result.any_violated

    def test_icu_contract(self):
        expr = patient_expression({"hr": 72, "sbp": 115, "spo2": 98, "rr": 16, "temp": 36.8})
        contract = icu_contract()
        result = contract.evaluate(Ledger(), expr)
        # Some terms will be PENDING (sustain needs history)
        assert result.any_pending

    def test_sepsis_screening_normal(self):
        expr = patient_expression({"rr": 16, "sbp": 115, "temp": 36.8, "hr": 72})
        contract = sepsis_screening_contract()
        result = contract.evaluate(Ledger(), expr)
        assert result.all_met

    def test_sepsis_screening_elevated(self):
        # Elevated RR + low SBP + fever → sepsis screen flags
        expr = patient_expression({"rr": 24, "sbp": 85, "temp": 39.0, "hr": 110})
        contract = sepsis_screening_contract()
        result = contract.evaluate(Ledger(), expr)
        assert result.any_violated

    def test_contract_with_patient_id(self):
        contract = icu_contract(patient_id="bed-7")
        assert "bed-7" in contract.name

    def test_contract_to_string(self):
        expr = patient_expression({"hr": 72, "sbp": 115, "spo2": 98, "rr": 16, "temp": 36.8})
        contract = standard_monitoring_contract()
        result = contract.evaluate(Ledger(), expr)
        s = result.to_string()
        assert "[+]" in s  # at least one MET


class TestAllVitalsDefined:
    """Verify all 8 vital signs have consistent band definitions."""

    def test_all_have_bands(self):
        for name, vital in VITAL_SIGNS.items():
            assert vital.band.critical_low < vital.band.normal_low, f"{name}: critical_low >= normal_low"
            assert vital.band.normal_low <= vital.band.normal_high, f"{name}: normal_low > normal_high"
            assert vital.band.normal_high < vital.band.critical_high or name == "spo2", f"{name}: normal_high >= critical_high"
            assert vital.band.critical_low < vital.band.baseline < vital.band.critical_high, f"{name}: baseline outside critical range"

    def test_all_baselines_are_intact(self):
        for name, vital in VITAL_SIGNS.items():
            health = classify_band(vital.band.baseline, vital.band)
            assert health == Health.INTACT, f"{name}: baseline classifies as {health.value}"
