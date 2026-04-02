"""Tests for typed Absence on Observation (v0.9.22)."""
import pytest
from margin import (
    Absence, Observation, Health, Confidence, Expression,
    Monitor, Parser, Thresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(name="co2", value=400.0, absence=None, absence_detail=None):
    return Observation(
        name=name,
        health=Health.INTACT,
        value=value,
        baseline=400.0,
        confidence=Confidence.HIGH,
        absence=absence,
        absence_detail=absence_detail,
    )


# ---------------------------------------------------------------------------
# Absence enum basics
# ---------------------------------------------------------------------------

class TestAbsenceEnum:
    def test_all_members_exist(self):
        assert Absence.NOT_MEASURED.value == "not_measured"
        assert Absence.BELOW_DETECTION.value == "below_detection"
        assert Absence.ABOVE_RANGE.value == "above_range"
        assert Absence.SENSOR_FAILED.value == "sensor_failed"
        assert Absence.REDACTED.value == "redacted"
        assert Absence.NOT_APPLICABLE.value == "not_applicable"
        assert Absence.PENDING.value == "pending"

    def test_round_trip_from_string(self):
        for member in Absence:
            assert Absence(member.value) is member


# ---------------------------------------------------------------------------
# Observation with absence=None (existing behavior unchanged)
# ---------------------------------------------------------------------------

class TestObservationNoAbsence:
    def test_is_absent_false(self):
        obs = _obs()
        assert obs.is_absent is False

    def test_absence_none_by_default(self):
        obs = _obs()
        assert obs.absence is None
        assert obs.absence_detail is None

    def test_sigma_unchanged(self):
        obs = _obs(value=440.0)
        assert obs.sigma == pytest.approx(0.1)

    def test_to_dict_no_absence_keys(self):
        d = _obs().to_dict()
        assert "absence" not in d
        assert "absence_detail" not in d

    def test_to_atom_unchanged(self):
        atom = _obs().to_atom()
        assert "ABSENT" not in atom
        assert "INTACT" in atom

    def test_from_dict_old_format(self):
        """Old serialized JSON (no absence key) deserializes correctly."""
        d = {
            "name": "co2",
            "health": "INTACT",
            "value": 400.0,
            "baseline": 400.0,
            "sigma": 0.0,
            "confidence": "high",
            "higher_is_better": True,
            "provenance": [],
        }
        obs = Observation.from_dict(d)
        assert obs.absence is None
        assert obs.absence_detail is None
        assert obs.is_absent is False


# ---------------------------------------------------------------------------
# Observation with absence set
# ---------------------------------------------------------------------------

class TestObservationWithAbsence:
    def test_is_absent_true(self):
        obs = _obs(absence=Absence.BELOW_DETECTION)
        assert obs.is_absent is True

    def test_absence_detail(self):
        obs = _obs(
            absence=Absence.BELOW_DETECTION,
            absence_detail="detection limit: 0.001 ppm",
        )
        assert obs.absence_detail == "detection limit: 0.001 ppm"

    def test_to_atom_shows_absence(self):
        obs = _obs(absence=Absence.SENSOR_FAILED)
        assert obs.to_atom() == "co2:ABSENT(sensor_failed)"

    def test_to_atom_below_detection(self):
        obs = _obs(absence=Absence.BELOW_DETECTION)
        assert obs.to_atom() == "co2:ABSENT(below_detection)"

    def test_to_dict_includes_absence(self):
        obs = _obs(absence=Absence.REDACTED)
        d = obs.to_dict()
        assert d["absence"] == "redacted"

    def test_to_dict_includes_absence_detail(self):
        obs = _obs(
            absence=Absence.REDACTED,
            absence_detail="HIPAA",
        )
        d = obs.to_dict()
        assert d["absence_detail"] == "HIPAA"

    def test_to_dict_no_detail_when_none(self):
        obs = _obs(absence=Absence.REDACTED)
        d = obs.to_dict()
        assert "absence_detail" not in d

    def test_round_trip_to_dict_from_dict(self):
        obs = _obs(
            absence=Absence.BELOW_DETECTION,
            absence_detail="LOD: 0.001",
        )
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        assert restored.absence is Absence.BELOW_DETECTION
        assert restored.absence_detail == "LOD: 0.001"
        assert restored.is_absent is True

    def test_round_trip_all_absence_types(self):
        for member in Absence:
            obs = _obs(absence=member)
            restored = Observation.from_dict(obs.to_dict())
            assert restored.absence is member

    def test_value_still_accessible(self):
        """Even when absent, .value is still whatever was set (caller decides convention)."""
        obs = _obs(value=0.0, absence=Absence.BELOW_DETECTION)
        assert obs.value == 0.0


# ---------------------------------------------------------------------------
# Expression with absent observations
# ---------------------------------------------------------------------------

class TestExpressionWithAbsence:
    def test_to_string_shows_absent(self):
        obs_present = _obs(name="temp", value=22.0)
        obs_absent = _obs(name="co2", absence=Absence.SENSOR_FAILED)
        expr = Expression(observations=[obs_present, obs_absent])
        s = expr.to_string()
        assert "ABSENT(sensor_failed)" in s
        assert "temp:INTACT" in s

    def test_degraded_does_not_include_absent(self):
        """Absent observations have health=INTACT (as constructed); they aren't degraded."""
        obs = _obs(absence=Absence.NOT_MEASURED)
        expr = Expression(observations=[obs])
        assert expr.degraded() == []

    def test_absent_helper(self):
        obs_present = _obs(name="temp", value=22.0)
        obs_absent = _obs(name="co2", absence=Absence.SENSOR_FAILED)
        expr = Expression(observations=[obs_present, obs_absent])
        result = expr.absent()
        assert len(result) == 1
        assert result[0].name == "co2"
        assert result[0].absence is Absence.SENSOR_FAILED

    def test_absent_empty_when_all_present(self):
        obs = _obs(name="temp", value=22.0)
        expr = Expression(observations=[obs])
        assert expr.absent() == []

    def test_absent_multiple(self):
        obs1 = _obs(name="co2", absence=Absence.SENSOR_FAILED)
        obs2 = _obs(name="ph", absence=Absence.BELOW_DETECTION)
        obs3 = _obs(name="temp", value=22.0)
        expr = Expression(observations=[obs1, obs2, obs3])
        result = expr.absent()
        assert len(result) == 2
        names = {o.name for o in result}
        assert names == {"co2", "ph"}


# ---------------------------------------------------------------------------
# Monitor skips absent observations in trackers
# ---------------------------------------------------------------------------

class TestMonitorAbsenceSkip:
    def _make_monitor(self):
        parser = Parser(
            baselines={"co2": 400.0, "temp": 22.0},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )
        return Monitor(parser, window=50)

    def test_normal_flow_unaffected(self):
        """Standard update flow still works — no regression from the guard."""
        m = self._make_monitor()
        for i in range(20):
            expr = m.update({"co2": 400.0 + i, "temp": 22.0})
        assert m._drift_trackers["co2"].n_observations == 20
        assert m._drift_trackers["temp"].n_observations == 20
        assert expr is not None

    def test_absent_skipped_in_drift_tracker(self):
        """Absent observations via absences= should not enter drift tracker."""
        m = self._make_monitor()
        for _ in range(5):
            m.update({"co2": 400.0, "temp": 22.0})
        n_before = m._drift_trackers["co2"].n_observations

        # co2 absent this step — drift tracker should not advance
        m.update({"temp": 22.0}, absences={"co2": Absence.SENSOR_FAILED})
        assert m._drift_trackers["co2"].n_observations == n_before
        # temp still counted
        assert m._drift_trackers["temp"].n_observations == 6

    def test_absent_skipped_in_anomaly_tracker(self):
        """Absent observations should not enter anomaly tracker."""
        m = self._make_monitor()
        for _ in range(5):
            m.update({"co2": 400.0, "temp": 22.0})
        n_before = m._anomaly_trackers["co2"].n_values

        m.update({"temp": 22.0}, absences={"co2": Absence.NOT_MEASURED})
        assert m._anomaly_trackers["co2"].n_values == n_before

    def test_absent_appears_in_expression(self):
        """Even though skipped in trackers, absent obs is in the Expression."""
        m = self._make_monitor()
        expr = m.update(
            {"temp": 22.0},
            absences={"co2": Absence.SENSOR_FAILED},
        )
        assert len(expr.observations) == 2
        assert len(expr.absent()) == 1
        assert expr.absent()[0].name == "co2"
        assert expr.absent()[0].absence is Absence.SENSOR_FAILED

    def test_absent_obs_has_health_ood(self):
        """Parser emits Health.OOD for absent components."""
        m = self._make_monitor()
        expr = m.update(
            {"temp": 22.0},
            absences={"co2": Absence.BELOW_DETECTION},
        )
        co2 = [o for o in expr.observations if o.name == "co2"][0]
        assert co2.health is Health.OOD
        assert co2.is_absent is True

    def test_mixed_present_and_absent_over_time(self):
        """co2 comes and goes — drift tracker only counts present steps."""
        m = self._make_monitor()
        # 3 present
        for _ in range(3):
            m.update({"co2": 400.0, "temp": 22.0})
        # 2 absent
        for _ in range(2):
            m.update({"temp": 22.0}, absences={"co2": Absence.PENDING})
        # 3 more present
        for _ in range(3):
            m.update({"co2": 405.0, "temp": 22.0})
        assert m._drift_trackers["co2"].n_observations == 6  # 3+3, not 8
        assert m._drift_trackers["temp"].n_observations == 8


# ---------------------------------------------------------------------------
# Parser.parse() with absences parameter
# ---------------------------------------------------------------------------

class TestParserAbsences:
    def _make_parser(self):
        return Parser(
            baselines={"co2": 400.0, "temp": 22.0, "ph": 7.0},
            thresholds=Thresholds(intact=0.8, ablated=0.2),
        )

    def test_no_absences_unchanged(self):
        p = self._make_parser()
        expr = p.parse({"co2": 400.0, "temp": 22.0})
        assert len(expr.observations) == 2
        assert all(not o.is_absent for o in expr.observations)

    def test_one_absent_component(self):
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"co2": Absence.SENSOR_FAILED},
        )
        assert len(expr.observations) == 2
        co2 = [o for o in expr.observations if o.name == "co2"][0]
        assert co2.is_absent is True
        assert co2.absence is Absence.SENSOR_FAILED
        assert co2.health is Health.OOD
        assert co2.value == 400.0  # baseline used as value

    def test_absent_value_equals_baseline(self):
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"ph": Absence.BELOW_DETECTION},
        )
        ph = [o for o in expr.observations if o.name == "ph"][0]
        assert ph.value == 7.0  # baseline
        assert ph.sigma == pytest.approx(0.0)  # value == baseline → sigma 0

    def test_absent_confidence_defaults_to_indeterminate(self):
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"co2": Absence.NOT_MEASURED},
        )
        co2 = [o for o in expr.observations if o.name == "co2"][0]
        assert co2.confidence is Confidence.INDETERMINATE

    def test_absent_confidence_overridable(self):
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"co2": Absence.NOT_MEASURED},
            confidences={"co2": Confidence.LOW},
        )
        co2 = [o for o in expr.observations if o.name == "co2"][0]
        assert co2.confidence is Confidence.LOW

    def test_absence_wins_over_value(self):
        """If a component appears in both values and absences, absence wins."""
        p = self._make_parser()
        expr = p.parse(
            {"co2": 999.0, "temp": 22.0},
            absences={"co2": Absence.REDACTED},
        )
        co2 = [o for o in expr.observations if o.name == "co2"][0]
        assert co2.is_absent is True
        assert co2.value == 400.0  # baseline, not the 999.0 that was passed

    def test_all_absent(self):
        p = self._make_parser()
        expr = p.parse(
            {},
            absences={
                "co2": Absence.SENSOR_FAILED,
                "temp": Absence.NOT_MEASURED,
            },
        )
        assert len(expr.observations) == 2
        assert all(o.is_absent for o in expr.observations)

    def test_absent_unknown_component(self):
        """Absent component not in baselines still emits an observation."""
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"humidity": Absence.NOT_APPLICABLE},
        )
        hum = [o for o in expr.observations if o.name == "humidity"][0]
        assert hum.is_absent is True
        assert hum.value == 0.0  # no baseline → 0.0

    def test_round_trip_absent_observation(self):
        p = self._make_parser()
        expr = p.parse(
            {"temp": 22.0},
            absences={"co2": Absence.BELOW_DETECTION},
        )
        d = expr.to_dict()
        restored = Expression.from_dict(d)
        co2 = [o for o in restored.observations if o.name == "co2"][0]
        assert co2.absence is Absence.BELOW_DETECTION
        assert co2.is_absent is True
