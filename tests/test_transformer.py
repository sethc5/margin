"""Tests for the transformer adapter layer."""

import pytest
from datetime import datetime

from margin.confidence import Confidence
from margin.health import Health, Thresholds
from margin.observation import Op, Expression
from margin.ledger import Record, Ledger

from transformer.signal import ProcessSignal, DEFAULT_SIGNAL_NAMES
from transformer.circuit import CircuitState, gates_to_confidence
from transformer.parsers import make_pythia_parser, make_from_sweep


class TestProcessSignal:
    def test_from_list(self):
        sig = ProcessSignal.from_list([0.5, -0.3, 0.8, 0.1] + [0.0] * 12)
        assert sig["confidence"] == 0.5
        assert sig["novelty"] == -0.3
        assert sig.n_signals == 16

    def test_named_dict(self):
        sig = ProcessSignal.zeros()
        d = sig.named_dict()
        assert len(d) == 16
        assert all(v == 0.0 for v in d.values())

    def test_core_state(self):
        sig = ProcessSignal.from_list([0.1, 0.2, 0.3, 0.4] + [0.0] * 12)
        assert sig.core_state == {"confidence": 0.1, "novelty": 0.2, "coherence": 0.3, "retrieval_mode": 0.4}

    def test_trajectory(self):
        vals = [0.0] * 4 + [0.5, -0.5, 0.3, -0.1] + [0.0] * 8
        sig = ProcessSignal.from_list(vals)
        assert sig.trajectory["converging"] == 0.5
        assert sig.trajectory["stalling"] == -0.1

    def test_confidence_as_uncertain(self):
        sig = ProcessSignal.from_list([0.7] + [0.0] * 15)
        uv = sig.confidence_as_uncertain()
        assert uv.point == 0.7
        assert uv.uncertainty == 0.05

    def test_validity_is_event_based(self):
        sig = ProcessSignal.from_list([0.0] * 16, prompt_hash="abc")
        v = sig.validity()
        assert v.mode == "event"
        assert "abc" in v.invalidating_event

    def test_roundtrip(self):
        sig = ProcessSignal.from_list([0.1, 0.2] + [0.0] * 14, hook_layer=14, prompt_hash="xyz")
        d = sig.to_dict()
        sig2 = ProcessSignal.from_dict(d)
        assert sig2.hook_layer == 14
        assert sig2.prompt_hash == "xyz"
        assert sig2["confidence"] == pytest.approx(0.1, abs=0.001)

    def test_repr(self):
        sig = ProcessSignal.from_list([0.9, -0.8, 0.7] + [0.0] * 13)
        r = repr(sig)
        assert "ProcessSignal" in r


class TestGatesToConfidence:
    def test_both_gates(self):
        assert gates_to_confidence(True, True, 5.0) == Confidence.HIGH

    def test_one_gate(self):
        assert gates_to_confidence(True, False, 5.0) == Confidence.MODERATE
        assert gates_to_confidence(False, True, 5.0) == Confidence.MODERATE

    def test_no_gates(self):
        assert gates_to_confidence(False, False, 5.0) == Confidence.LOW

    def test_no_gates_negative_gap(self):
        assert gates_to_confidence(False, False, -1.0) == Confidence.INDETERMINATE

    def test_one_gate_negative_gap(self):
        # One gate still fired — MODERATE, not INDETERMINATE
        assert gates_to_confidence(True, False, -1.0) == Confidence.MODERATE


class TestCircuitState:
    def test_from_step_result_intact(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.1, current_io_gap=4.8,
            gap_gate=False, geo_gate=False, gate_reason="neither",
            corrected=False,
        )
        assert cs.circuit_name == "IOI"
        assert cs.health() == Health.INTACT
        assert cs.gate_confidence == Confidence.LOW

    def test_from_step_result_ablated(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.1, current_io_gap=0.8,
            gap_gate=True, geo_gate=True, gate_reason="gap_threshold",
            corrected=True,
        )
        assert cs.health() == Health.RECOVERING  # corrected=True
        assert cs.gate_confidence == Confidence.HIGH

    def test_gap_delta(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=3.0,
            gap_gate=False, geo_gate=False, gate_reason="neither",
            corrected=False,
        )
        assert cs.gap_delta == pytest.approx(-2.0)

    def test_recovery_ratio(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=4.0,
            gap_gate=False, geo_gate=False, gate_reason="neither",
            corrected=False,
        )
        assert cs.recovery_ratio == pytest.approx(0.8)

    def test_to_observation(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=4.0,
            gap_gate=True, geo_gate=True, gate_reason="gap",
            corrected=False,
        )
        obs = cs.to_observation()
        assert obs.name == "IOI"
        assert obs.health == Health.INTACT
        assert obs.baseline == 5.0
        assert obs.value == 4.0
        assert obs.higher_is_better is True
        assert obs.measured_at is not None

    def test_gap_diff_uncertainty(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=3.0,
            gap_gate=False, geo_gate=False, gate_reason="neither",
            corrected=False, gap_uncertainty=0.2,
        )
        diff = cs.gap_diff()
        assert diff.point == pytest.approx(-2.0)
        assert diff.uncertainty > 0  # propagated from both gaps

    def test_with_signal(self):
        sig = ProcessSignal.from_list([0.5] + [0.0] * 15)
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=4.0,
            gap_gate=True, geo_gate=True, gate_reason="gap",
            corrected=False, signal=sig,
        )
        assert cs.signal is sig
        assert cs.to_dict()["signal"] is not None

    def test_to_dict(self):
        cs = CircuitState.from_step_result(
            circuit_name="NM", intact_io_gap=4.2, current_io_gap=1.0,
            gap_gate=True, geo_gate=False, gate_reason="gap_threshold",
            corrected=True, cluster="suppressed", cluster_score=0.88,
        )
        d = cs.to_dict()
        assert d["circuit_name"] == "NM"
        assert d["health"] == "RECOVERING"
        assert d["cluster"] == "suppressed"

    def test_repr(self):
        cs = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.0, current_io_gap=4.0,
            gap_gate=True, geo_gate=True, gate_reason="gap",
            corrected=False,
        )
        assert "IOI" in repr(cs)


class TestPythiaParser:
    def test_intact_ioi(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 4.8})
        assert e.health_of("IOI") == Health.INTACT

    def test_ablated_ioi(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 0.5})
        assert e.health_of("IOI") == Health.ABLATED

    def test_recovering_with_correction(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 1.0}, correction_magnitude=0.5, alpha=0.6)
        assert e.health_of("IOI") == Health.RECOVERING

    def test_multi_circuit(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 4.8, "NM": 1.3, "IH": 2.5, "SH": 0.3})
        assert e.health_of("IOI") == Health.INTACT
        assert e.health_of("NM") == Health.ABLATED    # 1.3 < 1.5 (ablated threshold)
        assert e.health_of("IH") == Health.DEGRADED   # 1.5 <= 2.5 < 3.5
        assert e.health_of("SH") == Health.DEGRADED   # 0.05 <= 0.3 < 0.40 (SH thresholds)

    def test_sh_custom_thresholds(self):
        p = make_pythia_parser()
        # SH at 0.45 should be INTACT (SH intact threshold is 0.40)
        e = p.parse({"SH": 0.45})
        assert e.health_of("SH") == Health.INTACT
        # SH at 0.03 should be ABLATED (SH ablated threshold is 0.05)
        e2 = p.parse({"SH": 0.03})
        assert e2.health_of("SH") == Health.ABLATED

    def test_correction_targets_worst(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 4.8, "NM": 0.5}, correction_magnitude=0.5, alpha=0.6)
        assert e.corrections[0].target == "NM"
        assert e.corrections[0].op == Op.RESTORE

    def test_expression_string(self):
        p = make_pythia_parser()
        e = p.parse({"IOI": 4.8, "NM": 1.3}, correction_magnitude=0.5, alpha=0.6)
        s = e.to_string()
        assert "IOI" in s
        assert "NM" in s


class TestMakeFromSweep:
    def test_basic(self):
        p = make_from_sweep({
            "IOI": {"baseline_gap": 3.5, "layer": 23},
            "NM": {"baseline_gap": 2.9, "layer": 21},
        })
        e = p.parse({"IOI": 3.0, "NM": 2.5})
        assert e.health_of("IOI") is not None
        assert e.health_of("NM") is not None

    def test_auto_thresholds(self):
        p = make_from_sweep({
            "IOI": {"baseline_gap": 4.0, "layer": 26},
        })
        # intact = 0.70 * 4.0 = 2.8, ablated = 0.30 * 4.0 = 1.2
        e = p.parse({"IOI": 3.0})
        assert e.health_of("IOI") == Health.INTACT  # 3.0 >= 2.8

        e2 = p.parse({"IOI": 1.0})
        assert e2.health_of("IOI") == Health.ABLATED  # 1.0 < 1.2

    def test_custom_thresholds(self):
        p = make_from_sweep(
            {"IOI": {"baseline_gap": 3.5, "layer": 23}},
            thresholds=Thresholds(intact=2.5, ablated=1.0),
        )
        e = p.parse({"IOI": 2.6})
        assert e.health_of("IOI") == Health.INTACT


class TestIntegration:
    """End-to-end: CircuitState → Observation → Ledger → render."""

    def test_full_loop(self):
        parser = make_pythia_parser()
        ledger = Ledger(label="pythia-6.9b-test")

        # Step 0: intact
        before_0 = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.1, current_io_gap=4.8,
            gap_gate=False, geo_gate=False, gate_reason="neither",
            corrected=False,
        )
        ledger.append(Record(
            step=0, tag=" Mary",
            before=before_0.to_observation(),
            fired=False,
        ))

        # Step 1: degraded, correction applied
        before_1 = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.1, current_io_gap=1.1,
            gap_gate=True, geo_gate=True, gate_reason="gap_threshold",
            corrected=True,
        )
        after_1 = CircuitState.from_step_result(
            circuit_name="IOI", intact_io_gap=5.1, current_io_gap=3.7,
            gap_gate=True, geo_gate=True, gate_reason="gap_threshold",
            corrected=True,
        )
        ledger.append(Record(
            step=1, tag=" went",
            before=before_1.to_observation(),
            after=after_1.to_observation(),
            fired=True, op=Op.RESTORE, alpha=0.6, magnitude=0.55,
        ))

        assert len(ledger) == 2
        assert ledger.n_fired == 1
        assert ledger.records[1].improvement > 0
        assert ledger.records[1].was_beneficial()

        rendered = ledger.render()
        assert "step   0" in rendered
        assert "step   1" in rendered
        assert "RESTORE" in rendered

        # Roundtrip
        ledger2 = Ledger.from_json(ledger.to_json())
        assert len(ledger2) == 2
        assert ledger2.render() == rendered
