"""Tests for adapters.neuro — neuroscience signal quality profiles."""

from datetime import datetime

from margin import Health, Confidence, Expression
from adapters.neuro import (
    NEURO_PROFILES, NeuroSignalProfile, NeuroSignal,
    parse_neuro, neuro_expression,
)


class TestProfiles:
    def test_all_profiles_exist(self):
        assert "eeg" in NEURO_PROFILES
        assert "emg" in NEURO_PROFILES
        assert "bci" in NEURO_PROFILES
        assert "spike_sorting" in NEURO_PROFILES
        assert "calcium_imaging" in NEURO_PROFILES
        assert "fmri" in NEURO_PROFILES

    def test_profiles_have_signals(self):
        for name, profile in NEURO_PROFILES.items():
            assert len(profile.signals) >= 7, f"{name} has too few signals"

    def test_eeg_polarity(self):
        eeg = NEURO_PROFILES["eeg"].signals
        assert eeg["impedance"].thresholds.higher_is_better is False
        assert eeg["snr"].thresholds.higher_is_better is True
        assert eeg["line_noise"].thresholds.higher_is_better is False
        assert eeg["artifact_rate"].thresholds.higher_is_better is False
        assert eeg["alpha_power"].thresholds.higher_is_better is True
        assert eeg["channel_correlation"].thresholds.higher_is_better is True

    def test_bci_polarity(self):
        bci = NEURO_PROFILES["bci"].signals
        assert bci["classification_accuracy"].thresholds.higher_is_better is True
        assert bci["latency"].thresholds.higher_is_better is False
        assert bci["channel_dropout"].thresholds.higher_is_better is False

    def test_spike_polarity(self):
        sp = NEURO_PROFILES["spike_sorting"].signals
        assert sp["noise_floor"].thresholds.higher_is_better is False
        assert sp["spike_snr"].thresholds.higher_is_better is True
        assert sp["isolation_score"].thresholds.higher_is_better is True
        assert sp["isi_violation_rate"].thresholds.higher_is_better is False

    def test_calcium_polarity(self):
        ca = NEURO_PROFILES["calcium_imaging"].signals
        assert ca["photobleaching_rate"].thresholds.higher_is_better is False
        assert ca["snr"].thresholds.higher_is_better is True
        assert ca["neuropil_contamination"].thresholds.higher_is_better is False

    def test_fmri_polarity(self):
        fm = NEURO_PROFILES["fmri"].signals
        assert fm["tsnr"].thresholds.higher_is_better is True
        assert fm["framewise_displacement"].thresholds.higher_is_better is False
        assert fm["dvars"].thresholds.higher_is_better is False
        assert fm["ghost_to_signal"].thresholds.higher_is_better is False


class TestParseNeuro:
    def test_healthy_eeg(self):
        readings = {"impedance": 5.0, "snr": 18.0, "artifact_rate": 0.02, "line_noise": 2.0}
        obs = parse_neuro(readings, profile="eeg")
        assert len(obs) == 4
        assert all(o.health == Health.INTACT for o in obs.values())

    def test_high_impedance(self):
        obs = parse_neuro({"impedance": 55.0}, profile="eeg")
        assert obs["impedance"].health == Health.ABLATED

    def test_low_snr(self):
        obs = parse_neuro({"snr": 2.0}, profile="eeg")
        assert obs["snr"].health == Health.ABLATED

    def test_bci_low_accuracy(self):
        obs = parse_neuro({"classification_accuracy": 0.35}, profile="bci")
        assert obs["classification_accuracy"].health == Health.ABLATED

    def test_bci_high_latency(self):
        obs = parse_neuro({"latency": 250.0}, profile="bci")
        assert obs["latency"].health == Health.ABLATED

    def test_spike_good_isolation(self):
        obs = parse_neuro({"isolation_score": 0.95, "spike_snr": 12.0}, profile="spike_sorting")
        assert obs["isolation_score"].health == Health.INTACT
        assert obs["spike_snr"].health == Health.INTACT

    def test_spike_bad_isi(self):
        obs = parse_neuro({"isi_violation_rate": 0.06}, profile="spike_sorting")
        assert obs["isi_violation_rate"].health == Health.ABLATED

    def test_calcium_photobleaching(self):
        obs = parse_neuro({"photobleaching_rate": 0.03}, profile="calcium_imaging")
        assert obs["photobleaching_rate"].health == Health.ABLATED

    def test_fmri_motion(self):
        obs = parse_neuro({"framewise_displacement": 1.0}, profile="fmri")
        assert obs["framewise_displacement"].health == Health.ABLATED

    def test_fmri_good_tsnr(self):
        obs = parse_neuro({"tsnr": 75.0}, profile="fmri")
        assert obs["tsnr"].health == Health.INTACT

    def test_unknown_signal_ignored(self):
        obs = parse_neuro({"impedance": 5.0, "unknown": 42.0}, profile="eeg")
        assert "impedance" in obs
        assert "unknown" not in obs

    def test_unknown_profile_falls_back(self):
        obs = parse_neuro({"impedance": 5.0}, profile="nonexistent")
        assert len(obs) == 1

    def test_timestamp(self):
        t = datetime(2026, 1, 1)
        obs = parse_neuro({"impedance": 5.0}, measured_at=t)
        assert obs["impedance"].measured_at == t


class TestNeuroExpression:
    def test_expression(self):
        readings = {"impedance": 5.0, "snr": 18.0, "artifact_rate": 0.02}
        expr = neuro_expression(readings, profile="eeg", session_id="ses-01")
        assert isinstance(expr, Expression)
        assert expr.label == "ses-01"
        assert len(expr.observations) == 3

    def test_expression_degraded(self):
        readings = {"impedance": 55.0, "snr": 2.0}
        expr = neuro_expression(readings, profile="eeg")
        assert len(expr.degraded()) == 2

    def test_expression_to_string(self):
        readings = {"impedance": 5.0, "snr": 18.0}
        expr = neuro_expression(readings, profile="eeg")
        s = expr.to_string()
        assert "impedance" in s
        assert "snr" in s

    def test_with_monitor(self):
        """Verify neuro signals work with streaming Monitor."""
        from margin import Monitor, Parser
        from adapters.neuro.signals import _EEG_SIGNALS

        baselines = {s.name: s.baseline for s in _EEG_SIGNALS.values()}
        component_thresholds = {s.name: s.thresholds for s in _EEG_SIGNALS.values()}
        first = next(iter(_EEG_SIGNALS.values()))
        parser = Parser(baselines=baselines, thresholds=first.thresholds,
                        component_thresholds=component_thresholds)
        monitor = Monitor(parser, window=20)

        for i in range(10):
            monitor.update({
                "impedance": 5.0 + i * 0.5,
                "snr": 18.0 - i * 0.3,
                "artifact_rate": 0.02 + i * 0.005,
            })
        assert monitor.expression is not None
        assert monitor.step == 10
