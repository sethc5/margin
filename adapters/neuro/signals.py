"""
Neuroscience signal quality as margin observations.

Each recording modality has sensors with domain-appropriate thresholds.
Polarity matters everywhere:
  - Impedance, artifact rate, noise floor → lower is better
  - SNR, spike rate, classification accuracy → higher is better
  - Frequency band power, electrode offset → band thresholds

Drift detection catches:
  - Electrode impedance creep over hours
  - Reference drift in long recordings
  - Photobleaching in calcium imaging
  - Scanner drift in fMRI

Anomaly detection catches:
  - Bad channels (sudden impedance spike)
  - Motion artifacts (transient signal contamination)
  - Amplifier saturation
  - Cross-talk (via correlation discovery)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from margin.health import Health, Thresholds, classify


@dataclass
class NeuroSignal:
    """Threshold profile for a neuroscience signal metric."""
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


@dataclass
class NeuroSignalProfile:
    """A named set of signal thresholds for a recording modality."""
    name: str
    description: str
    signals: dict[str, NeuroSignal]


# -----------------------------------------------------------------------
# EEG profile (scalp electroencephalography)
# MNE-Python, OpenBCI, BrainFlow compatible
# -----------------------------------------------------------------------

_EEG_SIGNALS: dict[str, NeuroSignal] = {
    "impedance": NeuroSignal(
        name="impedance", display_name="Electrode Impedance",
        thresholds=Thresholds(intact=10.0, ablated=50.0, higher_is_better=False),
        baseline=5.0, unit="kΩ",
    ),
    "snr": NeuroSignal(
        name="snr", display_name="Signal-to-Noise Ratio",
        thresholds=Thresholds(intact=10.0, ablated=3.0, higher_is_better=True),
        baseline=20.0, unit="dB",
    ),
    "line_noise": NeuroSignal(
        name="line_noise", display_name="Line Noise Power (50/60 Hz)",
        thresholds=Thresholds(intact=5.0, ablated=20.0, higher_is_better=False),
        baseline=2.0, unit="µV²/Hz",
    ),
    "artifact_rate": NeuroSignal(
        name="artifact_rate", display_name="Artifact Rate",
        thresholds=Thresholds(intact=0.05, ablated=0.20, higher_is_better=False),
        baseline=0.02, unit="ratio",
    ),
    "dc_offset": NeuroSignal(
        name="dc_offset", display_name="DC Offset",
        thresholds=Thresholds(intact=50.0, ablated=200.0, higher_is_better=False),
        baseline=10.0, unit="µV",
    ),
    "alpha_power": NeuroSignal(
        name="alpha_power", display_name="Alpha Band Power (8-13 Hz)",
        thresholds=Thresholds(intact=5.0, ablated=1.0, higher_is_better=True),
        baseline=15.0, unit="µV²/Hz",
    ),
    "flatline_fraction": NeuroSignal(
        name="flatline_fraction", display_name="Flatline Fraction",
        thresholds=Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "channel_correlation": NeuroSignal(
        name="channel_correlation", display_name="Mean Neighbor Correlation",
        thresholds=Thresholds(intact=0.5, ablated=0.15, higher_is_better=True),
        baseline=0.7, unit="r",
    ),
    "saturation_rate": NeuroSignal(
        name="saturation_rate", display_name="Amplifier Saturation Rate",
        thresholds=Thresholds(intact=0.001, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
}

# -----------------------------------------------------------------------
# EMG profile (electromyography)
# -----------------------------------------------------------------------

_EMG_SIGNALS: dict[str, NeuroSignal] = {
    "impedance": NeuroSignal(
        name="impedance", display_name="Electrode Impedance",
        thresholds=Thresholds(intact=15.0, ablated=50.0, higher_is_better=False),
        baseline=5.0, unit="kΩ",
    ),
    "snr": NeuroSignal(
        name="snr", display_name="Signal-to-Noise Ratio",
        thresholds=Thresholds(intact=15.0, ablated=5.0, higher_is_better=True),
        baseline=25.0, unit="dB",
    ),
    "baseline_noise": NeuroSignal(
        name="baseline_noise", display_name="Baseline Noise RMS",
        thresholds=Thresholds(intact=10.0, ablated=30.0, higher_is_better=False),
        baseline=5.0, unit="µV",
    ),
    "cross_talk": NeuroSignal(
        name="cross_talk", display_name="Cross-Talk Ratio",
        thresholds=Thresholds(intact=0.05, ablated=0.15, higher_is_better=False),
        baseline=0.02, unit="ratio",
    ),
    "motion_artifact_rate": NeuroSignal(
        name="motion_artifact_rate", display_name="Motion Artifact Rate",
        thresholds=Thresholds(intact=0.03, ablated=0.15, higher_is_better=False),
        baseline=0.01, unit="ratio",
    ),
    "rms_amplitude": NeuroSignal(
        name="rms_amplitude", display_name="RMS Amplitude (active)",
        thresholds=Thresholds(intact=50.0, ablated=10.0, higher_is_better=True),
        baseline=200.0, unit="µV",
    ),
    "median_frequency": NeuroSignal(
        name="median_frequency", display_name="Median Frequency",
        thresholds=Thresholds(intact=40.0, ablated=20.0, higher_is_better=True),
        baseline=80.0, unit="Hz",
    ),
}

# -----------------------------------------------------------------------
# BCI profile (brain-computer interface, real-time)
# BrainFlow, OpenBCI compatible
# -----------------------------------------------------------------------

_BCI_SIGNALS: dict[str, NeuroSignal] = {
    "impedance": NeuroSignal(
        name="impedance", display_name="Electrode Impedance",
        thresholds=Thresholds(intact=10.0, ablated=40.0, higher_is_better=False),
        baseline=5.0, unit="kΩ",
    ),
    "classification_accuracy": NeuroSignal(
        name="classification_accuracy", display_name="Online Classification Accuracy",
        thresholds=Thresholds(intact=0.70, ablated=0.40, higher_is_better=True),
        baseline=0.85, unit="ratio",
    ),
    "latency": NeuroSignal(
        name="latency", display_name="Processing Latency",
        thresholds=Thresholds(intact=50.0, ablated=200.0, higher_is_better=False),
        baseline=20.0, unit="ms",
    ),
    "feature_snr": NeuroSignal(
        name="feature_snr", display_name="Feature SNR",
        thresholds=Thresholds(intact=3.0, ablated=1.0, higher_is_better=True),
        baseline=5.0, unit="ratio",
    ),
    "artifact_rejection_rate": NeuroSignal(
        name="artifact_rejection_rate", display_name="Trial Rejection Rate",
        thresholds=Thresholds(intact=0.10, ablated=0.40, higher_is_better=False),
        baseline=0.05, unit="ratio",
    ),
    "channel_dropout": NeuroSignal(
        name="channel_dropout", display_name="Channel Dropout Rate",
        thresholds=Thresholds(intact=0.0, ablated=0.10, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
    "mu_power": NeuroSignal(
        name="mu_power", display_name="Mu Rhythm Power (8-12 Hz)",
        thresholds=Thresholds(intact=3.0, ablated=0.5, higher_is_better=True),
        baseline=8.0, unit="µV²/Hz",
    ),
    "sample_loss_rate": NeuroSignal(
        name="sample_loss_rate", display_name="Sample Loss Rate",
        thresholds=Thresholds(intact=0.001, ablated=0.01, higher_is_better=False),
        baseline=0.0, unit="ratio",
    ),
}

# -----------------------------------------------------------------------
# Spike sorting profile (extracellular electrophysiology)
# SpikeInterface, Neo compatible
# -----------------------------------------------------------------------

_SPIKE_SIGNALS: dict[str, NeuroSignal] = {
    "electrode_impedance": NeuroSignal(
        name="electrode_impedance", display_name="Electrode Impedance",
        thresholds=Thresholds(intact=1000.0, ablated=3000.0, higher_is_better=False),
        baseline=500.0, unit="kΩ",
    ),
    "noise_floor": NeuroSignal(
        name="noise_floor", display_name="Noise Floor (RMS)",
        thresholds=Thresholds(intact=10.0, ablated=25.0, higher_is_better=False),
        baseline=5.0, unit="µV",
    ),
    "spike_snr": NeuroSignal(
        name="spike_snr", display_name="Spike SNR",
        thresholds=Thresholds(intact=5.0, ablated=2.0, higher_is_better=True),
        baseline=10.0, unit="ratio",
    ),
    "firing_rate": NeuroSignal(
        name="firing_rate", display_name="Mean Firing Rate",
        thresholds=Thresholds(intact=1.0, ablated=0.1, higher_is_better=True),
        baseline=10.0, unit="Hz",
    ),
    "isolation_score": NeuroSignal(
        name="isolation_score", display_name="Unit Isolation Score",
        thresholds=Thresholds(intact=0.90, ablated=0.70, higher_is_better=True),
        baseline=0.95, unit="score",
    ),
    "isi_violation_rate": NeuroSignal(
        name="isi_violation_rate", display_name="ISI Violation Rate",
        thresholds=Thresholds(intact=0.01, ablated=0.05, higher_is_better=False),
        baseline=0.005, unit="ratio",
    ),
    "drift_um": NeuroSignal(
        name="drift_um", display_name="Electrode Drift",
        thresholds=Thresholds(intact=20.0, ablated=60.0, higher_is_better=False),
        baseline=5.0, unit="µm",
    ),
    "n_units": NeuroSignal(
        name="n_units", display_name="Sorted Units",
        thresholds=Thresholds(intact=3.0, ablated=1.0, higher_is_better=True),
        baseline=8.0, unit="count",
    ),
}

# -----------------------------------------------------------------------
# Calcium imaging profile
# CaImAn, Suite2p compatible
# -----------------------------------------------------------------------

_CALCIUM_SIGNALS: dict[str, NeuroSignal] = {
    "snr": NeuroSignal(
        name="snr", display_name="Fluorescence SNR",
        thresholds=Thresholds(intact=5.0, ablated=2.0, higher_is_better=True),
        baseline=10.0, unit="ratio",
    ),
    "photobleaching_rate": NeuroSignal(
        name="photobleaching_rate", display_name="Photobleaching Rate",
        thresholds=Thresholds(intact=0.005, ablated=0.02, higher_is_better=False),
        baseline=0.002, unit="%/frame",
    ),
    "motion_correction_shift": NeuroSignal(
        name="motion_correction_shift", display_name="Motion Correction Shift",
        thresholds=Thresholds(intact=3.0, ablated=10.0, higher_is_better=False),
        baseline=1.0, unit="pixels",
    ),
    "baseline_fluorescence": NeuroSignal(
        name="baseline_fluorescence", display_name="Baseline Fluorescence (F0)",
        thresholds=Thresholds(intact=500.0, ablated=100.0, higher_is_better=True),
        baseline=1000.0, unit="AU",
    ),
    "active_cell_fraction": NeuroSignal(
        name="active_cell_fraction", display_name="Active Cell Fraction",
        thresholds=Thresholds(intact=0.10, ablated=0.02, higher_is_better=True),
        baseline=0.25, unit="ratio",
    ),
    "neuropil_contamination": NeuroSignal(
        name="neuropil_contamination", display_name="Neuropil Contamination",
        thresholds=Thresholds(intact=0.3, ablated=0.6, higher_is_better=False),
        baseline=0.15, unit="ratio",
    ),
    "frame_rate_stability": NeuroSignal(
        name="frame_rate_stability", display_name="Frame Rate Stability",
        thresholds=Thresholds(intact=0.98, ablated=0.90, higher_is_better=True),
        baseline=1.0, unit="ratio",
    ),
    "z_drift": NeuroSignal(
        name="z_drift", display_name="Z-Plane Drift",
        thresholds=Thresholds(intact=2.0, ablated=5.0, higher_is_better=False),
        baseline=0.5, unit="µm",
    ),
}

# -----------------------------------------------------------------------
# fMRI profile (functional magnetic resonance imaging)
# Nilearn, fMRIPrep compatible
# -----------------------------------------------------------------------

_FMRI_SIGNALS: dict[str, NeuroSignal] = {
    "tsnr": NeuroSignal(
        name="tsnr", display_name="Temporal SNR",
        thresholds=Thresholds(intact=50.0, ablated=20.0, higher_is_better=True),
        baseline=80.0, unit="ratio",
    ),
    "framewise_displacement": NeuroSignal(
        name="framewise_displacement", display_name="Framewise Displacement",
        thresholds=Thresholds(intact=0.3, ablated=0.8, higher_is_better=False),
        baseline=0.1, unit="mm",
    ),
    "dvars": NeuroSignal(
        name="dvars", display_name="DVARS (Standardized)",
        thresholds=Thresholds(intact=1.3, ablated=2.0, higher_is_better=False),
        baseline=1.0, unit="std",
    ),
    "global_signal_std": NeuroSignal(
        name="global_signal_std", display_name="Global Signal Std",
        thresholds=Thresholds(intact=2.0, ablated=5.0, higher_is_better=False),
        baseline=1.0, unit="%",
    ),
    "outlier_fraction": NeuroSignal(
        name="outlier_fraction", display_name="Outlier Volume Fraction",
        thresholds=Thresholds(intact=0.05, ablated=0.15, higher_is_better=False),
        baseline=0.02, unit="ratio",
    ),
    "efc": NeuroSignal(
        name="efc", display_name="Entropy Focus Criterion",
        thresholds=Thresholds(intact=0.5, ablated=0.8, higher_is_better=False),
        baseline=0.35, unit="ratio",
    ),
    "coregistration_error": NeuroSignal(
        name="coregistration_error", display_name="Coregistration Error",
        thresholds=Thresholds(intact=2.0, ablated=5.0, higher_is_better=False),
        baseline=1.0, unit="mm",
    ),
    "ghost_to_signal": NeuroSignal(
        name="ghost_to_signal", display_name="Ghost-to-Signal Ratio",
        thresholds=Thresholds(intact=0.05, ablated=0.15, higher_is_better=False),
        baseline=0.02, unit="ratio",
    ),
}

# -----------------------------------------------------------------------
# Profile registry
# -----------------------------------------------------------------------

NEURO_PROFILES: dict[str, NeuroSignalProfile] = {
    "eeg": NeuroSignalProfile("eeg", "Scalp EEG (MNE-Python, OpenBCI, BrainFlow)", _EEG_SIGNALS),
    "emg": NeuroSignalProfile("emg", "Electromyography", _EMG_SIGNALS),
    "bci": NeuroSignalProfile("bci", "Brain-computer interface (real-time)", _BCI_SIGNALS),
    "spike_sorting": NeuroSignalProfile("spike_sorting", "Extracellular spike sorting (SpikeInterface)", _SPIKE_SIGNALS),
    "calcium_imaging": NeuroSignalProfile("calcium_imaging", "Calcium imaging (CaImAn, Suite2p)", _CALCIUM_SIGNALS),
    "fmri": NeuroSignalProfile("fmri", "Functional MRI (Nilearn, fMRIPrep)", _FMRI_SIGNALS),
}


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def parse_neuro(
    readings: dict[str, float],
    profile: str = "eeg",
    confidence: Confidence = Confidence.MODERATE,
    signals: Optional[dict[str, NeuroSignal]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse neuroscience signal readings into margin Observations.

    Args:
        readings:    {"impedance": 8.0, "snr": 15.0, "artifact_rate": 0.03, ...}
        profile:     "eeg", "emg", "bci", "spike_sorting", "calcium_imaging", "fmri"
        confidence:  measurement confidence
        signals:     override signal definitions
        measured_at: timestamp
    """
    if signals is None:
        p = NEURO_PROFILES.get(profile)
        if p is None:
            p = NEURO_PROFILES["eeg"]
        signals = p.signals

    observations = {}
    for name, value in readings.items():
        signal = signals.get(name)
        if signal is None:
            continue
        health = classify(value, confidence, thresholds=signal.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value,
            baseline=signal.baseline,
            confidence=confidence,
            higher_is_better=signal.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def neuro_expression(
    readings: dict[str, float],
    profile: str = "eeg",
    session_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    signals: Optional[dict[str, NeuroSignal]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a session-wide signal quality Expression."""
    obs = parse_neuro(readings, profile, confidence, signals, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=session_id,
    )
