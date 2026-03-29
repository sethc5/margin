"""
Neuroscience recording adapter for margin.

Signal quality monitoring for EEG, EMG, BCI, spike sorting, and calcium imaging.
Detects electrode drift, bad channels, motion artifacts, and signal degradation.

Compatible with MNE-Python, BrainFlow, OpenBCI, SpikeInterface, CaImAn.

Profiles: eeg, emg, bci, spike_sorting, calcium_imaging, fmri.
"""
from .signals import (
    NEURO_PROFILES, NeuroSignalProfile, NeuroSignal,
    parse_neuro, neuro_expression,
)
