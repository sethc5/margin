"""
Healthcare adapter for margin.

Maps clinical vital signs and lab results into typed health observations.
WHO/AHA standard ranges. Not a medical device — a typed vocabulary for
clinical data that already has established thresholds.
"""

from .vitals import (
    VitalSign, VITAL_SIGNS,
    parse_vitals, patient_expression,
    BandThresholds, classify_band,
)
from .contracts import (
    standard_monitoring_contract,
    icu_contract,
    sepsis_screening_contract,
)

__all__ = [
    "VitalSign", "VITAL_SIGNS",
    "parse_vitals", "patient_expression",
    "BandThresholds", "classify_band",
    "standard_monitoring_contract",
    "icu_contract",
    "sepsis_screening_contract",
]
