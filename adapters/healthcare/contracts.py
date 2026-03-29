"""
Clinical monitoring contracts: typed success criteria for patient care.

Standard clinical monitoring requirements expressed as margin contracts.
Each contract defines what "safe" looks like and evaluates the ledger
against it.

NOT A MEDICAL DEVICE. These encode standard clinical thresholds as
typed data. Clinical decisions require licensed practitioners.
"""

from __future__ import annotations

from margin.health import Health
from margin.contract import (
    Contract, HealthTarget, SustainHealth,
    RecoveryThreshold, NoHarmful,
)


def standard_monitoring_contract(patient_id: str = "") -> Contract:
    """
    General ward monitoring: all vitals should be INTACT or DEGRADED.
    No vital in ABLATED for more than 2 consecutive readings.
    """
    return Contract(
        name=f"standard-monitoring{':' + patient_id if patient_id else ''}",
        terms=[
            HealthTarget("hr-safe", "hr", Health.DEGRADED, or_better=True),
            HealthTarget("sbp-safe", "sbp", Health.DEGRADED, or_better=True),
            HealthTarget("dbp-safe", "dbp", Health.DEGRADED, or_better=True),
            HealthTarget("spo2-safe", "spo2", Health.DEGRADED, or_better=True),
            HealthTarget("temp-safe", "temp", Health.DEGRADED, or_better=True),
            HealthTarget("rr-safe", "rr", Health.DEGRADED, or_better=True),
        ],
    )


def icu_contract(patient_id: str = "") -> Contract:
    """
    ICU-level monitoring: all vitals INTACT, sustained for 6 readings.
    SpO2 must not drop below DEGRADED. Mean recovery above 0.7.
    """
    return Contract(
        name=f"icu-monitoring{':' + patient_id if patient_id else ''}",
        terms=[
            HealthTarget("hr-intact", "hr", Health.INTACT),
            HealthTarget("sbp-intact", "sbp", Health.INTACT),
            HealthTarget("spo2-intact", "spo2", Health.INTACT),
            HealthTarget("rr-intact", "rr", Health.INTACT),
            HealthTarget("temp-intact", "temp", Health.INTACT),
            SustainHealth("vitals-stable", "hr", Health.INTACT, for_steps=6),
            SustainHealth("spo2-stable", "spo2", Health.INTACT, for_steps=6),
            RecoveryThreshold("recovery-adequate", min_recovery=0.7, over_steps=12),
            NoHarmful("no-adverse-interventions", over_steps=12),
        ],
    )


def sepsis_screening_contract(patient_id: str = "") -> Contract:
    """
    Sepsis screening criteria (qSOFA-inspired).

    Flags when:
    - Respiratory rate is elevated (DEGRADED or worse)
    - Systolic BP is low (DEGRADED or worse)
    - Mental status change (modeled via heart rate variability / temp)

    These are early warning thresholds, not diagnostic criteria.
    """
    return Contract(
        name=f"sepsis-screen{':' + patient_id if patient_id else ''}",
        terms=[
            HealthTarget("rr-normal", "rr", Health.INTACT),
            HealthTarget("sbp-normal", "sbp", Health.INTACT),
            HealthTarget("temp-normal", "temp", Health.INTACT),
            HealthTarget("hr-normal", "hr", Health.INTACT),
        ],
    )
