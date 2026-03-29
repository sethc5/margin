# Healthcare Adapter for Margin

**Clinical vital signs and lab results as typed health observations.**

Maps standard clinical measurements into margin's typed vocabulary.
Every vital sign gets a health classification (INTACT / DEGRADED / ABLATED),
sigma normalization for cross-vital comparison, and clinical contracts
for monitoring requirements.

**NOT A MEDICAL DEVICE.** This is a typed data vocabulary for clinical
measurements that already have established thresholds. Clinical decisions
require licensed practitioners.

## Quick example

```python
from adapters.healthcare import parse_vitals, patient_expression

# Raw bedside readings
readings = {"hr": 88, "sbp": 145, "spo2": 93, "temp": 38.5, "rr": 22}

expr = patient_expression(readings, patient_id="bed-4")
print(expr.to_string())
# [hr:INTACT(+0.22σ)] [sbp:DEGRADED(-0.26σ)] [spo2:DEGRADED(-0.05σ)]
# [temp:DEGRADED(-0.05σ)] [rr:DEGRADED(+0.38σ)]
```

All five vitals on one normalized scale. Sigma is always positive = healthier,
regardless of whether the vital is too high or too low.

## Band thresholds

Clinical vitals have **bands** — both too high and too low are unhealthy.
Margin handles one polarity per threshold, so each vital is classified
against both boundaries and the worse result wins:

```python
from adapters.healthcare import classify_band, VITAL_SIGNS

# Heart rate 45 bpm — below normal_low (60)
health = classify_band(45, VITAL_SIGNS["hr"].band)
# Health.ABLATED (below critical_low=40? No, 45 > 40 → DEGRADED)

# Temperature 39.5°C — above normal_high (37.2)
health = classify_band(39.5, VITAL_SIGNS["temp"].band)
# Health.DEGRADED (above normal but below critical_high=40.0)
```

## Standard ranges

| Vital | Normal range | Critical low | Critical high | Unit |
|---|---|---|---|---|
| Heart Rate | 60–100 | 40 | 150 | bpm |
| Systolic BP | 90–120 | 70 | 180 | mmHg |
| Diastolic BP | 60–80 | 40 | 120 | mmHg |
| SpO2 | 95–100 | 90 | 100 | % |
| Temperature | 36.1–37.2 | 35.0 | 40.0 | °C |
| Respiratory Rate | 12–20 | 8 | 30 | /min |
| Blood Glucose | 70–100 | 54 | 250 | mg/dL |
| MAP | 70–100 | 60 | 110 | mmHg |

Override with custom `VitalSign` definitions for pediatric, geriatric,
or condition-specific ranges.

## Clinical contracts

Pre-built monitoring requirements:

```python
from adapters.healthcare import standard_monitoring_contract, icu_contract
from margin import Ledger

contract = icu_contract(patient_id="bed-4")
result = contract.evaluate(ledger, current_expression)
print(result.to_string())
# Contract(icu-monitoring:bed-4):
#   [+] hr-intact: hr:INTACT vs target INTACT
#   [!] spo2-intact: spo2:DEGRADED vs target INTACT
#   [?] vitals-stable: 3/6 steps
#   [+] recovery-adequate: recovery 0.82 >= 0.7
#   [+] no-adverse-interventions: 0 harmful in 12 steps
```

Three contracts included:
- **standard_monitoring** — general ward: no vitals in ABLATED
- **icu_contract** — ICU: all vitals INTACT, sustained for 6 readings
- **sepsis_screening** — qSOFA-inspired early warning flags

## Integration targets

- **GNU Health** — map `gnuhealth.patient.evaluation` vitals to `parse_vitals()`
- **OpenMRS** — map `Obs` concepts (concept IDs for vitals) to `VITAL_SIGNS`
- **OHDSI/OMOP** — map `measurement` table rows to margin Observations
- **HL7 FHIR** — map `Observation` resources to margin Observations
- **OpenICE** — multi-device vital aggregation via `CompositeObservation`
```
