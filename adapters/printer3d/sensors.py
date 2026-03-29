"""
3D printer sensors as margin observations.

Sensor profiles for FDM, resin/SLA, and high-speed CoreXY printers.
Thresholds reflect real failure modes:
  - Hotend too cold → under-extrusion, clogs
  - Hotend too hot → heat creep, filament degradation
  - Bed temp drift → warping, adhesion failure
  - Extruder current spike → jam or grinding
  - Layer deviation → print quality failure

Band thresholds (both too-high AND too-low are bad) use lower-is-better
with the "ablated" threshold above the healthy range — margin classifies
deviation in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from margin.observation import Observation, Expression
from margin.confidence import Confidence
from margin.health import Health, Thresholds, classify


@dataclass
class PrinterSensor:
    """Threshold profile for a 3D printer sensor."""
    name: str
    display_name: str
    thresholds: Thresholds
    baseline: float
    unit: str = ""


@dataclass
class PrinterSensorProfile:
    """A named set of sensor thresholds for a printer type."""
    name: str
    description: str
    sensors: dict[str, PrinterSensor]


# -----------------------------------------------------------------------
# FDM profile (standard filament printers: Ender, Prusa, etc.)
# -----------------------------------------------------------------------

_FDM_SENSORS: dict[str, PrinterSensor] = {
    "hotend_temp_error": PrinterSensor(
        name="hotend_temp_error", display_name="Hotend Temp Error (±°C from target)",
        thresholds=Thresholds(intact=3.0, ablated=10.0, higher_is_better=False),
        baseline=0.0, unit="°C",
    ),
    "bed_temp_error": PrinterSensor(
        name="bed_temp_error", display_name="Bed Temp Error (±°C from target)",
        thresholds=Thresholds(intact=2.0, ablated=8.0, higher_is_better=False),
        baseline=0.0, unit="°C",
    ),
    "extruder_current": PrinterSensor(
        name="extruder_current", display_name="Extruder Motor Current",
        thresholds=Thresholds(intact=0.8, ablated=1.5, higher_is_better=False),
        baseline=0.5, unit="A",
    ),
    "layer_height_deviation": PrinterSensor(
        name="layer_height_deviation", display_name="Layer Height Deviation",
        thresholds=Thresholds(intact=0.02, ablated=0.08, higher_is_better=False),
        baseline=0.0, unit="mm",
    ),
    "filament_flow_rate": PrinterSensor(
        name="filament_flow_rate", display_name="Filament Flow Rate",
        thresholds=Thresholds(intact=90.0, ablated=70.0, higher_is_better=True),
        baseline=100.0, unit="%",
    ),
    "part_cooling_fan": PrinterSensor(
        name="part_cooling_fan", display_name="Part Cooling Fan Speed",
        thresholds=Thresholds(intact=80.0, ablated=40.0, higher_is_better=True),
        baseline=100.0, unit="%",
    ),
    "x_axis_vibration": PrinterSensor(
        name="x_axis_vibration", display_name="X-Axis Vibration",
        thresholds=Thresholds(intact=1.5, ablated=5.0, higher_is_better=False),
        baseline=0.5, unit="mm/s²",
    ),
    "y_axis_vibration": PrinterSensor(
        name="y_axis_vibration", display_name="Y-Axis Vibration",
        thresholds=Thresholds(intact=1.5, ablated=5.0, higher_is_better=False),
        baseline=0.5, unit="mm/s²",
    ),
    "print_progress": PrinterSensor(
        name="print_progress", display_name="Print Progress",
        thresholds=Thresholds(intact=1.0, ablated=0.0, higher_is_better=True),
        baseline=0.5, unit="ratio",
    ),
    "ambient_temp": PrinterSensor(
        name="ambient_temp", display_name="Ambient/Enclosure Temperature",
        thresholds=Thresholds(intact=35.0, ablated=45.0, higher_is_better=False),
        baseline=25.0, unit="°C",
    ),
    "z_offset_drift": PrinterSensor(
        name="z_offset_drift", display_name="Z-Offset Drift from Calibration",
        thresholds=Thresholds(intact=0.02, ablated=0.08, higher_is_better=False),
        baseline=0.0, unit="mm",
    ),
    "retraction_count": PrinterSensor(
        name="retraction_count", display_name="Retractions per Minute",
        thresholds=Thresholds(intact=30.0, ablated=80.0, higher_is_better=False),
        baseline=10.0, unit="retracts/min",
    ),
}

# -----------------------------------------------------------------------
# Resin / SLA profile
# -----------------------------------------------------------------------

_RESIN_SENSORS: dict[str, PrinterSensor] = {
    "uv_power": PrinterSensor(
        name="uv_power", display_name="UV LED Power",
        thresholds=Thresholds(intact=90.0, ablated=60.0, higher_is_better=True),
        baseline=100.0, unit="%",
    ),
    "resin_temp": PrinterSensor(
        name="resin_temp", display_name="Resin Temperature",
        thresholds=Thresholds(intact=28.0, ablated=35.0, higher_is_better=False),
        baseline=24.0, unit="°C",
    ),
    "resin_level": PrinterSensor(
        name="resin_level", display_name="Resin Vat Level",
        thresholds=Thresholds(intact=40.0, ablated=15.0, higher_is_better=True),
        baseline=80.0, unit="%",
    ),
    "z_axis_force": PrinterSensor(
        name="z_axis_force", display_name="Z-Axis Peel Force",
        thresholds=Thresholds(intact=15.0, ablated=40.0, higher_is_better=False),
        baseline=8.0, unit="N",
    ),
    "layer_cure_time_error": PrinterSensor(
        name="layer_cure_time_error", display_name="Layer Cure Time Error",
        thresholds=Thresholds(intact=0.5, ablated=2.0, higher_is_better=False),
        baseline=0.0, unit="s",
    ),
    "fep_tension": PrinterSensor(
        name="fep_tension", display_name="FEP Film Tension",
        thresholds=Thresholds(intact=250.0, ablated=150.0, higher_is_better=True),
        baseline=300.0, unit="Hz",
    ),
    "lcd_screen_hours": PrinterSensor(
        name="lcd_screen_hours", display_name="LCD Screen Hours",
        thresholds=Thresholds(intact=1500.0, ablated=2000.0, higher_is_better=False),
        baseline=0.0, unit="hours",
    ),
    "ambient_temp": PrinterSensor(
        name="ambient_temp", display_name="Ambient Temperature",
        thresholds=Thresholds(intact=28.0, ablated=35.0, higher_is_better=False),
        baseline=23.0, unit="°C",
    ),
}

# -----------------------------------------------------------------------
# CoreXY / high-speed FDM profile (Voron, Bambu, RatRig)
# -----------------------------------------------------------------------

_COREXY_SENSORS: dict[str, PrinterSensor] = {
    "hotend_temp_error": PrinterSensor(
        name="hotend_temp_error", display_name="Hotend Temp Error",
        thresholds=Thresholds(intact=2.0, ablated=8.0, higher_is_better=False),
        baseline=0.0, unit="°C",
    ),
    "bed_temp_error": PrinterSensor(
        name="bed_temp_error", display_name="Bed Temp Error",
        thresholds=Thresholds(intact=1.5, ablated=6.0, higher_is_better=False),
        baseline=0.0, unit="°C",
    ),
    "chamber_temp": PrinterSensor(
        name="chamber_temp", display_name="Chamber Temperature",
        thresholds=Thresholds(intact=55.0, ablated=65.0, higher_is_better=False),
        baseline=45.0, unit="°C",
    ),
    "input_shaper_accel": PrinterSensor(
        name="input_shaper_accel", display_name="Input Shaper Max Accel",
        thresholds=Thresholds(intact=5000.0, ablated=2000.0, higher_is_better=True),
        baseline=8000.0, unit="mm/s²",
    ),
    "pressure_advance": PrinterSensor(
        name="pressure_advance", display_name="Pressure Advance Deviation",
        thresholds=Thresholds(intact=0.01, ablated=0.04, higher_is_better=False),
        baseline=0.0, unit="s",
    ),
    "print_speed": PrinterSensor(
        name="print_speed", display_name="Actual Print Speed",
        thresholds=Thresholds(intact=200.0, ablated=80.0, higher_is_better=True),
        baseline=300.0, unit="mm/s",
    ),
    "extruder_current": PrinterSensor(
        name="extruder_current", display_name="Extruder Motor Current",
        thresholds=Thresholds(intact=1.0, ablated=1.8, higher_is_better=False),
        baseline=0.6, unit="A",
    ),
    "belt_tension_a": PrinterSensor(
        name="belt_tension_a", display_name="Belt A Tension",
        thresholds=Thresholds(intact=110.0, ablated=80.0, higher_is_better=True),
        baseline=125.0, unit="Hz",
    ),
    "belt_tension_b": PrinterSensor(
        name="belt_tension_b", display_name="Belt B Tension",
        thresholds=Thresholds(intact=110.0, ablated=80.0, higher_is_better=True),
        baseline=125.0, unit="Hz",
    ),
    "filament_flow_rate": PrinterSensor(
        name="filament_flow_rate", display_name="Filament Flow Rate",
        thresholds=Thresholds(intact=92.0, ablated=75.0, higher_is_better=True),
        baseline=100.0, unit="%",
    ),
    "mcu_temperature": PrinterSensor(
        name="mcu_temperature", display_name="MCU Temperature",
        thresholds=Thresholds(intact=60.0, ablated=80.0, higher_is_better=False),
        baseline=40.0, unit="°C",
    ),
    "stepper_x_temp": PrinterSensor(
        name="stepper_x_temp", display_name="X Stepper Temperature",
        thresholds=Thresholds(intact=55.0, ablated=75.0, higher_is_better=False),
        baseline=35.0, unit="°C",
    ),
    "stepper_y_temp": PrinterSensor(
        name="stepper_y_temp", display_name="Y Stepper Temperature",
        thresholds=Thresholds(intact=55.0, ablated=75.0, higher_is_better=False),
        baseline=35.0, unit="°C",
    ),
}

# -----------------------------------------------------------------------
# Profile registry
# -----------------------------------------------------------------------

PRINTER_PROFILES: dict[str, PrinterSensorProfile] = {
    "fdm": PrinterSensorProfile("fdm", "Standard FDM printers (Ender, Prusa, etc.)", _FDM_SENSORS),
    "resin": PrinterSensorProfile("resin", "Resin / SLA / MSLA printers", _RESIN_SENSORS),
    "corexy": PrinterSensorProfile("corexy", "High-speed CoreXY (Voron, Bambu, RatRig)", _COREXY_SENSORS),
}


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

def parse_printer(
    readings: dict[str, float],
    profile: str = "fdm",
    confidence: Confidence = Confidence.MODERATE,
    sensors: Optional[dict[str, PrinterSensor]] = None,
    measured_at: Optional[datetime] = None,
) -> dict[str, Observation]:
    """
    Parse 3D printer sensor readings into margin Observations.

    Args:
        readings:    {"hotend_temp_error": 1.5, "extruder_current": 0.6, ...}
        profile:     "fdm", "resin", "corexy"
        confidence:  measurement confidence
        sensors:     override sensor definitions
        measured_at: timestamp
    """
    if sensors is None:
        p = PRINTER_PROFILES.get(profile)
        if p is None:
            p = PRINTER_PROFILES["fdm"]
        sensors = p.sensors

    observations = {}
    for name, value in readings.items():
        sensor = sensors.get(name)
        if sensor is None:
            continue
        health = classify(value, confidence, thresholds=sensor.thresholds)
        observations[name] = Observation(
            name=name, health=health, value=value,
            baseline=sensor.baseline,
            confidence=confidence,
            higher_is_better=sensor.thresholds.higher_is_better,
            measured_at=measured_at,
        )
    return observations


def printer_expression(
    readings: dict[str, float],
    profile: str = "fdm",
    printer_id: str = "",
    confidence: Confidence = Confidence.MODERATE,
    sensors: Optional[dict[str, PrinterSensor]] = None,
    measured_at: Optional[datetime] = None,
) -> Expression:
    """Build a printer-wide health Expression."""
    obs = parse_printer(readings, profile, confidence, sensors, measured_at)
    return Expression(
        observations=list(obs.values()),
        confidence=min((o.confidence for o in obs.values()), default=Confidence.INDETERMINATE),
        label=printer_id,
    )
