"""
3D printer adapter for margin.

Typed health classification for FDM/SLA/resin printers.
Hotend temperature, bed adhesion, extruder current, layer quality —
all classified with correct polarity and print-phase-aware profiles.

Profiles: fdm, resin, corexy (high-speed FDM).
"""
from .sensors import (
    PRINTER_PROFILES, PrinterSensorProfile, PrinterSensor,
    parse_printer, printer_expression,
)
