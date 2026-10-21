"""Medtronic MiniMed pump CareLink export format (720G and its CareLink layout).

A variant of the Guardian Connect CareLink export (`formats/medtronic.py`): the same semicolon-
delimited, comma-decimal file with metadata rows, a "-------" separator line and a header, but from an
insulin pump. Its header is the Guardian one widened from 48 to 54 columns, with new columns inserted
mid-header (so it is declared in full here rather than derived), and `Alarm` renamed `Alert`.

File structure:
- Lines 1-5: patient and export metadata
- Line 6: "-------;<device>;Pump" section separator
- Line 7: column header; lines 8+: the pump section's rows
- Mid-file: "-------;<device>;Sensor", the header again, then the sensor section's rows

The sensor section of a pump worn without a Medtronic sensor holds only `Event Marker` day markers
("Start of the day" / "End of the day"), so a pump export alone carries no glucose; it is joined with
a CGM export by `FormatProcessor.fold_insulin_onto_grid` (docs/PLAN_pump.md).

How the pump section reads (measured on four 720G exports, docs/PLAN_pump.md §2.2):
- `Basal Rate (U/h)` is a rate change, in force until the next. It is the effective rate for an
  absolute temp basal and for a suspend (the pump logs the 0), but the *scheduled* rate during a
  `Percent` temp basal, whose percentage the parser applies.
- Each bolus is logged twice under one `Bolus Number` and `Bolus Type`: when programmed (`Bolus Volume
  Selected (U)` only) and when finished (`Bolus Volume Delivered (U)` too).
- `Suspend` carries the state the pump entered; `NORMAL_PUMPING` ends a suspension.
"""

from typing import List
import polars as pl
from cgm_format.interface.schema import (
    ColumnSchema,
    CGMSchemaDefinition,
    EnumLiteral,
    regenerate_schema_json as _regenerate,
)
from cgm_format.formats.medtronic import (
    MEDTRONIC_HEADER_LINE,
    MEDTRONIC_DATA_START_LINE,
    MEDTRONIC_METADATA_LINES,
)


# =============================================================================
# File Format Constants
# =============================================================================

MEDTRONIC_PUMP_HEADER_LINE = MEDTRONIC_HEADER_LINE
MEDTRONIC_PUMP_DATA_START_LINE = MEDTRONIC_DATA_START_LINE
MEDTRONIC_PUMP_METADATA_LINES = MEDTRONIC_METADATA_LINES

# A section starts at a line opening with this marker: "-------;<device>;<section>".
MEDTRONIC_SECTION_MARKER = "-------"
MEDTRONIC_PUMP_SECTION = "Pump"
MEDTRONIC_SENSOR_SECTION = "Sensor"

# Present in the pump export's header and absent from the Guardian Connect one, so a pump export is
# told apart before the generic Medtronic pattern (which both carry) can claim it.
MEDTRONIC_PUMP_DETECTION_PATTERNS = [
    "Estimated Reservoir Volume after Fill (U)",
]


# =============================================================================
# Vendor vocabularies the parser branches on
# =============================================================================

class MedtronicBolusType(EnumLiteral):
    """`Bolus Type` values. A square bolus (alone or as a dual's second part) is delivered over
    `Bolus Duration`; a normal one at once."""
    NORMAL = "Normal"
    SQUARE = "Square"
    DUAL_NORMAL_PART = "Dual (normal part)"
    DUAL_SQUARE_PART = "Dual (square part)"


MEDTRONIC_SPREAD_BOLUS_TYPES: tuple[str, ...] = (
    MedtronicBolusType.SQUARE,
    MedtronicBolusType.DUAL_SQUARE_PART,
)


class MedtronicTempBasalType(EnumLiteral):
    """`Temp Basal Type` values. A `Rate` temp is logged as the effective basal rate; a `Percent`
    temp is not, so the parser applies `Temp Basal Amount` percent for `Temp Basal Duration`."""
    RATE = "Rate"
    PERCENT = "Percent"


# The `Suspend` value that ends a suspension. Every other value starts one.
MEDTRONIC_RESUME_STATE = "NORMAL_PUMPING"

# `Event Marker` values in the sensor section that carry nothing but the day boundary.
MEDTRONIC_DAY_MARKERS: tuple[str, ...] = ("Start of the day", "End of the day")


# =============================================================================
# Raw File Column Names
# =============================================================================

class MedtronicPumpColumn(EnumLiteral):
    """Column names in a MiniMed pump CareLink export, in file order."""
    INDEX = "Index"
    DATE = "Date"
    TIME = "Time"
    NEW_DEVICE_TIME = "New Device Time"
    BG_SOURCE = "BG Source"
    BG_READING = "BG Reading (mg/dL)"
    LINKED_BG_METER_ID = "Linked BG Meter ID"
    BASAL_RATE = "Basal Rate (U/h)"
    TEMP_BASAL_AMOUNT = "Temp Basal Amount"
    TEMP_BASAL_TYPE = "Temp Basal Type"
    TEMP_BASAL_DURATION = "Temp Basal Duration (h:mm:ss)"
    BOLUS_TYPE = "Bolus Type"
    BOLUS_VOLUME_SELECTED = "Bolus Volume Selected (U)"
    BOLUS_VOLUME_DELIVERED = "Bolus Volume Delivered (U)"
    BOLUS_DURATION = "Bolus Duration (h:mm:ss)"
    PRIME_TYPE = "Prime Type"
    PRIME_VOLUME_DELIVERED = "Prime Volume Delivered (U)"
    ESTIMATED_RESERVOIR_VOLUME = "Estimated Reservoir Volume after Fill (U)"
    ALERT = "Alert"
    USER_CLEARED_ALERTS = "User Cleared Alerts"
    SUSPEND = "Suspend"
    REWIND = "Rewind"
    BWZ_ESTIMATE = "BWZ Estimate (U)"
    BWZ_TARGET_HIGH_BG = "BWZ Target High BG (mg/dL)"
    BWZ_TARGET_LOW_BG = "BWZ Target Low BG (mg/dL)"
    BWZ_CARB_RATIO = "BWZ Carb Ratio (g/U)"
    BWZ_INSULIN_SENSITIVITY = "BWZ Insulin Sensitivity (mg/dL/U)"
    BWZ_CARB_INPUT = "BWZ Carb Input (grams)"
    BWZ_BG_INPUT = "BWZ BG Input (mg/dL)"
    BWZ_CORRECTION_ESTIMATE = "BWZ Correction Estimate (U)"
    BWZ_FOOD_ESTIMATE = "BWZ Food Estimate (U)"
    BWZ_ACTIVE_INSULIN = "BWZ Active Insulin (U)"
    BWZ_STATUS = "BWZ Status"
    SENSOR_CALIBRATION_BG = "Sensor Calibration BG (mg/dL)"
    SENSOR_GLUCOSE = "Sensor Glucose (mg/dL)"
    ISIG_VALUE = "ISIG Value"
    EVENT_MARKER = "Event Marker"
    BOLUS_NUMBER = "Bolus Number"
    BOLUS_CANCELLATION_REASON = "Bolus Cancellation Reason"
    BWZ_UNABSORBED_INSULIN_TOTAL = "BWZ Unabsorbed Insulin Total (U)"
    FINAL_BOLUS_ESTIMATE = "Final Bolus Estimate"
    SCROLL_STEP_SIZE = "Scroll Step Size"
    INSULIN_ACTION_CURVE_TIME = "Insulin Action Curve Time"
    SENSOR_CALIBRATION_REJECTED_REASON = "Sensor Calibration Rejected Reason"
    PRESET_BOLUS = "Preset Bolus"
    BOLUS_SOURCE = "Bolus Source"
    BLE_NETWORK_DEVICE = "BLE Network Device"
    DEVICE_UPDATE_EVENT = "Device Update Event"
    NETWORK_DEVICE_ASSOCIATED_REASON = "Network Device Associated Reason"
    NETWORK_DEVICE_DISASSOCIATED_REASON = "Network Device Disassociated Reason"
    NETWORK_DEVICE_DISCONNECTED_REASON = "Network Device Disconnected Reason"
    SENSOR_EXCEPTION = "Sensor Exception"
    PRESET_TEMP_BASAL_NAME = "Preset Temp Basal Name"
    SENSOR_STATE = "Sensor State"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names in file order."""
        return [member.value for member in cls]


# =============================================================================
# Raw File Format Schema
# =============================================================================

# Every cell is read as text: numbers carry comma decimals and "-------" placeholders, so the parser
# converts the ones it uses itself. Units are declared for documentation and Frictionless export.
_SERVICE = (
    MedtronicPumpColumn.INDEX,
    MedtronicPumpColumn.DATE,
    MedtronicPumpColumn.TIME,
    MedtronicPumpColumn.NEW_DEVICE_TIME,
)
_REQUIRED = (MedtronicPumpColumn.INDEX, MedtronicPumpColumn.DATE, MedtronicPumpColumn.TIME)
_UNITS = {
    MedtronicPumpColumn.BG_READING: "mg/dL",
    MedtronicPumpColumn.BASAL_RATE: "U/h",
    MedtronicPumpColumn.BOLUS_VOLUME_SELECTED: "U",
    MedtronicPumpColumn.BOLUS_VOLUME_DELIVERED: "U",
    MedtronicPumpColumn.PRIME_VOLUME_DELIVERED: "U",
    MedtronicPumpColumn.ESTIMATED_RESERVOIR_VOLUME: "U",
    MedtronicPumpColumn.BWZ_ESTIMATE: "U",
    MedtronicPumpColumn.BWZ_TARGET_HIGH_BG: "mg/dL",
    MedtronicPumpColumn.BWZ_TARGET_LOW_BG: "mg/dL",
    MedtronicPumpColumn.BWZ_CARB_RATIO: "g/U",
    MedtronicPumpColumn.BWZ_INSULIN_SENSITIVITY: "mg/dL/U",
    MedtronicPumpColumn.BWZ_CARB_INPUT: "grams",
    MedtronicPumpColumn.BWZ_BG_INPUT: "mg/dL",
    MedtronicPumpColumn.BWZ_CORRECTION_ESTIMATE: "U",
    MedtronicPumpColumn.BWZ_FOOD_ESTIMATE: "U",
    MedtronicPumpColumn.BWZ_ACTIVE_INSULIN: "U",
    MedtronicPumpColumn.SENSOR_CALIBRATION_BG: "mg/dL",
    MedtronicPumpColumn.SENSOR_GLUCOSE: "mg/dL",
    MedtronicPumpColumn.BWZ_UNABSORBED_INSULIN_TOTAL: "U",
}


def _column(name: MedtronicPumpColumn) -> ColumnSchema:
    col: ColumnSchema = {
        "name": name,
        "dtype": pl.Utf8,
        "description": f"CareLink pump export column {name.value!r}, as text",
        "constraints": {"required": name in _REQUIRED},
    }
    if name in _UNITS:
        col["unit"] = _UNITS[name]
    return col


MEDTRONIC_PUMP_SCHEMA = CGMSchemaDefinition(
    service_columns=tuple(_column(c) for c in _SERVICE),
    data_columns=tuple(_column(c) for c in MedtronicPumpColumn if c not in _SERVICE),
    header_line=MEDTRONIC_PUMP_HEADER_LINE,
    data_start_line=MEDTRONIC_PUMP_DATA_START_LINE,
    metadata_lines=MEDTRONIC_PUMP_METADATA_LINES,
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate medtronic_pump.json from the current schema definition."""
    _regenerate(MEDTRONIC_PUMP_SCHEMA, __file__)
