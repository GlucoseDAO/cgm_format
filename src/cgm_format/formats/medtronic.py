"""Medtronic Guardian Connect / CareLink CGM export format schema definition.

This module defines the specific schema for Medtronic Guardian Connect CGM data format,
including raw file column definitions and metadata structure.

Medtronic CareLink exports are semicolon-delimited CSV files with European decimal
notation (comma as decimal separator). Numeric columns may contain "-------" placeholder
strings. The file can contain multiple device sections (Pump, Sensor) separated by
"-------" lines, each with its own header row.

File structure:
- Lines 1-5: Patient metadata (name, dates, device info)
- Line 6: "-------" device section separator
- Line 7: Column headers (48 semicolon-separated columns)
- Lines 8+: Data rows
- Mid-file: Additional "-------" separators and repeated header rows for extra device sections

Example metadata:
Last Name;First Name;Patient ID;System ID;Start Date;End Date;Device;Guardian Connect

Example data row (sensor glucose):
2747,00000;2021/07/25;00:32:50;;;;;;;;;;;;;;;;;;;;;;;;;;;;;55;;;;;;;;;;;;;;;;

Example data row (event marker):
0,0;2021/07/25;09:35:28;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Insulin: 27,00;;;;;;;;;;;;;;
"""

from typing import List
import polars as pl
from cgm_format.interface.schema import (
    ColumnSchema,
    CGMSchemaDefinition,
    EnumLiteral,
)


# =============================================================================
# File Format Constants
# =============================================================================

MEDTRONIC_HEADER_LINE = 7
MEDTRONIC_DATA_START_LINE = 8
MEDTRONIC_METADATA_LINES = (1, 2, 3, 4, 5, 6)

MEDTRONIC_CSV_SEPARATOR = ";"

MEDTRONIC_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y/%m/%d %H:%M:%S",   # Primary CareLink format: 2021/07/25 09:35:28
    "%Y-%m-%d %H:%M:%S",   # ISO-like variant
    "%Y-%m-%dT%H:%M:%S",   # ISO 8601 variant
)

MEDTRONIC_REQUIRED_HEADERS: tuple[str, ...] = (
    "Index",
    "Date",
    "Time",
    "Sensor Glucose (mg/dL)",
)

MEDTRONIC_DETECTION_PATTERNS = [
    "Sensor Glucose (mg/dL)",   # Medtronic-specific glucose column name
]

MEDTRONIC_SCHEMA_OVERRIDES_UTF8: tuple[str, ...] = (
    "Sensor Glucose (mg/dL)",
    "BG Reading (mg/dL)",
    "Basal Rate (U/h)",
    "Bolus Volume Delivered (U)",
    "BWZ Carb Input (grams)",
    "Sensor Calibration BG (mg/dL)",
    "Event Marker",
    "Alarm",
)


# =============================================================================
# Raw File Column Names (as they appear in Medtronic CareLink CSV exports)
# =============================================================================

class MedtronicColumn(EnumLiteral):
    """Column names in Medtronic Guardian Connect / CareLink export files."""
    INDEX = "Index"
    DATE = "Date"
    TIME = "Time"
    NEW_DEVICE_TIME = "New Device Time"
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
    ALARM = "Alarm"
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
    NETWORK_DEVICE_ASSOCIATED_REASON = "Network Device Associated Reason"
    NETWORK_DEVICE_DISASSOCIATED_REASON = "Network Device Disassociated Reason"
    NETWORK_DEVICE_DISCONNECTED_REASON = "Network Device Disconnected Reason"
    SENSOR_EXCEPTION = "Sensor Exception"
    PRESET_TEMP_BASAL_NAME = "Preset Temp Basal Name"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names in canonical CSV order."""
        return [
            cls.INDEX, cls.DATE, cls.TIME, cls.NEW_DEVICE_TIME,
            cls.BG_READING, cls.LINKED_BG_METER_ID, cls.BASAL_RATE,
            cls.TEMP_BASAL_AMOUNT, cls.TEMP_BASAL_TYPE, cls.TEMP_BASAL_DURATION,
            cls.BOLUS_TYPE, cls.BOLUS_VOLUME_SELECTED, cls.BOLUS_VOLUME_DELIVERED,
            cls.BOLUS_DURATION, cls.PRIME_TYPE, cls.PRIME_VOLUME_DELIVERED,
            cls.ALARM, cls.SUSPEND, cls.REWIND,
            cls.BWZ_ESTIMATE, cls.BWZ_TARGET_HIGH_BG, cls.BWZ_TARGET_LOW_BG,
            cls.BWZ_CARB_RATIO, cls.BWZ_INSULIN_SENSITIVITY, cls.BWZ_CARB_INPUT,
            cls.BWZ_BG_INPUT, cls.BWZ_CORRECTION_ESTIMATE, cls.BWZ_FOOD_ESTIMATE,
            cls.BWZ_ACTIVE_INSULIN, cls.BWZ_STATUS,
            cls.SENSOR_CALIBRATION_BG, cls.SENSOR_GLUCOSE, cls.ISIG_VALUE,
            cls.EVENT_MARKER, cls.BOLUS_NUMBER, cls.BOLUS_CANCELLATION_REASON,
            cls.BWZ_UNABSORBED_INSULIN_TOTAL, cls.FINAL_BOLUS_ESTIMATE,
            cls.SCROLL_STEP_SIZE, cls.INSULIN_ACTION_CURVE_TIME,
            cls.SENSOR_CALIBRATION_REJECTED_REASON, cls.PRESET_BOLUS,
            cls.BOLUS_SOURCE, cls.NETWORK_DEVICE_ASSOCIATED_REASON,
            cls.NETWORK_DEVICE_DISASSOCIATED_REASON,
            cls.NETWORK_DEVICE_DISCONNECTED_REASON,
            cls.SENSOR_EXCEPTION, cls.PRESET_TEMP_BASAL_NAME,
        ]


# =============================================================================
# Medtronic Raw File Format Schema
# =============================================================================

MEDTRONIC_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": MedtronicColumn.INDEX,
            "dtype": pl.Utf8,
            "description": "Row index (European decimal format, e.g. '0,0', '1,00000')",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.DATE,
            "dtype": pl.Utf8,
            "description": "Date of the event (e.g. '2021/07/25')",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.TIME,
            "dtype": pl.Utf8,
            "description": "Time of the event (e.g. '09:35:28')",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.NEW_DEVICE_TIME,
            "dtype": pl.Utf8,
            "description": "New device time after time change",
            "constraints": {"required": False}
        },
    ),
    data_columns=(
        {
            "name": MedtronicColumn.BG_READING,
            "dtype": pl.Utf8,
            "description": "Fingerstick BG meter reading (may contain '-------' placeholders)",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.LINKED_BG_METER_ID,
            "dtype": pl.Utf8,
            "description": "Linked blood glucose meter identifier",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BASAL_RATE,
            "dtype": pl.Utf8,
            "description": "Programmed basal insulin rate (may contain '-------')",
            "unit": "U/h",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_AMOUNT,
            "dtype": pl.Utf8,
            "description": "Temporary basal rate amount",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_TYPE,
            "dtype": pl.Utf8,
            "description": "Temporary basal type",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_DURATION,
            "dtype": pl.Utf8,
            "description": "Duration of temporary basal in h:mm:ss",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of bolus delivery",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_VOLUME_SELECTED,
            "dtype": pl.Utf8,
            "description": "Selected bolus volume (may contain '-------')",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_VOLUME_DELIVERED,
            "dtype": pl.Utf8,
            "description": "Delivered bolus volume (may contain '-------')",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_DURATION,
            "dtype": pl.Utf8,
            "description": "Duration of extended bolus in h:mm:ss",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRIME_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of infusion set prime",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRIME_VOLUME_DELIVERED,
            "dtype": pl.Utf8,
            "description": "Volume delivered during prime",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.ALARM,
            "dtype": pl.Utf8,
            "description": "Alarm event text (e.g. 'LOW SG', 'CAL NOW', 'RISE RATE')",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SUSPEND,
            "dtype": pl.Utf8,
            "description": "Pump suspend event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.REWIND,
            "dtype": pl.Utf8,
            "description": "Reservoir rewind event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_ESTIMATE,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard estimated dose",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_TARGET_HIGH_BG,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard target high BG",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_TARGET_LOW_BG,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard target low BG",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_CARB_RATIO,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard carb ratio",
            "unit": "g/U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_INSULIN_SENSITIVITY,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard insulin sensitivity factor",
            "unit": "mg/dL/U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_CARB_INPUT,
            "dtype": pl.Utf8,
            "description": "Carbohydrate grams entered into Bolus Wizard (may contain '-------')",
            "unit": "grams",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_BG_INPUT,
            "dtype": pl.Utf8,
            "description": "BG value entered into Bolus Wizard",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_CORRECTION_ESTIMATE,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard correction dose estimate",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_FOOD_ESTIMATE,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard food dose estimate",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_ACTIVE_INSULIN,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard active insulin on board",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_STATUS,
            "dtype": pl.Utf8,
            "description": "Bolus Wizard status",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SENSOR_CALIBRATION_BG,
            "dtype": pl.Utf8,
            "description": "BG value used for sensor calibration (may contain '-------')",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SENSOR_GLUCOSE,
            "dtype": pl.Utf8,
            "description": "CGM sensor glucose reading (may contain '-------')",
            "unit": "mg/dL",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.ISIG_VALUE,
            "dtype": pl.Utf8,
            "description": "Raw sensor signal value",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.EVENT_MARKER,
            "dtype": pl.Utf8,
            "description": "Free-text event marker (e.g. 'Insulin: 27,00', 'Meal: 60,00grams')",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_NUMBER,
            "dtype": pl.Utf8,
            "description": "Bolus delivery number",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_CANCELLATION_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for bolus cancellation",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_UNABSORBED_INSULIN_TOTAL,
            "dtype": pl.Utf8,
            "description": "Total unabsorbed insulin from Bolus Wizard",
            "unit": "U",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.FINAL_BOLUS_ESTIMATE,
            "dtype": pl.Utf8,
            "description": "Final bolus estimate after adjustments",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SCROLL_STEP_SIZE,
            "dtype": pl.Utf8,
            "description": "Scroll step size setting",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.INSULIN_ACTION_CURVE_TIME,
            "dtype": pl.Utf8,
            "description": "Insulin action curve time setting",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SENSOR_CALIBRATION_REJECTED_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for sensor calibration rejection",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRESET_BOLUS,
            "dtype": pl.Utf8,
            "description": "Preset bolus configuration",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_SOURCE,
            "dtype": pl.Utf8,
            "description": "Source of bolus command",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.NETWORK_DEVICE_ASSOCIATED_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for network device association",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.NETWORK_DEVICE_DISASSOCIATED_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for network device disassociation",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.NETWORK_DEVICE_DISCONNECTED_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for network device disconnection",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SENSOR_EXCEPTION,
            "dtype": pl.Utf8,
            "description": "Sensor exception event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRESET_TEMP_BASAL_NAME,
            "dtype": pl.Utf8,
            "description": "Name of preset temporary basal pattern",
            "constraints": {"required": False}
        },
    ),
    header_line=MEDTRONIC_HEADER_LINE,
    data_start_line=MEDTRONIC_DATA_START_LINE,
    metadata_lines=MEDTRONIC_METADATA_LINES
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate medtronic.json from the current schema definition.

    Run this after modifying enums or schema to keep medtronic.json in sync:
        python3 -c "from formats.medtronic import regenerate_schema_json; regenerate_schema_json()"
    """
    from cgm_format.interface.schema import regenerate_schema_json as _regenerate
    _regenerate(MEDTRONIC_SCHEMA, __file__)
