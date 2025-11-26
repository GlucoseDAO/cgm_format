"""Medtronic Guardian Connect CGM export format schema definition.

This module defines the specific schema for Medtronic Guardian Connect CGM data format,
including raw file column definitions, event types, and metadata structure.

File structure:
- Line 1: Header row with patient metadata columns
- Line 2: Patient information (Last Name, First Name, Patient ID, System ID, dates, device info)
- Line 3: Empty line
- Line 4: Notice text ("Device data shown may exceed selected date range.")
- Line 5: Empty line
- Line 6: Device identifier row (separator line with device type and serial)
- Line 7: Column headers for data rows
- Line 8+: Data rows with sensor glucose, insulin, bolus, basal rate, and event data

Example header rows:
- Row 1: Last Name;First Name;Patient ID;System ID;Start Date;End Date;Device;Guardian Connect
- Row 2: "Livia";"Zaharia";"";"";"26.06.2021 00:00:00";"25.07.2021 23:59:59";"Serial Number";GCZ7PA-UYGI-AZWP-LWF1
- Row 6: -------;Guardian Connect;Pump;GCZ7PA-UYGI-AZWP-LWF1;-------

Example data row:
1,00000;2021/07/25;09:35:28;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Insulin: 27,00;;;;;;;;;;;;;;

Note: Uses semicolon (;) as delimiter instead of comma.
"""

from typing import List
from enum import Enum
import polars as pl
from interface.schema import (
    ColumnSchema,
    CGMSchemaDefinition,
    EnumLiteral,
)


# =============================================================================
# File Format Constants
# =============================================================================

# File structure: Row 1-2 = patient metadata, Row 3-6 = device info/separators, Row 7 = header, Row 8+ = data
MEDTRONIC_HEADER_LINE = 7
MEDTRONIC_DATA_START_LINE = 8
MEDTRONIC_METADATA_LINES = tuple(range(1, MEDTRONIC_HEADER_LINE))  # Rows 1-6 are metadata to skip

# Multiple timestamp formats across Medtronic versions (tuple for probing)
MEDTRONIC_TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M:%S",  # Standard format: 2021/07/25 09:35:28
    "%d.%m.%Y %H:%M:%S",  # European format in metadata: 26.06.2021 00:00:00
)

# Format detection patterns (unique identifiers in CSV headers/content)
MEDTRONIC_DETECTION_PATTERNS = [
    "Guardian Connect",                      # Unique device identifier
    "Sensor Glucose (mg/dL)",               # Medtronic-specific column
    "BWZ Estimate (U)",                     # Bolus Wizard column (unique to Medtronic)
    "Transmitter Time (Long Integer)",      # Not present - distinguishes from Dexcom
    "Index;Date;Time;New Device Time",      # Unique column combination with semicolons
]


# =============================================================================
# Medtronic Event/Record Type Enums
# =============================================================================

class MedtronicBolusType(EnumLiteral):
    """Type of bolus insulin delivery."""
    NORMAL = "Normal"
    SQUARE = "Square"  # Extended/square wave bolus
    DUAL = "Dual"      # Dual wave bolus (normal + square)
    EMPTY = ""


class MedtronicTempBasalType(EnumLiteral):
    """Type of temporary basal rate."""
    PERCENT = "Percent"  # Percentage of normal basal
    ABSOLUTE = "Absolute"  # Absolute rate
    EMPTY = ""


class MedtronicPrimeType(EnumLiteral):
    """Type of insulin pump priming."""
    MANUAL = "Manual"
    CANNULA_FILL = "Cannula Fill"
    TUBING_FILL = "Tubing Fill"
    EMPTY = ""


# =============================================================================
# Raw File Column Names (as they appear in Medtronic CSV exports)
# =============================================================================

class MedtronicColumn(EnumLiteral):
    """Column names in Medtronic Guardian Connect export files."""
    # Service/Identifier columns
    INDEX = "Index"
    DATE = "Date"
    TIME = "Time"
    NEW_DEVICE_TIME = "New Device Time"
    
    # Glucose readings
    BG_READING = "BG Reading (mg/dL)"
    SENSOR_GLUCOSE = "Sensor Glucose (mg/dL)"
    SENSOR_CALIBRATION_BG = "Sensor Calibration BG (mg/dL)"
    BG_INPUT = "BWZ BG Input (mg/dL)"
    LINKED_BG_METER_ID = "Linked BG Meter ID"
    ISIG_VALUE = "ISIG Value"
    
    # Basal insulin
    BASAL_RATE = "Basal Rate (U/h)"
    TEMP_BASAL_AMOUNT = "Temp Basal Amount"
    TEMP_BASAL_TYPE = "Temp Basal Type"
    TEMP_BASAL_DURATION = "Temp Basal Duration (h:mm:ss)"
    PRESET_TEMP_BASAL_NAME = "Preset Temp Basal Name"
    
    # Bolus insulin
    BOLUS_TYPE = "Bolus Type"
    BOLUS_VOLUME_SELECTED = "Bolus Volume Selected (U)"
    BOLUS_VOLUME_DELIVERED = "Bolus Volume Delivered (U)"
    BOLUS_DURATION = "Bolus Duration (h:mm:ss)"
    BOLUS_NUMBER = "Bolus Number"
    BOLUS_CANCELLATION_REASON = "Bolus Cancellation Reason"
    PRESET_BOLUS = "Preset Bolus"
    BOLUS_SOURCE = "Bolus Source"
    
    # Bolus Wizard (BWZ) calculations
    BWZ_ESTIMATE = "BWZ Estimate (U)"
    BWZ_TARGET_HIGH_BG = "BWZ Target High BG (mg/dL)"
    BWZ_TARGET_LOW_BG = "BWZ Target Low BG (mg/dL)"
    BWZ_CARB_RATIO = "BWZ Carb Ratio (g/U)"
    BWZ_INSULIN_SENSITIVITY = "BWZ Insulin Sensitivity (mg/dL/U)"
    BWZ_CARB_INPUT = "BWZ Carb Input (grams)"
    BWZ_CORRECTION_ESTIMATE = "BWZ Correction Estimate (U)"
    BWZ_FOOD_ESTIMATE = "BWZ Food Estimate (U)"
    BWZ_ACTIVE_INSULIN = "BWZ Active Insulin (U)"
    BWZ_UNABSORBED_INSULIN_TOTAL = "BWZ Unabsorbed Insulin Total (U)"
    BWZ_STATUS = "BWZ Status"
    FINAL_BOLUS_ESTIMATE = "Final Bolus Estimate"
    
    # Pump maintenance
    PRIME_TYPE = "Prime Type"
    PRIME_VOLUME_DELIVERED = "Prime Volume Delivered (U)"
    REWIND = "Rewind"
    
    # Alarms and system events
    ALARM = "Alarm"
    SUSPEND = "Suspend"
    SENSOR_EXCEPTION = "Sensor Exception"
    SENSOR_CALIBRATION_REJECTED_REASON = "Sensor Calibration Rejected Reason"
    
    # Device network
    NETWORK_DEVICE_ASSOCIATED_REASON = "Network Device Associated Reason"
    NETWORK_DEVICE_DISASSOCIATED_REASON = "Network Device Disassociated Reason"
    NETWORK_DEVICE_DISCONNECTED_REASON = "Network Device Disconnected Reason"
    
    # Events and settings
    EVENT_MARKER = "Event Marker"
    SCROLL_STEP_SIZE = "Scroll Step Size"
    INSULIN_ACTION_CURVE_TIME = "Insulin Action Curve Time"
    
    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names in order as they appear in Medtronic CSV files."""
        return [
            cls.INDEX, cls.DATE, cls.TIME, cls.NEW_DEVICE_TIME,
            cls.BG_READING, cls.LINKED_BG_METER_ID,
            cls.BASAL_RATE, cls.TEMP_BASAL_AMOUNT, cls.TEMP_BASAL_TYPE, cls.TEMP_BASAL_DURATION,
            cls.BOLUS_TYPE, cls.BOLUS_VOLUME_SELECTED, cls.BOLUS_VOLUME_DELIVERED, cls.BOLUS_DURATION,
            cls.PRIME_TYPE, cls.PRIME_VOLUME_DELIVERED,
            cls.ALARM, cls.SUSPEND, cls.REWIND,
            cls.BWZ_ESTIMATE, cls.BWZ_TARGET_HIGH_BG, cls.BWZ_TARGET_LOW_BG,
            cls.BWZ_CARB_RATIO, cls.BWZ_INSULIN_SENSITIVITY, cls.BWZ_CARB_INPUT, cls.BG_INPUT,
            cls.BWZ_CORRECTION_ESTIMATE, cls.BWZ_FOOD_ESTIMATE, cls.BWZ_ACTIVE_INSULIN, cls.BWZ_STATUS,
            cls.SENSOR_CALIBRATION_BG, cls.SENSOR_GLUCOSE, cls.ISIG_VALUE,
            cls.EVENT_MARKER, cls.BOLUS_NUMBER, cls.BOLUS_CANCELLATION_REASON,
            cls.BWZ_UNABSORBED_INSULIN_TOTAL, cls.FINAL_BOLUS_ESTIMATE,
            cls.SCROLL_STEP_SIZE, cls.INSULIN_ACTION_CURVE_TIME,
            cls.SENSOR_CALIBRATION_REJECTED_REASON, cls.PRESET_BOLUS, cls.BOLUS_SOURCE,
            cls.NETWORK_DEVICE_ASSOCIATED_REASON, cls.NETWORK_DEVICE_DISASSOCIATED_REASON,
            cls.NETWORK_DEVICE_DISCONNECTED_REASON, cls.SENSOR_EXCEPTION, cls.PRESET_TEMP_BASAL_NAME
        ]


# =============================================================================
# Medtronic Guardian Connect Raw File Format Schema
# =============================================================================

MEDTRONIC_SCHEMA = CGMSchemaDefinition(
    service_columns=[
        {
            "name": MedtronicColumn.INDEX,
            "dtype": pl.Utf8,  # Can have format like "1,00000" with commas
            "description": "Sequential index of the record in the export",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.DATE,
            "dtype": pl.Utf8,  # String format: "YYYY/MM/DD"
            "description": "Date of the event in YYYY/MM/DD format",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.TIME,
            "dtype": pl.Utf8,  # String format: "HH:MM:SS"
            "description": "Time of the event in HH:MM:SS format",
            "constraints": {"required": True}
        },
        {
            "name": MedtronicColumn.NEW_DEVICE_TIME,
            "dtype": pl.Utf8,
            "description": "Timestamp when device time was changed",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.LINKED_BG_METER_ID,
            "dtype": pl.Utf8,
            "description": "Identifier for linked blood glucose meter",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of bolus insulin delivery (Normal, Square, Dual)",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of temporary basal rate (Percent, Absolute)",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRIME_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of pump priming event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.ALARM,
            "dtype": pl.Utf8,
            "description": "Alarm message or type",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.SUSPEND,
            "dtype": pl.Utf8,
            "description": "Pump suspension event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.REWIND,
            "dtype": pl.Utf8,
            "description": "Pump rewind event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BWZ_STATUS,
            "dtype": pl.Utf8,
            "description": "Status of Bolus Wizard calculation",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.EVENT_MARKER,
            "dtype": pl.Utf8,
            "description": "User-logged event marker (e.g., 'Insulin: 27,00')",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_NUMBER,
            "dtype": pl.Utf8,
            "description": "Unique identifier for bolus event",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_CANCELLATION_REASON,
            "dtype": pl.Utf8,
            "description": "Reason for bolus cancellation",
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
            "description": "Name of preset bolus used",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.BOLUS_SOURCE,
            "dtype": pl.Utf8,
            "description": "Source of bolus command (pump, remote, etc.)",
            "constraints": {"required": False}
        },
        {
            "name": MedtronicColumn.PRESET_TEMP_BASAL_NAME,
            "dtype": pl.Utf8,
            "description": "Name of preset temporary basal rate used",
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
            "description": "Sensor exception or error message",
            "constraints": {"required": False}
        },
    ],
    data_columns=[
        # Glucose readings
        {
            "name": MedtronicColumn.BG_READING,
            "dtype": pl.Float64,
            "description": "Blood glucose reading from linked meter",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.SENSOR_GLUCOSE,
            "dtype": pl.Float64,
            "description": "CGM sensor glucose reading",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.SENSOR_CALIBRATION_BG,
            "dtype": pl.Float64,
            "description": "Blood glucose value used for sensor calibration",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BG_INPUT,
            "dtype": pl.Float64,
            "description": "Blood glucose value input to Bolus Wizard",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.ISIG_VALUE,
            "dtype": pl.Float64,
            "description": "Interstitial Signal (ISIG) raw sensor value",
            "constraints": {"minimum": 0}
        },
        
        # Basal insulin
        {
            "name": MedtronicColumn.BASAL_RATE,
            "dtype": pl.Float64,
            "description": "Basal insulin rate",
            "unit": "U/h",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_AMOUNT,
            "dtype": pl.Float64,
            "description": "Temporary basal rate amount (percentage or absolute)",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.TEMP_BASAL_DURATION,
            "dtype": pl.Utf8,  # String format: "HH:MM:SS"
            "description": "Duration of temporary basal rate in HH:MM:SS format",
            "unit": "h:mm:ss",
            "constraints": {"required": False}
        },
        
        # Bolus insulin
        {
            "name": MedtronicColumn.BOLUS_VOLUME_SELECTED,
            "dtype": pl.Float64,
            "description": "Bolus insulin volume selected for delivery",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BOLUS_VOLUME_DELIVERED,
            "dtype": pl.Float64,
            "description": "Bolus insulin volume actually delivered",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BOLUS_DURATION,
            "dtype": pl.Utf8,  # String format: "HH:MM:SS"
            "description": "Duration of bolus delivery in HH:MM:SS format",
            "unit": "h:mm:ss",
            "constraints": {"required": False}
        },
        
        # Bolus Wizard calculations
        {
            "name": MedtronicColumn.BWZ_ESTIMATE,
            "dtype": pl.Float64,
            "description": "Bolus Wizard insulin dose estimate",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_TARGET_HIGH_BG,
            "dtype": pl.Float64,
            "description": "Bolus Wizard high target blood glucose",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_TARGET_LOW_BG,
            "dtype": pl.Float64,
            "description": "Bolus Wizard low target blood glucose",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_CARB_RATIO,
            "dtype": pl.Float64,
            "description": "Bolus Wizard carbohydrate to insulin ratio",
            "unit": "g/U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_INSULIN_SENSITIVITY,
            "dtype": pl.Float64,
            "description": "Bolus Wizard insulin sensitivity factor",
            "unit": "mg/dL/U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_CARB_INPUT,
            "dtype": pl.Float64,
            "description": "Carbohydrate input to Bolus Wizard",
            "unit": "grams",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_CORRECTION_ESTIMATE,
            "dtype": pl.Float64,
            "description": "Bolus Wizard correction insulin estimate",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_FOOD_ESTIMATE,
            "dtype": pl.Float64,
            "description": "Bolus Wizard food/meal insulin estimate",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_ACTIVE_INSULIN,
            "dtype": pl.Float64,
            "description": "Active insulin (insulin on board) at time of Bolus Wizard use",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.BWZ_UNABSORBED_INSULIN_TOTAL,
            "dtype": pl.Float64,
            "description": "Total unabsorbed insulin from previous boluses",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.FINAL_BOLUS_ESTIMATE,
            "dtype": pl.Float64,
            "description": "Final bolus estimate after user adjustments",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        
        # Pump maintenance
        {
            "name": MedtronicColumn.PRIME_VOLUME_DELIVERED,
            "dtype": pl.Float64,
            "description": "Insulin volume delivered during priming",
            "unit": "U",
            "constraints": {"minimum": 0}
        },
        
        # Settings
        {
            "name": MedtronicColumn.SCROLL_STEP_SIZE,
            "dtype": pl.Float64,
            "description": "Scroll step size setting",
            "constraints": {"minimum": 0}
        },
        {
            "name": MedtronicColumn.INSULIN_ACTION_CURVE_TIME,
            "dtype": pl.Float64,
            "description": "Insulin action curve time setting",
            "unit": "hours",
            "constraints": {"minimum": 0}
        },
    ],
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
    from interface.schema import regenerate_schema_json as _regenerate
    _regenerate(MEDTRONIC_SCHEMA, __file__)
