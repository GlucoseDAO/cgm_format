"""Dexcom G6/G7 CGM export format schema definition.

This module defines the specific schema for Dexcom G6/G7 CGM data format,
including raw file column definitions, event types, and metadata structure.

File structure:
- Lines 1-11: Header rows containing metadata (patient info, alerts, device info)
- Line 12+: Data rows with EGV (Estimated Glucose Values) and events

Example header rows:
- Row 1: Column names
- Row 2: FirstName
- Row 3: LastName  
- Row 4: Device info
- Rows 5-11: Alert configurations (Fall, High, Low, Signal Loss, Rise, Urgent Low, Urgent Low Soon)

Example data row:
11,2025-05-01 0:01:47,EGV,,,,android G6,165,,,,,4237574,8AM1EY
"""

from typing import List, Dict, Tuple
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

# File structure: Row 1 = header, Rows 2-11 = metadata, Row 12+ = data
DEXCOM_HEADER_LINE = 1
DEXCOM_DATA_START_LINE = 12
DEXCOM_METADATA_LINES = tuple(range(DEXCOM_HEADER_LINE+1, DEXCOM_DATA_START_LINE))  # Rows 2-11 are metadata to skip

# Multiple timestamp formats across Clarity versions (tuple for probing)
DEXCOM_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",  # Older Clarity exports: 2019-10-14T16:42:37
    "%Y-%m-%d %H:%M:%S",  # Newer Clarity exports: 2025-05-01 0:01:47
)

# Glucose value replacements for High/Low readings
DEXCOM_HIGH_GLUCOSE_DEFAULT = 401  # mg/dL - Standard upper limit for CGM sensors
DEXCOM_LOW_GLUCOSE_DEFAULT = 39    # mg/dL - Standard lower limit for CGM sensors

# Format detection patterns (unique identifiers in CSV headers/content)
DEXCOM_DETECTION_PATTERNS = [
    "Timestamp (YYYY-MM-DDThh:mm:ss)",  # Unique timestamp column header format
    "Event Type,Event Subtype",         # Unique column combination
    "Transmitter ID",                   # Dexcom-specific column
    "Transmitter Time (Long Integer)",  # Dexcom-specific column
]


# =============================================================================
# Dexcom Event Type Enums
# =============================================================================

class DexcomEventType(EnumLiteral):
    """Type of recorded event in Dexcom data (Event Type column)."""
    ALERT = "Alert"
    CALIBRATION = "Calibration"
    CARBOHYDRATES = "Carbs"
    EGV = "EGV"
    EXERCISE = "Exercise"
    HEALTH = "Health"
    INSULIN = "Insulin"


class DexcomEventSubtype(EnumLiteral):
    """Subtype of recorded event in Dexcom data (Event Subtype column)."""
    # Alert subtypes
    ALERT_FALL = "Fall"
    ALERT_HIGH = "High"
    ALERT_LOW = "Low"
    ALERT_SIGNAL_LOSS = "Signal Loss"
    ALERT_RISE = "Rise"
    ALERT_URGENT_LOW = "Urgent Low"
    ALERT_URGENT_LOW_SOON = "Urgent Low Soon"
    
    # EGV subtypes
    EGV_HIGH = "High"
    EGV_LOW = "Low"
    
    # Exercise subtypes
    EXERCISE_HEAVY = "Heavy"
    EXERCISE_LIGHT = "Light"
    EXERCISE_MEDIUM = "Medium"
    
    # Health subtypes
    HEALTH_ALCOHOL = "Alcohol"
    HEALTH_CYCLE = "Cycle"
    HEALTH_ILLNESS = "Illness"
    HEALTH_LOW_SYMPTOMS = "Low Symptoms"
    HEALTH_STRESS = "Stress"
    
    # Insulin subtypes
    INSULIN_LONG_ACTING = "Long-Acting"
    INSULIN_FAST_ACTING = "Fast-Acting"
    
    # Empty subtype (for events without subtypes)
    EMPTY = ""


class DexcomEventTypeSubtype(Enum):
    """Combined Event Type + Subtype pairs for mapping to unified format."""
    
    # Alerts
    ALERT_FALL = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_FALL)
    ALERT_HIGH = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_HIGH)
    ALERT_LOW = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_LOW)
    ALERT_SIGNAL_LOSS = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_SIGNAL_LOSS)
    ALERT_RISE = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_RISE)
    ALERT_URGENT_LOW = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_URGENT_LOW)
    ALERT_URGENT_LOW_SOON = (DexcomEventType.ALERT, DexcomEventSubtype.ALERT_URGENT_LOW_SOON)
    
    # Calibration
    CALIBRATION = (DexcomEventType.CALIBRATION, DexcomEventSubtype.EMPTY)
    
    # Carbohydrates
    CARBS = (DexcomEventType.CARBOHYDRATES, DexcomEventSubtype.EMPTY)
    
    # EGV (Estimated Glucose Value)
    EGV = (DexcomEventType.EGV, DexcomEventSubtype.EMPTY)
    EGV_HIGH = (DexcomEventType.EGV, DexcomEventSubtype.EGV_HIGH)
    EGV_LOW = (DexcomEventType.EGV, DexcomEventSubtype.EGV_LOW)
    
    # Exercise
    EXERCISE_HEAVY = (DexcomEventType.EXERCISE, DexcomEventSubtype.EXERCISE_HEAVY)
    EXERCISE_LIGHT = (DexcomEventType.EXERCISE, DexcomEventSubtype.EXERCISE_LIGHT)
    EXERCISE_MEDIUM = (DexcomEventType.EXERCISE, DexcomEventSubtype.EXERCISE_MEDIUM)
    
    # Health
    HEALTH_ALCOHOL = (DexcomEventType.HEALTH, DexcomEventSubtype.HEALTH_ALCOHOL)
    HEALTH_CYCLE = (DexcomEventType.HEALTH, DexcomEventSubtype.HEALTH_CYCLE)
    HEALTH_ILLNESS = (DexcomEventType.HEALTH, DexcomEventSubtype.HEALTH_ILLNESS)
    HEALTH_LOW_SYMPTOMS = (DexcomEventType.HEALTH, DexcomEventSubtype.HEALTH_LOW_SYMPTOMS)
    HEALTH_STRESS = (DexcomEventType.HEALTH, DexcomEventSubtype.HEALTH_STRESS)
    
    # Insulin
    INSULIN_LONG_ACTING = (DexcomEventType.INSULIN, DexcomEventSubtype.INSULIN_LONG_ACTING)
    INSULIN_FAST_ACTING = (DexcomEventType.INSULIN, DexcomEventSubtype.INSULIN_FAST_ACTING)


# =============================================================================
# Raw File Column Names (as they appear in Dexcom CSV exports)
# =============================================================================

class DexcomColumn(EnumLiteral):
    """Column names in Dexcom G6/G7 export files."""
    INDEX = "Index"
    TIMESTAMP = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    EVENT_TYPE = "Event Type"
    EVENT_SUBTYPE = "Event Subtype"
    PATIENT_INFO = "Patient Info"
    DEVICE_INFO = "Device Info"
    SOURCE_DEVICE_ID = "Source Device ID"
    GLUCOSE_VALUE = "Glucose Value (mg/dL)"
    INSULIN_VALUE = "Insulin Value (u)"
    CARB_VALUE = "Carb Value (grams)"
    DURATION = "Duration (hh:mm:ss)"
    GLUCOSE_RATE_OF_CHANGE = "Glucose Rate of Change (mg/dL/min)"
    TRANSMITTER_TIME = "Transmitter Time (Long Integer)"
    TRANSMITTER_ID = "Transmitter ID"
    
    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names in order."""
        return [
            cls.INDEX, cls.TIMESTAMP, cls.EVENT_TYPE, cls.EVENT_SUBTYPE,
            cls.PATIENT_INFO, cls.DEVICE_INFO, cls.SOURCE_DEVICE_ID,
            cls.GLUCOSE_VALUE, cls.INSULIN_VALUE, cls.CARB_VALUE,
            cls.DURATION, cls.GLUCOSE_RATE_OF_CHANGE, cls.TRANSMITTER_TIME,
            cls.TRANSMITTER_ID
        ]


# =============================================================================
# Dexcom Raw File Format Schema
# =============================================================================

DEXCOM_SCHEMA = CGMSchemaDefinition(
    service_columns=[
        {
            "name": DexcomColumn.INDEX,
            "dtype": pl.Int64,
            "description": "Sequential index of the record in the export",
            "constraints": {"required": True}
        },
        # NOTE: TIMESTAMP comes second in the actual CSV file, before EVENT_TYPE
        {
            "name": DexcomColumn.TIMESTAMP,
            "dtype": pl.Utf8,  # String format: "YYYY-MM-DD HH:MM:SS"
            "description": "Timestamp of the event in YYYY-MM-DD HH:MM:SS format",
            "constraints": {"required": True}
        },
        {
            "name": DexcomColumn.EVENT_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of recorded event",
            "constraints": {
                "required": True,
                "enum": [e.value for e in DexcomEventType]
            }
        },
        {
            "name": DexcomColumn.EVENT_SUBTYPE,
            "dtype": pl.Utf8,
            "description": "Subtype of recorded event (may be empty)",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.PATIENT_INFO,
            "dtype": pl.Utf8,
            "description": "Patient information field",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.DEVICE_INFO,
            "dtype": pl.Utf8,
            "description": "Device information (e.g., 'android G6', 'iOS G7')",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.SOURCE_DEVICE_ID,
            "dtype": pl.Utf8,
            "description": "Source device identifier",
            "constraints": {"required": False}
        },
    ],
    data_columns=[
        {
            "name": DexcomColumn.GLUCOSE_VALUE,
            "dtype": pl.Float64,
            "description": "Blood glucose reading from CGM sensor",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomColumn.INSULIN_VALUE,
            "dtype": pl.Float64,
            "description": "Insulin dose (type determined by Event Subtype)",
            "unit": "u",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomColumn.CARB_VALUE,
            "dtype": pl.Float64,
            "description": "Carbohydrate intake",
            "unit": "grams",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomColumn.DURATION,
            "dtype": pl.Utf8,  # String format: "HH:MM:SS"
            "description": "Duration of exercise activity in HH:MM:SS format",
            "unit": "hh:mm:ss",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.GLUCOSE_RATE_OF_CHANGE,
            "dtype": pl.Float64,
            "description": "Rate of change of glucose levels",
            "unit": "mg/dL/min",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.TRANSMITTER_TIME,
            "dtype": pl.Int64,
            "description": "Transmitter time as long integer (epoch-like timestamp)",
            "constraints": {"required": False}
        },
        {
            "name": DexcomColumn.TRANSMITTER_ID,
            "dtype": pl.Utf8,
            "description": "Transmitter identifier (e.g., '8AM1EY')",
            "constraints": {"required": False}
        },
    ],
    header_line=DEXCOM_HEADER_LINE,
    data_start_line=DEXCOM_DATA_START_LINE,
    metadata_lines=DEXCOM_METADATA_LINES
)



# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate dexcom.json from the current schema definition.
    
    Run this after modifying enums or schema to keep dexcom.json in sync:
        python3 -c "from formats.dexcom import regenerate_schema_json; regenerate_schema_json()"
    """
    from interface.schema import regenerate_schema_json as _regenerate
    _regenerate(DEXCOM_SCHEMA, __file__)
    
    



