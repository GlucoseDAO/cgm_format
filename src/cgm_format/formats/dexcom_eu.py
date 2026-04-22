"""Dexcom G6/G7 European CGM export format (mmol/L units).

Same file structure as standard Dexcom (rows 1-11 header/metadata, row 12+ data)
but with glucose values in mmol/L instead of mg/dL.

Column differences vs standard Dexcom:
  - "Glucose Value (mmol/L)" instead of "Glucose Value (mg/dL)"
  - "Glucose Rate of Change (mmol/L/min)" instead of "Glucose Rate of Change (mg/dL/min)"

All other columns, event types, metadata layout, and timestamp formats are identical.
"""

from typing import List
import polars as pl
from cgm_format.interface.schema import (
    CGMSchemaDefinition,
    EnumLiteral,
    regenerate_schema_json as _regenerate,
)

# Shared Dexcom constants
from cgm_format.formats.dexcom import (
    DEXCOM_TIMESTAMP_FORMATS,
    DEXCOM_HIGH_GLUCOSE_DEFAULT,
    DEXCOM_LOW_GLUCOSE_DEFAULT,
    DexcomEventType,
    DexcomEventSubtype,
    DexcomEventTypeSubtype,
)

# EU exports include an extra "Sensor" metadata row (G7), so data starts
# one line later than the standard mg/dL export.
# Line 1 = header, Lines 2-12 = metadata, Line 13+ = data
DEXCOM_EU_HEADER_LINE = 1
DEXCOM_EU_DATA_START_LINE = 13
DEXCOM_EU_METADATA_LINES = tuple(range(DEXCOM_EU_HEADER_LINE + 1, DEXCOM_EU_DATA_START_LINE))  # Rows 2-12

# mmol/L → mg/dL conversion factor (standard clinical value)
MMOL_TO_MGDL = 18.0182

# Detection patterns — "Glucose Value (mmol/L)" uniquely identifies the EU variant
DEXCOM_EU_DETECTION_PATTERNS = [
    "Glucose Value (mmol/L)",
]


# =============================================================================
# EU Column Names (only GLUCOSE_VALUE and GLUCOSE_RATE_OF_CHANGE differ)
# =============================================================================

class DexcomEUColumn(EnumLiteral):
    """Column names in European Dexcom G6/G7 export files (mmol/L units)."""
    INDEX = "Index"
    TIMESTAMP = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    EVENT_TYPE = "Event Type"
    EVENT_SUBTYPE = "Event Subtype"
    PATIENT_INFO = "Patient Info"
    DEVICE_INFO = "Device Info"
    SOURCE_DEVICE_ID = "Source Device ID"
    GLUCOSE_VALUE = "Glucose Value (mmol/L)"
    INSULIN_VALUE = "Insulin Value (u)"
    CARB_VALUE = "Carb Value (grams)"
    DURATION = "Duration (hh:mm:ss)"
    GLUCOSE_RATE_OF_CHANGE = "Glucose Rate of Change (mmol/L/min)"
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
# EU Raw File Format Schema
# =============================================================================

DEXCOM_EU_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": DexcomEUColumn.INDEX,
            "dtype": pl.Int64,
            "description": "Sequential index of the record in the export",
            "constraints": {"required": True}
        },
        {
            "name": DexcomEUColumn.TIMESTAMP,
            "dtype": pl.Utf8,
            "description": "Timestamp of the event in YYYY-MM-DD HH:MM:SS format",
            "constraints": {"required": True}
        },
        {
            "name": DexcomEUColumn.EVENT_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of recorded event",
            "constraints": {
                "required": True,
                "enum": [e.value for e in DexcomEventType]
            }
        },
        {
            "name": DexcomEUColumn.EVENT_SUBTYPE,
            "dtype": pl.Utf8,
            "description": "Subtype of recorded event (may be empty)",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.PATIENT_INFO,
            "dtype": pl.Utf8,
            "description": "Patient information field",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.DEVICE_INFO,
            "dtype": pl.Utf8,
            "description": "Device information (e.g., 'android G6', 'iOS G7')",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.SOURCE_DEVICE_ID,
            "dtype": pl.Utf8,
            "description": "Source device identifier",
            "constraints": {"required": False}
        },
    ),
    data_columns=(
        {
            "name": DexcomEUColumn.GLUCOSE_VALUE,
            "dtype": pl.Float64,
            "description": "Blood glucose reading from CGM sensor",
            "unit": "mmol/L",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomEUColumn.INSULIN_VALUE,
            "dtype": pl.Float64,
            "description": "Insulin dose (type determined by Event Subtype)",
            "unit": "u",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomEUColumn.CARB_VALUE,
            "dtype": pl.Float64,
            "description": "Carbohydrate intake",
            "unit": "grams",
            "constraints": {"minimum": 0}
        },
        {
            "name": DexcomEUColumn.DURATION,
            "dtype": pl.Utf8,
            "description": "Duration of exercise activity in HH:MM:SS format",
            "unit": "hh:mm:ss",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.GLUCOSE_RATE_OF_CHANGE,
            "dtype": pl.Float64,
            "description": "Rate of change of glucose levels",
            "unit": "mmol/L/min",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.TRANSMITTER_TIME,
            "dtype": pl.Int64,
            "description": "Transmitter time as long integer (epoch-like timestamp)",
            "constraints": {"required": False}
        },
        {
            "name": DexcomEUColumn.TRANSMITTER_ID,
            "dtype": pl.Utf8,
            "description": "Transmitter identifier (e.g., '8AM1EY')",
            "constraints": {"required": False}
        },
    ),
    header_line=DEXCOM_EU_HEADER_LINE,
    data_start_line=DEXCOM_EU_DATA_START_LINE,
    metadata_lines=DEXCOM_EU_METADATA_LINES
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate dexcom_eu.json from the current schema definition."""
    _regenerate(DEXCOM_EU_SCHEMA, __file__)
