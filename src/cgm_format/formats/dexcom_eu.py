"""Dexcom G6/G7 European CGM export format (mmol/L units).

Same file structure as standard Dexcom (rows 1-11 header/metadata, row 12+ data)
but with glucose values in mmol/L instead of mg/dL.

Column differences vs standard Dexcom:
  - "Glucose Value (mmol/L)" instead of "Glucose Value (mg/dL)"
  - "Glucose Rate of Change (mmol/L/min)" instead of "Glucose Rate of Change (mg/dL/min)"

All other columns, event types, metadata layout, and timestamp formats are identical.
"""

from typing import List
from cgm_format.interface.schema import (
    EnumLiteral,
    derive_schema,
    regenerate_schema_json as _regenerate,
)

# Base schema to derive from + its column vocabulary (for the rename keys)
from cgm_format.formats.dexcom import (
    DexcomColumn,
    DEXCOM_SCHEMA,
)
# Re-exported for backward compatibility (public symbol); the canonical
# definition now lives with the unit-conversion table in unified.py.
from cgm_format.formats.unified import MMOL_TO_MGDL

# EU exports include an extra "Sensor" metadata row (G7), so data starts
# one line later than the standard mg/dL export.
# Line 1 = header, Lines 2-12 = metadata, Line 13+ = data
DEXCOM_EU_HEADER_LINE = 1
DEXCOM_EU_DATA_START_LINE = 13
DEXCOM_EU_METADATA_LINES = tuple(range(DEXCOM_EU_HEADER_LINE + 1, DEXCOM_EU_DATA_START_LINE))  # Rows 2-12

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

# The EU export is standard Dexcom with two glucose columns relabeled to mmol/L
# and one extra metadata row. Rather than re-declare all 14 columns, derive it
# from DEXCOM_SCHEMA and patch only the real deltas. The mmol/L `unit` is what
# drives the parser's declarative mmol/L→mg/dL conversion (UNIT_CONVERSIONS);
# there is no EU-specific conversion code.
DEXCOM_EU_SCHEMA = derive_schema(
    DEXCOM_SCHEMA,
    renames={
        DexcomColumn.GLUCOSE_VALUE: DexcomEUColumn.GLUCOSE_VALUE.value,
        DexcomColumn.GLUCOSE_RATE_OF_CHANGE: DexcomEUColumn.GLUCOSE_RATE_OF_CHANGE.value,
    },
    units={
        DexcomEUColumn.GLUCOSE_VALUE.value: "mmol/L",
        DexcomEUColumn.GLUCOSE_RATE_OF_CHANGE.value: "mmol/L/min",
    },
    data_start_line=DEXCOM_EU_DATA_START_LINE,
    metadata_lines=DEXCOM_EU_METADATA_LINES,
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate dexcom_eu.json from the current schema definition."""
    _regenerate(DEXCOM_EU_SCHEMA, __file__)
