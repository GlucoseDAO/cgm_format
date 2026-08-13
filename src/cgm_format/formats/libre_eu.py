"""FreeStyle Libre European / mmol/L CGM export format.

Same file structure as standard Libre (row 1 metadata, row 2 header, row 3+ data)
but with glucose values in mmol/L instead of mg/dL, plus two extra ketone columns
that later LibreView exports append.

Column differences vs standard Libre:
  - "Historic Glucose mmol/L" instead of "Historic Glucose mg/dL"
  - "Scan Glucose mmol/L" instead of "Scan Glucose mg/dL"
  - "Strip Glucose mmol/L" instead of "Strip Glucose mg/dL"
  - trailing "Historic Ketone mmol/L" and "Scan Ketone mmol/L"

All other columns, record types, metadata layout, and timestamp formats are
identical. The long-acting insulin alias ("Long-Acting Insulin (units)") is
inherited from LIBRE_SCHEMA.
"""

from typing import List
import polars as pl
from cgm_format.interface.schema import (
    ColumnSchema,
    EnumLiteral,
    derive_schema,
    regenerate_schema_json as _regenerate,
)
from cgm_format.formats.libre import (
    LibreColumn,
    LIBRE_SCHEMA,
    LIBRE_HEADER_LINE,
    LIBRE_DATA_START_LINE,
    LIBRE_METADATA_LINES,
)

# File geometry is identical to the mg/dL Libre export.
LIBRE_EU_HEADER_LINE = LIBRE_HEADER_LINE
LIBRE_EU_DATA_START_LINE = LIBRE_DATA_START_LINE
LIBRE_EU_METADATA_LINES = LIBRE_METADATA_LINES

# Detection patterns — "Historic Glucose mmol/L" uniquely identifies the EU variant
LIBRE_EU_DETECTION_PATTERNS = [
    "Historic Glucose mmol/L",
]


# =============================================================================
# EU Column Names (glucose columns + two appended ketone columns differ)
# =============================================================================

class LibreEUColumn(EnumLiteral):
    """Column names in European FreeStyle Libre export files (mmol/L units)."""
    DEVICE = "Device"
    SERIAL_NUMBER = "Serial Number"
    DEVICE_TIMESTAMP = "Device Timestamp"
    RECORD_TYPE = "Record Type"
    HISTORIC_GLUCOSE = "Historic Glucose mmol/L"
    SCAN_GLUCOSE = "Scan Glucose mmol/L"
    NON_NUMERIC_RAPID_INSULIN = "Non-numeric Rapid-Acting Insulin"
    RAPID_INSULIN = "Rapid-Acting Insulin (units)"
    NON_NUMERIC_FOOD = "Non-numeric Food"
    CARBOHYDRATES_GRAMS = "Carbohydrates (grams)"
    CARBOHYDRATES_SERVINGS = "Carbohydrates (servings)"
    NON_NUMERIC_LONG_INSULIN = "Non-numeric Long-Acting Insulin"
    LONG_INSULIN = "Long-Acting Insulin Value (units)"
    NOTES = "Notes"
    STRIP_GLUCOSE = "Strip Glucose mmol/L"
    KETONE = "Ketone mmol/L"
    MEAL_INSULIN = "Meal Insulin (units)"
    CORRECTION_INSULIN = "Correction Insulin (units)"
    USER_CHANGE_INSULIN = "User Change Insulin (units)"
    HISTORIC_KETONE = "Historic Ketone mmol/L"
    SCAN_KETONE = "Scan Ketone mmol/L"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Get all column names in order."""
        return [
            cls.DEVICE, cls.SERIAL_NUMBER, cls.DEVICE_TIMESTAMP, cls.RECORD_TYPE,
            cls.HISTORIC_GLUCOSE, cls.SCAN_GLUCOSE, cls.NON_NUMERIC_RAPID_INSULIN,
            cls.RAPID_INSULIN, cls.NON_NUMERIC_FOOD, cls.CARBOHYDRATES_GRAMS,
            cls.CARBOHYDRATES_SERVINGS, cls.NON_NUMERIC_LONG_INSULIN, cls.LONG_INSULIN,
            cls.NOTES, cls.STRIP_GLUCOSE, cls.KETONE, cls.MEAL_INSULIN,
            cls.CORRECTION_INSULIN, cls.USER_CHANGE_INSULIN,
            cls.HISTORIC_KETONE, cls.SCAN_KETONE,
        ]


# =============================================================================
# EU Raw File Format Schema
# =============================================================================

# The EU export is standard Libre with three glucose columns relabeled to mmol/L
# and two ketone columns appended. Rather than re-declare every column, derive it
# from LIBRE_SCHEMA and patch only the real deltas. The mmol/L `unit` is what
# drives the parser's declarative mmol/L→mg/dL conversion (UNIT_CONVERSIONS);
# there is no EU-specific conversion code.
_APPENDED_KETONE_COLUMNS: tuple[ColumnSchema, ...] = (
    {
        "name": LibreEUColumn.HISTORIC_KETONE,
        "dtype": pl.Float64,
        "description": "Historic ketone level (LibreView mmol/L export)",
        "unit": "mmol/L",
        "constraints": {"minimum": 0},
    },
    {
        "name": LibreEUColumn.SCAN_KETONE,
        "dtype": pl.Float64,
        "description": "Scan ketone level (LibreView mmol/L export)",
        "unit": "mmol/L",
        "constraints": {"minimum": 0},
    },
)

LIBRE_EU_SCHEMA = derive_schema(
    LIBRE_SCHEMA,
    renames={
        LibreColumn.HISTORIC_GLUCOSE: LibreEUColumn.HISTORIC_GLUCOSE.value,
        LibreColumn.SCAN_GLUCOSE: LibreEUColumn.SCAN_GLUCOSE.value,
        LibreColumn.STRIP_GLUCOSE: LibreEUColumn.STRIP_GLUCOSE.value,
    },
    units={
        LibreEUColumn.HISTORIC_GLUCOSE.value: "mmol/L",
        LibreEUColumn.SCAN_GLUCOSE.value: "mmol/L",
        LibreEUColumn.STRIP_GLUCOSE.value: "mmol/L",
    },
    append_data_columns=_APPENDED_KETONE_COLUMNS,
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate libre_eu.json from the current schema definition."""
    _regenerate(LIBRE_EU_SCHEMA, __file__)
