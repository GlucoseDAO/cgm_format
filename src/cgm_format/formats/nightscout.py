"""Nightscout CGM data format schema definition.

Nightscout is an open-source cloud-based CGM data platform.  Data can be
obtained in two ways:

1. **REST API** (``/api/v1/entries.json``, ``/api/v1/treatments.json``): native
   JSON, handled by :meth:`~cgm_format.format_parser.FormatParser.parse_nightscout`
   and its wrappers.

2. **nightscout-exporter CSV** (community tool): a combined CSV file with two
   sections separated by ``#`` comment lines.  This is what the schema and
   ``detect_format`` pipeline target.

.. note::
   The built-in Nightscout ``/api/v1/entries.csv`` endpoint is *defective*:
   it returns headerless CSV with only 5 hard-coded columns.  The treatments
   endpoint does not support CSV at all.  We therefore **do not** support the
   Nightscout API CSV — use the JSON API or the nightscout-exporter CSV.

nightscout-exporter CSV example::

    # CGM ENTRIES
    Date,Time,Glucose (mg/dL),Type,Device,Trend,ID
    "3/31/2026","7:51:03 PM","202","sgv","","4","abc123"
    ...

    # TREATMENTS (Insulin, Carbs, Exercise)
    Date,Time,Event Type,Insulin (U),Carbs (g),Notes,ID
    "3/31/2026","8:12:21 PM","SMB","0.1","","","def456"
    ...

Entries JSON example::

    {"sgv": 202, "dateString": "2026-03-31T19:51:03.000Z", "direction": "Flat", "type": "sgv", ...}

Treatments JSON example::

    {"eventType": "SMB", "insulin": 0.1, "created_at": "2026-03-31T20:12:21.726Z", ...}
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

NIGHTSCOUT_HEADER_LINE = 1
NIGHTSCOUT_DATA_START_LINE = 2
NIGHTSCOUT_METADATA_LINES = ()

# Timestamp formats used by the JSON API (ISO 8601 variants)
NIGHTSCOUT_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.fZ",   # ISO 8601 with millis + Z: 2026-03-31T19:51:03.000Z
    "%Y-%m-%dT%H:%M:%SZ",      # ISO 8601 without millis + Z: 2026-03-31T19:51:03Z
    "%Y-%m-%dT%H:%M:%S%.f",    # ISO 8601 with millis, no Z
    "%Y-%m-%dT%H:%M:%S",       # ISO 8601 plain
)

# Detection patterns for nightscout-exporter CSV files
NIGHTSCOUT_ENTRIES_DETECTION_PATTERNS: list[str] = [
    "# CGM ENTRIES",                    # nightscout-exporter section header
    "Date,Time,Glucose (mg/dL)",        # nightscout-exporter entries header
]

NIGHTSCOUT_TREATMENTS_DETECTION_PATTERNS: list[str] = [
    "# TREATMENTS",                     # nightscout-exporter section header
    "Date,Time,Event Type,Insulin",     # nightscout-exporter treatments header
]

NIGHTSCOUT_DETECTION_PATTERNS: list[str] = NIGHTSCOUT_ENTRIES_DETECTION_PATTERNS


# =============================================================================
# Nightscout Event / Direction Enums
# =============================================================================

class NightscoutTreatmentEventType(EnumLiteral):
    """Treatment event types from Nightscout treatments endpoint."""
    BOLUS = "Bolus"
    SMB = "SMB"
    TEMP_BASAL = "Temp Basal"
    CARB_CORRECTION = "Carb Correction"
    MEAL_BOLUS = "Meal Bolus"
    CORRECTION_BOLUS = "Correction Bolus"
    SENSOR_START = "Sensor Start"
    SITE_CHANGE = "Site Change"
    INSULIN_CHANGE = "Insulin Change"
    TEMPORARY_TARGET = "Temporary Target"
    NOTE = "Note"
    PROFILE_SWITCH = "Profile Switch"
    ANNOUNCEMENT = "Announcement"


class NightscoutDirection(EnumLiteral):
    """Glucose trend direction from Nightscout entries."""
    NONE = "NONE"
    DOUBLE_UP = "DoubleUp"
    SINGLE_UP = "SingleUp"
    FORTY_FIVE_UP = "FortyFiveUp"
    FLAT = "Flat"
    FORTY_FIVE_DOWN = "FortyFiveDown"
    SINGLE_DOWN = "SingleDown"
    DOUBLE_DOWN = "DoubleDown"
    NOT_COMPUTABLE = "NOT COMPUTABLE"
    RATE_OUT_OF_RANGE = "RATE OUT OF RANGE"


# =============================================================================
# Raw Column Names
# =============================================================================

class NightscoutEntryColumn(EnumLiteral):
    """Column names for Nightscout entries.

    Two sets of names are defined:

    - **Exporter CSV** columns (``Date``, ``Time``, ``Glucose (mg/dL)``, …) —
      used by the nightscout-exporter CSV format and the frictionless schema.
    - **JSON API** field names (``dateString``, ``sgv``, …) — used by the
      JSON parser internally.
    """
    # nightscout-exporter CSV columns
    DATE = "Date"
    TIME = "Time"
    GLUCOSE_MGDL = "Glucose (mg/dL)"
    TYPE = "Type"
    DEVICE = "Device"
    TREND = "Trend"
    ID = "ID"

    # JSON API field names (not part of the CSV schema)
    DATE_STRING = "dateString"   # JSON: ISO 8601 timestamp
    DATE_EPOCH = "date"          # JSON: epoch millis
    SGV = "sgv"                  # JSON: sensor glucose value
    DIRECTION = "direction"      # JSON: trend direction string

    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Columns in nightscout-exporter CSV entries section."""
        return [cls.DATE, cls.TIME, cls.GLUCOSE_MGDL, cls.TYPE, cls.DEVICE, cls.TREND, cls.ID]

    @classmethod
    def get_all_columns(cls) -> List[str]:
        return cls.get_csv_columns()


class NightscoutTreatmentColumn(EnumLiteral):
    """Column names for Nightscout treatments.

    Two sets: exporter CSV columns and JSON API field names.
    """
    # nightscout-exporter CSV columns
    DATE = "Date"
    TIME = "Time"
    EVENT_TYPE = "Event Type"
    INSULIN_U = "Insulin (U)"
    CARBS_G = "Carbs (g)"
    NOTES = "Notes"
    ID = "ID"

    # JSON API field names
    CREATED_AT = "created_at"
    EVENT_TYPE_JSON = "eventType"
    INSULIN = "insulin"
    CARBS = "carbs"
    DURATION = "duration"
    RATE = "rate"
    ABSOLUTE = "absolute"
    ENTERED_BY = "enteredBy"
    UTC_OFFSET = "utcOffset"
    MILLS = "mills"

    @classmethod
    def get_csv_columns(cls) -> List[str]:
        """Columns in nightscout-exporter CSV treatments section."""
        return [cls.DATE, cls.TIME, cls.EVENT_TYPE, cls.INSULIN_U, cls.CARBS_G, cls.NOTES, cls.ID]

    @classmethod
    def get_all_columns(cls) -> List[str]:
        return cls.get_csv_columns()


# =============================================================================
# Nightscout Raw File Format Schemas
# =============================================================================

NIGHTSCOUT_ENTRIES_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": NightscoutEntryColumn.DATE,
            "dtype": pl.Utf8,
            "description": "Locale-formatted date string (e.g. 3/31/2026)",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutEntryColumn.TIME,
            "dtype": pl.Utf8,
            "description": "Locale-formatted time string (e.g. 7:51:03 PM)",
            "constraints": {"required": True},
        },
    ),
    data_columns=(
        {
            "name": NightscoutEntryColumn.GLUCOSE_MGDL,
            "dtype": pl.Utf8,
            "description": "Sensor glucose value (string from exporter, cast to int during parsing)",
            "unit": "mg/dL",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.TYPE,
            "dtype": pl.Utf8,
            "description": "Entry type (sgv, mbg, cal, etc.)",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.DEVICE,
            "dtype": pl.Utf8,
            "description": "Source device identifier",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.TREND,
            "dtype": pl.Utf8,
            "description": "Trend direction (numeric string from exporter)",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.ID,
            "dtype": pl.Utf8,
            "description": "Nightscout entry _id",
            "constraints": {"required": False},
        },
    ),
    header_line=NIGHTSCOUT_HEADER_LINE,
    data_start_line=NIGHTSCOUT_DATA_START_LINE,
    metadata_lines=NIGHTSCOUT_METADATA_LINES,
)

NIGHTSCOUT_TREATMENTS_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": NightscoutTreatmentColumn.DATE,
            "dtype": pl.Utf8,
            "description": "Locale-formatted date string",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutTreatmentColumn.TIME,
            "dtype": pl.Utf8,
            "description": "Locale-formatted time string",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutTreatmentColumn.EVENT_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of treatment event (SMB, Bolus, Temp Basal, etc.)",
            "constraints": {"required": True},
        },
    ),
    data_columns=(
        {
            "name": NightscoutTreatmentColumn.INSULIN_U,
            "dtype": pl.Utf8,
            "description": "Insulin dose (string from exporter, cast to float during parsing)",
            "unit": "U",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.CARBS_G,
            "dtype": pl.Utf8,
            "description": "Carbohydrate intake (string from exporter)",
            "unit": "grams",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.NOTES,
            "dtype": pl.Utf8,
            "description": "Treatment notes",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.ID,
            "dtype": pl.Utf8,
            "description": "Nightscout treatment _id",
            "constraints": {"required": False},
        },
    ),
    header_line=NIGHTSCOUT_HEADER_LINE,
    data_start_line=NIGHTSCOUT_DATA_START_LINE,
    metadata_lines=NIGHTSCOUT_METADATA_LINES,
)


def regenerate_schema_json() -> None:
    """Regenerate nightscout.json from the current schema definition."""
    from cgm_format.interface.schema import regenerate_schema_json as _regenerate
    _regenerate(NIGHTSCOUT_ENTRIES_SCHEMA, __file__)
