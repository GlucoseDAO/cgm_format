"""Nightscout CGM data format schema definition.

Nightscout is an open-source cloud-based CGM data platform. Unlike vendor CSV exports,
Nightscout data comes from a REST API and is natively JSON (though CSV exports are also
supported). Data is split across two endpoints:

- **Entries** (`/api/v1/entries.json`): Sensor glucose values (SGV) with direction trends
- **Treatments** (`/api/v1/treatments.json`): Insulin, carbs, temp basals, site changes, etc.

Entries CSV example:
    dateString,date,sgv,direction,type,glucose,unfiltered,uncalibrated,sessionStartDate,utcOffset
    2026-03-31T19:51:03.000Z,1774986663000,202,Flat,sgv,202,202,202,,0

Treatments CSV example:
    created_at,eventType,insulin,carbs,duration,rate,absolute,enteredBy,utcOffset,mills,notes
    2026-03-31T20:12:21.726Z,SMB,0.1,,0,,,iAPS,0,,

Entries JSON example:
    {"sgv": 202, "dateString": "2026-03-31T19:51:03.000Z", "direction": "Flat", "type": "sgv", ...}

Treatments JSON example:
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

NIGHTSCOUT_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.fZ",   # ISO 8601 with millis + Z: 2026-03-31T19:51:03.000Z
    "%Y-%m-%dT%H:%M:%SZ",      # ISO 8601 without millis + Z: 2026-03-31T19:51:03Z
    "%Y-%m-%dT%H:%M:%S%.f",    # ISO 8601 with millis, no Z
    "%Y-%m-%dT%H:%M:%S",       # ISO 8601 plain
)

NIGHTSCOUT_ENTRIES_DETECTION_PATTERNS: list[str] = [
    "dateString,date,sgv",          # CSV entries header
    "\"sgv\"",                       # JSON entries key
    "direction,type,glucose",        # CSV entries header fragment
]

NIGHTSCOUT_TREATMENTS_DETECTION_PATTERNS: list[str] = [
    "created_at,eventType,insulin",  # CSV treatments header
    "\"eventType\"",                  # JSON treatments key
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
    """Column names in Nightscout entries CSV exports."""
    DATE_STRING = "dateString"
    DATE = "date"
    SGV = "sgv"
    DIRECTION = "direction"
    TYPE = "type"
    GLUCOSE = "glucose"
    UNFILTERED = "unfiltered"
    UNCALIBRATED = "uncalibrated"
    SESSION_START_DATE = "sessionStartDate"
    UTC_OFFSET = "utcOffset"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        return [
            cls.DATE_STRING, cls.DATE, cls.SGV, cls.DIRECTION, cls.TYPE,
            cls.GLUCOSE, cls.UNFILTERED, cls.UNCALIBRATED,
            cls.SESSION_START_DATE, cls.UTC_OFFSET,
        ]


class NightscoutTreatmentColumn(EnumLiteral):
    """Column names in Nightscout treatments CSV exports."""
    CREATED_AT = "created_at"
    EVENT_TYPE = "eventType"
    INSULIN = "insulin"
    CARBS = "carbs"
    DURATION = "duration"
    RATE = "rate"
    ABSOLUTE = "absolute"
    ENTERED_BY = "enteredBy"
    UTC_OFFSET = "utcOffset"
    MILLS = "mills"
    NOTES = "notes"

    @classmethod
    def get_all_columns(cls) -> List[str]:
        return [
            cls.CREATED_AT, cls.EVENT_TYPE, cls.INSULIN, cls.CARBS,
            cls.DURATION, cls.RATE, cls.ABSOLUTE, cls.ENTERED_BY,
            cls.UTC_OFFSET, cls.MILLS, cls.NOTES,
        ]


# =============================================================================
# Nightscout Raw File Format Schemas
# =============================================================================

NIGHTSCOUT_ENTRIES_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": NightscoutEntryColumn.DATE_STRING,
            "dtype": pl.Utf8,
            "description": "ISO 8601 timestamp string",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutEntryColumn.DATE,
            "dtype": pl.Int64,
            "description": "Unix epoch timestamp in milliseconds",
            "constraints": {"required": True},
        },
    ),
    data_columns=(
        {
            "name": NightscoutEntryColumn.SGV,
            "dtype": pl.Int64,
            "description": "Sensor glucose value",
            "unit": "mg/dL",
            "constraints": {"minimum": 0},
        },
        {
            "name": NightscoutEntryColumn.DIRECTION,
            "dtype": pl.Utf8,
            "description": "Glucose trend direction (Flat, SingleUp, etc.)",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.TYPE,
            "dtype": pl.Utf8,
            "description": "Entry type (sgv, mbg, cal, etc.)",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.GLUCOSE,
            "dtype": pl.Int64,
            "description": "Glucose value (alias for sgv)",
            "unit": "mg/dL",
            "constraints": {"minimum": 0},
        },
        {
            "name": NightscoutEntryColumn.UNFILTERED,
            "dtype": pl.Int64,
            "description": "Unfiltered raw sensor value",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.UNCALIBRATED,
            "dtype": pl.Int64,
            "description": "Uncalibrated raw sensor value",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.SESSION_START_DATE,
            "dtype": pl.Utf8,
            "description": "Sensor session start date",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutEntryColumn.UTC_OFFSET,
            "dtype": pl.Int64,
            "description": "UTC offset in minutes",
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
            "name": NightscoutTreatmentColumn.CREATED_AT,
            "dtype": pl.Utf8,
            "description": "ISO 8601 timestamp of the treatment",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutTreatmentColumn.EVENT_TYPE,
            "dtype": pl.Utf8,
            "description": "Type of treatment event",
            "constraints": {"required": True},
        },
        {
            "name": NightscoutTreatmentColumn.ENTERED_BY,
            "dtype": pl.Utf8,
            "description": "Source system (e.g. iAPS, Loop)",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.UTC_OFFSET,
            "dtype": pl.Int64,
            "description": "UTC offset in minutes",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.MILLS,
            "dtype": pl.Int64,
            "description": "Unix epoch timestamp in milliseconds",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.NOTES,
            "dtype": pl.Utf8,
            "description": "Treatment notes",
            "constraints": {"required": False},
        },
    ),
    data_columns=(
        {
            "name": NightscoutTreatmentColumn.INSULIN,
            "dtype": pl.Float64,
            "description": "Insulin dose",
            "unit": "U",
            "constraints": {"minimum": 0},
        },
        {
            "name": NightscoutTreatmentColumn.CARBS,
            "dtype": pl.Float64,
            "description": "Carbohydrate intake",
            "unit": "grams",
            "constraints": {"minimum": 0},
        },
        {
            "name": NightscoutTreatmentColumn.DURATION,
            "dtype": pl.Float64,
            "description": "Duration of treatment in minutes",
            "unit": "minutes",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.RATE,
            "dtype": pl.Float64,
            "description": "Basal rate",
            "unit": "U/h",
            "constraints": {"required": False},
        },
        {
            "name": NightscoutTreatmentColumn.ABSOLUTE,
            "dtype": pl.Float64,
            "description": "Absolute basal rate",
            "unit": "U/h",
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
