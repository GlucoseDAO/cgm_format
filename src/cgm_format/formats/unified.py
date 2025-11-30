"""CGM Unified Format Definition.

This module defines the specific schema for the unified CGM data format,
using the base schema infrastructure from interface.schema.
"""

from enum import Flag, auto
import polars as pl
from cgm_format.interface.schema import (
    ColumnSchema,
    CGMSchemaDefinition,
    EnumLiteral,
)

# TODO: support truncated form of UnifiedFormat without service columns

# =============================================================================
# File Format Constants
# =============================================================================

# File structure: Row 1 = header, Row 2+ = data (standard CSV format)
UNIFIED_HEADER_LINE = 1
UNIFIED_DATA_START_LINE = 2
UNIFIED_METADATA_LINES = ()  # No metadata lines to skip

# Multiple timestamp formats for unified format (tuple for consistency)
UNIFIED_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",  # ISO 8601: 2024-05-01T12:30:45
)

# Format detection patterns (unique identifiers in CSV headers/content)
UNIFIED_DETECTION_PATTERNS = [
    "sequence_id",      # Unique service column in unified format
    "event_type",       # Unique service column (lowercase with underscore)
    "quality",          # Unique service column
]

# =============================================================================
# Unified Event Type Enums
# =============================================================================

class UnifiedEventType(EnumLiteral):
    """Type of recorded event in CGM data.
    
    Each event type has an 7-character code that uniquely identifies it.
    These codes map to Dexcom EVENT_TYPE+SUBTYPE combinations.
    """
    # Core glucose readings
    GLUCOSE = "EGV_READ"  # Normal CGM value (Estimated Glucose Value)
    # GLUCOSE_HIGH = "HIGH_EGV"  # High glucose reading replaced by value + ILL Quality
    # GLUCOSE_LOW = "LOW_EGV"  # Low glucose reading replaced by value + ILL Quality
    
    # Calibration
    CALIBRATION = "CALIBRAT"  # Sensor calibration event
    
    # Carbohydrates
    CARBOHYDRATES = "CARBS_IN"  # Carbohydrate intake
    
    # Insulin
    INSULIN_FAST = "INS_FAST"  # Fast-acting (bolus) insulin
    INSULIN_SLOW = "INS_SLOW"  # Long-acting (basal) insulin
    
    # Exercise
    EXERCISE_LIGHT = "XRCS_LTE"  # Light exercise
    EXERCISE_MEDIUM = "XRCS_MED"  # Medium exercise
    EXERCISE_HEAVY = "XRCS_HVY"  # Heavy exercise
    
    # Alerts
    ALERT_HIGH = "ALRT_HIG"  # High glucose alert
    ALERT_LOW = "ALRT_LOG"  # Low glucose alert
    ALERT_URGENT_LOW = "ALRT_ULG"  # Urgent low glucose alert
    ALERT_URGENT_LOW_SOON = "ALRT_ULS"  # Urgent low soon alert
    ALERT_RISE = "ALRT_RIS"  # Rapid rise alert
    ALERT_FALL = "ALRT_FAL"  # Rapid fall alert
    ALERT_SIGNAL_LOSS = "ALRT_SIG"  # Signal loss alert
    
    # Health events
    HEALTH_ILLNESS = "HLTH_ILL"  # Illness
    HEALTH_STRESS = "HLTH_STR"  # Stress
    HEALTH_LOW_SYMPTOMS = "HLTH_LSY"  # Low symptoms
    HEALTH_CYCLE = "HLTH_CYC"  # Menstrual cycle
    HEALTH_ALCOHOL = "HLTH_ALC"  # Alcohol consumption
    
    # System events
    OTHER = "OTHEREVT"  # Other/unknown event type


class Quality(Flag):
    """Data quality indicator."""

    OUT_OF_RANGE = auto()  # Out-of-range or flagged values
    SENSOR_CALIBRATION = auto()  # excluded 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD
    IMPUTATION = auto()  # Imputed/interpolated data
    TIME_DUPLICATE = auto()  # Event time is non-unique

GOOD_QUALITY = Quality(0)

# CGM Unified Format Schema
CGM_SCHEMA = CGMSchemaDefinition(
    service_columns=[
        {
            "name": "sequence_id",
            "dtype": pl.Int64,
            "description": "Unique identifier for the data sequence",
            "constraints": {"required": True}
        },
        {
            "name": "event_type",
            "dtype": pl.Utf8,  # Enum as string in Polars
            "description": "Type of recorded event (8-char code mapping to Dexcom EVENT_TYPE+SUBTYPE)",
            "constraints": {
                "required": True,
                "enum": [e.value for e in UnifiedEventType]
            }
        },
        {
            "name": "quality",
            "dtype": pl.Int64,
            "description": "Data quality indicator (0=GOOD, 1=ILL, 2=SENSOR_CALIBRATION)",
            "constraints": {
                "required": True,
                "enum": [e.value for e in Quality]
            }
        },
    ],
    data_columns=[
        {
            "name": "datetime",
            "dtype": pl.Datetime('ms'),
            "description": "Timestamp of the event in ISO 8601 format",
            "constraints": {"required": True}
        },
        {
            "name": "glucose",
            "dtype": pl.Float64,
            "description": "Blood glucose reading from CGM sensor",
            "unit": "mg/dL",
            "constraints": {"minimum": 0}
        },
        {
            "name": "carbs",
            "dtype": pl.Float64,
            "description": "Carbohydrate intake",
            "unit": "g",
            "constraints": {"minimum": 0}
        },
        {
            "name": "insulin_slow",
            "dtype": pl.Float64,
            "description": "Long-acting (basal) insulin dose",
            "unit": "u",
            "constraints": {"minimum": 0}
        },
        {
            "name": "insulin_fast",
            "dtype": pl.Float64,
            "description": "Short-acting (bolus) insulin dose",
            "unit": "u",
            "constraints": {"minimum": 0}
        },
        {
            "name": "exercise",
            "dtype": pl.Int64,
            "description": "Duration of exercise activity",
            "unit": "seconds",
            "constraints": {"minimum": 0}
        },
    ],
    header_line=UNIFIED_HEADER_LINE,
    data_start_line=UNIFIED_DATA_START_LINE,
    metadata_lines=UNIFIED_METADATA_LINES,
    # Primary key: All data columns (service columns are metadata)
    # Rows with identical data values are true duplicates
    primary_key=["datetime", "glucose", "carbs", "insulin_slow", "insulin_fast", "exercise"]
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate unified.json from the current schema definition.
    
    Run this after modifying enums or schema to keep unified.json in sync:
        python3 -c "from formats.unified import regenerate_schema_json; regenerate_schema_json()"
    """
    from cgm_format.interface.schema import regenerate_schema_json as _regenerate
    _regenerate(CGM_SCHEMA, __file__)

