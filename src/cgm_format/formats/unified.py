"""CGM Unified Format Definition.

This module defines the specific schema for the unified CGM data format,
using the base schema infrastructure from interface.schema.
"""

import json
from enum import Flag, auto
from pathlib import Path
from typing import Mapping, Union
import polars as pl
from cgm_format.interface.schema import (
    ColumnSchema,
    CGMSchemaDefinition,
    EnumLiteral,
    derive_schema,
    regenerate_schema_json as _regenerate_json,
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
    "%Y-%m-%dT%H:%M:%S%.f",  # ISO 8601 with milliseconds: 2024-05-01T12:30:45.000
    "%Y-%m-%dT%H:%M:%S",     # ISO 8601 without milliseconds: 2024-05-01T12:30:45
)

# Format detection patterns (unique identifiers in CSV headers/content)
UNIFIED_DETECTION_PATTERNS = [
    "sequence_id",      # Unique service column in unified format
    "event_type",       # Unique service column (lowercase with underscore)
    "quality",          # Unique service column
]

# The extended unified format shares its file geometry with the core one — it is
# the same CSV with more data columns.
UNIFIED_EXTENDED_HEADER_LINE = UNIFIED_HEADER_LINE
UNIFIED_EXTENDED_DATA_START_LINE = UNIFIED_DATA_START_LINE
UNIFIED_EXTENDED_METADATA_LINES = UNIFIED_METADATA_LINES
UNIFIED_EXTENDED_TIMESTAMP_FORMATS = UNIFIED_TIMESTAMP_FORMATS

# An extended round-trip CSV also carries every core unified pattern
# (`sequence_id`, `event_type`, `quality`), so UNIFIED_EXTENDED must be
# registered BEFORE UNIFIED_CGM and must key on something only it has. The
# `annotations` column is the last column of the extended header, and the
# leading comma anchors the match to a header field rather than to the word
# appearing inside somebody's free-text note.
UNIFIED_EXTENDED_DETECTION_PATTERNS = [
    ",annotations",
]

# =============================================================================
# Unit conversion (declarative)
# =============================================================================

# mmol/L → mg/dL conversion factor (standard clinical value).
MMOL_TO_MGDL = 18.0182

# Canonical unified glucose unit. Vendor parsers scale to this via
# FormatParser._glucose_to_canonical, which reads the column's declared unit
# from its schema and looks up the factor in UNIT_CONVERSIONS.
CANONICAL_GLUCOSE_UNIT = "mg/dL"

# Factors to scale a vendor column from its declared source unit to the
# canonical unified unit (glucose in mg/dL, exercise in seconds). A vendor/
# regional variant that only differs in units is then expressed purely as a
# `unit` in its (derived) schema — the parser reads the declared unit and
# applies the factor here, so no per-variant conversion code is needed.
# Keyed (source_unit, target_unit); equal units / missing pairs are a no-op.
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("mmol/L", "mg/dL"): MMOL_TO_MGDL,
    ("min", "s"): 60.0,
    ("h", "s"): 3600.0,
}

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
    IMPUTATION = "IMPUTATN"  # Imputed/interpolated data DEPRECATED!

class Quality(Flag):
    """Data quality indicator.

    Bitwise flags stored as Int64. Combine with ``|``, test with ``&``;
    ``Quality(0)`` (``GOOD_QUALITY``) means no flag is set.
    """

    OUT_OF_RANGE = auto()  # 1  — Out-of-range or flagged values
    SENSOR_CALIBRATION = auto()  # 2  — excluded 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD
    IMPUTATION = auto()  # 4  — Imputed/interpolated data
    TIME_DUPLICATE = auto()  # 8  — Event time is non-unique
    SYNCHRONIZATION = auto()  # 16 — Event time was synchronized
    # 32 — Value was synthesized by merging two concurrent sensor tracks (e.g.
    # the mean of a Libre and a Dexcom reading at the same timestamp). Distinct
    # from IMPUTATION: nothing was missing, but the number emitted is one no
    # single device produced. A row fed by exactly one sensor is that sensor's
    # reading and is NOT flagged.
    TRACK_MERGE = auto()

GOOD_QUALITY = Quality(0)


# =============================================================================
# Annotations — deterministic JSON serialization
# =============================================================================

# A value an annotation entry may carry. JSON scalars only: anything richer
# needs a typed column, not a stringly-typed escape hatch.
AnnotationValue = Union[str, int, float, bool, None]
AnnotationMapping = Mapping[str, AnnotationValue]


def annotations_to_json(annotations: AnnotationMapping | None) -> str | None:
    """Serialize an annotation mapping to the canonical `annotations` cell.

    This is the single serializer for the extended schema's `annotations`
    column, and it is load-bearing rather than cosmetic: `annotations` is a
    *data* column, so it participates in `primary_key` and in
    `get_stable_sort_keys()`. Two runs that emit different bytes for the same
    mapping would sort rows differently and break the byte-level round-trip and
    idempotency guarantees — flakily, which is worse than breaking them
    outright. Hence sorted keys, no insignificant whitespace, and the shortest
    round-trip float repr (Python's default).

    ``allow_nan=False`` is deliberate: the stdlib default emits bare ``NaN`` /
    ``Infinity``, which is not valid JSON and poisons every downstream reader.
    A non-finite annotation value is a caller bug and raises here.

    "No annotation" is ``None`` — never ``""`` and never ``"{}"``. Polars
    writes a null as an empty CSV field and reads an empty field back as null,
    so an empty-string cell would not survive a round-trip unchanged. An empty
    mapping means the caller had nothing to record, so it maps to ``None`` too.

    A key present with a ``None`` value is kept: that is "the source named this
    field and gave us nothing", which is a different statement from the key
    being absent, and the three-valued rule says not to collapse them.

    Args:
        annotations: Mapping of annotation key to JSON scalar, or None.

    Returns:
        A compact JSON object string, or None when there is nothing to record.

    Raises:
        ValueError: If a value is NaN or infinite.
    """
    if not annotations:
        return None
    return json.dumps(
        dict(annotations),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

# CGM Unified Format Schema
CGM_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": "sequence_id",
            "dtype": pl.Int64,
            "description": "Unique identifier for the data sequence",
            "constraints": {"required": True}
        },
        {
            "name": "original_datetime",
            "dtype": pl.Datetime('ms'),
            "description": "Original timestamp before any modifications (preserved from conversion)",
            "constraints": {"required": True}
        },
        {
            "name": "quality",
            "dtype": pl.Int64,
            "description": (
                "Data quality indicator — bitwise Quality flags stored as Int64 "
                "(0=GOOD_QUALITY, 1=OUT_OF_RANGE, 2=SENSOR_CALIBRATION, "
                "4=IMPUTATION, 8=TIME_DUPLICATE, 16=SYNCHRONIZATION, "
                "32=TRACK_MERGE; combine with |, test with &)"
            ),
            "constraints": {
                "required": True,
                "enum": [0] + [e.value for e in Quality]  # Include 0 for GOOD_QUALITY
            }
        },
        {
            "name": "event_type",
            "dtype": pl.Utf8,  # Enum as string in Polars
            "description": "Type of recorded event (8-char code mapping to Dexcom EVENT_TYPE+SUBTYPE)",
            "constraints": {
                "required": True,
                "enum": [e.value for e in UnifiedEventType]
            }
        }
    ),
    data_columns=(
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
            # "s", not "seconds": UNIT_CONVERSIONS is keyed ("min","s") /
            # ("h","s"), so the long spelling made every lookup a silent no-op.
            "unit": "s",
            "constraints": {"minimum": 0}
        },
    ),
    header_line=UNIFIED_HEADER_LINE,
    data_start_line=UNIFIED_DATA_START_LINE,
    metadata_lines=UNIFIED_METADATA_LINES,
    # Primary key: All data columns (service columns are metadata)
    # Rows with identical data values are true duplicates
    primary_key=("datetime", "glucose", "carbs", "insulin_slow", "insulin_fast", "exercise")
)

# =============================================================================
# Extended Unified Format Schema
# =============================================================================

# Data columns the extended schema appends to the core six. The order below IS
# the widened total ordering: `get_stable_sort_keys()` returns every column in
# schema order, so appending here changes how extended frames sort. Grouped
# food -> wearable -> analyte, with `annotations` last.
#
# `annotations` is a DATA column, not a service column. That is deliberate: it
# therefore joins `primary_key` and the sort keys, which is what keeps two
# annotation-only rows sharing a timestamp distinguishable from each other. The
# cost is that it shows up in `get_polars_schema(data_only=True)`;
# `FormatProcessor.to_core_df()` is the narrowing escape hatch for callers who
# want the core six back.
EXTENDED_DATA_COLUMNS: tuple[ColumnSchema, ...] = (
    # --- food: the macronutrient decomposition of a logged meal ---
    {
        "name": "calories",
        "dtype": pl.Float64,
        "description": "Energy content of a logged meal",
        "unit": "kcal",
        "constraints": {"minimum": 0},
    },
    {
        "name": "protein",
        "dtype": pl.Float64,
        "description": "Protein content of a logged meal",
        "unit": "g",
        "constraints": {"minimum": 0},
    },
    {
        "name": "fat",
        "dtype": pl.Float64,
        "description": "Fat content of a logged meal",
        "unit": "g",
        "constraints": {"minimum": 0},
    },
    {
        "name": "fiber",
        "dtype": pl.Float64,
        "description": "Dietary fibre content of a logged meal",
        "unit": "g",
        "constraints": {"minimum": 0},
    },
    # --- wearable: streams a fitness tracker or chest strap reports ---
    {
        "name": "heart_rate",
        "dtype": pl.Float64,
        "description": "Heart rate reported by a wearable device",
        "unit": "bpm",
        "constraints": {"minimum": 0},
    },
    {
        "name": "breathing_rate",
        "dtype": pl.Float64,
        "description": "Breathing rate reported by a wearable device",
        "unit": "breaths/min",
        "constraints": {"minimum": 0},
    },
    {
        "name": "acceleration",
        "dtype": pl.Float64,
        "description": "Accelerometer magnitude reported by a wearable device",
        "unit": "g",
        "constraints": {"minimum": 0},
    },
    {
        "name": "mets",
        "dtype": pl.Float64,
        "description": "Metabolic equivalent of task reported by a wearable device",
        "unit": "MET",
        "constraints": {"minimum": 0},
    },
    {
        "name": "activity_calories",
        "dtype": pl.Float64,
        "description": "Energy expenditure reported by a wearable device",
        "unit": "kcal",
        "constraints": {"minimum": 0},
    },
    {
        "name": "steps",
        "dtype": pl.Int64,
        "description": "Step count reported by a wearable device",
        "unit": "count",
        "constraints": {"minimum": 0},
    },
    # --- analyte: a second measured analyte beside glucose ---
    {
        "name": "ketones",
        "dtype": pl.Float64,
        "description": "Blood ketone (beta-hydroxybutyrate) reading",
        # Clinical ketones are reported in mmol/L worldwide, so mmol/L IS the
        # canonical unit here. This column is deliberately NOT routed through
        # FormatParser._glucose_to_canonical: that helper converts to the
        # canonical *glucose* unit, and applying one analyte's convention to
        # another would silently multiply every ketone value by 18.0182.
        "unit": "mmol/L",
        "constraints": {"minimum": 0},
    },
    # --- annotations: last, always ---
    {
        "name": "annotations",
        "dtype": pl.Utf8,
        "description": (
            "JSON object holding source detail with no typed home (meal photo "
            "path, free-text description, the raw vendor column a value came "
            "from). Serialize with annotations_to_json(); null when absent"
        ),
    },
)

# The extended target schema, derived from the core one. CGM_SCHEMA above is
# NOT modified — the extended schema is an opt-in target, so every existing
# consumer keeps the frame it has today.
#
# primary_key is passed explicitly: derive_schema does not widen it when columns
# are appended, and leaving it at the core six would call two rows true
# duplicates while their macros, wearable streams or annotations differ.
CGM_SCHEMA_EXTENDED = derive_schema(
    CGM_SCHEMA,
    append_data_columns=EXTENDED_DATA_COLUMNS,
    primary_key=(
        "datetime", "glucose", "carbs", "insulin_slow", "insulin_fast", "exercise",
        "calories", "protein", "fat", "fiber",
        "heart_rate", "breathing_rate", "acceleration", "mets",
        "activity_calories", "steps",
        "ketones",
        "annotations",
    ),
)


# =============================================================================
# Schema JSON Export Helper
# =============================================================================

def regenerate_schema_json() -> None:
    """Regenerate unified.json from the current schema definition.

    Run this after modifying enums or schema to keep unified.json in sync:
        python3 -c "from formats.unified import regenerate_schema_json; regenerate_schema_json()"
    """
    _regenerate_json(CGM_SCHEMA, __file__)


def regenerate_extended_schema_json() -> None:
    """Regenerate unified_extended.json from CGM_SCHEMA_EXTENDED.

    Kept separate from regenerate_schema_json() because this module declares two
    schemas but the batch regenerator derives one JSON filename per module.
    """
    sibling = Path(__file__).with_name("unified_extended.py")
    _regenerate_json(CGM_SCHEMA_EXTENDED, str(sibling))

