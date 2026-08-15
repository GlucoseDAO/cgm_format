# CGM Unified Format Specification

This document defines the unified data format used to standardize CGM data across different vendors (Dexcom, Libre, etc.).

> For information about the processing pipeline that produces this format, see `PIPELINE.md`

## Overview

The output is a **Polars DataFrame** with strict schema constraints.

> **Schema Definition:** The authoritative schema is defined in `formats/unified.py` using the `CGMSchemaDefinition` class. All data types listed below are Polars types (e.g., `Int64`, `Float64`, `Datetime`, `Utf8`).

### Service Columns

| Column | Type | Description |
|--------|------|-------------|
| `sequence_id` | `Int64` | Unique identifier for the data sequence |
| `original_datetime` | `Datetime` | Original timestamp before any modifications (preserved from conversion) |
| `quality` | `Int64` | Data quality indicator (bitwise flags, 0=GOOD) |
| `event_type` | `Utf8` | Type of recorded event (8-char code mapping to Dexcom EVENT_TYPE+SUBTYPE) |

#### Event Type Enum

Each event type uses a 7-8 character code:

**Core Glucose Readings:**

- `EGV_READ` - Normal CGM value (Estimated Glucose Value). Libre historic (Record Type 0) and
  scan (Record Type 1) readings both map here.

**Calibration:**

- `CALIBRAT` - Sensor calibration event. Libre strip / finger-prick readings (Record Type 2) map here.

**Carbohydrates:**

- `CARBS_IN` - Carbohydrate intake

**Insulin:**

- `INS_FAST` - Fast-acting (bolus) insulin
- `INS_SLOW` - Long-acting (basal) insulin

**Exercise:**

- `XRCS_LTE` - Light exercise
- `XRCS_MED` - Medium exercise
- `XRCS_HVY` - Heavy exercise

**Alerts:**

- `ALRT_HIG` - High glucose alert
- `ALRT_LOG` - Low glucose alert
- `ALRT_ULG` - Urgent low glucose alert
- `ALRT_ULS` - Urgent low soon alert
- `ALRT_RIS` - Rapid rise alert
- `ALRT_FAL` - Rapid fall alert
- `ALRT_SIG` - Signal loss alert

**Health Events:**

- `HLTH_ILL` - Illness
- `HLTH_STR` - Stress
- `HLTH_LSY` - Low symptoms
- `HLTH_CYC` - Menstrual cycle
- `HLTH_ALC` - Alcohol consumption

**System Events:**

- `IMPUTATN` - Imputed/interpolated data (deprecated - use quality flag instead)
- `OTHEREVT` - Other/unknown event type

#### Quality Flags

The quality field uses bitwise flags (Python `Flag` enum) to indicate data issues:

- `0` = GOOD_QUALITY - Valid, high-quality data (no flags set)
- `1` = OUT_OF_RANGE - Out-of-range or flagged values
- `2` = SENSOR_CALIBRATION - 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD
- `4` = IMPUTATION - Imputed/interpolated data
- `8` = TIME_DUPLICATE - Event time is non-unique
- `16` = SYNCHRONIZATION - Event time was synchronized

Multiple flags can be combined (e.g., `3` = OUT_OF_RANGE | SENSOR_CALIBRATION).

### Data Columns

The following columns are passed to the LLM:

| Column | Type | Unit | Description | Constraints |
|--------|------|------|-------------|-------------|
| `datetime` | `Datetime` | - | Timestamp of the event in ISO 8601 format | Required |
| `glucose` | `Float64` | mg/dL | Blood glucose reading from CGM sensor | ≥ 0 |
| `carbs` | `Float64` | g | Carbohydrate intake | ≥ 0 |
| `insulin_slow` | `Float64` | u | Long-acting (basal) insulin dose | ≥ 0 |
| `insulin_fast` | `Float64` | u | Short-acting (bolus) insulin dose | ≥ 0 |
| `exercise` | `Int64` | seconds | Duration of exercise activity | ≥ 0 |

### Primary Key

The schema defines a primary key consisting of all data columns:
- `(datetime, glucose, carbs, insulin_slow, insulin_fast, exercise)`

Rows with identical data values across these columns are considered true duplicates. Service columns (`sequence_id`, `original_datetime`, `quality`, `event_type`) are metadata and not part of the primary key.

### Stable Sorting

For deterministic row ordering, the schema uses all columns in priority order:
1. `sequence_id` - Group by sequence
2. `original_datetime` - Temporal order (preserves original timing)
3. `quality` - Clean data first (0 = no flags)
4. `event_type` - Consistent event ordering
5. All data columns - Final tiebreaker for identical events

This ensures completely deterministic ordering even when multiple events have the same timestamp, quality, and type.

## Schema Usage

The schema is implemented using the `CGMSchemaDefinition` class from `interface/schema.py`, which provides:

- **Polars schema dictionary**: `CGM_SCHEMA.get_polars_schema(data_only=False)`
- **Column names list**: `CGM_SCHEMA.get_column_names(data_only=False)`
- **Cast expressions**: `CGM_SCHEMA.get_cast_expressions(data_only=False)`
- **Inference schema**: `CGM_SCHEMA.get_inference_schema()` - Returns schema with only data columns (for ML)
- **Stable sort keys**: `CGM_SCHEMA.get_stable_sort_keys()` - Returns all columns for deterministic sorting
- **Frictionless Data export**: `CGM_SCHEMA.to_frictionless_schema()`
- **Schema validation**: `CGM_SCHEMA.validate_dataframe(df, enforce=False)`
- **Schema enforcement**: `CGM_SCHEMA.validate_dataframe(df, enforce=True)`

Set `data_only=True` to work with only the data columns (excluding service columns).

### Schema Validation

The schema system provides two modes for working with DataFrames:

**Validation Mode** (`enforce=False`):
- Checks that all expected columns are present in the correct order
- Verifies that column types match the schema exactly
- Raises errors if schema doesn't match:
  - `MissingColumnError` - Required column is missing
  - `ExtraColumnError` - Unexpected column present
  - `ColumnOrderError` - Columns not in schema order
  - `ColumnTypeError` - Column type doesn't match schema

**Enforcement Mode** (`enforce=True`):
- Adds missing columns with null values (e.g., `original_datetime`)
- Removes extra columns not in schema
- Casts columns to correct types (strict for most types, non-strict for numeric types to handle nulls)
- Reorders columns to match schema
- Applies stable sorting using `get_stable_sort_keys()` for deterministic row ordering

Example:
```python
from cgm_format.formats.unified import CGM_SCHEMA

# Validate that DataFrame matches schema (strict)
validated_df = CGM_SCHEMA.validate_dataframe(df, enforce=False)

# Enforce schema on DataFrame (add missing, cast types, reorder)
enforced_df = CGM_SCHEMA.validate_dataframe(df, enforce=True)
```

### Regenerating Schema JSON

To regenerate `unified.json` after modifying the schema:

```python
python3 -c "from formats.unified import regenerate_schema_json; regenerate_schema_json()"
```

## The Extended Schema

`CGM_SCHEMA_EXTENDED` (0.10.0) is `CGM_SCHEMA` plus the channels research corpora carry and the six
core data columns have no home for. It sits **beside** the core schema, which is unchanged — an
existing consumer sees exactly the frame it saw before.

| Group | Columns | Unit |
|---|---|---|
| Food | `calories`, `protein`, `fat`, `fiber` | kcal, g, g, g |
| Wearable | `heart_rate`, `breathing_rate`, `acceleration`, `mets`, `activity_calories`, `steps` | bpm, breaths/min, g, —, kcal, count |
| Analyte | `ketones` | mmol/L |
| Escape hatch | `annotations` | JSON object |

The core columns keep their positions, so the core schema is an exact **prefix** of the extended
one. That is what lets `FormatProcessor.to_core_df()` narrow a frame by projection alone.

`annotations` is a **data** column, not a service column, so it joins `primary_key` and the sort
keys. That is deliberate: it is what keeps two annotation-only rows at the same timestamp
distinguishable, and CGMacros has 1,553 such rows. It also means the JSON must serialize
deterministically — sorted keys, stable floats — because the byte-level round-trip guarantee depends
on it. Use `annotations_to_json`; do not build the string by hand.

`ketones` is mmol/L and is **not** routed through the glucose conversion. Clinical ketones are
already mmol/L, and borrowing another analyte's convention would scale them by 18.

Frames targeting the extended schema are processed by `ExtendedFormatProcessor`. Handing one to the
core `FormatProcessor` raises under the default validation mode rather than silently narrowing.

## Tracks are alternatives, never shards

A multi-track source — CGMacros wears a Libre and a Dexcom over the same ten days — yields one frame
per sensor from `parse_tracks()`. Rows belonging to *neither* device (meals, macronutrients, heart
rate, photo annotations) are **replicated into every track**, so each frame is a complete
self-contained view of that period as seen through one sensor.

The consequence matters more than the rule: **concatenating two track frames double-counts every
meal.** A consumer who reasonably assumes tracks are shards of one dataset and stacks them gets
carbohydrate totals that are exactly twice reality, with nothing raised anywhere.

```python
libre = tracks["libre"]
dexcom = tracks["dexcom"]

pl.concat([libre, dexcom])   # WRONG — every meal counted twice
libre                         # right — pick one
libre["glucose"] - dexcom["glucose"]   # right — or compare them
```

Pick one, or compare them. Never add them.

## Format Detection

The unified format can be detected by the presence of these unique identifiers in CSV headers:

- `sequence_id`
- `original_datetime`
- `event_type`
- `quality`

## Timestamp Format

Timestamps use ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`

Example: `2024-05-01T12:30:45`
