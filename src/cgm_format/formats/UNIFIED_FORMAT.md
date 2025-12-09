# CGM Unified Format Specification

This document defines the unified data format used to standardize CGM data across different vendors (Dexcom, Libre, etc.).

> For information about the processing pipeline that produces this format, see `interface/PIPELINE.md`

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

- `EGV_READ` - Normal CGM value (Estimated Glucose Value)

**Calibration:**

- `CALIBRAT` - Sensor calibration event

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

## Format Detection

The unified format can be detected by the presence of these unique identifiers in CSV headers:

- `sequence_id`
- `original_datetime`
- `event_type`
- `quality`

## Timestamp Format

Timestamps use ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`

Example: `2024-05-01T12:30:45`
