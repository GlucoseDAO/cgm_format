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
| `event_type` | `Utf8` | Type of recorded event (8-char code mapping to Dexcom EVENT_TYPE+SUBTYPE) |
| `quality` | `Int64` | Data quality indicator (0=GOOD, 1=ILL, 2=SENSOR_CALIBRATION) |

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

- `IMPUTATN` - Imputed/interpolated data
- `OTHEREVT` - Other/unknown event type

#### Quality Enum

- `0` = GOOD - Valid, high-quality data
- `1` = ILL - Out-of-range or flagged values
- `2` = SENSOR_CALIBRATION - 24hr period after gap ≥ 2:45:00

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

## Schema Usage

The schema is implemented using the `CGMSchemaDefinition` class from `interface/schema.py`, which provides:

- **Polars schema dictionary**: `CGM_SCHEMA.get_polars_schema(data_only=False)`
- **Column names list**: `CGM_SCHEMA.get_column_names(data_only=False)`
- **Cast expressions**: `CGM_SCHEMA.get_cast_expressions(data_only=False)`
- **Frictionless Data export**: `CGM_SCHEMA.to_frictionless_schema()`

Set `data_only=True` to work with only the data columns (excluding service columns).

### Regenerating Schema JSON

To regenerate `unified.json` after modifying the schema:

```python
python3 -c "from formats.unified import regenerate_schema_json; regenerate_schema_json()"
```

## Format Detection

The unified format can be detected by the presence of these unique identifiers in CSV headers:

- `sequence_id`
- `event_type`
- `quality`

## Timestamp Format

Timestamps use ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`

Example: `2024-05-01T12:30:45`
