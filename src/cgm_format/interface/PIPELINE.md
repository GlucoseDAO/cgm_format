# CGM Data Processing Pipeline

This document describes the complete CGM data processing pipeline, from raw vendor files to inference-ready data.

## Overview

The pipeline is separated into two main concerns:

1. **CGMParser** (FormatParser) - Vendor-specific parsing to unified format (Stages 1-3)
2. **CGMProcessor** (FormatProcessor) - Vendor-agnostic unified format processing (Stages 4-5)

All processing operations are designed to be **idempotent** - applying the same operation multiple times produces the same result as applying it once. This is achieved through the use of `original_datetime` preservation and quality flags.

## Quick Start Example

```python
from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor

# Stage 1-3: Parse vendor-specific data to unified format
parser = FormatParser()
unified_df = parser.parse_file("data/dexcom_export.csv")
# ✓ Sequences automatically detected and assigned

# Stage 4: Process unified format
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=19)
unified_df = processor.interpolate_gaps(unified_df)  # Fill small gaps
unified_df = processor.synchronize_timestamps(unified_df)  # Align to grid

# Stage 5: Prepare for inference
inference_df, warnings = processor.prepare_for_inference(
    unified_df,
    minimum_duration_minutes=60,
    maximum_wanted_duration=480
)

# Check warnings
if warnings:
    print(f"Warnings: {warnings}")

# Optional: Strip service columns for ML model
from cgm_format.format_processor import FormatProcessor
data_only = FormatProcessor.to_data_only_df(
    inference_df,
    drop_service_columns=True,
    drop_duplicates=True,
    glucose_only=True
)
```

## Supported Input Formats

Raw CGM data from multiple vendors:

- **Dexcom** - Dexcom CGM CSV exports
- **Libre** - FreeStyle Libre CSV exports
- **Unified** - Pre-processed unified format (for deserialized data and roundtrip compatibility)

## Pipeline Stages

### Stage 1: Preprocess Raw Data

**Method:** `CGMParser.decode_raw_data(raw_data: Union[bytes, str]) -> str`

Cleans raw input data:

- Remove BOM (Byte Order Mark) artifacts
- Fix encoding issues
- Strip vendor-specific junk characters

**Input:** Raw file contents (bytes or string)

**Output:** Cleaned string data ready for parsing

### Stage 2: Format Detection

**Method:** `CGMParser.detect_format(text_data: str) -> SupportedCGMFormat`

Identifies the vendor format based on header patterns in the CSV string.

**Input:** Preprocessed string data

**Output:** `SupportedCGMFormat` enum value (DEXCOM, LIBRE, or UNIFIED_CGM)

**Errors:**

- `UnknownFormatError` - Format cannot be determined

### Stage 3: Vendor-Specific Parsing

**Method:** `CGMParser.parse_to_unified(text_data: str, format_type: SupportedCGMFormat) -> UnifiedFormat`

Converts vendor-specific CSV to unified format. This stage handles:

- CSV validation and sanity checks
- Vendor-specific quirks (High/Low glucose values, timestamp format probing, etc.)
- Column mapping to unified schema
- Populating service fields (`sequence_id`, `event_type`, `quality`)
- **Creating `original_datetime` column** - Preserves original timestamps before any modifications (essential for idempotency)
- Out-of-range glucose marking (flags sensor errors like "High"/"Low" readings)
- Schema enforcement and validation via `CGM_SCHEMA.validate_dataframe()`
- Sequence detection via `detect_and_assign_sequences()` (see below)

**Input:** Preprocessed string data and detected format type

**Output:** Polars DataFrame in unified format (see `formats/UNIFIED_FORMAT.md`) with `sequence_id` assigned

**Errors:**

- `MalformedDataError` - CSV is unparseable, has zero valid rows, or conversion fails
- `ZeroValidInputError` - No valid data rows found after processing

#### Sequence Detection (Final Parsing Step)

**Method:** `CGMParser.detect_and_assign_sequences(dataframe, expected_interval_minutes=5, large_gap_threshold_minutes=15) -> UnifiedFormat`

This is called automatically at the end of `parse_to_unified()` and is the final parsing step. It detects large gaps in **glucose events only** and assigns `sequence_id` values:

**Two-pass approach:**
1. Detect sequences based on glucose event gaps only (non-glucose events don't create sequences)
2. Assign non-glucose events (insulin, carbs, exercise) to nearest glucose sequence by time

**Why glucose-only gap detection?**
- Prevents non-glucose events from "bridging" glucose gaps
- Ensures discontinuous glucose data is properly split into separate sequences
- Non-glucose events are assigned to the nearest glucose sequence via `join_asof`

**Sequence ID values:**
- `sequence_id >= 1` - Assigned to a glucose sequence
- `sequence_id = 0` - Unassigned (no glucose events available for assignment)

**Idempotency:** If sequence_id already exists with multiple sequences, this method validates and potentially splits sequences with internal large gaps.

**Input:** DataFrame in unified format (with or without sequence_id)

**Output:** DataFrame with sequence_id column assigned based on glucose gaps

### Stage 4: Postprocessing (Unified Operations)

After Stage 3, all vendor-specific processing is complete. The following operations work on unified format data regardless of original vendor. All operations are designed for **idempotency** through the use of `original_datetime` and quality flags.

#### Timestamp Synchronization

**Method:** `CGMProcessor.synchronize_timestamps(dataframe: UnifiedFormat) -> UnifiedFormat`

Aligns timestamps to minute boundaries and creates fixed-frequency data with consistent intervals. This is a **lossless operation** - it keeps ALL source rows and only rounds their timestamps to the grid.

**When to call:** Should be called after `interpolate_gaps()` when sequences are created and small gaps are filled.

**Operations:**

1. Rounds timestamps to nearest minute using grid alignment
2. Each source row independently maps to its nearest grid point
3. Marks all rows with `Quality.SYNCHRONIZATION` flag
4. Uses `original_datetime` for grid calculations (ensures idempotency)

**Grid Alignment Logic:**
- Grid start is based on first `original_datetime` in sequence, rounded to nearest minute
- All timestamps are rounded to grid points: `grid_start + N * expected_interval_minutes`
- Uses "round half up" behavior for consistency with interpolation

**Idempotency:** Multiple applications produce identical results because:
- Grid calculations use `original_datetime` (never modified)
- Quality flags are additive (SYNCHRONIZATION flag persists)
- Grid alignment is deterministic based on sequence's first timestamp

**Input:** DataFrame with sequence IDs (preprocessed by `interpolate_gaps()`)

**Output:** DataFrame with synchronized timestamps at fixed intervals, all rows marked with SYNCHRONIZATION flag

**Errors:**

- `ZeroValidInputError` - DataFrame is empty or has no data

#### Gap Interpolation

**Method:** `CGMProcessor.interpolate_gaps(dataframe: UnifiedFormat) -> UnifiedFormat`

Fills small gaps in continuous glucose data with linearly interpolated values. **Important:** This method expects `sequence_id` to already exist in the dataframe (assigned by parser).

**Operations:**

- Detects gaps between glucose events > `expected_interval_minutes` and <= `small_gap_max_minutes`
- Adds rows with `event_type='EGV_READ'` and `Quality.IMPUTATION` flag for missing data points
- Inherits quality flags from neighboring glucose readings (combines with bitwise OR)
- Updates `ProcessingWarning.IMPUTATION` in processor's warning list if gaps were filled
- Only interpolates between valid glucose readings (non-glucose events don't affect gap detection)

**Snap-to-Grid Mode (`snap_to_grid=True`, default):**
- Interpolated points are placed on the sequence grid (same grid as `synchronize_timestamps`)
- Adds both `Quality.IMPUTATION` and `Quality.SYNCHRONIZATION` flags
- Uses grid-aligned timestamps for boundary calculations (ensures idempotency with sync)
- Guarantees: `interpolate → sync` and `sync → interpolate` are equivalent

**Non-Grid Mode (`snap_to_grid=False`):**
- Interpolated points placed at regular intervals from previous timestamp
- Only adds `Quality.IMPUTATION` flag (no SYNCHRONIZATION)
- May not align with synchronization grid

**Idempotency:** Multiple applications produce identical results because:
- Uses `original_datetime` for gap detection (never modified by interpolation)
- In snap-to-grid mode, uses same grid calculation as synchronization
- Existing imputed points are skipped (already have IMPUTATION flag)

**Input:** DataFrame with potential gaps and sequence_id column

**Output:** DataFrame with interpolated values marked with IMPUTATION (and SYNCHRONIZATION if snap_to_grid) flags

#### Calibration Period Marking

**Method:** `CGMProcessor.mark_calibration_periods(dataframe: UnifiedFormat) -> UnifiedFormat`

Marks 24-hour periods after large gaps (≥ CALIBRATION_GAP_THRESHOLD = 2:45:00) with `Quality.SENSOR_CALIBRATION` flag. This indicates data quality may be reduced during sensor warm-up/calibration.

**Operations:**

1. Detects gaps ≥ 2:45:00 using `original_datetime` (idempotent regardless of sync)
2. Marks all data points within 24 hours after gap end with SENSOR_CALIBRATION flag
3. Uses bitwise OR to add flag on top of existing quality flags

**Idempotency:** Uses `original_datetime` for gap detection, ensuring consistent results after synchronization.

**Input:** DataFrame with sequences and original_datetime column

**Output:** DataFrame with quality flags updated for calibration periods

#### Time Duplicate Marking

**Method:** `CGMProcessor.mark_time_duplicates(dataframe: UnifiedFormat) -> UnifiedFormat`

Marks events with duplicate timestamps, keeping first occurrence clean.

**Keep-first logic:**
- First event at a timestamp: no flag added
- Subsequent events with same timestamp: marked with `Quality.TIME_DUPLICATE` flag

**When called:** Automatically called in `prepare_for_inference()` before warning collection.

**Input:** DataFrame with datetime column

**Output:** DataFrame with TIME_DUPLICATE flag added to duplicate timestamp rows

### Stage 5: Inference Preparation

**Method:** `CGMProcessor.prepare_for_inference(dataframe, minimum_duration_minutes=60, maximum_wanted_duration=480) -> InferenceResult`

Prepares processed data for machine learning inference. Returns full UnifiedFormat with all columns (use `to_data_only_df()` to strip service columns if needed).

#### Input Parameters

| Parameter                  | Type            | Default | Description                                 |
| -------------------------- | --------------- | ------- | ------------------------------------------- |
| `dataframe`                | `UnifiedFormat` | -       | Fully processed DataFrame in unified format |
| `minimum_duration_minutes` | `int`           | 60      | Minimum required sequence duration          |
| `maximum_wanted_duration`  | `int`           | 480     | Maximum desired sequence duration           |

#### Operations

1. **Validate non-empty data**: Check for zero valid data points (raises `ZeroValidInputError`)
2. **Select latest valid sequence**: Keep only the most recent sequence with valid glucose data
   - Tries sequences starting from most recent timestamp
   - Skips sequences with no glucose data
   - Falls back to previous sequences if latest doesn't meet minimum duration
3. **Truncate to maximum duration**: If sequence exceeds `maximum_wanted_duration`
   - **Truncates from the beginning**, keeping the **latest (most recent)** data
   - Preserves the most recent `maximum_wanted_duration` minutes of data
   - Example: For 600 minutes of data with max duration of 480 minutes, keeps the last 480 minutes
4. **Mark time duplicates**: Calls `mark_time_duplicates()` to flag duplicate timestamps
5. **Mark calibration periods**: Calls `mark_calibration_periods()` to flag post-gap calibration windows
6. **Collect warnings** based on truncated data quality:
   - `TOO_SHORT`: sequence duration < minimum_duration_minutes (after truncation)
   - `CALIBRATION`: contains calibration events OR SENSOR_CALIBRATION quality flag
   - `OUT_OF_RANGE`: contains OUT_OF_RANGE quality flags (sensor errors)
   - `IMPUTATION`: contains IMPUTATION quality flags (imputed/interpolated data)
   - `TIME_DUPLICATES`: contains non-unique timestamps OR TIME_DUPLICATE quality flags

#### Output

Returns `InferenceResult` tuple: `(unified_format_dataframe, warnings)`

- `unified_format_dataframe`: Full DataFrame with all columns including service columns
  - `sequence_id`, `original_datetime`, `event_type`, `quality` - Service/metadata columns
  - `datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise` - Data columns
- `warnings`: `ProcessingWarning` flags combined with bitwise OR

#### Processing Warnings

Warnings are implemented as flags and can be combined:

- `ProcessingWarning.TOO_SHORT` - Minimum duration requirement not met
- `ProcessingWarning.CALIBRATION` - Contains calibration events or 24hr post-calibration gap period
- `ProcessingWarning.OUT_OF_RANGE` - Contains sensor out-of-range errors ("High"/"Low" readings)
- `ProcessingWarning.IMPUTATION` - Contains imputed/interpolated gaps
- `ProcessingWarning.TIME_DUPLICATES` - Contains non-unique time entries

Example:

```python
data, warnings = processor.prepare_for_inference(df)
if warnings & ProcessingWarning.TOO_SHORT:
    print("Warning: Sequence is too short")
if warnings & ProcessingWarning.IMPUTATION:
    print("Warning: Contains imputed data")
```

#### Stripping Service Columns

**Static Method:** `FormatProcessor.to_data_only_df(unified_df, drop_service_columns=True, drop_duplicates=False, glucose_only=False) -> pl.DataFrame`

Optional pipeline-terminating function that removes metadata columns for ML models:

**Operations:**
- Filter to glucose-only events if `glucose_only=True` (drops non-EGV events)
- Drop duplicate timestamps if `drop_duplicates=True` (keeps first occurrence)
- Strip service columns if `drop_service_columns=True`: removes `sequence_id`, `event_type`, `quality`
- Keeps only data columns: `datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise`

**Example:**
```python
# Get full unified format with warnings
unified_df, warnings = processor.prepare_for_inference(df)

# Strip service columns for ML model
data_only = FormatProcessor.to_data_only_df(
    unified_df, 
    drop_service_columns=True,
    drop_duplicates=True,
    glucose_only=True
)
```

#### Errors

- `ZeroValidInputError` - No valid data points or no sequences meet minimum duration with glucose data

## Constants

| Constant                          | Value                  | Description                                                     | Stage               |
| --------------------------------- | ---------------------- | --------------------------------------------------------------- | ------------------- |
| `EXPECTED_INTERVAL_MINUTES`       | 5                      | Expected data collection interval                               | All stages          |
| `SMALL_GAP_MAX_MINUTES`           | 19.0                   | Maximum gap size to interpolate (3 intervals + 80% tolerance)   | Stage 4 (Processor) |
| `CALIBRATION_GAP_THRESHOLD`       | 9900 seconds (2:45:00) | Minimum gap duration to trigger sensor calibration quality flag | Stage 4 (Processor) |
| `CALIBRATION_PERIOD_HOURS`        | 24                     | Duration of calibration period after large gap                  | Stage 4 (Processor) |
| `MINIMUM_DURATION_MINUTES`        | 60                     | Default minimum sequence duration for inference                 | Stage 5 (Processor) |
| `MAXIMUM_WANTED_DURATION_MINUTES` | 480                    | Default maximum sequence duration for inference                 | Stage 5 (Processor) |

**Note:** Gap thresholds and calibration marking are applied during processing (Stage 4), not parsing (Stage 3). Parsing only performs sequence detection based on large gaps.

## Serialization

### CSV Export

**Method:** `CGMParser.to_csv_string(dataframe: UnifiedFormat) -> str`

Serializes unified format DataFrame to CSV string for storage or transmission.

**Schema Validation:** Can be configured via `FormatParser.validation_mode`:
- `ValidationMethod.INPUT` - Validates input DataFrame matches schema before export
- `ValidationMethod.INPUT_FORCED` - Enforces schema on input (casts, reorders) before export
- `ValidationMethod(0)` - No validation (fastest)

**Input:** DataFrame in unified format

**Output:** CSV string representation

### CSV File Export

**Method:** `CGMParser.to_csv_file(dataframe: UnifiedFormat, file_path: str) -> None`

Saves unified format DataFrame directly to CSV file.

**Input:** DataFrame in unified format and file path

**Output:** CSV file written to disk

## Compatibility Layer

### Pandas Conversion

Optional pandas support (requires `pandas` and `pyarrow` packages):

- `to_pandas(df: pl.DataFrame) -> pd.DataFrame` - Convert Polars to pandas
- `to_polars(df: pd.DataFrame) -> pl.DataFrame` - Convert pandas to Polars

**Note:** These functions raise `ImportError` if pandas/pyarrow are not installed.

### Event Type Splitting

**Static Method:** `FormatProcessor.split_glucose_events(unified_df: UnifiedFormat) -> Tuple[UnifiedFormat, UnifiedFormat]`

Splits a UnifiedFormat DataFrame into glucose readings and other events:

- **Glucose DataFrame**: Contains only `EGV_READ` events (including imputed ones marked with quality flag)
- **Events DataFrame**: Contains all other event types (insulin, carbs, exercise, calibration, etc.)

Both output DataFrames maintain full UnifiedFormat schema with all columns. This is a non-destructive split operation.

**Example:**
```python
glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)
# Process glucose and events separately
glucose_df = processor.interpolate_gaps(glucose_df)
# Merge back if needed
combined = pl.concat([glucose_df, events_df]).sort(CGM_SCHEMA.get_stable_sort_keys())
```

## Schema System

The unified format uses a robust schema system defined in `interface/schema.py` with the `CGMSchemaDefinition` class.

### Schema Definition

The authoritative schema (`CGM_SCHEMA`) is defined in `formats/unified.py` and provides:

- **Column definitions**: Service columns (sequence_id, original_datetime, event_type, quality) and data columns (datetime, glucose, carbs, insulin_slow, insulin_fast, exercise)
- **Type safety**: Polars dtype enforcement for all columns
- **Primary key**: Combination of all data columns for true duplicate detection
- **Stable sorting**: Deterministic row ordering using all columns in priority order

### Schema Validation

The schema system provides two modes:

**Validation Mode** (`enforce=False`):
- Checks all expected columns are present in correct order
- Verifies column types match schema exactly
- Raises errors on mismatch:
  - `MissingColumnError` - Required column is missing
  - `ExtraColumnError` - Unexpected column present
  - `ColumnOrderError` - Columns not in schema order
  - `ColumnTypeError` - Column type doesn't match schema

**Enforcement Mode** (`enforce=True`):
- Adds missing columns with null values (e.g., `original_datetime`)
- Removes extra columns not in schema
- Casts columns to correct types (strict for most, non-strict for numeric to handle nulls)
- Reorders columns to match schema
- Applies stable sorting using `get_stable_sort_keys()` for deterministic row ordering

### Configuration

**Parser Validation:** `FormatParser.validation_mode` (class variable)
- `ValidationMethod.INPUT` - Validate input DataFrames
- `ValidationMethod.OUTPUT` - Validate output DataFrames
- `ValidationMethod.INPUT_FORCED` - Enforce schema on input
- `ValidationMethod.OUTPUT_FORCED` - Enforce schema on output
- `ValidationMethod(0)` - No validation (fastest)

Default: `ValidationMethod.INPUT` (validate inputs, trust outputs)

**Processor Validation:** `FormatProcessor(validation_mode=...)` (instance parameter)
- Same flags as parser
- Default: `ValidationMethod.INPUT`

### Example Usage

```python
from cgm_format.formats.unified import CGM_SCHEMA

# Validate that DataFrame matches schema (strict)
validated_df = CGM_SCHEMA.validate_dataframe(df, enforce=False)

# Enforce schema on DataFrame (add missing, cast types, reorder, sort)
enforced_df = CGM_SCHEMA.validate_dataframe(df, enforce=True)

# Get column names
columns = CGM_SCHEMA.get_column_names(data_only=False)

# Get stable sort keys for deterministic ordering
sort_keys = CGM_SCHEMA.get_stable_sort_keys()
df = df.sort(sort_keys)
```

## Error Handling

The pipeline defines the following error types:

| Error                 | Base Class   | Description                                          |
| --------------------- | ------------ | ---------------------------------------------------- |
| `MalformedDataError`  | `ValueError` | Data cannot be parsed or converted properly          |
| `MissingColumnError`  | `MalformedDataError` | Required column is missing from the DataFrame |
| `ExtraColumnError`    | `MalformedDataError` | Unexpected column present in the DataFrame    |
| `ColumnOrderError`    | `MalformedDataError` | Columns not in correct schema order           |
| `ColumnTypeError`     | `MalformedDataError` | Column type doesn't match schema              |
| `UnknownFormatError`  | `ValueError` | Format cannot be determined                          |
| `ZeroValidInputError` | `ValueError` | No valid data points in the sequence                 |

## Type Aliases

- `UnifiedFormat = pl.DataFrame` - Type alias highlighting unified format DataFrames
- `InferenceResult = Tuple[pl.DataFrame, ProcessingWarning]` - (unified_dataframe, warnings)

## Testing

The pipeline is extensively tested with comprehensive test coverage (see `tests/README.md`):

### Test Coverage

- **test_format_detection_validation.py** - Format detection for all vendors, Frictionless schema validation
- **test_integration_pipeline.py** - Full end-to-end pipeline integration on real data (no mocking)
- **test_format_processor.py** - FormatProcessor implementation: sync, interpolation, inference prep, calibration marking
- **test_format_converter.py** - Format parser: detection, unified parsing, roundtrip verification, sequence detection
- **test_roundtrip_datetime.py** - Datetime type preservation through roundtrip conversions
- **test_idempotency.py** - Idempotency and commutativity of processing operations
- **test_schema.py** - Schema definition and Frictionless validation tests
- **test_utils.py** - Utility methods (split_glucose_events, to_data_only_df)

All tests use real data files and verify:
- Data integrity and consistency
- Timestamp ordering and idempotency
- Lossless operations (no data loss)
- Schema compliance
- Error handling

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_idempotency.py

# Run with verbose output
uv run pytest tests/ -v
```

## Convenience Methods

The parser provides convenience methods for common input sources:

### File Input

**Method:** `CGMParser.parse_file(file_path: Union[str, Path]) -> UnifiedFormat`

Parse CGM data directly from file path. Automatically detects format and handles encoding.

```python
from cgm_format.format_parser import FormatParser

# Parse from file path
df = FormatParser.parse_file("data/dexcom_export.csv")
```

### Base64 Input

**Method:** `CGMParser.parse_base64(base64_data: str) -> UnifiedFormat`

Parse CGM data from base64 encoded string. Useful for web API endpoints.

```python
# Parse from base64 encoded data
df = FormatParser.parse_base64(base64_string)
```

### Bytes Input

**Method:** `CGMParser.parse_from_bytes(raw_data: bytes) -> UnifiedFormat`

Parse CGM data from raw bytes. Chains decode → detect → parse.

```python
# Parse from raw bytes
with open("data.csv", "rb") as f:
    df = FormatParser.parse_from_bytes(f.read())
```

### String Input

**Method:** `CGMParser.parse_from_string(text_data: str) -> UnifiedFormat`

Parse CGM data from cleaned string. Assumes data is already decoded. Chains detect → parse.

```python
# Parse from string
df = FormatParser.parse_from_string(csv_string)
```

All convenience methods automatically:
- Detect format (Dexcom, Libre, or Unified)
- Handle encoding issues and BOM artifacts
- Parse to unified format
- Assign sequence_id values
- Validate schema

## Design Principles

### Idempotency

All processing operations are idempotent through careful design:

1. **Original Timestamp Preservation**: `original_datetime` column preserves original timestamps before any modifications
2. **Grid Calculations**: Synchronization and interpolation use `original_datetime` for grid calculations
3. **Quality Flags**: Additive quality flags (bitwise OR) preserve operation history
4. **Deterministic Sorting**: Stable sort keys ensure consistent row ordering

### Losslessness

Processing operations preserve all data:

1. **Synchronization**: Keeps ALL source rows, only rounds timestamps to grid
2. **Interpolation**: Adds new rows, never removes existing data
3. **Schema Enforcement**: Adds missing columns, preserves existing data
4. **Sequence Detection**: Pure annotation, doesn't modify or remove data

### Separation of Concerns

1. **Parser (Stages 1-3)**: Vendor-specific → Unified format
2. **Processor (Stages 4-5)**: Vendor-agnostic unified operations
3. **Schema System**: Centralized validation and enforcement
4. **Quality Flags**: Fine-grained data quality tracking at row level
5. **Processing Warnings**: Coarse-grained quality assessment at sequence level
