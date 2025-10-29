# CGM Data Processing Pipeline

This document describes the complete CGM data processing pipeline, from raw vendor files to inference-ready data.

## Overview

The pipeline is separated into two main concerns:

1. **CGMParser** - Vendor-specific parsing to unified format (Stages 1-3)
2. **CGMProcessor** - Vendor-agnostic unified format processing (Stages 4-5)

## Supported Input Formats

Raw CGM data from multiple vendors:

- **Dexcom** - Dexcom CGM CSV exports
- **Libre** - FreeStyle Libre CSV exports
- **Unified** - Pre-processed unified format (for deserialized data)

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
- Vendor-specific quirks (High/Low glucose values, timezone fixes, etc.)
- Column mapping to unified schema
- Populating service fields (`sequence_id`, `event_type`, `quality`)
- High/low glucose marking (flags values outside healthy ranges)

**Input:** Preprocessed string data and detected format type

**Output:** Polars DataFrame in unified format (see `formats/UNIFIED_FORMAT.md`)

**Errors:**

- `MalformedDataError` - CSV is unparseable, has zero valid rows, or conversion fails

### Stage 4: Postprocessing (Unified Operations)

After Stage 3, all vendor-specific processing is complete. The following operations work on unified format data regardless of original vendor.

#### Timestamp Synchronization

**Method:** `CGMProcessor.synchronize_timestamps(dataframe: UnifiedFormat) -> UnifiedFormat`

Aligns timestamps to minute boundaries and creates fixed-frequency data with consistent intervals.

This method should be called after `interpolate_gaps()` when sequences are already created and small gaps are filled.

**Operations:**

1. Rounds timestamps to nearest minute (removes seconds, sets to 00)
2. Creates fixed-frequency timestamps with `expected_interval_minutes` spacing
3. Linearly interpolates glucose values between data points
4. Shifts discrete events (carbs, insulin, exercise) to nearest timestamps

**Input:** DataFrame with sequence IDs (preprocessed by `interpolate_gaps()`)

**Output:** DataFrame with synchronized timestamps at fixed intervals

**Errors:**

- `ZeroValidInputError` - DataFrame is empty or has no data
- `ValueError` - Data has gaps larger than `small_gap_max_minutes` (not preprocessed)
- `ValueError` - Data missing `sequence_id` column (run `interpolate_gaps()` first)

#### Gap Interpolation

**Method:** `CGMProcessor.interpolate_gaps(dataframe: UnifiedFormat) -> UnifiedFormat`

Fills gaps in continuous glucose data with imputed values.

- Adds rows with `event_type='IMPUTATN'` for missing data points
- Sets appropriate quality flags on imputed values
- Updates `ProcessingWarning.IMPUTATION` flag if gaps were filled
- In case of larg gaps, more than

**Input:** DataFrame with potential gaps

**Output:** DataFrame with interpolated values and imputation events

### Stage 5: Inference Preparation

**Method:** `CGMProcessor.prepare_for_inference(dataframe, minimum_duration_minutes, maximum_wanted_duration) -> InferenceResult`

Prepares processed data for machine learning inference.

#### Input Parameters

| Parameter                  | Type            | Default | Description                                 |
| -------------------------- | --------------- | ------- | ------------------------------------------- |
| `dataframe`                | `UnifiedFormat` | -       | Fully processed DataFrame in unified format |
| `minimum_duration_minutes` | `int`           | 60      | Minimum required sequence duration          |
| `maximum_wanted_duration`  | `int`           | 480     | Maximum desired sequence duration           |

#### Operations

1. Check for zero valid data points (raises `ZeroValidInputError`)
2. Keep only the last (latest) sequence based on most recent timestamps
   - If multiple sequences exist, identifies the sequence with the maximum (most recent) timestamp
   - Keeps only that sequence and discards all others
   - Single sequence data is unaffected
3. Collect warnings based on data quality:
   - `TOO_SHORT`: sequence duration < minimum_duration_minutes
   - `CALIBRATION`: contains calibration events
   - `QUALITY`: contains ILL or SENSOR_CALIBRATION quality flags
   - `IMPUTATION`: contains imputed events
4. Truncate sequences exceeding `maximum_wanted_duration`
   - **Truncates from the beginning**, keeping the **latest (most recent)** data
   - Preserves the most recent `maximum_wanted_duration` minutes of data
   - Example: For 60 minutes of data with max duration of 30 minutes, keeps the last 30 minutes
5. Extract data columns only (excludes service columns: `sequence_id`, `event_type`, `quality`)

#### Output

Returns `InferenceResult` tuple: `(data_only_dataframe, warnings)`

- `data_only_dataframe`: DataFrame with only data columns (datetime, glucose, carbs, insulin_slow, insulin_fast, exercise)
- `warnings`: `ProcessingWarning` flags (can be combined with bitwise OR)

#### Processing Warnings

Warnings are implemented as flags and can be combined:

- `ProcessingWarning.TOO_SHORT` - Minimum duration requirement not met
- `ProcessingWarning.CALIBRATION` - Output sequence contains calibration events
- `ProcessingWarning.QUALITY` - Contains ill or sensor calibration quality events
- `ProcessingWarning.IMPUTATION` - Contains imputed gaps

Example:

```python
warnings = ProcessingWarning.TOO_SHORT | ProcessingWarning.QUALITY
```

#### Errors

- `ZeroValidInputError` - No valid data points in the sequence

## Constants

| Constant                          | Value                  | Description                                                     | Stage               |
| --------------------------------- | ---------------------- | --------------------------------------------------------------- | ------------------- |
| `CALIBRATION_GAP_THRESHOLD`       | 9900 seconds (2:45:00) | Minimum gap duration to trigger sensor calibration quality flag | Stage 3 (Parser)    |
| `MINIMUM_DURATION_MINUTES`        | 60                     | Default minimum sequence duration for inference                 | Stage 5 (Processor) |
| `MAXIMUM_WANTED_DURATION_MINUTES` | 480                    | Default maximum sequence duration for inference                 | Stage 5 (Processor) |

**Note:** `CALIBRATION_GAP_THRESHOLD` is used during parsing (Stage 3) to mark data quality, not in the processor (Stages 4-5).

## Serialization

### CSV Export

**Method:** `CGMParser.to_csv_string(dataframe: UnifiedFormat) -> str`

Serializes unified format DataFrame to CSV string for storage or transmission.

**Input:** DataFrame in unified format

**Output:** CSV string representation

## Compatibility Layer

### Pandas Conversion

Optional pandas support (requires `pandas` and `pyarrow` packages):

- `to_pandas(df: pl.DataFrame) -> pd.DataFrame` - Convert Polars to pandas
- `to_polars(df: pd.DataFrame) -> pl.DataFrame` - Convert pandas to Polars

**Note:** These functions raise `ImportError` if pandas/pyarrow are not installed.

## Error Handling

The pipeline defines the following error types:

| Error                 | Base Class   | Description                                 |
| --------------------- | ------------ | ------------------------------------------- |
| `MalformedDataError`  | `ValueError` | Data cannot be parsed or converted properly |
| `UnknownFormatError`  | `ValueError` | Format cannot be determined                 |
| `ZeroValidInputError` | `ValueError` | No valid data points in the sequence        |

## Type Aliases

- `UnifiedFormat = pl.DataFrame` - Type alias highlighting unified format DataFrames
- `ValidationResult = Tuple[pl.DataFrame, int, int]` - (dataframe, bad_rows, valid_rows)
- `InferenceResult = Tuple[pl.DataFrame, ProcessingWarning]` - (dataframe, warnings)
