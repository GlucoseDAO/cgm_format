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

Aligns timestamps to minute boundaries for consistent time-series analysis.

**Input:** DataFrame in unified format

**Output:** DataFrame with synchronized timestamps

#### Gap Interpolation

**Method:** `CGMProcessor.interpolate_gaps(dataframe: UnifiedFormat) -> UnifiedFormat`

Fills gaps in continuous glucose data with imputed values.

- Adds rows with `event_type='IMPUTATN'` for missing data points
- Sets appropriate quality flags on imputed values
- Updates `ProcessingWarning.IMPUTATION` flag if gaps were filled

**Input:** DataFrame with potential gaps

**Output:** DataFrame with interpolated values and imputation events

### Stage 5: Inference Preparation

**Method:** `CGMProcessor.prepare_for_inference(dataframe, minimum_duration_minutes, maximum_wanted_duration) -> InferenceResult`

Prepares processed data for machine learning inference.

#### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataframe` | `UnifiedFormat` | - | Fully processed DataFrame in unified format |
| `minimum_duration_minutes` | `int` | 60 | Minimum required sequence duration |
| `maximum_wanted_duration` | `int` | 480 | Maximum desired sequence duration |

#### Operations

- Truncate to data columns only (excludes service columns: `sequence_id`, `event_type`, `quality`)
- Truncate sequences exceeding `maximum_wanted_duration`
- Raise global warning flags based on individual row quality
- Validate minimum duration requirements

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

| Constant | Value | Description |
|----------|-------|-------------|
| `CALIBRATION_GAP_THRESHOLD` | 9900 seconds (2:45:00) | Minimum gap duration to trigger sensor calibration quality flag |
| `MINIMUM_DURATION_MINUTES` | 60 | Default minimum sequence duration for inference |
| `MAXIMUM_WANTED_DURATION_MINUTES` | 480 | Default maximum sequence duration for inference |

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

| Error | Base Class | Description |
|-------|------------|-------------|
| `MalformedDataError` | `ValueError` | Data cannot be parsed or converted properly |
| `UnknownFormatError` | `ValueError` | Format cannot be determined |
| `ZeroValidInputError` | `ValueError` | No valid data points in the sequence |

## Type Aliases

- `UnifiedFormat = pl.DataFrame` - Type alias highlighting unified format DataFrames
- `ValidationResult = Tuple[pl.DataFrame, int, int]` - (dataframe, bad_rows, valid_rows)
- `InferenceResult = Tuple[pl.DataFrame, ProcessingWarning]` - (dataframe, warnings)
