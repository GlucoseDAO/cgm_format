# cgm_format

Python library for converting vendor-specific Continuous Glucose Monitoring (CGM) data (Dexcom, Libre) into a standardized unified format for ML training and inference.

[![Tests](https://github.com/GlucoseDAO/cgm_format/actions/workflows/test.yml/badge.svg)](https://github.com/GlucoseDAO/cgm_format/actions/workflows/test.yml)

## Features

- **Vendor format detection**: Automatic detection of Dexcom, Libre, and Unified formats
- **Robust parsing**: Handles BOM marks, encoding artifacts, and vendor-specific CSV quirks
- **Unified schema**: Standardized data format with service columns (metadata) and data columns
- **Idempotent processing**: All operations are idempotent - applying them multiple times produces the same result
- **Schema validation**: Comprehensive validation and enforcement system with Frictionless Data Table Schema support
- **Type-safe**: Polars-based with strict type definitions and enum support
- **Quality tracking**: Fine-grained data quality tracking via bitwise flags
- **Extensively tested**: Comprehensive test suite with real data (no mocking)
- **Extensible**: Clean abstract interfaces for adding new vendor formats

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip3 install -e .

# Optional dependencies
uv pip install -e ".[extra]"  # pandas, pyarrow, frictionless
uv pip install -e ".[cli]"    # typer, rich (for cgm-cli tool)
uv pip install -e ".[dev]"    # pytest + all extras
```

## CLI Tool

The package includes a comprehensive CLI tool for CGM data processing with 8 commands:

**Commands:**
- `detect` - Format detection
- `parse` - Parse to unified format  
- `validate` - Schema validation (single file)
- `report` - Comprehensive validation report with Frictionless (directory)
- `process` - Interpolation & synchronization
- `pipeline` - Full 6-stage processing pipeline
- `info` - File information & statistics
- `batch` - Batch process directories

**Installation:**
```bash
# Install with CLI support
uv pip install -e ".[cli]"  # Adds typer + rich dependencies

# Verify installation
cgm-cli --help
```

**Usage:**
```bash
# Three ways to run
cgm-cli <command>                        # Installed command
python -m cgm_format.cgm_cli <command>   # Module execution  
python scripts/cgm_cli.py <command>      # Direct script

# Examples
cgm-cli detect data/patient.csv                    # Detect format
cgm-cli parse data/dexcom.csv -o unified.csv       # Parse to unified
cgm-cli pipeline data/libre.csv -o output.csv      # Full pipeline
cgm-cli report data/ -o report.txt --frictionless  # Validation report
cgm-cli batch data/raw/ -o data/processed/         # Batch process

# Run interactive example
uv run python examples/example_cli_usage.py --skip-slow
```

**Key Features:**
- Rich terminal output (colors, progress bars, formatted tables)
- Frictionless validation with automatic suppression of known vendor quirks
- Comprehensive statistics (glucose, quality flags, event counts)
- Processing warnings and quality issue detection
- Batch validation reports for entire directories

See [scripts/README.md](scripts/README.md) for complete command reference.

## Quick Start

### Basic Parsing

```python
from cgm_format import FormatParser
import polars as pl

# Parse any supported CGM file (Dexcom, Libre, or Unified)
unified_df = FormatParser.parse_file("data/example.csv")

# Or parse from base64 (useful for web APIs)
unified_df = FormatParser.parse_base64(base64_encoded_csv)

# Access the data
print(unified_df.head())

# Save to unified format
FormatParser.to_csv_file(unified_df, "output.csv")
```

### Complete Inference Pipeline

```python
from cgm_format import FormatParser, FormatProcessor

# Stage 1-3: Parse vendor format to unified
unified_df = FormatParser.parse_file("data/dexcom_export.csv")

# Stage 4-5: Process for inference
processor = FormatProcessor(
    expected_interval_minutes=5,
    small_gap_max_minutes=15
)

# Fill gaps and create sequences
processed_df = processor.interpolate_gaps(unified_df)

# Prepare final inference data (returns full UnifiedFormat)
unified_df, warnings = processor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=180,      # Require 3 hours minimum (default: 60)
    maximum_wanted_duration=1440       # Truncate to last 24 hours if longer (default: 480)
)

# Strip service columns for ML model
inference_df = FormatProcessor.to_data_only_df(unified_df)

# Feed to ML model
predictions = your_model.predict(inference_df)
```

### Split Glucose and Events

```python
from cgm_format import FormatParser, FormatProcessor

# Parse mixed data
unified_df = FormatParser.parse_file("data/cgm_with_events.csv")

# Split into glucose readings and other events (insulin, carbs, etc.)
glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)

# Process glucose data separately
processor = FormatProcessor()
glucose_df = processor.interpolate_gaps(glucose_df)
unified_df, warnings = processor.prepare_for_inference(glucose_df)

# Strip service columns if needed for ML
inference_df = FormatProcessor.to_data_only_df(unified_df)

# Analyze events separately
insulin_events = events_df.filter(pl.col('event_type').str.contains('INSULIN'))
```

**See [USAGE.md](USAGE.md) for complete inference workflows and [examples/usage_example.py](examples/usage_example.py) for runnable examples.**

## Unified Format Schema

The library converts all vendor formats to a standardized schema with two types of columns:

### Service Columns (Metadata)

| Column | Type | Description |
|--------|------|-------------|
| `sequence_id` | `Int64` | Unique sequence identifier (split by large gaps in glucose data) |
| `original_datetime` | `Datetime` | Original timestamp before any modifications (preserved for idempotency) |
| `event_type` | `Utf8` | Event type (8-char code: EGV_READ, INS_FAST, CARBS_IN, etc.) |
| `quality` | `Int64` | Data quality flags (bitwise): 0=GOOD, 1=OUT_OF_RANGE, 2=SENSOR_CALIBRATION, 4=IMPUTATION, 8=TIME_DUPLICATE, 16=SYNCHRONIZATION |

### Data Columns

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `datetime` | `Datetime` | - | Timestamp (ISO 8601) |
| `glucose` | `Float64` | mg/dL | Blood glucose reading |
| `carbs` | `Float64` | g | Carbohydrate intake |
| `insulin_slow` | `Float64` | u | Long-acting insulin dose |
| `insulin_fast` | `Float64` | u | Short-acting insulin dose |
| `exercise` | `Int64` | seconds | Exercise duration |

See [`formats/UNIFIED_FORMAT.md`](formats/UNIFIED_FORMAT.md) for complete specification and event type enums.

## Processing Pipeline

The library implements a comprehensive processing pipeline with two main stages:

### Parsing (Stages 1-3): CGMParser Interface

Vendor-specific parsing to unified format with automatic sequence detection.

### Processing (Stages 4-5): CGMProcessor Interface

Vendor-agnostic operations on unified data. All operations are **idempotent** through `original_datetime` preservation and quality flags.

**Complete Pipeline Example:**

```python
from cgm_format import FormatParser, FormatProcessor

# Stages 1-3: Parse to unified format (sequences automatically assigned)
unified_df = FormatParser.parse_file("data/dexcom_export.csv")

# Stage 4: Interpolate gaps and mark calibration periods
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=19)
unified_df = processor.interpolate_gaps(unified_df)

# Optional: Synchronize timestamps to fixed grid
unified_df = processor.synchronize_timestamps(unified_df)

# Stage 5: Prepare for inference with quality checks
inference_df, warnings = processor.prepare_for_inference(
    unified_df,
    minimum_duration_minutes=60,
    maximum_wanted_duration=480
)

# Strip service columns for ML model
data_only = FormatProcessor.to_data_only_df(inference_df)
```

See [`interface/PIPELINE.md`](src/cgm_format/interface/PIPELINE.md) for complete documentation.

### Stage 1: Preprocess Raw Data

Remove BOM marks, encoding artifacts, and normalize text encoding.

```python
text_data = FormatParser.decode_raw_data(raw_bytes)
```

### Stage 2: Format Detection

Automatically detect vendor format from CSV headers.

```python
from cgm_format.interface.cgm_interface import SupportedCGMFormat

format_type = FormatParser.detect_format(text_data)
# Returns: SupportedCGMFormat.DEXCOM, .LIBRE, or .UNIFIED_CGM
```

### Stage 3: Vendor-Specific Parsing

Parse vendor CSV to unified format, handling vendor-specific quirks and automatically detecting sequences:

- Dexcom: High/Low glucose markers, variable-length rows, metadata rows
- Libre: Record type filtering, timestamp format variations
- **Sequence detection**: Automatically splits data at large gaps (>15 min) in glucose readings
- **Original timestamp preservation**: Creates `original_datetime` column for idempotency

```python
unified_df = FormatParser.parse_to_unified(text_data, format_type)
# ✓ Sequences automatically assigned based on glucose gaps
# ✓ original_datetime preserved for idempotent processing
```

All stages can be chained with convenience methods:

```python
# Parse from file path (recommended) - sequences auto-detected
unified_df = FormatParser.parse_file("data.csv")

# Parse from base64 string (web APIs)
unified_df = FormatParser.parse_base64(base64_encoded_csv)

# Parse from bytes (lower-level)
unified_df = FormatParser.parse_from_bytes(raw_data)

# Parse from string (manual control)
unified_df = FormatParser.parse_from_string(text_data)
```

### Stage 4: Gap Interpolation and Calibration Marking

The `FormatProcessor.interpolate_gaps()` method handles data continuity and quality marking:

```python
from cgm_format import FormatProcessor

processor = FormatProcessor(
    expected_interval_minutes=5,    # Normal CGM reading interval
    small_gap_max_minutes=19,       # Max gap size to interpolate (3 intervals + 80% tolerance)
    snap_to_grid=True               # Align interpolated points to synchronization grid (default)
)

# Fill small gaps with linear interpolation
processed_df = processor.interpolate_gaps(unified_df)
```

**What it does:**

1. **Gap Detection**: Identifies gaps in continuous glucose monitoring data (only glucose events)
2. **Small Gap Interpolation**: Fills gaps (>5 min, ≤19 min) with linearly interpolated glucose values
3. **Snap-to-Grid Mode** (default): Interpolated points align with synchronization grid
   - Adds both `IMPUTATION` and `SYNCHRONIZATION` quality flags
   - Guarantees idempotency: `interpolate → sync` ≡ `sync → interpolate`
4. **Calibration Period Marking**: Called automatically in `prepare_for_inference()`
   - Marks 24-hour periods after gaps ≥2h45m with `SENSOR_CALIBRATION` quality flag
5. **Warning Collection**: Tracks imputation events via `ProcessingWarning.IMPUTATION`
6. **Idempotency**: Uses `original_datetime` for gap detection (never modified)

**Example - Analyze sequences created:**

```python
# Check sequences
sequence_count = processed_df['sequence_id'].n_unique()
print(f"Created {sequence_count} sequences")

# Analyze each sequence
import polars as pl
sequence_info = processed_df.group_by('sequence_id').agg([
    pl.col('datetime').min().alias('start_time'),
    pl.col('datetime').max().alias('end_time'),
    pl.col('datetime').count().alias('num_points'),
])

for row in sequence_info.iter_rows(named=True):
    duration_hours = (row['end_time'] - row['start_time']).total_seconds() / 3600
    print(f"Sequence {row['sequence_id']}: {duration_hours:.1f}h, {row['num_points']} points")
```

### Stage 5: Timestamp Synchronization (Optional)

Align timestamps to fixed-frequency intervals for ML models requiring regular time steps. This is a **lossless operation** - it keeps ALL source rows and only rounds their timestamps to the grid:

```python
# After interpolate_gaps(), synchronize to exact intervals
synchronized_df = processor.synchronize_timestamps(processed_df)

# Now all timestamps are at exact 5-minute intervals: 10:00:00, 10:05:00, 10:10:00, etc.
```

**What it does:**

1. Rounds timestamps to nearest minute boundary (removes seconds)
2. Each source row independently maps to its nearest grid point
3. Marks all rows with `SYNCHRONIZATION` quality flag
4. Uses `original_datetime` for grid calculations (ensures idempotency)
5. Preserves sequence boundaries (processes each sequence independently)

**Idempotency:** Multiple applications produce identical results because grid calculations use `original_datetime` (never modified) and quality flags are additive.

**When to use:** Time-series models expecting fixed intervals (LSTM, transformers, ARIMA)  
**When to skip:** Models handling irregular timestamps, or when original timing is critical

### Stage 6: Inference Preparation

The `prepare_for_inference()` method performs final quality assurance and returns full UnifiedFormat:

```python
# Prepare final inference-ready data (returns full UnifiedFormat)
unified_df, warnings = processor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=180,      # Require 3 hours minimum (default: 60)
    maximum_wanted_duration=1440       # Truncate to last 24 hours if longer (default: 480)
)

# Optionally strip service columns for ML models
inference_df = FormatProcessor.to_data_only_df(unified_df)

# Check for quality issues
from cgm_format.interface.cgm_interface import ProcessingWarning

if warnings & ProcessingWarning.TOO_SHORT:
    print("Warning: Sequence shorter than minimum duration")
if warnings & ProcessingWarning.OUT_OF_RANGE:
    print("Warning: Data contains sensor out-of-range errors")
if warnings & ProcessingWarning.IMPUTATION:
    print("Warning: Data contains interpolated values")
if warnings & ProcessingWarning.CALIBRATION:
    print("Warning: Data contains calibration events or post-calibration periods")
if warnings & ProcessingWarning.TIME_DUPLICATES:
    print("Warning: Data contains duplicate timestamps")
```

**What it does:**

1. **Validation**: Raises `ZeroValidInputError` if no valid glucose data exists
2. **Sequence Selection**: Keeps only the **latest** valid sequence (most recent timestamps)
   - Tries sequences from most recent, falls back if too short
3. **Truncation**: Keeps last N minutes if exceeding `maximum_wanted_duration`
4. **Time Duplicate Marking**: Marks duplicate timestamps with `TIME_DUPLICATE` quality flag
5. **Calibration Period Marking**: Marks 24h periods after gaps ≥2h45m with `SENSOR_CALIBRATION` flag
6. **Quality Checks**: Collects warnings for:
   - `TOO_SHORT`: sequence duration < minimum_duration_minutes
   - `OUT_OF_RANGE`: sensor out-of-range errors ("High"/"Low" readings)
   - `CALIBRATION`: calibration events or 24hr post-calibration gap periods
   - `IMPUTATION`: imputed/interpolated data
   - `TIME_DUPLICATES`: non-unique timestamps
7. **Returns**: Full UnifiedFormat with all columns (use `to_data_only_df()` to strip service columns)

**Output DataFrame:**

```python
# inference_df contains only data columns:
# ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']

# Feed directly to ML model
predictions = your_model.predict(inference_df)
```

### Complete Processor Configuration

```python
from cgm_format import FormatProcessor
from cgm_format.interface.cgm_interface import MINIMUM_DURATION_MINUTES, MAXIMUM_WANTED_DURATION_MINUTES

# Initialize processor with custom intervals
processor = FormatProcessor(
    expected_interval_minutes=5,     # CGM reading interval (5 min for Dexcom, 15 min for Libre)
    small_gap_max_minutes=19,        # Max gap to interpolate (3 intervals + 80% tolerance)
    snap_to_grid=True                # Align interpolated points to sync grid (default, ensures idempotency)
)

# Stage 4: Fill gaps
processed_df = processor.interpolate_gaps(unified_df)

# Stage 5 (Optional): Synchronize to fixed intervals
# synchronized_df = processor.synchronize_timestamps(processed_df)

# Stage 6: Prepare for inference (returns full UnifiedFormat)
unified_df, warnings = processor.prepare_for_inference(
    processed_df,  # or synchronized_df if using Stage 5
    minimum_duration_minutes=MINIMUM_DURATION_MINUTES,        # Default: 60 (1 hour)
    maximum_wanted_duration=MAXIMUM_WANTED_DURATION_MINUTES   # Default: 480 (8 hours)
)

# Optional: Strip service columns for ML models
inference_df = FormatProcessor.to_data_only_df(unified_df)

# Check warnings
if processor.has_warnings():
    all_warnings = processor.get_warnings()
    print(f"Processing collected {len(all_warnings)} warnings")
```

## Advanced Usage

### Working with Schemas

```python
from cgm_format.formats.unified import CGM_SCHEMA, UnifiedEventType, Quality

# Get Polars schema
polars_schema = CGM_SCHEMA.get_polars_schema()
data_only_schema = CGM_SCHEMA.get_polars_schema(data_only=True)

# Get column names
all_columns = CGM_SCHEMA.get_column_names()
data_columns = CGM_SCHEMA.get_column_names(data_only=True)

# Get cast expressions for Polars
cast_exprs = CGM_SCHEMA.get_cast_expressions()
df = df.with_columns(cast_exprs)

# Use enums
event = UnifiedEventType.GLUCOSE  # "EGV_READ"
quality = 0                       # GOOD_QUALITY (no flags)
```

### Batch Processing with Inference Preparation

```python
from pathlib import Path
from cgm_format import FormatParser, FormatProcessor
import polars as pl

data_dir = Path("data")
output_dir = Path("data/inference_ready")
output_dir.mkdir(exist_ok=True)

processor = FormatProcessor()
results = []

for csv_file in data_dir.glob("*.csv"):
    try:
        # Parse to unified format
        unified_df = FormatParser.parse_from_file(csv_file)
        
        # Process for inference
        processed_df = processor.interpolate_gaps(unified_df)
        unified_df, warnings = processor.prepare_for_inference(processed_df)
        inference_df = FormatProcessor.to_data_only_df(unified_df)
        
        # Add patient identifier
        patient_id = csv_file.stem
        inference_df = inference_df.with_columns([
            pl.lit(patient_id).alias('patient_id')
        ])
        
        results.append(inference_df)
        
        # Save individual file
        output_file = output_dir / f"{patient_id}_inference.csv"
        FormatParser.to_csv_file(inference_df, str(output_file))
        
        warning_str = f"warnings={warnings.value}" if warnings else "OK"
        print(f"✓ {csv_file.name}: {len(inference_df)} records, {warning_str}")
        
    except Exception as e:
        print(f"✗ Failed {csv_file.name}: {e}")

# Combine all processed data
if results:
    combined_df = pl.concat(results)
    FormatParser.to_csv_file(combined_df, str(output_dir / "combined_inference.csv"))
    print(f"\n✓ Combined {len(results)} files into single dataset")
```

### Format Detection and Validation

```python
from examples.example_schema_usage import run_format_detection_and_validation
from pathlib import Path

# Validate all files in data directory
run_format_detection_and_validation(
    data_dir=Path("data"),
    parsed_dir=Path("data/parsed"),
    output_file=Path("validation_report.txt")
)
```

This generates a detailed report with:

- Format detection statistics
- Frictionless schema validation results (if library installed)
- Known vendor quirks automatically suppressed

## Supported Formats

### Dexcom Clarity Export

- CSV with metadata rows (rows 2-11)
- Variable-length rows (non-EGV events missing trailing columns)
- High/Low glucose markers for out-of-range values
- Event types: EGV, Insulin, Carbs, Exercise
- Multiple timestamp format variants

### FreeStyle Libre

- CSV with metadata row 1, header row 2
- Record type filtering (0=glucose, 4=insulin, 5=food)
- Multiple timestamp format variants
- Separate rapid/long insulin columns

### Unified Format

- Standardized CSV with header row 1
- ISO 8601 timestamps
- Service columns + data columns
- Validates existing unified format files

## Project Structure

```text
cgm_format/
├── src/
│   └── cgm_format/              # Main package
│       ├── __init__.py          # Package exports (FormatParser, FormatProcessor)
│       ├── format_parser.py  # FormatParser implementation (Stages 1-3)
│       ├── format_processor.py  # FormatProcessor implementation (Stages 4-6)
│       ├── interface/           # Abstract interfaces and schema infrastructure
│       │   ├── cgm_interface.py # CGMParser and CGMProcessor interfaces
│       │   ├── schema.py        # Base schema definition system
│       │   └── PIPELINE.md      # Pipeline documentation
│       └── formats/             # Format-specific schemas and definitions
│           ├── unified.py       # Unified format schema and enums
│           ├── unified.json     # Frictionless schema export
│           ├── dexcom.py        # Dexcom format schema and constants
│           ├── dexcom.json      # Frictionless schema for Dexcom
│           ├── libre.py         # Libre format schema and constants
│           ├── libre.json       # Frictionless schema for Libre
│           └── UNIFIED_FORMAT.md # Unified format specification
├── examples/                    # Example scripts
│   ├── usage_example.py         # Runnable usage examples
│   └── example_schema_usage.py  # Format detection & validation examples
├── tests/                       # Pytest test suite
│   ├── test_format_parser.py # Parsing and conversion tests
│   ├── test_format_processor.py # Processing tests
│   └── test_schema.py           # Schema validation tests
├── data/                        # Test data and parsed outputs
│   └── parsed/                  # Converted unified format files
├── pyproject.toml               # Package configuration (hatchling)
├── USAGE.md                     # Complete usage guide for inference
└── README.md                    # This file
```

## Architecture

### Two-Layer Interface Design

**CGMParser** (Stages 1-3): Vendor-specific parsing to unified format

- `decode_raw_data()` - Encoding cleanup
- `detect_format()` - Format detection
- `parse_to_unified()` - Vendor CSV → UnifiedFormat with sequence detection
- `detect_and_assign_sequences()` - Glucose-gap-based sequence assignment (automatic)

**CGMProcessor** (Stages 4-5): Vendor-agnostic operations on unified data

- `interpolate_gaps()` - Gap detection and interpolation with calibration marking
- `synchronize_timestamps()` - Timestamp alignment to fixed intervals (lossless)
- `mark_calibration_periods()` - 24hr post-gap quality marking
- `mark_time_duplicates()` - Duplicate timestamp flagging
- `prepare_for_inference()` - ML preparation with quality checks and truncation

The current implementation:
- `FormatParser` implements the `CGMParser` interface (Stages 1-3)
- `FormatProcessor` implements the `CGMProcessor` interface (Stages 4-5)

All operations are **idempotent** through `original_datetime` preservation and quality flags.

### Processing Stages Implementation

**Stage 1-3 (FormatParser):**
- BOM removal and encoding normalization
- Pattern-based format detection (first 15 lines)
- Vendor-specific CSV parsing with quirk handling
- Timestamp format probing (handles multiple formats per vendor)
- Column mapping to unified schema
- Service field population (sequence_id, event_type, quality, original_datetime)
- Glucose-only gap detection and sequence assignment (two-pass approach)
- Schema validation and enforcement

**Stage 4 (FormatProcessor.interpolate_gaps):**
- Time difference calculation between consecutive glucose readings
- Small gap interpolation (> expected_interval, ≤ small_gap_max_minutes)
- Linear interpolation with snap-to-grid mode for idempotency
- Imputation row creation with `Quality.IMPUTATION` + `Quality.SYNCHRONIZATION` flags
- Warning collection for imputed data
- Uses `original_datetime` for gap detection (ensures idempotency)

**Stage 5 (FormatProcessor.synchronize_timestamps):**
- Timestamp rounding to minute boundaries using grid alignment
- Each source row maps to nearest grid point (lossless operation)
- Grid calculations use `original_datetime` (ensures idempotency)
- All rows marked with `Quality.SYNCHRONIZATION` flag
- Preserves sequence boundaries (processes each independently)

**Stage 6 (FormatProcessor.prepare_for_inference):**
- Zero-data validation (raises `ZeroValidInputError`)
- Latest valid sequence selection with fallback
- Time duplicate marking with `Quality.TIME_DUPLICATE` flag
- Calibration period marking (24h after gaps ≥2h45m) with `Quality.SENSOR_CALIBRATION` flag
- Duration verification with `TOO_SHORT` warning
- Quality flag detection (`OUT_OF_RANGE`, `SENSOR_CALIBRATION`, `IMPUTATION`, `TIME_DUPLICATES`)
- Sequence truncation from beginning (preserves most recent data)
- Optional service column removal via `to_data_only_df()`
- Warning flag aggregation and return

### Processing Configuration Parameters

**FormatProcessor initialization:**

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `expected_interval_minutes` | 5 | Normal reading interval | Grid spacing for synchronization; gap detection baseline |
| `small_gap_max_minutes` | 19 | Max gap to interpolate | Gaps > this are not filled; gaps ≤ this are filled with interpolation |
| `snap_to_grid` | True | Align interpolated points to grid | When True, ensures idempotency between interpolate and sync operations |

**Common configurations:**

```python
# Dexcom G6/G7 (5-minute readings)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=19)

# FreeStyle Libre (manual scans, typically 15 min)
processor = FormatProcessor(expected_interval_minutes=15, small_gap_max_minutes=57)  # 3 intervals + 80%

# Strict quality (minimal imputation)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=10)

# Lenient (more gap filling for sparse data)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=30)
```

**prepare_for_inference parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minimum_duration_minutes` | 60 | Minimum sequence duration required (warns if shorter) |
| `maximum_wanted_duration` | 480 | Maximum duration to keep (truncates from beginning) |

**Constants from interface:**

```python
from cgm_format.interface.cgm_interface import (
    MINIMUM_DURATION_MINUTES,           # 60 (1 hour)
    MAXIMUM_WANTED_DURATION_MINUTES,    # 480 (8 hours)
    CALIBRATION_GAP_THRESHOLD,          # 9900 seconds (2h45m)
    CALIBRATION_PERIOD_HOURS,           # 24 hours
)
```

### Schema System

Schemas are defined using `CGMSchemaDefinition` from `interface/schema.py`:

- **Type-safe**: Polars dtypes with strict validation
- **Vendor-specific**: Each format has its own schema with quirks documented
- **Validation modes**: Validate (raise on mismatch) or enforce (cast and fix)
- **Frictionless export**: Auto-generate validation schemas
- **Dialect support**: CSV parsing hints (header rows, comment rows, etc.)
- **Stable sorting**: Deterministic row ordering for idempotency

**Configuration:**

```python
from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor
from cgm_format.interface.cgm_interface import ValidationMethod

# Parser validation (class variable)
FormatParser.validation_mode = ValidationMethod.INPUT  # Validate inputs (default)
FormatParser.validation_mode = ValidationMethod.INPUT_FORCED  # Enforce schema on inputs

# Processor validation (instance parameter)
processor = FormatProcessor(validation_mode=ValidationMethod.INPUT)
```

**Schema usage:**

```python
from cgm_format.formats.unified import CGM_SCHEMA

# Validate DataFrame matches schema (raises on mismatch)
validated_df = CGM_SCHEMA.validate_dataframe(df, enforce=False)

# Enforce schema (add missing columns, cast types, reorder, sort)
enforced_df = CGM_SCHEMA.validate_dataframe(df, enforce=True)

# Get stable sort keys for deterministic ordering
sort_keys = CGM_SCHEMA.get_stable_sort_keys()
df = df.sort(sort_keys)
```

## Error Handling

### Exceptions

| Exception | Base | Description |
|-----------|------|-------------|
| `UnknownFormatError` | `ValueError` | Format cannot be detected |
| `MalformedDataError` | `ValueError` | CSV parsing or conversion failed |
| `MissingColumnError` | `MalformedDataError` | Required column missing from DataFrame |
| `ExtraColumnError` | `MalformedDataError` | Unexpected column present in DataFrame |
| `ColumnOrderError` | `MalformedDataError` | Columns not in correct schema order |
| `ColumnTypeError` | `MalformedDataError` | Column type doesn't match schema |
| `ZeroValidInputError` | `ValueError` | No valid data points found |

### Processing Warnings

The `FormatProcessor` collects quality warnings during processing:

| Warning Flag | Description | Triggered By |
|--------------|-------------|--------------|
| `ProcessingWarning.TOO_SHORT` | Sequence duration < minimum_duration_minutes | `prepare_for_inference()` |
| `ProcessingWarning.OUT_OF_RANGE` | Data contains OUT_OF_RANGE quality flag (sensor errors) | `prepare_for_inference()` |
| `ProcessingWarning.CALIBRATION` | Data contains calibration events or SENSOR_CALIBRATION quality flag | `prepare_for_inference()` |
| `ProcessingWarning.IMPUTATION` | Data contains IMPUTATION quality flag (interpolated data) | `interpolate_gaps()` |
| `ProcessingWarning.TIME_DUPLICATES` | Data contains TIME_DUPLICATE quality flag | `prepare_for_inference()` |

**Usage:**

```python
processor = FormatProcessor()
processed_df = processor.interpolate_gaps(unified_df)
inference_df, warnings = processor.prepare_for_inference(processed_df)

# Check individual warnings using bitwise AND
if warnings & ProcessingWarning.OUT_OF_RANGE:
    print("Sensor out-of-range errors detected")
if warnings & ProcessingWarning.CALIBRATION:
    print("Calibration events or post-calibration periods present")

# Get all warnings as list
all_warnings = processor.get_warnings()
print(f"Collected {len(all_warnings)} warnings")

# Check if any warnings exist
if processor.has_warnings():
    print("Processing completed with warnings")
```

## Testing

The library has comprehensive test coverage with real data (no mocking):

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_format_processor.py -v

# Run idempotency tests
uv run pytest tests/test_idempotency.py -v

# Generate validation report
uv run python examples/example_schema_usage.py

# Run usage examples with real data
uv run python examples/usage_example.py
```

**Test Coverage:**

- **test_format_detection_validation.py** - Format detection, Frictionless schema validation
- **test_integration_pipeline.py** - Full end-to-end pipeline on real data (no mocking)
- **test_format_processor.py** - Processor implementation: sync, interpolation, inference prep
- **test_format_converter.py** - Parser: detection, parsing, roundtrip, sequence detection
- **test_roundtrip_datetime.py** - Datetime type preservation through conversions
- **test_idempotency.py** - Idempotency and commutativity of operations
- **test_schema.py** - Schema validation and Frictionless conversion
- **test_utils.py** - Utility methods (split_glucose_events, to_data_only_df)

All tests verify:
- Data integrity and consistency
- Timestamp ordering and idempotency
- Lossless operations (no data loss)
- Schema compliance
- Error handling

See [`tests/README.md`](tests/README.md) for detailed test documentation.

## Development

### Regenerating Schema JSON Files

After modifying schema definitions:

```bash
# Regenerate unified.json
python3 -c "from cgm_format.formats.unified import regenerate_schema_json; regenerate_schema_json()"

# Regenerate dexcom.json
python3 -c "from cgm_format.formats.dexcom import regenerate_schema_json; regenerate_schema_json()"

# Regenerate libre.json
python3 -c "from cgm_format.formats.libre import regenerate_schema_json; regenerate_schema_json()"
```

### Adding New Vendor Formats

1. Create schema in `src/cgm_format/formats/your_vendor.py` using `CGMSchemaDefinition`
2. Add format to `SupportedCGMFormat` enum in `src/cgm_format/interface/cgm_interface.py`
3. Add detection patterns and implement parsing in `src/cgm_format/format_parser.py`
4. Add tests in `tests/test_format_parser.py`

## Requirements

- Python 3.10+
- polars 1.34.0+

Optional:

- pandas 2.3.3+ (compatibility layer)
- pyarrow 21.0.0+ (pandas conversion)
- frictionless 5.18.1+ (schema validation)
- pytest 8.0.0+ (testing)

## Documentation

- **[USAGE.md](USAGE.md)** - Complete usage guide for inference workflows
- **[examples/usage_example.py](examples/usage_example.py)** - Runnable examples with real data
- **[src/cgm_format/interface/PIPELINE.md](src/cgm_format/interface/PIPELINE.md)** - Detailed pipeline architecture
- **[src/cgm_format/formats/UNIFIED_FORMAT.md](src/cgm_format/formats/UNIFIED_FORMAT.md)** - Unified schema specification
- **[examples/example_schema_usage.py](examples/example_schema_usage.py)** - Schema validation examples

## License

See [LICENSE](LICENSE) file.
