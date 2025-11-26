# cgm_format

Python library for converting vendor-specific Continuous Glucose Monitoring (CGM) data (Dexcom, Libre) into a standardized unified format for ML training and inference.

## Features

- **Vendor format detection**: Automatic detection of Dexcom, Libre, and Unified formats
- **Robust parsing**: Handles BOM marks, encoding artifacts, and vendor-specific CSV quirks
- **Unified schema**: Standardized data format with service columns (metadata) and data columns
- **Schema validation**: Frictionless Data Table Schema support for validation
- **Type-safe**: Polars-based with strict type definitions and enum support
- **Extensible**: Clean abstract interfaces for adding new vendor formats

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip3 install -e .

# Optional dependencies
uv pip install -e ".[extra]"  # pandas, pyarrow, frictionless
uv pip install -e ".[dev]"    # pytest
```

## Quick Start

### Basic Parsing

```python
from cgm_format import FormatParser
import polars as pl

# Parse any supported CGM file (Dexcom, Libre, or Unified)
unified_df = FormatParser.parse_from_file("data/example.csv")

# Access the data
print(unified_df.head())

# Save to unified format
FormatParser.to_csv_file(unified_df, "output.csv")
```

### Complete Inference Pipeline

```python
from cgm_format import FormatParser, FormatProcessor

# Stage 1-3: Parse vendor format to unified
unified_df = FormatParser.parse_from_file("data/dexcom_export.csv")

# Stage 4-5: Process for inference
processor = FormatProcessor(
    expected_interval_minutes=5,
    small_gap_max_minutes=15
)

# Fill gaps and create sequences
processed_df = processor.interpolate_gaps(unified_df)

# Prepare final inference data
inference_df, warnings = processor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=180,
    maximum_wanted_duration=1440
)

# Feed to ML model
predictions = your_model.predict(inference_df)
```

**See [USAGE.md](USAGE.md) for complete inference workflows and [examples/usage_example.py](examples/usage_example.py) for runnable examples.**

## Unified Format Schema

The library converts all vendor formats to a standardized schema with two types of columns:

### Service Columns (Metadata)

| Column | Type | Description |
|--------|------|-------------|
| `sequence_id` | `Int64` | Unique sequence identifier |
| `event_type` | `Utf8` | Event type (8-char code: EGV_READ, INS_FAST, CARBS_IN, etc.) |
| `quality` | `Int64` | Data quality (0=GOOD, 1=ILL, 2=SENSOR_CALIBRATION) |

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

The library implements a 3-stage parsing pipeline defined in the `CGMParser` interface:

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

Parse vendor CSV to unified format, handling vendor-specific quirks:

- Dexcom: High/Low glucose markers, variable-length rows, metadata rows
- Libre: Record type filtering, timestamp format variations

```python
unified_df = FormatParser.parse_to_unified(text_data, format_type)
```

All stages can be chained with convenience methods:

```python
# Parse from file
unified_df = FormatParser.parse_from_file("data.csv")

# Parse from bytes
unified_df = FormatParser.parse_from_bytes(raw_data)

# Parse from string
unified_df = FormatParser.parse_from_string(text_data)
```

See [`interface/PIPELINE.md`](interface/PIPELINE.md) for complete pipeline documentation.

### Stage 4: Gap Interpolation and Sequence Creation

The `FormatProcessor.interpolate_gaps()` method handles data continuity:

```python
from cgm_format import FormatProcessor

processor = FormatProcessor(
    expected_interval_minutes=5,    # Normal CGM reading interval
    small_gap_max_minutes=15        # Max gap size to interpolate
)

# Detect gaps, create sequences, and interpolate missing values
processed_df = processor.interpolate_gaps(unified_df)
```

**What it does:**

1. **Gap Detection**: Identifies gaps in continuous glucose monitoring data
2. **Sequence Creation**: Splits data at large gaps (>15 min default) into separate sequences
3. **Small Gap Interpolation**: Fills small gaps (≤15 min) with linearly interpolated glucose values
4. **Calibration Marking**: Marks 24-hour periods after gaps ≥2h45m as `Quality.SENSOR_CALIBRATION`
5. **Warning Collection**: Tracks imputation events via `ProcessingWarning.IMPUTATION`

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

Align timestamps to fixed-frequency intervals for ML models requiring regular time steps:

```python
# After interpolate_gaps(), synchronize to exact intervals
synchronized_df = processor.synchronize_timestamps(processed_df)

# Now all timestamps are at exact 5-minute intervals: 10:00:00, 10:05:00, 10:10:00, etc.
```

**What it does:**

1. Rounds timestamps to nearest minute boundary (removes seconds)
2. Creates fixed-frequency timestamps at `expected_interval_minutes` intervals
3. Linearly interpolates glucose values between measurements
4. Shifts discrete events (carbs, insulin, exercise) to nearest timestamp
5. Preserves sequence boundaries (processes each sequence independently)

**When to use:** Time-series models expecting fixed intervals (LSTM, transformers, ARIMA)  
**When to skip:** Models handling irregular timestamps, or when original timing is critical

### Stage 6: Inference Preparation

The `prepare_for_inference()` method performs final quality assurance and data extraction:

```python
# Prepare final inference-ready data
inference_df, warnings = processor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=180,      # Require 3 hours minimum
    maximum_wanted_duration=1440       # Truncate to last 24 hours if longer
)

# Check for quality issues
from cgm_format.interface.cgm_interface import ProcessingWarning

if warnings & ProcessingWarning.TOO_SHORT:
    print("Warning: Sequence shorter than minimum duration")
if warnings & ProcessingWarning.QUALITY:
    print("Warning: Data contains quality issues (ILL or SENSOR_CALIBRATION)")
if warnings & ProcessingWarning.IMPUTATION:
    print("Warning: Data contains interpolated values")
```

**What it does:**

1. **Validation**: Raises `ZeroValidInputError` if no valid glucose data exists
2. **Sequence Selection**: Keeps only the **latest** sequence (most recent timestamps)
3. **Duration Checks**: Warns if sequence < `minimum_duration_minutes`
4. **Quality Checks**: Collects warnings for calibration events and quality flags
5. **Truncation**: Keeps last N minutes if exceeding `maximum_wanted_duration`
6. **Column Extraction**: Returns only data columns (removes service metadata)

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
    small_gap_max_minutes=15         # Max gap to interpolate (larger gaps create new sequences)
)

# Stage 4: Fill gaps and create sequences
processed_df = processor.interpolate_gaps(unified_df)

# Stage 5 (Optional): Synchronize to fixed intervals
# synchronized_df = processor.synchronize_timestamps(processed_df)

# Stage 6: Prepare for inference
inference_df, warnings = processor.prepare_for_inference(
    processed_df,  # or synchronized_df if using Stage 5
    minimum_duration_minutes=MINIMUM_DURATION_MINUTES,        # Default: 180 (3 hours)
    maximum_wanted_duration=MAXIMUM_WANTED_DURATION_MINUTES   # Default: 1440 (24 hours)
)

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
quality = Quality.GOOD            # 0
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
        inference_df, warnings = processor.prepare_for_inference(processed_df)
        
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
│       ├── format_converter.py  # FormatParser implementation (Stages 1-3)
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
│   ├── test_format_converter.py # Parsing and conversion tests
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
- `parse_to_unified()` - Vendor CSV → UnifiedFormat

**CGMProcessor** (Stages 4-5): Vendor-agnostic operations on unified data

- `synchronize_timestamps()` - Timestamp alignment to fixed intervals
- `interpolate_gaps()` - Gap detection, sequence creation, and interpolation
- `prepare_for_inference()` - ML preparation with quality checks and truncation

The current implementation:
- `FormatParser` implements the `CGMParser` interface (Stages 1-3)
- `FormatProcessor` implements the `CGMProcessor` interface (Stages 4-5)

### Processing Stages Implementation

**Stage 1-3 (FormatParser):**
- BOM removal and encoding normalization
- Pattern-based format detection (first 15 lines)
- Vendor-specific CSV parsing with quirk handling
- Column mapping to unified schema
- Service field population (sequence_id, event_type, quality)

**Stage 4 (FormatProcessor.interpolate_gaps):**
- Time difference calculation between consecutive readings
- Sequence boundary detection (gaps > `small_gap_max_minutes`)
- Linear interpolation for small gaps (≤ `small_gap_max_minutes`)
- Imputation event creation with `event_type='IMPUTATN'`
- Calibration period marking (24h after gaps ≥ 2h45m)
- Warning collection for imputed data

**Stage 5 (FormatProcessor.synchronize_timestamps):**
- Timestamp rounding to minute boundaries
- Fixed-frequency grid generation at `expected_interval_minutes`
- Asof join (backward/forward) for value alignment
- Linear glucose interpolation between grid points
- Discrete event shifting to nearest timestamp

**Stage 6 (FormatProcessor.prepare_for_inference):**
- Zero-data validation (raises `ZeroValidInputError`)
- Latest sequence selection (max timestamp)
- Duration verification with `TOO_SHORT` warning
- Quality flag detection (`ILL`, `SENSOR_CALIBRATION`)
- Sequence truncation from beginning (preserves most recent data)
- Service column removal (data columns only)
- Warning flag aggregation and return

### Processing Configuration Parameters

**FormatProcessor initialization:**

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `expected_interval_minutes` | 5 | Normal reading interval | Grid spacing for synchronization; gap detection baseline |
| `small_gap_max_minutes` | 15 | Max gap to interpolate | Gaps > this create new sequences; gaps ≤ this are filled |

**Common configurations:**

```python
# Dexcom G6/G7 (5-minute readings)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=15)

# FreeStyle Libre (manual scans, typically 15 min)
processor = FormatProcessor(expected_interval_minutes=15, small_gap_max_minutes=45)

# Strict quality (minimal imputation)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=10)

# Lenient (more gap filling for sparse data)
processor = FormatProcessor(expected_interval_minutes=5, small_gap_max_minutes=30)
```

**prepare_for_inference parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minimum_duration_minutes` | 180 | Minimum sequence duration required (warns if shorter) |
| `maximum_wanted_duration` | 1440 | Maximum duration to keep (truncates from beginning) |

**Constants from interface:**

```python
from cgm_format.interface.cgm_interface import (
    MINIMUM_DURATION_MINUTES,           # 180 (3 hours)
    MAXIMUM_WANTED_DURATION_MINUTES,    # 1440 (24 hours)
    CALIBRATION_GAP_THRESHOLD,          # 9900 seconds (2h45m)
)
```

### Schema System

Schemas are defined using `CGMSchemaDefinition` from `interface/schema.py`:

- **Type-safe**: Polars dtypes with constraints
- **Vendor-specific**: Each format has its own schema with quirks documented
- **Frictionless export**: Auto-generate validation schemas
- **Dialect support**: CSV parsing hints (header rows, comment rows, etc.)

## Error Handling

### Exceptions

| Exception | Base | Description |
|-----------|------|-------------|
| `UnknownFormatError` | `ValueError` | Format cannot be detected |
| `MalformedDataError` | `ValueError` | CSV parsing or conversion failed |
| `ZeroValidInputError` | `ValueError` | No valid data points found |

### Processing Warnings

The `FormatProcessor` collects quality warnings during processing:

| Warning Flag | Description | Triggered By |
|--------------|-------------|--------------|
| `ProcessingWarning.TOO_SHORT` | Sequence duration < minimum_duration_minutes | `prepare_for_inference()` |
| `ProcessingWarning.QUALITY` | Data contains ILL or SENSOR_CALIBRATION quality flags | `prepare_for_inference()` |
| `ProcessingWarning.IMPUTATION` | Data contains interpolated values | `interpolate_gaps()` |
| `ProcessingWarning.CALIBRATION` | Data contains calibration events | `prepare_for_inference()` |

**Usage:**

```python
processor = FormatProcessor()
processed_df = processor.interpolate_gaps(unified_df)
inference_df, warnings = processor.prepare_for_inference(processed_df)

# Check individual warnings
if warnings & ProcessingWarning.QUALITY:
    print("Quality issues detected")

# Get all warnings as list
all_warnings = processor.get_warnings()
print(f"Collected {len(all_warnings)} warnings")

# Check if any warnings exist
if processor.has_warnings():
    print("Processing completed with warnings")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_format_converter.py -v

# Generate validation report
uv run python examples/example_schema_usage.py

# Run usage examples with real data
uv run python examples/usage_example.py
```

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
3. Add detection patterns and implement parsing in `src/cgm_format/format_converter.py`
4. Add tests in `tests/test_format_converter.py`

## Requirements

- Python 3.12+
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
