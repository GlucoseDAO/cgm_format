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

```python
from format_converter import FormatParser
import polars as pl

# Parse any supported CGM file (Dexcom, Libre, or Unified)
unified_df = FormatParser.parse_from_file("data/example.csv")

# Access the data
print(unified_df.head())

# Save to unified format
FormatParser.to_csv_file(unified_df, "output.csv")
```

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
from interface.cgm_interface import SupportedCGMFormat

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

## Advanced Usage

### Working with Schemas

```python
from formats.unified import CGM_SCHEMA, UnifiedEventType, Quality

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

### Batch Processing

```python
from pathlib import Path
from format_converter import FormatParser

data_dir = Path("data")
parsed_dir = Path("data/parsed")
parsed_dir.mkdir(exist_ok=True)

for csv_file in data_dir.glob("*.csv"):
    try:
        # Parse to unified format
        unified_df = FormatParser.parse_from_file(csv_file)
        
        # Save with standardized name
        output_file = parsed_dir / f"{csv_file.stem}_unified.csv"
        FormatParser.to_csv_file(unified_df, output_file)
        
        print(f"✓ Processed {csv_file.name}")
    except Exception as e:
        print(f"✗ Failed {csv_file.name}: {e}")
```

### Format Detection and Validation

```python
from example_schema_usage import run_format_detection_and_validation
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
├── interface/                   # Abstract interfaces and schema infrastructure
│   ├── cgm_interface.py         # CGMParser and CGMProcessor interfaces
│   ├── schema.py                # Base schema definition system
│   └── PIPELINE.md              # Pipeline documentation
├── formats/                     # Format-specific schemas and definitions
│   ├── unified.py               # Unified format schema and enums
│   ├── unified.json             # Frictionless schema export
│   ├── dexcom.py                # Dexcom format schema and constants
│   ├── dexcom.json              # Frictionless schema for Dexcom
│   ├── libre.py                 # Libre format schema and constants
│   ├── libre.json               # Frictionless schema for Libre
│   └── UNIFIED_FORMAT.md        # Unified format specification
├── format_converter.py          # Main FormatParser implementation
├── example_schema_usage.py      # Format detection & validation examples
├── tests/                       # Pytest test suite
│   ├── test_format_converter.py # Parsing and conversion tests
│   └── test_schema.py           # Schema validation tests
└── data/                        # Test data and parsed outputs
    └── parsed/                  # Converted unified format files
```

## Architecture

### Two-Layer Interface Design

**CGMParser** (Stages 1-3): Vendor-specific parsing to unified format

- `decode_raw_data()` - Encoding cleanup
- `detect_format()` - Format detection
- `parse_to_unified()` - Vendor CSV → UnifiedFormat

**CGMProcessor** (Stages 4-5): Vendor-agnostic operations on unified data

- `synchronize_timestamps()` - Timestamp alignment (not yet implemented)
- `interpolate_gaps()` - Gap filling (not yet implemented)
- `prepare_for_inference()` - ML preparation (not yet implemented)

The current implementation (`FormatParser`) implements the `CGMParser` interface.

### Schema System

Schemas are defined using `CGMSchemaDefinition` from `interface/schema.py`:

- **Type-safe**: Polars dtypes with constraints
- **Vendor-specific**: Each format has its own schema with quirks documented
- **Frictionless export**: Auto-generate validation schemas
- **Dialect support**: CSV parsing hints (header rows, comment rows, etc.)

## Error Handling

| Exception | Base | Description |
|-----------|------|-------------|
| `UnknownFormatError` | `ValueError` | Format cannot be detected |
| `MalformedDataError` | `ValueError` | CSV parsing or conversion failed |
| `ZeroValidInputError` | `ValueError` | No valid data points found |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_format_converter.py -v

# Generate validation report
python3 example_schema_usage.py
```

## Development

### Regenerating Schema JSON Files

After modifying schema definitions:

```bash
# Regenerate unified.json
python3 -c "from formats.unified import regenerate_schema_json; regenerate_schema_json()"

# Regenerate dexcom.json
python3 -c "from formats.dexcom import regenerate_schema_json; regenerate_schema_json()"

# Regenerate libre.json
python3 -c "from formats.libre import regenerate_schema_json; regenerate_schema_json()"
```

### Adding New Vendor Formats

1. Create schema in `formats/your_vendor.py` using `CGMSchemaDefinition`
2. Add format to `SupportedCGMFormat` enum in `interface/cgm_interface.py`
3. Add detection patterns and implement parsing in `format_converter.py`
4. Add tests in `tests/test_format_converter.py`

## Requirements

- Python 3.12+
- polars 1.34.0+

Optional:

- pandas 2.3.3+ (compatibility layer)
- pyarrow 21.0.0+ (pandas conversion)
- frictionless 5.18.1+ (schema validation)
- pytest 8.0.0+ (testing)

## License

See [LICENSE](LICENSE) file.
