# Scripts

Helper scripts and CLI tools for the cgm_format project.

## cgm_cli.py

**Comprehensive CLI tool for CGM data processing** - provides access to all parser, processor, and validation features.

### Installation

```bash
# Install with CLI dependencies
uv add --optional cli cgm-format

# Or install in development mode
uv sync --extra cli
```

### Usage Modes

The CLI can be used in three ways:

```bash
# 1. As an installed command (after pip/uv install)
cgm-cli <command> [options]

# 2. As a Python module
python -m cgm_format.cgm_cli <command> [options]

# 3. As a direct script
python scripts/cgm_cli.py <command> [options]
```

### Commands

#### Format Detection & Parsing

**detect** - Detect the format of a CGM data file
```bash
cgm-cli detect input.csv [--verbose]
```

**parse** - Parse a CGM data file to unified format
```bash
cgm-cli parse input.csv --output unified.csv [--stats] [--preview]
```

**validate** - Validate a CSV file against its schema
```bash
cgm-cli validate input.csv [--format unified|dexcom|libre] [--verbose]
```

**report** - Generate comprehensive validation report for directory
```bash
cgm-cli report data/ --output validation_report.txt \
    [--pattern "*.csv"] \
    [--frictionless/--no-frictionless] \
    [--suppress-known/--show-all]
```

Features:
- Batch format detection for all files
- Optional Frictionless schema validation
- Automatic suppression of known vendor quirks (Dexcom High/Low, variable-length rows)
- Detailed text report with format breakdown, validation results, and schema info
- Similar to `examples/example_schema_usage.py` functionality

#### Data Processing

**process** - Process unified format data (interpolate, synchronize)
```bash
cgm-cli process unified.csv --output processed.csv \
    [--interpolate/--no-interpolate] \
    [--sync/--no-sync] \
    [--interval 5] \
    [--max-gap 19]
```

**pipeline** - Run the complete processing pipeline
```bash
cgm-cli pipeline input.csv --output final.csv \
    [--interval 5] \
    [--max-gap 19] \
    [--min-duration 15] \
    [--max-duration 1440] \
    [--glucose-only] \
    [--drop-duplicates]
```

Pipeline stages:
1. Parse vendor format to unified
2. Detect and assign sequences
3. Interpolate gaps
4. Synchronize timestamps
5. Prepare for inference (quality checks)
6. Convert to data-only format

#### File Information

**info** - Show information about a CGM data file
```bash
cgm-cli info input.csv [--detailed]
```

#### Batch Processing

**batch** - Batch process multiple CGM data files
```bash
cgm-cli batch data/ --output processed/ \
    [--pattern "*.csv"] \
    [--command parse|process|pipeline] \
    [--continue/--stop]
```

### Examples

```bash
# Detect format of a file
cgm-cli detect data/patient_export.csv

# Parse Dexcom file to unified format
cgm-cli parse data/dexcom_export.csv -o data/unified.csv --stats

# Run full pipeline with glucose-only output
cgm-cli pipeline data/libre_export.csv -o output.csv --glucose-only

# Get detailed info about a file
cgm-cli info data/unified.csv --detailed

# Validate unified format file
cgm-cli validate data/unified.csv

# Generate comprehensive validation report (like example_schema_usage.py)
cgm-cli report data/ -o validation_report.txt --frictionless

# Batch process all CSV files in a directory
cgm-cli batch data/raw/ --output data/processed/ --command pipeline

# Process with custom parameters
cgm-cli process data/unified.csv -o processed.csv \
    --interval 5 \
    --max-gap 20 \
    --interpolate \
    --sync
```

### Features

- **Rich output** - Colored terminal output with progress indicators
- **Comprehensive statistics** - Show glucose stats, event counts, quality flags
- **Processing warnings** - Detect and report data quality issues
- **Batch processing** - Process entire directories at once
- **Flexible configuration** - Control all processing parameters
- **Multiple output formats** - Full unified or data-only formats

---

## regenerate_all_schemas.py

Automatically regenerates all JSON schema files from their Python schema definitions.

### Usage

```bash
# As executable (recommended - uses uv automatically)
./scripts/regenerate_all_schemas.py

# Or using uv explicitly
uv run python scripts/regenerate_all_schemas.py

# Or directly with Python (requires dependencies installed)
python scripts/regenerate_all_schemas.py
```

### What it does

1. Discovers all format modules in `src/cgm_format/formats/` (excluding `__init__.py` and `*_WIP.py`)
2. Dynamically imports each module
3. Calls `regenerate_schema_json()` function if it exists
4. Generates/updates corresponding `.json` schema files

### Output

The script regenerates:
- `dexcom.json` - Dexcom G6/G7 format schema
- `libre.json` - FreeStyle Libre 3 format schema
- `unified.json` - Unified CGM format schema

### When to run

Run this script after:
- Modifying enum values in format definitions
- Adding/removing columns in schemas
- Changing column descriptions or constraints
- Any other schema-related changes

This ensures the JSON schema files stay in sync with the Python schema definitions.

## scrub_synthetic_libre.py

Creates synthetic FreeStyle Libre CGM data from real data for CI testing and demos.

### Usage

```bash
uv run python scripts/scrub_synthetic_libre.py INPUT_FILE OUTPUT_FILE [--seed SEED]
```

### Arguments

- `INPUT_FILE` - Path to input FreeStyle Libre CSV file
- `OUTPUT_FILE` - Path to output synthetic CSV file
- `--seed` - Random seed for reproducibility (default: 42)

### Example

```bash
uv run python scripts/scrub_synthetic_libre.py \
    data/FreeStyle_Libre_3__11-12-2024.csv \
    data/FreeStyle_Libre_3_synthetic.csv
```

### Transformations Applied

1. **Serial Number** - Replaces with random UUID in same format
2. **Dates** - Changes to 1961-04-12 base date (Gagarin's space flight) while preserving relative timing
3. **Patient Name** - Replaces with "Gagarin"
4. **Patient Notes** - Removes all patient notes (Cyrillic text)
5. **Glucose Values** - Applies random baseline offset (10-20 mg/dL) + noise (±1) to:
   - Historic Glucose mg/dL
   - Scan Glucose mg/dL
   - Strip Glucose mg/dL
6. **Timestamps** - Shifts all timestamps by random minutes (multiple of 5)

### When to use

- Creating synthetic test data for CI/CD pipelines
- Generating demo data without exposing real patient information
- Testing format parsers with realistic but fake data

## scrub_synthetic_dexcom.py

Creates synthetic Dexcom CGM data from real Clarity export files for CI testing and demos.

### Usage

```bash
uv run python scripts/scrub_synthetic_dexcom.py INPUT_FILE OUTPUT_FILE [--seed SEED]
```

### Arguments

- `INPUT_FILE` - Path to input Dexcom Clarity CSV file
- `OUTPUT_FILE` - Path to output synthetic CSV file
- `--seed` - Random seed for reproducibility (default: 42)

### Example

```bash
uv run python scripts/scrub_synthetic_dexcom.py \
    data/Clarity_Export__Patient_2025-05-14_154517.csv \
    data/Clarity_Export_synthetic.csv
```

### Transformations Applied

1. **Transmitter ID** - Replaces with random 6-character alphanumeric ID
2. **Dates** - Changes to 1961-04-12 base date (Gagarin's space flight) while preserving relative timing
3. **Patient Name** - Replaces with "Gagarin"
4. **Glucose Values** - Applies random baseline offset (10-20 mg/dL) + noise (±1) to EGV readings
5. **Timestamps** - Shifts all timestamps by random minutes (multiple of 5)
6. **Transmitter Time** - Adjusts transmitter time accordingly (in seconds)

### When to use

- Creating synthetic test data for CI/CD pipelines
- Generating demo data without exposing real patient information
- Testing Dexcom format parsers with realistic but fake data

## Notes

All scrubber scripts:
- Use Polars for efficient data processing
- Support reproducible output via `--seed` parameter
- Preserve CSV structure and data relationships
- Include full type hints
- Use typer for CLI interface

