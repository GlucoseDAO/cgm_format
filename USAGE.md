# Usage Guide: CGM Data Processing for Inference

This guide demonstrates the complete workflow for processing CGM data from vendor formats (Dexcom, Libre) through to inference-ready datasets.

## Overview

The processing pipeline consists of two main components:

1. **FormatParser** (Stages 1-3): Parse vendor-specific CSV to unified format
2. **FormatProcessor** (Stages 4-5): Process unified data for ML inference

```text
Vendor CSV → Parse → Unified Format → Interpolate → Synchronize → Inference Ready
            (Stage 1-3)              (Stage 4)     (Stage 5)
```

**Note**: `FormatProcessor` uses classmethods only - no need to instantiate. Call methods directly on the class.

## Quick Start: End-to-End Inference Pipeline

```python
from cgm_format import FormatParser, FormatProcessor

# Stage 1-3: Parse vendor format to unified
unified_df = FormatParser.parse_file("data/dexcom_export.csv")

# Stage 4a: Detect and assign sequences based on gaps
# FormatProcessor uses classmethods - call directly on class
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=19  # Gaps > 19 min create new sequences
)

# Stage 4b: Interpolate gaps within sequences
unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,    # CGM reads every 5 minutes
    small_gap_max_minutes=19        # Interpolate gaps up to 19 min (default: 3 intervals + 80% tolerance)
)

# Stage 4b: Align to fixed-frequency timestamps (optional but recommended)
unified_df = FormatProcessor.synchronize_timestamps(
    unified_df,
    expected_interval_minutes=5
)

# Stage 5: Prepare for inference with quality checks
inference_df, warning_flags = FormatProcessor.prepare_for_inference(
    unified_df,
    minimum_duration_minutes=15,       # 15 minutes minimum (default: 60)
    maximum_wanted_duration=24 * 60    # 24 hours maximum (1440 minutes, default: 480)
)

# Convert to glucose-only data format for ML model
glucose_only_df = FormatProcessor.to_data_only_df(
    inference_df,
    drop_service_columns=False,  # Keep metadata columns like sequence_id, quality
    drop_duplicates=True,        # Remove duplicate timestamps
    glucose_only=True            # Filter to glucose readings only
)

# Check data quality warnings
if warning_flags:
    print(f"Processing warnings: {warning_flags}")

# Feed to ML model
model_predictions = your_model.predict(glucose_only_df)
```

## Stage-by-Stage Workflow

### Stage 1-3: Parsing Vendor Formats

The `FormatParser` handles all vendor-specific quirks and converts to unified format.

#### Parse from File (Recommended)

```python
from cgm_format import FormatParser
from pathlib import Path

# Automatically detects Dexcom, Libre, or Unified format
file_path = Path("path/to/cgm_export.csv")
unified_df = FormatParser.parse_file(file_path)

print(f"Parsed {len(unified_df)} data points")
print(unified_df.head())
```

#### Check Format Support

```python
# Check if a file format is supported before parsing
with open("unknown_file.csv", 'rb') as f:
    raw_data = f.read()
    
if FormatParser.format_supported(raw_data):
    unified_df = FormatParser.parse_from_bytes(raw_data)
else:
    print("Unsupported format - skipping file")
```

#### Parse from Base64 (Web APIs)

```python
# Useful for web applications that receive base64-encoded CSV uploads
base64_data = request.form['cgm_file']  # Base64 string from web form
unified_df = FormatParser.parse_base64(base64_data)

# Or handle data URIs
data_uri = "data:text/csv;base64,Q29udGVudCxIZXJl..."
if data_uri.startswith("data:"):
    base64_part = data_uri.split(",")[1]
    unified_df = FormatParser.parse_base64(base64_part)
```

#### Parse with Manual Stages (Advanced)

```python
from cgm_format.interface.cgm_interface import SupportedCGMFormat

# Read raw file
with open("data.csv", 'rb') as f:
    raw_data = f.read()

# Stage 1: Decode and clean encoding
text_data = FormatParser.decode_raw_data(raw_data)

# Stage 2: Detect format
format_type = FormatParser.detect_format(text_data)
print(f"Detected format: {format_type.name}")

# Stage 3: Parse to unified
unified_df = FormatParser.parse_to_unified(text_data, format_type)
```

#### Handle Dexcom High/Low Values

```python
from cgm_format.formats.unified import Quality
import polars as pl

# Dexcom marks out-of-range readings as "High" or "Low"
# These are replaced with numeric placeholders (default: 401, 39 mg/dL)
# and marked with the OUT_OF_RANGE quality flag

unified_df = FormatParser.parse_file("dexcom_export.csv")

# Check for out-of-range readings
out_of_range_count = unified_df.filter(
    (pl.col('quality') & Quality.OUT_OF_RANGE.value) != 0
).height

print(f"Found {out_of_range_count} out-of-range readings")
```

### Stage 4: Gap Interpolation and Sequence Creation

The processing workflow starts with sequence detection, followed by gap interpolation.

#### Step 1: Detect and Assign Sequences

**Note**: `FormatProcessor` uses classmethods - no need to instantiate. Call methods directly on the class.

First, explicitly detect sequences based on large gaps in glucose data:

```python
# Detect and assign sequences based on gaps in glucose readings
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,       # Normal CGM interval
    large_gap_threshold_minutes=19     # Gaps > 19 min create new sequences (default)
)

# Check sequences created
sequence_count = unified_df['sequence_id'].n_unique()
print(f"Created {sequence_count} sequence(s)")
```

**Note**: This method is automatically called by `interpolate_gaps()` if sequences aren't assigned yet, but it's best practice to call it explicitly for clarity.

#### Step 2: Interpolate Small Gaps

After sequences are assigned, interpolate small gaps within each sequence:

```python
# FormatProcessor uses classmethods - no need to instantiate
unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,    # Normal CGM interval
    small_gap_max_minutes=19,       # Max gap to interpolate (default)
    snap_to_grid=True               # Align interpolated points to grid (default, ensures idempotency)
)

# Check if imputation occurred
from cgm_format.formats.unified import Quality

imputed_count = unified_df.filter(
    (pl.col('quality') & Quality.IMPUTATION.value) != 0
).height

if imputed_count > 0:
    print(f"Data contains {imputed_count} interpolated values")
```

#### Understanding Sequences

Data is split into sequences when gaps exceed `large_gap_threshold_minutes` (default 19 minutes).

**Best Practice**: Explicitly call `detect_and_assign_sequences()` before other processing steps:

```python
import polars as pl

# Step 1: Detect sequences
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=19
)

# Step 2: Analyze sequences
sequence_info = unified_df.group_by('sequence_id').agg([
    pl.col('datetime').min().alias('start_time'),
    pl.col('datetime').max().alias('end_time'),
    pl.col('datetime').count().alias('num_points'),
])

for row in sequence_info.iter_rows(named=True):
    duration = (row['end_time'] - row['start_time']).total_seconds() / 3600
    print(f"Sequence {row['sequence_id']}: "
          f"{duration:.1f} hours, {row['num_points']} points")
```

#### Snap-to-Grid Mode

**Snap-to-Grid Mode** (`snap_to_grid=True`, default):
- Interpolated points are placed on the sequence grid (same grid as `synchronize_timestamps`)
- Ensures idempotency: `interpolate → sync` and `sync → interpolate` produce equivalent results
- Adds both `Quality.IMPUTATION` and `Quality.SYNCHRONIZATION` flags to interpolated points

**Non-Grid Mode** (`snap_to_grid=False`):
- Interpolated points placed at regular intervals from previous timestamp
- Only adds `Quality.IMPUTATION` flag (no SYNCHRONIZATION)
- May not align with synchronization grid

### Stage 5: Timestamp Synchronization (Optional)

Aligns timestamps to exact minute boundaries with fixed intervals.

**Lossless Operation**: Synchronization keeps ALL source rows and only rounds their timestamps to the grid.

**Idempotency**: `synchronize_timestamps()` and `interpolate_gaps()` can be called in any order:
- Both use the same grid alignment based on the first timestamp in each sequence
- You can safely call: sync→interpolate OR interpolate→sync (results are equivalent)
- Calling the same operation twice has no additional effect

```python
# Option 1: Interpolate then synchronize (recommended)
unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=19
)
synchronized_df = FormatProcessor.synchronize_timestamps(
    unified_df,
    expected_interval_minutes=5
)

# Option 2: Synchronize then interpolate (also works - produces same result)
# synchronized_df = FormatProcessor.synchronize_timestamps(unified_df, expected_interval_minutes=5)
# synchronized_df = FormatProcessor.interpolate_gaps(synchronized_df, expected_interval_minutes=5, small_gap_max_minutes=19)

# All timestamps are now at exact 5-minute intervals
print(synchronized_df['datetime'].head(10))
```

### Stage 6: Inference Preparation

The `prepare_for_inference()` method performs final QA and returns full UnifiedFormat.

**Operations performed**:
1. Validation: Checks for zero valid data points (raises `ZeroValidInputError`)
2. Sequence Selection: Keeps only the **latest** sequence (most recent timestamps)
3. **Marks time duplicates**: Adds `TIME_DUPLICATE` quality flag to duplicate timestamps
4. **Marks calibration periods**: Adds `SENSOR_CALIBRATION` flag for 24h after large gaps (≥2h 45min)
5. Duration Check: Warns if sequence < minimum_duration_minutes
6. Quality Checks: Collects warnings for calibration, quality issues, imputation
7. Truncation: Keeps last N minutes if exceeding maximum_wanted_duration
8. Returns: Full UnifiedFormat with all columns

```python
inference_df, warning_flags = FormatProcessor.prepare_for_inference(
    synchronized_df,
    minimum_duration_minutes=15,       # 15 minutes minimum (default: 60)
    maximum_wanted_duration=24 * 60    # 24 hours maximum (1440 minutes, default: 480)
)

# Convert to glucose-only data for ML model
glucose_only_df = FormatProcessor.to_data_only_df(
    inference_df,
    drop_service_columns=False,  # Keep metadata columns (sequence_id, quality, event_type)
    drop_duplicates=True,        # Remove duplicate timestamps
    glucose_only=True            # Filter to glucose readings only (EGV_READ events)
)

# Check warnings
from cgm_format.interface.cgm_interface import ProcessingWarning

if warning_flags:
    if warning_flags & ProcessingWarning.TOO_SHORT:
        print("Warning: Sequence shorter than minimum duration")
    
    if warning_flags & ProcessingWarning.QUALITY:
        print("Warning: Data contains quality issues")
    
    if warning_flags & ProcessingWarning.IMPUTATION:
        print("Warning: Data contains interpolated values")
    
    if warning_flags & ProcessingWarning.CALIBRATION:
        print("Warning: Data contains calibration events")
    
    if warning_flags & ProcessingWarning.TIME_DUPLICATES:
        print("Warning: Data contains duplicate timestamps")
```

#### Inference DataFrame Structure

After `to_data_only_df()` with `glucose_only=True`:

```python
# glucose_only_df columns when drop_service_columns=False:
# - sequence_id: Int64 (sequence identifier)
# - original_datetime: Datetime (original timestamp before synchronization)
# - quality: Int64 (quality flags: 0=GOOD, 1=OUT_OF_RANGE, 2=SENSOR_CALIBRATION, 4=IMPUTATION, 8=TIME_DUPLICATE, 16=SYNCHRONIZATION)
# - event_type: Utf8 (always "EGV_READ" for glucose-only data)
# - datetime: Datetime (timestamp, may be synchronized to grid)
# - glucose: Float64 (mg/dL, always present for glucose-only data)
# - carbs: Float64 (grams, nullable)
# - insulin_slow: Float64 (units, nullable)
# - insulin_fast: Float64 (units, nullable)
# - exercise: Int64 (seconds, nullable)

print(glucose_only_df.columns)
# ['sequence_id', 'original_datetime', 'quality', 'event_type', 'datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']

# If you want only data columns without metadata, use drop_service_columns=True:
data_only_df = FormatProcessor.to_data_only_df(
    inference_df,
    drop_service_columns=True,
    glucose_only=True
)
print(data_only_df.columns)
# ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
```

## Common Workflows

### Workflow 1: Basic Inference Pipeline

For most ML use cases, use the complete pipeline:

```python
from cgm_format import FormatParser, FormatProcessor
from pathlib import Path

# Parse
file_path = Path("cgm_data.csv")
unified_df = FormatParser.parse_file(file_path)

# Process - all methods are classmethods
# Step 1: Detect and assign sequences
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=19
)

# Step 2: Interpolate small gaps within sequences
unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=19
)

# Step 3: Synchronize timestamps to fixed grid
unified_df = FormatProcessor.synchronize_timestamps(
    unified_df,
    expected_interval_minutes=5
)

# Prepare for inference
inference_df, warning_flags = FormatProcessor.prepare_for_inference(
    unified_df,
    minimum_duration_minutes=15,
    maximum_wanted_duration=24 * 60
)

# Convert to glucose-only data
glucose_only_df = FormatProcessor.to_data_only_df(
    inference_df,
    drop_service_columns=False,
    drop_duplicates=True,
    glucose_only=True
)

# Check quality
if warning_flags:
    print(f"Data quality warnings: {warning_flags}")
    # Decide whether to proceed or reject data

# Feed to model
predictions = model.predict(glucose_only_df)
```

### Workflow 2: Batch Processing Multiple Files

```python
from pathlib import Path
from cgm_format import FormatParser, FormatProcessor
import polars as pl

data_dir = Path("data/exports")
results = []

for csv_file in data_dir.glob("*.csv"):
    try:
        # Check if format is supported
        with open(csv_file, 'rb') as f:
            if not FormatParser.format_supported(f.read()):
                print(f"✗ {csv_file.name}: Unsupported format")
                continue
        
        # Parse and process
        unified_df = FormatParser.parse_file(csv_file)
        
        # Step 1: Detect sequences
        unified_df = FormatProcessor.detect_and_assign_sequences(
            unified_df,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=19
        )
        
        # Step 2: Interpolate gaps
        unified_df = FormatProcessor.interpolate_gaps(
            unified_df,
            expected_interval_minutes=5,
            small_gap_max_minutes=19
        )
        
        # Step 3: Synchronize timestamps
        unified_df = FormatProcessor.synchronize_timestamps(
            unified_df,
            expected_interval_minutes=5
        )
        
        # Prepare for inference
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            unified_df,
            minimum_duration_minutes=15,
            maximum_wanted_duration=24 * 60
        )
        
        # Convert to glucose-only data
        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        
        # Add patient_id from filename
        patient_id = csv_file.stem
        glucose_only_df = glucose_only_df.with_columns([
            pl.lit(patient_id).alias('patient_id')
        ])
        
        results.append(glucose_only_df)
        print(f"✓ {csv_file.name}: {len(glucose_only_df)} points, warnings={warning_flags}")
        
    except Exception as e:
        print(f"✗ {csv_file.name}: {e}")

# Combine all patient data
all_data = pl.concat(results)
```

### Workflow 3: Custom Preprocessing for Research

```python
from cgm_format import FormatParser, FormatProcessor
from cgm_format.formats.unified import Quality
import polars as pl

# Parse
unified_df = FormatParser.parse_file("research_data.csv")

# Step 1: Detect sequences with custom threshold
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=10  # Stricter sequence splitting
)

# Step 2: Custom gap handling - only interpolate very small gaps
unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=10  # Only interpolate very small gaps
)

# Filter out any imputed or calibration data using quality flags
high_quality_df = unified_df.filter(
    (pl.col('quality') & (Quality.IMPUTATION.value | Quality.SENSOR_CALIBRATION.value)) == 0
)

# Synchronize to exact intervals
synchronized_df = FormatProcessor.synchronize_timestamps(
    high_quality_df,
    expected_interval_minutes=5
)

# Prepare with strict duration requirements
inference_df, warnings = FormatProcessor.prepare_for_inference(
    synchronized_df,
    minimum_duration_minutes=360,  # Require 6 hours
    maximum_wanted_duration=2880   # Allow 48 hours
)

if warnings:
    print("Warning: Data may not meet research quality standards")
```

### Workflow 4: Splitting Glucose and Events

For workflows that need to process glucose readings separately from other events:

```python
# Parse data with multiple event types
unified_df = FormatParser.parse_file("data/cgm_with_events.csv")

# Split into glucose readings and other events
glucose_df, events_df = FormatProcessor.split_glucose_events(unified_df)

# glucose_df contains: EGV_READ events (including interpolated ones marked with IMPUTATION quality flag)
# events_df contains: INSULIN_FAST, INSULIN_SLOW, CARBS, EXERCISE, CALIBRATION, etc.

# Process glucose data for inference
# Step 1: Detect sequences
glucose_df = FormatProcessor.detect_and_assign_sequences(
    glucose_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=19
)

# Step 2: Interpolate gaps
glucose_df = FormatProcessor.interpolate_gaps(
    glucose_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=19
)

# Step 3: Prepare for inference
inference_df, warnings = FormatProcessor.prepare_for_inference(
    glucose_df,
    minimum_duration_minutes=60,
    maximum_wanted_duration=480
)

# Strip service columns if needed for ML
glucose_only_df = FormatProcessor.to_data_only_df(inference_df)

# Analyze events separately
insulin_doses = events_df.filter(
    pl.col('event_type').str.contains('INSULIN')
).select(['datetime', 'insulin_fast', 'insulin_slow'])

carb_intake = events_df.filter(
    pl.col('event_type') == 'CARBS_IN'
).select(['datetime', 'carbs'])
```

## Error Handling

### Common Exceptions

```python
from cgm_format.interface.cgm_interface import (
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError
)

try:
    unified_df = FormatParser.parse_file("data.csv")
    
except UnknownFormatError as e:
    print(f"Could not detect format: {e}")
    # File is not a supported CGM format
    
except MalformedDataError as e:
    print(f"Parsing failed: {e}")
    # CSV is corrupted or doesn't match expected structure
    
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    unified_df = FormatProcessor.interpolate_gaps(unified_df)
    inference_df, warnings = FormatProcessor.prepare_for_inference(unified_df)
    
except ZeroValidInputError as e:
    print(f"No valid data: {e}")
    # File has no glucose readings or is empty
```

## Processing Configuration

### Tuning Parameters

```python
# Conservative: strict quality, minimal imputation
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=10  # Stricter - split at smaller gaps
)

unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=10  # Small interpolation window
)

# Lenient: more gap filling for sparse data
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=5,
    large_gap_threshold_minutes=30  # More lenient sequence grouping
)

unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=30  # Larger interpolation window
)

# Match your CGM device specifications
# Libre scans every 15 minutes
unified_df = FormatProcessor.detect_and_assign_sequences(
    unified_df,
    expected_interval_minutes=15,
    large_gap_threshold_minutes=45
)

unified_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=15,
    small_gap_max_minutes=45
)
```

### Tuning Inference Parameters

```python
# Short-term prediction (next 30 minutes)
inference_df, warnings = FormatProcessor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=60,   # Need 1 hour history (default: 60)
    maximum_wanted_duration=180    # Use last 3 hours (default: 480)
)

# Long-term analysis (daily patterns)
inference_df, warnings = FormatProcessor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=720,   # Need 12 hours minimum
    maximum_wanted_duration=10080   # Use last 7 days
)
```

## Validating Data Quality

### Inspect Unified Format

```python
import polars as pl
from cgm_format.formats.unified import UnifiedEventType, Quality

unified_df = FormatParser.parse_file("data.csv")

# Basic statistics
print(f"Total records: {len(unified_df)}")
print(f"Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")

# Glucose statistics
glucose_stats = unified_df.select([
    pl.col('glucose').min().alias('min_glucose'),
    pl.col('glucose').max().alias('max_glucose'),
    pl.col('glucose').mean().alias('mean_glucose'),
    pl.col('glucose').std().alias('std_glucose'),
])
print(glucose_stats)

# Event type distribution
event_counts = unified_df.group_by('event_type').agg(
    pl.count().alias('count')
).sort('count', descending=True)
print(event_counts)

# Quality distribution
quality_counts = unified_df.group_by('quality').agg(
    pl.count().alias('count')
)
print(quality_counts)
```

### Check Processing Impact

```python
from cgm_format.formats.unified import Quality

# Before processing
print(f"Original records: {len(unified_df)}")

# Detect sequences
unified_df = FormatProcessor.detect_and_assign_sequences(unified_df)
sequence_count = unified_df['sequence_id'].n_unique()
print(f"Created {sequence_count} sequence(s)")

# After interpolation
processed_df = FormatProcessor.interpolate_gaps(
    unified_df,
    expected_interval_minutes=5,
    small_gap_max_minutes=19
)
print(f"After interpolation: {len(processed_df)}")

imputed_count = processed_df.filter(
    (pl.col('quality') & Quality.IMPUTATION.value) != 0
).height
print(f"Imputed records: {imputed_count}")

# After inference prep
inference_df, warnings = FormatProcessor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=60,
    maximum_wanted_duration=480
)
print(f"Final inference records: {len(inference_df)}")
```

## Integration with ML Models

### scikit-learn Compatible

```python
# Prepare data
unified_df = FormatParser.parse_file("training_data.csv")
unified_df = FormatProcessor.interpolate_gaps(unified_df)
inference_df, warnings = FormatProcessor.prepare_for_inference(unified_df)

# Convert to numpy for sklearn
X = inference_df.select(['glucose', 'carbs', 'insulin_fast', 'exercise']).to_numpy()
y = inference_df['glucose'].shift(-6).to_numpy()  # Predict 30 min ahead (6 * 5min)

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X[:-6], y[:-6])  # Drop last 6 rows (no target)
```

### PyTorch Time Series

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CGMDataset(Dataset):
    def __init__(self, df, sequence_length=12):
        # df is inference_df from prepare_for_inference()
        self.glucose = df['glucose'].to_numpy()
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.glucose) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.glucose[idx:idx+self.sequence_length]
        y = self.glucose[idx+self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Create dataset from processed data
dataset = CGMDataset(inference_df, sequence_length=12)  # 1 hour history
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Best Practices

### 1. Always Check Format Support

```python
# Check before parsing to skip unsupported files
with open(file_path, 'rb') as f:
    raw_data = f.read()
    if FormatParser.format_supported(raw_data):
        unified_df = FormatParser.parse_from_bytes(raw_data)
    else:
        print(f"Skipping unsupported format: {file_path}")
```

### 2. Always Check Warnings

```python
inference_df, warnings = FormatProcessor.prepare_for_inference(processed_df)

if warnings:
    # Log warnings for later analysis
    logger.warning(f"Data quality issues: {warnings}")
    
    # Decide on action based on warning severity
    from cgm_format.interface.cgm_interface import ProcessingWarning
    if warnings & ProcessingWarning.TOO_SHORT:
        raise ValueError("Insufficient data for inference")
```

### 3. Handle Multiple Sequences

```python
# prepare_for_inference() automatically selects the LATEST sequence
# If you need all sequences, process them separately:

unique_sequences = processed_df['sequence_id'].unique()

for seq_id in unique_sequences:
    seq_df = processed_df.filter(pl.col('sequence_id') == seq_id)
    
    # Process each sequence individually
    inference_df, warnings = FormatProcessor.prepare_for_inference(
        seq_df,
        minimum_duration_minutes=60,
        maximum_wanted_duration=480
    )
    
    # Process this sequence
    predictions = model.predict(inference_df)
```

### 4. Preserve Raw Data

```python
# Keep original unified format for auditing
unified_df = FormatParser.parse_file("patient_123.csv")
FormatParser.to_csv_file(unified_df, "archive/patient_123_unified.csv")

# Then process for inference
processed_df = FormatProcessor.interpolate_gaps(unified_df)
inference_df, warnings = FormatProcessor.prepare_for_inference(processed_df)
```

### 5. Version Your Pipeline

```python
PIPELINE_VERSION = "1.0.0"
PROCESSOR_CONFIG = {
    'expected_interval_minutes': 5,      # Default: 5 minutes
    'small_gap_max_minutes': 19,         # Default: 19 minutes (3 intervals + 80% tolerance)
    'minimum_duration_minutes': 60,      # Default: 60 minutes
    'maximum_wanted_duration': 480,      # Default: 480 minutes (8 hours)
}

# Include in metadata
metadata = {
    'pipeline_version': PIPELINE_VERSION,
    'config': PROCESSOR_CONFIG,
    'processed_at': datetime.now().isoformat(),
    'warnings': str(warnings),
}
```

## See Also

- [README.md](README.md) - Project overview and installation
- [src/cgm_format/interface/PIPELINE.md](src/cgm_format/interface/PIPELINE.md) - Complete pipeline documentation
- [src/cgm_format/formats/UNIFIED_FORMAT.md](src/cgm_format/formats/UNIFIED_FORMAT.md) - Unified schema specification
- [examples/usage_example.py](examples/usage_example.py) - Complete usage examples
- [tests/README.md](tests/README.md) - Test suite documentation
