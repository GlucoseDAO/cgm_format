# Usage Guide: CGM Data Processing for Inference

This guide demonstrates the complete workflow for processing CGM data from vendor formats (Dexcom, Libre) through to inference-ready datasets.

## Overview

The processing pipeline consists of two main components:

1. **FormatParser** (Stages 1-3): Parse vendor-specific CSV to unified format
2. **FormatProcessor** (Stages 4-5): Process unified data for ML inference

```
Vendor CSV → Parse → Unified Format → Interpolate → Synchronize → Inference Ready
            (Stage 1-3)              (Stage 4)     (Stage 5)
```

## Quick Start: End-to-End Inference Pipeline

```python
from cgm_format import FormatParser
from cgm_format import FormatProcessor

# Stage 1-3: Parse vendor format to unified
unified_df = FormatParser.parse_from_file("data/dexcom_export.csv")

# Stage 4-5: Process for inference
processor = FormatProcessor(
    expected_interval_minutes=5,    # CGM reads every 5 minutes
    small_gap_max_minutes=15        # Interpolate gaps up to 15 minutes
)

# Stage 4a: Fill gaps and create sequences
processed_df = processor.interpolate_gaps(unified_df)

# Stage 4b: Align to fixed-frequency timestamps
synchronized_df = processor.synchronize_timestamps(processed_df)

# Stage 5: Prepare final inference data
inference_df, warnings = processor.prepare_for_inference(
    synchronized_df,
    minimum_duration_minutes=180,      # Require 3 hours minimum
    maximum_wanted_duration=1440       # Truncate to 24 hours max
)

# Check data quality warnings
if processor.has_warnings():
    print(f"Processing warnings: {processor.get_warnings()}")

# Feed to ML model
model_predictions = your_model.predict(inference_df)
```

## Stage-by-Stage Workflow

### Stage 1-3: Parsing Vendor Formats

The `FormatParser` handles all vendor-specific quirks and converts to unified format.

#### Parse from File (Automatic Detection)

```python
from cgm_format import FormatParser

# Automatically detects Dexcom, Libre, or Unified format
unified_df = FormatParser.parse_from_file("path/to/cgm_export.csv")

print(f"Parsed {len(unified_df)} data points")
print(unified_df.head())
```

#### Parse with Manual Stages (Advanced)

```python
from cgm_format import FormatParser
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
from cgm_format import FormatParser

# Dexcom marks out-of-range readings as "High" or "Low"
# These are replaced with numeric placeholders (default: 401, 39 mg/dL)
# and marked as Quality.ILL to indicate sensor error

unified_df = FormatParser.parse_from_file("dexcom_export.csv")

# Check for out-of-range readings
from cgm_format.formats.unified import Quality

out_of_range_count = unified_df.filter(
    pl.col('quality') == Quality.ILL.value
).height

print(f"Found {out_of_range_count} out-of-range readings")
```

### Stage 4: Gap Interpolation and Sequence Creation

The `interpolate_gaps()` method performs two critical operations:

1. **Sequence Creation**: Splits data at large gaps (>15 min by default)
2. **Gap Filling**: Interpolates small gaps with imputed glucose values

```python
from cgm_format import FormatProcessor

processor = FormatProcessor(
    expected_interval_minutes=5,    # Normal CGM interval
    small_gap_max_minutes=15        # Max gap to interpolate
)

# Fills small gaps and creates sequences
processed_df = processor.interpolate_gaps(unified_df)

# Check sequences created
sequence_count = processed_df['sequence_id'].n_unique()
print(f"Created {sequence_count} sequences")

# Check if imputation occurred
if processor.has_warnings():
    from cgm_format.interface.cgm_interface import ProcessingWarning
    if ProcessingWarning.IMPUTATION in processor.get_warnings():
        print("Data contains interpolated values")
```

#### Understanding Sequences

Data is split into sequences when gaps exceed `small_gap_max_minutes`:

```python
import polars as pl

# Analyze sequences
sequence_info = processed_df.group_by('sequence_id').agg([
    pl.col('datetime').min().alias('start_time'),
    pl.col('datetime').max().alias('end_time'),
    pl.col('datetime').count().alias('num_points'),
])

for row in sequence_info.iter_rows(named=True):
    duration = (row['end_time'] - row['start_time']).total_seconds() / 3600
    print(f"Sequence {row['sequence_id']}: "
          f"{duration:.1f} hours, {row['num_points']} points")
```

#### Calibration Period Marking

After large gaps (≥2h 45min), the next 24 hours are marked as `Quality.SENSOR_CALIBRATION`:

```python
from cgm_format.formats.unified import Quality
from cgm_format.interface.cgm_interface import CALIBRATION_GAP_THRESHOLD

calibration_points = processed_df.filter(
    pl.col('quality') == Quality.SENSOR_CALIBRATION.value
).height

print(f"Calibration period points: {calibration_points}")
print(f"Threshold: {CALIBRATION_GAP_THRESHOLD} seconds")
```

### Stage 5: Timestamp Synchronization (Optional)

Aligns timestamps to exact minute boundaries with fixed intervals.

```python
# After interpolate_gaps(), synchronize to fixed frequency
synchronized_df = processor.synchronize_timestamps(processed_df)

# Now all timestamps are at exact 5-minute intervals
# Example: 10:00:00, 10:05:00, 10:10:00, etc.
print(synchronized_df['datetime'].head(10))
```

**When to use:**
- ML models expecting fixed-frequency time series
- Time-series forecasting with regular intervals
- Simplifying temporal feature engineering

**When to skip:**
- Models handling irregular timestamps
- Preserving original measurement times is critical
- Already using prepare_for_inference() with truncation

### Stage 6: Inference Preparation

The `prepare_for_inference()` method performs final QA and extraction:

```python
# Prepare data for ML model
inference_df, warnings = processor.prepare_for_inference(
    synchronized_df,
    minimum_duration_minutes=180,      # 3 hours minimum
    maximum_wanted_duration=1440       # 24 hours maximum
)

# warnings is a ProcessingWarning flags enum
from cgm_format.interface.cgm_interface import ProcessingWarning

if warnings & ProcessingWarning.TOO_SHORT:
    print("Warning: Sequence shorter than minimum duration")

if warnings & ProcessingWarning.QUALITY:
    print("Warning: Data contains quality issues")

if warnings & ProcessingWarning.IMPUTATION:
    print("Warning: Data contains interpolated values")

if warnings & ProcessingWarning.CALIBRATION:
    print("Warning: Data contains calibration events")
```

#### What prepare_for_inference() Does

1. **Validation**: Checks for zero valid data points (raises `ZeroValidInputError`)
2. **Sequence Selection**: Keeps only the **latest** sequence (most recent timestamps)
3. **Duration Check**: Warns if sequence < minimum_duration_minutes
4. **Quality Checks**: Collects warnings for calibration, quality issues, imputation
5. **Truncation**: Keeps last N minutes if exceeding maximum_wanted_duration
6. **Column Extraction**: Returns only data columns (removes service columns)

#### Inference DataFrame Structure

The returned DataFrame contains only data columns:

```python
# inference_df columns:
# - datetime: Timestamp
# - glucose: Float64 (mg/dL)
# - carbs: Float64 (grams)
# - insulin_slow: Float64 (units)
# - insulin_fast: Float64 (units)
# - exercise: Int64 (seconds)

print(inference_df.columns)
# ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
```

## Common Workflows

### Workflow 1: Basic Inference Pipeline

For most ML use cases, use the complete pipeline:

```python
from cgm_format import FormatParser
from cgm_format import FormatProcessor

# Parse
unified_df = FormatParser.parse_from_file("cgm_data.csv")

# Process
processor = FormatProcessor()
processed_df = processor.interpolate_gaps(unified_df)
inference_df, warnings = processor.prepare_for_inference(processed_df)

# Check quality
if warnings:
    print(f"Data quality warnings: {warnings}")
    # Decide whether to proceed or reject data

# Feed to model
predictions = model.predict(inference_df)
```

### Workflow 2: Batch Processing Multiple Files

```python
from pathlib import Path
from cgm_format import FormatParser
from cgm_format import FormatProcessor
import polars as pl

data_dir = Path("data/exports")
processor = FormatProcessor()

results = []

for csv_file in data_dir.glob("*.csv"):
    try:
        # Parse and process
        unified_df = FormatParser.parse_from_file(csv_file)
        processed_df = processor.interpolate_gaps(unified_df)
        inference_df, warnings = processor.prepare_for_inference(processed_df)
        
        # Add patient_id from filename
        patient_id = csv_file.stem
        inference_df = inference_df.with_columns([
            pl.lit(patient_id).alias('patient_id')
        ])
        
        results.append(inference_df)
        print(f"✓ {csv_file.name}: {len(inference_df)} points, warnings={warnings}")
        
    except Exception as e:
        print(f"✗ {csv_file.name}: {e}")

# Combine all patient data
all_data = pl.concat(results)
```

### Workflow 3: Custom Preprocessing for Research

```python
from cgm_format import FormatParser
from cgm_format import FormatProcessor
import polars as pl

# Parse
unified_df = FormatParser.parse_from_file("research_data.csv")

# Custom gap handling: stricter interpolation
processor = FormatProcessor(
    expected_interval_minutes=5,
    small_gap_max_minutes=10  # Only interpolate very small gaps
)

# Process with strict quality control
processed_df = processor.interpolate_gaps(unified_df)

# Filter out any imputed or calibration data
high_quality_df = processed_df.filter(
    (pl.col('event_type') != 'IMPUTATN') &
    (pl.col('quality') == 0)  # Quality.GOOD
)

# Synchronize to exact intervals
synchronized_df = processor.synchronize_timestamps(high_quality_df)

# Prepare for inference without warnings
inference_df, warnings = processor.prepare_for_inference(
    synchronized_df,
    minimum_duration_minutes=360,  # Require 6 hours
    maximum_wanted_duration=2880   # Allow 48 hours
)

if warnings:
    print("Warning: Data may not meet research quality standards")
```

### Workflow 4: Real-Time Inference (Streaming)

```python
from cgm_format import FormatParser
from cgm_format import FormatProcessor

class CGMInferenceService:
    def __init__(self):
        self.processor = FormatProcessor()
    
    def process_new_data(self, csv_bytes: bytes):
        """Process incoming CGM export for immediate inference."""
        try:
            # Parse
            unified_df = FormatParser.parse_from_bytes(csv_bytes)
            
            # Process
            processed_df = self.processor.interpolate_gaps(unified_df)
            inference_df, warnings = self.processor.prepare_for_inference(
                processed_df,
                minimum_duration_minutes=60,  # Need 1 hour minimum
                maximum_wanted_duration=180   # Use last 3 hours
            )
            
            # Return inference-ready data with quality info
            return {
                'data': inference_df,
                'warnings': warnings,
                'num_points': len(inference_df),
                'has_quality_issues': bool(warnings)
            }
            
        except Exception as e:
            return {'error': str(e)}

# Usage
service = CGMInferenceService()
result = service.process_new_data(uploaded_file_bytes)

if 'error' not in result:
    predictions = model.predict(result['data'])
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
    unified_df = FormatParser.parse_from_file("data.csv")
    
except UnknownFormatError as e:
    print(f"Could not detect format: {e}")
    # File is not a supported CGM format
    
except MalformedDataError as e:
    print(f"Parsing failed: {e}")
    # CSV is corrupted or doesn't match expected structure
    
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    processor = FormatProcessor()
    processed_df = processor.interpolate_gaps(unified_df)
    inference_df, warnings = processor.prepare_for_inference(processed_df)
    
except ZeroValidInputError as e:
    print(f"No valid data: {e}")
    # File has no glucose readings or is empty
```

## Processing Configuration

### Tuning FormatProcessor Parameters

```python
# Conservative: strict quality, minimal imputation
strict_processor = FormatProcessor(
    expected_interval_minutes=5,
    small_gap_max_minutes=10  # Small interpolation window
)

# Lenient: more gap filling for sparse data
lenient_processor = FormatProcessor(
    expected_interval_minutes=5,
    small_gap_max_minutes=30  # Larger interpolation window
)

# Match your CGM device specifications
libre_processor = FormatProcessor(
    expected_interval_minutes=15,  # Libre scans every 15 min
    small_gap_max_minutes=45
)
```

### Tuning Inference Parameters

```python
# Short-term prediction (next 30 minutes)
inference_df, warnings = processor.prepare_for_inference(
    processed_df,
    minimum_duration_minutes=60,   # Need 1 hour history
    maximum_wanted_duration=180    # Use last 3 hours
)

# Long-term analysis (daily patterns)
inference_df, warnings = processor.prepare_for_inference(
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

unified_df = FormatParser.parse_from_file("data.csv")

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
processor = FormatProcessor()

# Before processing
print(f"Original records: {len(unified_df)}")

# After interpolation
processed_df = processor.interpolate_gaps(unified_df)
print(f"After interpolation: {len(processed_df)}")

imputed_count = processed_df.filter(
    pl.col('event_type') == UnifiedEventType.IMPUTATION.value
).height
print(f"Imputed records: {imputed_count}")

# After inference prep
inference_df, warnings = processor.prepare_for_inference(processed_df)
print(f"Final inference records: {len(inference_df)}")
```

## Integration with ML Models

### scikit-learn Compatible

```python
from cgm_format import FormatParser
from cgm_format import FormatProcessor
import polars as pl

# Prepare data
unified_df = FormatParser.parse_from_file("training_data.csv")
processor = FormatProcessor()
processed_df = processor.interpolate_gaps(unified_df)
inference_df, warnings = processor.prepare_for_inference(processed_df)

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

### 1. Always Check Warnings

```python
inference_df, warnings = processor.prepare_for_inference(processed_df)

if warnings:
    # Log warnings for later analysis
    logger.warning(f"Data quality issues: {warnings}")
    
    # Decide on action based on warning severity
    if ProcessingWarning.TOO_SHORT in processor.get_warnings():
        raise ValueError("Insufficient data for inference")
```

### 2. Handle Multiple Sequences

```python
# prepare_for_inference() automatically selects the LATEST sequence
# If you need all sequences, process them separately:

unique_sequences = processed_df['sequence_id'].unique()

for seq_id in unique_sequences:
    seq_df = processed_df.filter(pl.col('sequence_id') == seq_id)
    
    # Create new processor for each sequence (clean warnings)
    seq_processor = FormatProcessor()
    inference_df, warnings = seq_processor.prepare_for_inference(seq_df)
    
    # Process this sequence
    predictions = model.predict(inference_df)
```

### 3. Preserve Raw Data

```python
# Keep original unified format for auditing
unified_df = FormatParser.parse_from_file("patient_123.csv")
FormatParser.to_csv_file(unified_df, "archive/patient_123_unified.csv")

# Then process for inference
processor = FormatProcessor()
processed_df = processor.interpolate_gaps(unified_df)
inference_df, warnings = processor.prepare_for_inference(processed_df)
```

### 4. Version Your Pipeline

```python
PIPELINE_VERSION = "1.0.0"
PROCESSOR_CONFIG = {
    'expected_interval_minutes': 5,
    'small_gap_max_minutes': 15,
    'minimum_duration_minutes': 180,
    'maximum_wanted_duration': 1440,
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
- [interface/PIPELINE.md](interface/PIPELINE.md) - Complete pipeline documentation
- [formats/UNIFIED_FORMAT.md](formats/UNIFIED_FORMAT.md) - Unified schema specification
- [example_schema_usage.py](example_schema_usage.py) - Schema validation examples


