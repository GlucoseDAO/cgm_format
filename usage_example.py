"""Practical usage examples for CGM data processing pipeline.

This script demonstrates common workflows for processing CGM data from vendor
formats (Dexcom, Libre) through to inference-ready datasets.

Run examples:
    uv run usage_example.py
"""

from pathlib import Path
from typing import Optional
import polars as pl

from format_converter import FormatParser
from format_processor import FormatProcessor
from interface.cgm_interface import (
    ProcessingWarning,
    ZeroValidInputError,
    UnknownFormatError,
    MalformedDataError,
)
from formats.unified import UnifiedEventType, Quality


def example_1_basic_pipeline(file_path: Path) -> pl.DataFrame:
    """Example 1: Basic end-to-end inference pipeline.
    
    This is the most common workflow for ML inference.
    
    Args:
        file_path: Path to CGM export file (any supported format)
        
    Returns:
        Inference-ready DataFrame
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Inference Pipeline")
    print("="*70)
    
    # Stage 1-3: Parse vendor format to unified
    print(f"\n1. Parsing file: {file_path.name}")
    unified_df = FormatParser.parse_from_file(file_path)
    print(f"   ✓ Parsed {len(unified_df)} records")
    print(f"   Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")
    
    # Stage 4-5: Process for inference
    print("\n2. Processing data for inference...")
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15
    )
    
    # Fill gaps and create sequences
    processed_df = processor.interpolate_gaps(unified_df)
    sequence_count = processed_df['sequence_id'].n_unique()
    print(f"   ✓ Created {sequence_count} sequence(s)")
    
    # Prepare final inference data
    inference_df, warnings = processor.prepare_for_inference(
        processed_df,
        minimum_duration_minutes=180,    # 3 hours minimum
        maximum_wanted_duration=1440     # 24 hours maximum
    )
    print(f"   ✓ Prepared {len(inference_df)} inference records")
    
    # Check warnings
    if processor.has_warnings():
        print("\n3. Data Quality Warnings:")
        for warning in processor.get_warnings():
            print(f"   ⚠ {warning.name}")
    else:
        print("\n3. ✓ No data quality warnings")
    
    print("\n4. Inference DataFrame:")
    print(inference_df.head())
    
    return inference_df


def example_2_quality_inspection(file_path: Path) -> None:
    """Example 2: Inspect data quality at each processing stage.
    
    Args:
        file_path: Path to CGM export file
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Data Quality Inspection")
    print("="*70)
    
    # Parse
    unified_df = FormatParser.parse_from_file(file_path)
    
    print("\n1. Original Data Statistics:")
    print(f"   Total records: {len(unified_df)}")
    
    # Glucose statistics
    glucose_stats = unified_df.select([
        pl.col('glucose').count().alias('glucose_count'),
        pl.col('glucose').min().alias('min_glucose'),
        pl.col('glucose').max().alias('max_glucose'),
        pl.col('glucose').mean().alias('mean_glucose'),
    ])
    print(f"   Glucose records: {glucose_stats['glucose_count'][0]}")
    print(f"   Glucose range: {glucose_stats['min_glucose'][0]:.1f} - {glucose_stats['max_glucose'][0]:.1f} mg/dL")
    print(f"   Mean glucose: {glucose_stats['mean_glucose'][0]:.1f} mg/dL")
    
    # Event type distribution
    print("\n2. Event Type Distribution:")
    event_counts = unified_df.group_by('event_type').agg(
        pl.count().alias('count')
    ).sort('count', descending=True)
    for row in event_counts.iter_rows(named=True):
        print(f"   {row['event_type']:15s}: {row['count']:5d} records")
    
    # Quality distribution
    print("\n3. Quality Distribution:")
    quality_counts = unified_df.group_by('quality').agg(
        pl.count().alias('count')
    )
    quality_names = {0: "GOOD", 1: "ILL", 2: "SENSOR_CALIBRATION"}
    for row in quality_counts.iter_rows(named=True):
        quality_name = quality_names.get(row['quality'], "UNKNOWN")
        print(f"   {quality_name:20s}: {row['count']:5d} records")
    
    # Process and check impact
    print("\n4. Processing Impact:")
    processor = FormatProcessor()
    processed_df = processor.interpolate_gaps(unified_df)
    
    imputed_count = processed_df.filter(
        pl.col('event_type') == UnifiedEventType.IMPUTATION.value
    ).height
    print(f"   Records after interpolation: {len(processed_df)}")
    print(f"   Imputed records added: {imputed_count}")
    
    # Sequence analysis
    print("\n5. Sequence Analysis:")
    sequence_info = processed_df.group_by('sequence_id').agg([
        pl.col('datetime').min().alias('start_time'),
        pl.col('datetime').max().alias('end_time'),
        pl.col('datetime').count().alias('num_points'),
    ]).sort('start_time')
    
    for row in sequence_info.iter_rows(named=True):
        duration_hours = (row['end_time'] - row['start_time']).total_seconds() / 3600
        print(f"   Sequence {row['sequence_id']}: "
              f"{duration_hours:.1f} hours, {row['num_points']} points")


def example_3_batch_processing(data_dir: Path, output_dir: Path) -> None:
    """Example 3: Batch process multiple CGM files.
    
    Args:
        data_dir: Directory containing CGM export files
        output_dir: Directory to save processed files
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing")
    print("="*70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    processor = FormatProcessor()
    results = []
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"\nProcessing {len(csv_files)} files from {data_dir}...")
    
    for csv_file in csv_files:
        print(f"\n  Processing: {csv_file.name}")
        
        try:
            # Parse and process
            unified_df = FormatParser.parse_from_file(csv_file)
            processed_df = processor.interpolate_gaps(unified_df)
            inference_df, warnings = processor.prepare_for_inference(processed_df)
            
            # Add filename as identifier
            patient_id = csv_file.stem
            inference_df = inference_df.with_columns([
                pl.lit(patient_id).alias('patient_id')
            ])
            
            # Save individual processed file
            output_file = output_dir / f"{patient_id}_inference.csv"
            FormatParser.to_csv_file(inference_df, str(output_file))
            
            results.append(inference_df)
            
            warning_str = f"warnings={warnings.value}" if warnings else "no warnings"
            print(f"    ✓ Processed {len(inference_df)} records, {warning_str}")
            
        except UnknownFormatError as e:
            print(f"    ✗ Unknown format: {e}")
        except MalformedDataError as e:
            print(f"    ✗ Malformed data: {e}")
        except ZeroValidInputError as e:
            print(f"    ✗ No valid data: {e}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Combine all results
    if results:
        print(f"\nCombining {len(results)} processed files...")
        combined_df = pl.concat(results)
        combined_file = output_dir / "combined_inference.csv"
        FormatParser.to_csv_file(combined_df, str(combined_file))
        print(f"  ✓ Saved combined dataset: {combined_file}")
        print(f"  Total records: {len(combined_df)}")


def example_4_custom_processing(file_path: Path) -> pl.DataFrame:
    """Example 4: Custom processing with strict quality control.
    
    For research or production use cases requiring high data quality.
    
    Args:
        file_path: Path to CGM export file
        
    Returns:
        High-quality inference DataFrame
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Processing (Strict Quality)")
    print("="*70)
    
    # Parse
    print(f"\n1. Parsing: {file_path.name}")
    unified_df = FormatParser.parse_from_file(file_path)
    print(f"   Original records: {len(unified_df)}")
    
    # Custom gap handling: only interpolate very small gaps
    print("\n2. Processing with strict gap handling...")
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=10  # Only fill gaps ≤10 minutes
    )
    
    processed_df = processor.interpolate_gaps(unified_df)
    
    # Filter out imputed and low-quality data
    print("\n3. Filtering for high quality data...")
    high_quality_df = processed_df.filter(
        (pl.col('event_type') != UnifiedEventType.IMPUTATION.value) &
        (pl.col('quality') == Quality.GOOD.value)
    )
    
    filtered_count = len(processed_df) - len(high_quality_df)
    print(f"   Filtered out {filtered_count} records")
    print(f"   Remaining: {len(high_quality_df)} records")
    
    # Synchronize to exact intervals
    print("\n4. Synchronizing timestamps to fixed intervals...")
    synchronized_df = processor.synchronize_timestamps(high_quality_df)
    print(f"   Synchronized: {len(synchronized_df)} records")
    
    # Prepare with strict duration requirements
    print("\n5. Preparing for inference (strict requirements)...")
    inference_df, warnings = processor.prepare_for_inference(
        synchronized_df,
        minimum_duration_minutes=360,  # Require 6 hours
        maximum_wanted_duration=2880   # Allow up to 48 hours
    )
    
    print(f"   Final records: {len(inference_df)}")
    
    if warnings:
        print("\n⚠ Quality Warnings:")
        for warning in processor.get_warnings():
            print(f"   - {warning.name}")
        raise ValueError("Data does not meet strict quality requirements")
    else:
        print("\n✓ Data meets strict quality requirements")
    
    return inference_df


def example_5_format_detection(file_path: Path) -> None:
    """Example 5: Manual format detection and vendor-specific handling.
    
    Args:
        file_path: Path to CGM export file
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Manual Format Detection")
    print("="*70)
    
    # Read raw file
    print(f"\n1. Reading raw file: {file_path.name}")
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    # Decode
    print("\n2. Decoding and cleaning...")
    text_data = FormatParser.decode_raw_data(raw_data)
    print(f"   Decoded {len(text_data)} characters")
    
    # Detect format
    print("\n3. Detecting format...")
    format_type = FormatParser.detect_format(text_data)
    print(f"   Detected format: {format_type.name}")
    
    # Show sample lines
    print("\n4. Sample data (first 5 lines):")
    lines = text_data.split('\n')[:5]
    for i, line in enumerate(lines, 1):
        preview = line[:80] + "..." if len(line) > 80 else line
        print(f"   Line {i}: {preview}")
    
    # Parse
    print("\n5. Parsing to unified format...")
    unified_df = FormatParser.parse_to_unified(text_data, format_type)
    print(f"   ✓ Parsed {len(unified_df)} records")
    
    # Format-specific information
    if format_type.name == "DEXCOM":
        print("\n6. Dexcom-specific checks:")
        out_of_range = unified_df.filter(pl.col('quality') == Quality.ILL.value).height
        print(f"   Out-of-range readings (High/Low): {out_of_range}")
        
    elif format_type.name == "LIBRE":
        print("\n6. Libre-specific checks:")
        insulin_records = unified_df.filter(
            (pl.col('event_type') == UnifiedEventType.INSULIN_FAST.value) |
            (pl.col('event_type') == UnifiedEventType.INSULIN_SLOW.value)
        ).height
        print(f"   Insulin records: {insulin_records}")


def example_6_error_handling() -> None:
    """Example 6: Comprehensive error handling."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling")
    print("="*70)
    
    test_files = [
        ("data/valid_file.csv", "Valid file"),
        ("data/unknown_format.csv", "Unknown format"),
        ("data/corrupted.csv", "Corrupted file"),
        ("data/nonexistent.csv", "Missing file"),
    ]
    
    for file_path, description in test_files:
        print(f"\nTesting: {description}")
        
        try:
            # Parse
            unified_df = FormatParser.parse_from_file(file_path)
            
            # Process
            processor = FormatProcessor()
            processed_df = processor.interpolate_gaps(unified_df)
            inference_df, warnings = processor.prepare_for_inference(processed_df)
            
            print(f"  ✓ Success: {len(inference_df)} records")
            
        except FileNotFoundError as e:
            print(f"  ✗ File not found: {e}")
            
        except UnknownFormatError as e:
            print(f"  ✗ Unknown format: {e}")
            print("     → File is not a supported CGM format (Dexcom, Libre, Unified)")
            
        except MalformedDataError as e:
            print(f"  ✗ Malformed data: {e}")
            print("     → CSV structure doesn't match expected format")
            
        except ZeroValidInputError as e:
            print(f"  ✗ No valid data: {e}")
            print("     → File contains no usable glucose readings")
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")


def example_7_ml_integration(file_path: Path) -> None:
    """Example 7: Prepare data for ML model integration.
    
    Args:
        file_path: Path to CGM export file
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: ML Model Integration")
    print("="*70)
    
    # Process data
    print("\n1. Processing data...")
    unified_df = FormatParser.parse_from_file(file_path)
    processor = FormatProcessor()
    processed_df = processor.interpolate_gaps(unified_df)
    inference_df, warnings = processor.prepare_for_inference(processed_df)
    
    print(f"   ✓ Prepared {len(inference_df)} records for inference")
    
    # Extract features
    print("\n2. Extracting features for ML model...")
    
    # Example: Prepare feature matrix
    features = inference_df.select([
        'glucose',
        'carbs',
        'insulin_fast',
        'insulin_slow',
        'exercise'
    ])
    
    # Fill nulls with 0 (no carbs/insulin/exercise)
    features = features.with_columns([
        pl.col('carbs').fill_null(0.0),
        pl.col('insulin_fast').fill_null(0.0),
        pl.col('insulin_slow').fill_null(0.0),
        pl.col('exercise').fill_null(0),
    ])
    
    print("   Feature matrix shape:", features.shape)
    print("   Features:", features.columns)
    
    # Create target (predict glucose 30 minutes ahead = 6 intervals * 5min)
    print("\n3. Creating prediction target...")
    target = inference_df['glucose'].shift(-6)  # 30 minutes ahead
    
    # Remove last 6 rows (no target available)
    features_train = features[:-6]
    target_train = target[:-6]
    
    print(f"   Training samples: {len(features_train)}")
    
    # Convert to numpy (for sklearn/pytorch)
    print("\n4. Converting to numpy arrays...")
    X = features_train.to_numpy()
    y = target_train.to_numpy()
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Ready for model.fit(X, y)")
    
    # Show sample
    print("\n5. Sample data (first 3 rows):")
    print(features_train.head(3))


def main() -> None:
    """Run all usage examples."""
    print("\n" + "="*70)
    print("CGM FORMAT LIBRARY - USAGE EXAMPLES")
    print("="*70)
    
    # Check for test data
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"\n⚠ Warning: Data directory '{data_dir}' not found")
        print("   Please create 'data/' directory with sample CGM files")
        print("   Supported formats: Dexcom, Libre, Unified")
        return
    
    # Find sample files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"\n⚠ Warning: No CSV files found in '{data_dir}'")
        print("   Please add sample CGM export files to run examples")
        return
    
    sample_file = csv_files[0]
    print(f"\nUsing sample file: {sample_file.name}")
    
    # Run examples
    try:
        example_1_basic_pipeline(sample_file)
        example_2_quality_inspection(sample_file)
        example_5_format_detection(sample_file)
        example_7_ml_integration(sample_file)
        
        # Batch processing (if multiple files)
        if len(csv_files) > 1:
            output_dir = Path("data/processed_examples")
            example_3_batch_processing(data_dir, output_dir)
        
        # Custom processing (might fail if data quality is insufficient)
        try:
            example_4_custom_processing(sample_file)
        except ValueError as e:
            print(f"\n⚠ Custom processing skipped: {e}")
        
        # Error handling examples
        example_6_error_handling()
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

