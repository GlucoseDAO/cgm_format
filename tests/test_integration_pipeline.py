"""Integration tests for full CGM data processing pipeline.

This test runs the complete pipeline on real data files from the data directory:
1. Parse files using FormatParser
2. Interpolate gaps with FormatProcessor
3. Synchronize timestamps
4. Prepare for inference
5. Convert to data-only format

No mocking - uses actual data files and tests the entire workflow.
"""

from pathlib import Path
from typing import List, Tuple
import polars as pl
import pytest

from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor
from cgm_format.interface.cgm_interface import (
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    ProcessingWarning,
)


# Data directory relative to project root
DATA_DIR = Path(__file__).parent.parent / "data"


def get_test_files() -> List[Path]:
    """Get all CSV files from the data directory."""
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    # Exclude the parsed subdirectory
    csv_files = [f for f in csv_files if "parsed" not in str(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {DATA_DIR}")
    
    return csv_files


class TestFullPipelineIntegration:
    """Integration tests for the full CGM processing pipeline."""
    
    @pytest.mark.parametrize("file_path", get_test_files(), ids=lambda p: p.name)
    def test_pipeline_single_file(self, file_path: Path) -> None:
        """Test full pipeline on a single data file.
        
        This test is parametrized to run on each file individually,
        so pytest will show exactly which files pass or fail.
        
        Args:
            file_path: Path to the CGM data file to test
        """
        # Skip unsupported formats using format_supported
        with open(file_path, 'rb') as f:
            if not FormatParser.format_supported(f.read()):
                pytest.skip(f"Unsupported format: {file_path.name}")

        print(f"\n{'='*70}")
        print(f"Testing: {file_path.name}")
        print(f"{'='*70}\n")
        
        # Process the file through the full pipeline
        result = self._process_single_file(file_path)
        
        # Display results
        print(f"✅ SUCCESS: {result['filename']}")
        print(f"   Parsed: {result['parsed_rows']} rows")
        print(f"   Interpolated: {result['interpolated_rows']} rows")
        print(f"   Sequences: {result['sequence_count']}")
        print(f"   Inference ready: {result['inference_rows']} rows")
        print(f"   Final data-only: {result['final_rows']} rows")
        print(f"   Warnings: {result['warning_flags']}")
        
        # Assert basic expectations
        assert result['parsed_rows'] > 0, "Should have parsed rows"
        assert result['final_rows'] > 0, "Should have final output rows"
    
    def _process_single_file(self, file_path: Path) -> dict:
        """Process a single file through the full pipeline.
        
        Args:
            file_path: Path to the CGM data file
            
        Returns:
            Dictionary with processing results and metrics
        """
        # Stage 1-3: Parse vendor format to unified
        unified_df = FormatParser.parse_file(file_path)
        parsed_rows = len(unified_df)
        
        assert parsed_rows > 0, f"No rows parsed from {file_path.name}"
        assert 'datetime' in unified_df.columns, "Missing datetime column"
        assert 'glucose' in unified_df.columns, "Missing glucose column"
        
        # Stage 4: Process and prepare for inference
        # FormatProcessor now uses classmethods - no need to instantiate
        
        # Step 1: Detect and assign sequences
        interpolated_df = FormatProcessor.detect_and_assign_sequences(
            unified_df,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=19
        )
        
        # Step 2: Interpolate gaps (sequences already created during parsing)
        interpolated_df = FormatProcessor.interpolate_gaps(
            interpolated_df,
            expected_interval_minutes=5,
            small_gap_max_minutes=19  # Default: 19 min (3 intervals + 80% tolerance)
        )
        interpolated_rows = len(interpolated_df)
        
        assert interpolated_rows >= parsed_rows, "Interpolation should not reduce rows"
        assert 'sequence_id' in interpolated_df.columns, "Missing sequence_id after interpolation"
        
        sequence_count = interpolated_df['sequence_id'].n_unique()
        
        # Synchronize timestamps
        synchronized_df = FormatProcessor.synchronize_timestamps(
            interpolated_df,
            expected_interval_minutes=5
        )
        
        # Stage 5: Prepare for inference with quality checks
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            synchronized_df,
            minimum_duration_minutes=15,  # 15 minutes minimum
            maximum_wanted_duration=24 * 60  # 24 hours maximum (1440 minutes)
        )
        
        inference_rows = len(inference_df)
        
        assert inference_rows > 0, f"No inference rows for {file_path.name}"
        
        # Stage 6: Convert to data-only format
        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        
        final_rows = len(glucose_only_df)
        
        assert final_rows > 0, f"No final rows for {file_path.name}"
        assert 'datetime' in glucose_only_df.columns, "Missing datetime in final output"
        assert 'glucose' in glucose_only_df.columns, "Missing glucose in final output"
        
        # Verify no duplicate timestamps
        if final_rows > 1:
            duplicates = glucose_only_df.select(
                pl.col('datetime').is_duplicated().sum()
            ).item()
            assert duplicates == 0, f"Found {duplicates} duplicate timestamps"
        
        return {
            'filename': file_path.name,
            'parsed_rows': parsed_rows,
            'interpolated_rows': interpolated_rows,
            'sequence_count': sequence_count,
            'inference_rows': inference_rows,
            'final_rows': final_rows,
            'warning_flags': warning_flags.value if warning_flags else 0,
        }
    
    def test_pipeline_summary(self) -> None:
        """Generate a summary report of all files in the data directory.
        
        This test always passes but provides useful summary statistics.
        """
        test_files = get_test_files()
        
        print(f"\n{'='*70}")
        print(f"INTEGRATION TEST SUMMARY REPORT")
        print(f"{'='*70}\n")
        print(f"Total files in data directory: {len(test_files)}")
        print(f"\nTo see individual file results, run:")
        print(f"  pytest tests/test_integration_pipeline.py::TestFullPipelineIntegration::test_pipeline_single_file -v")
        print(f"\n{'='*70}\n")
    
    def test_pipeline_single_file_detailed(self) -> None:
        """Detailed test of pipeline on a single file with extensive validation."""
        test_files = get_test_files()
        
        # Use first file for detailed testing
        file_path = test_files[0]
        
        print(f"\n{'='*70}")
        print(f"DETAILED PIPELINE TEST: {file_path.name}")
        print(f"{'='*70}\n")
        
        # Parse
        print("1. Parsing file...")
        unified_df = FormatParser.parse_file(file_path)
        print(f"   ✅ Parsed {len(unified_df)} rows")
        print(f"   Columns: {unified_df.columns}")
        print(f"   Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")
        
        # Validate parsed data
        assert len(unified_df) > 0
        assert unified_df['datetime'].is_sorted()
        
        glucose_count = unified_df['glucose'].count()
        print(f"   Glucose readings: {glucose_count}")
        assert glucose_count > 0, "No glucose readings found"
        
        # Process
        print("\n2. Processing with interpolation...")
        # FormatProcessor now uses classmethods
        
        # Step 1: Detect sequences
        interpolated_df = FormatProcessor.detect_and_assign_sequences(
            unified_df,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=19
        )
        
        # Step 2: Interpolate gaps
        interpolated_df = FormatProcessor.interpolate_gaps(
            interpolated_df,
            expected_interval_minutes=5,
            small_gap_max_minutes=19  # Default
        )
        print(f"   ✅ Interpolated to {len(interpolated_df)} rows")
        
        sequence_count = interpolated_df['sequence_id'].n_unique()
        print(f"   Data contains {sequence_count} sequence(s)")
        
        # Validate sequences
        sequence_ids = interpolated_df['sequence_id'].unique().sort()
        for seq_id in sequence_ids:
            seq_df = interpolated_df.filter(pl.col('sequence_id') == seq_id)
            seq_len = len(seq_df)
            duration = (seq_df['datetime'].max() - seq_df['datetime'].min()).total_seconds() / 60
            print(f"   Sequence {seq_id}: {seq_len} points, {duration:.1f} minutes")
        
        # Synchronize
        print("\n3. Synchronizing timestamps...")
        synchronized_df = FormatProcessor.synchronize_timestamps(
            interpolated_df,
            expected_interval_minutes=5
        )
        print(f"   ✅ Synchronized to {len(synchronized_df)} rows")
        
        # Prepare for inference
        print("\n4. Preparing for inference...")
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(
            synchronized_df,
            minimum_duration_minutes=15,
            maximum_wanted_duration=24 * 60
        )
        print(f"   ✅ Prepared {len(inference_df)} rows for inference")
        print(f"   Warning flags: {warning_flags.value if warning_flags else 0}")
        
        if warning_flags:
            print("   ⚠️  Warnings:")

            for warning in ProcessingWarning:
                if warning_flags & warning:
                    print(f"      - {warning.name}")
        
        # Convert to data-only
        print("\n5. Converting to data-only format...")
        glucose_only_df = FormatProcessor.to_data_only_df(
            inference_df,
            drop_service_columns=False,
            drop_duplicates=True,
            glucose_only=True
        )
        print(f"   ✅ Final data-only: {len(glucose_only_df)} rows")
        
        # Validate final output
        assert len(glucose_only_df) > 0
        assert 'datetime' in glucose_only_df.columns
        assert 'glucose' in glucose_only_df.columns
        
        # Check for duplicates
        duplicates = glucose_only_df.select(
            pl.col('datetime').is_duplicated().sum()
        ).item()
        print(f"   Duplicate timestamps: {duplicates}")
        assert duplicates == 0, "Should have no duplicate timestamps"
        
        # Check glucose values are valid
        glucose_stats = glucose_only_df.select([
            pl.col('glucose').min().alias('min'),
            pl.col('glucose').max().alias('max'),
            pl.col('glucose').mean().alias('mean'),
        ])
        
        min_g = glucose_stats['min'].item()
        max_g = glucose_stats['max'].item()
        mean_g = glucose_stats['mean'].item()
        
        print(f"\n6. Glucose statistics:")
        print(f"   Min: {min_g:.1f} mg/dL")
        print(f"   Max: {max_g:.1f} mg/dL")
        print(f"   Mean: {mean_g:.1f} mg/dL")
        
        # Validate glucose range (typical CGM range)
        assert min_g >= 20, f"Glucose too low: {min_g}"
        assert max_g <= 600, f"Glucose too high: {max_g}"
        assert 50 <= mean_g <= 300, f"Mean glucose out of expected range: {mean_g}"
        
        print(f"\n{'='*70}")
        print("DETAILED TEST PASSED")
        print(f"{'='*70}\n")
    
    def test_pipeline_error_handling(self) -> None:
        """Test that pipeline handles various error conditions gracefully."""
        # Test with invalid file path
        with pytest.raises(FileNotFoundError):
            FormatParser.parse_file(Path("/nonexistent/file.csv"))
        
        # Test with empty dataframe - it handles it gracefully by returning empty
        empty_df = pl.DataFrame({
            'datetime': [],
            'glucose': [],
            'event_type': [],
            'quality': [],
            'sequence_id': [],
            'original_datetime': [],
            'carbs': [],
            'insulin_fast': [],
            'insulin_slow': [],
            'exercise': [],
        })
        
        # Empty dataframe should return empty result
        result = FormatProcessor.interpolate_gaps(empty_df)
        assert len(result) == 0, "Empty input should produce empty output"
    
    def test_pipeline_data_consistency(self) -> None:
        """Test that pipeline maintains data consistency and ordering."""
        test_files = get_test_files()
        file_path = test_files[0]
        
        # Parse and process
        unified_df = FormatParser.parse_file(file_path)
        
        # Step 1: Detect sequences
        interpolated_df = FormatProcessor.detect_and_assign_sequences(unified_df)
        
        # Step 2: Interpolate gaps
        interpolated_df = FormatProcessor.interpolate_gaps(interpolated_df)
        
        # Step 3: Synchronize timestamps
        synchronized_df = FormatProcessor.synchronize_timestamps(interpolated_df)
        
        # Step 4: Prepare for inference
        inference_df, warning_flags = FormatProcessor.prepare_for_inference(synchronized_df)
        
        # Verify timestamps are sorted
        assert inference_df['datetime'].is_sorted(), "Timestamps should be sorted"
        
        # Verify we have at least one sequence
        sequence_ids = inference_df['sequence_id'].unique().sort()
        assert len(sequence_ids) > 0, "Should have at least one sequence"
        
        # Verify each sequence has sorted timestamps
        for seq_id in sequence_ids:
            seq_df = inference_df.filter(pl.col('sequence_id') == seq_id)
            assert seq_df['datetime'].is_sorted(), f"Sequence {seq_id} timestamps not sorted"
            assert len(seq_df) > 0, f"Sequence {seq_id} should have data"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

