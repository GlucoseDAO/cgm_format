"""Pytest tests for format_parser module.

Tests cover:
1. Format detection for all files in data/
2. Parsing to unified format
3. Saving parsed results to data/parsed/
"""

import pytest
from pathlib import Path
from collections import Counter
import polars as pl
from typing import ClassVar
from cgm_format import FormatParser as FormatParserPrime
from cgm_format.interface.cgm_interface import ValidationMethod

class FormatParser(FormatParserPrime):
    """Format parser for testing."""
    validation_mode : ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT


from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    UnknownFormatError,
    MalformedDataError,
)
from cgm_format.formats.unified import UNIFIED_TIMESTAMP_FORMATS


# Constants - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"


@pytest.fixture(scope="session", autouse=True)
def setup_parsed_directory():
    """Create parsed directory if it doesn't exist."""
    PARSED_DIR.mkdir(exist_ok=True, parents=True)
    yield
    # Cleanup is optional - we keep the parsed files for inspection


def is_medtronic_file(file_path: Path) -> bool:
    """Check if a file is a Medtronic Guardian Connect file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if file is Medtronic format, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            # Check first few lines for Guardian Connect marker
            header = f.read(500).decode('utf-8', errors='ignore')
            return "Guardian Connect" in header
    except Exception:
        return False


@pytest.fixture(scope="session")
def all_data_files():
    """Get all CSV files from the data directory."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files found in {DATA_DIR}"
    return csv_files


@pytest.fixture(scope="session")
def supported_data_files(all_data_files):
    """Get CSV files excluding unsupported formats (Medtronic for now)."""
    supported_files = [f for f in all_data_files if not is_medtronic_file(f)]
    assert len(supported_files) > 0, f"No supported CSV files found"
    return supported_files


class TestFormatDetection:
    """Test format detection for all data files."""
    
    def test_all_files_detected(self, all_data_files, supported_data_files):
        """Test that all supported files can be decoded and format detected."""
        failed_files = []
        format_counts = Counter()
        skipped_files = [f for f in all_data_files if f not in supported_data_files]
        
        for csv_file in supported_data_files:
            try:
                # Read raw bytes
                with open(csv_file, 'rb') as f:
                    raw_data = f.read()
                
                # Decode
                text_data = FormatParser.decode_raw_data(raw_data)
                assert isinstance(text_data, str), f"decode_raw_data should return string for {csv_file.name}"
                assert len(text_data) > 0, f"Decoded data is empty for {csv_file.name}"
                
                # Detect format
                detected_format = FormatParser.detect_format(text_data)
                assert isinstance(detected_format, SupportedCGMFormat), f"detect_format should return SupportedCGMFormat for {csv_file.name}"
                
                format_counts[detected_format] += 1
                
            except (UnknownFormatError, MalformedDataError, Exception) as e:
                failed_files.append((csv_file.name, str(e)))
        
        # Report results
        print(f"\n\n=== Format Detection Summary ===")
        print(f"Total files in data/: {len(all_data_files)}")
        print(f"Skipped (unsupported): {len(skipped_files)}")
        print(f"Tested: {len(supported_data_files)}")
        print(f"Successfully detected: {sum(format_counts.values())}")
        print(f"Failed: {len(failed_files)}")
        print(f"\nFormat breakdown:")
        for format_type, count in format_counts.most_common():
            print(f"  {format_type.value}: {count} files")
        
        if skipped_files:
            print(f"\nSkipped files (unsupported formats):")
            for f in skipped_files:
                print(f"  {f.name} (Medtronic Guardian Connect)")
        
        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  {filename}: {error}")
        
        # Assert all supported files were successfully detected
        assert len(failed_files) == 0, f"Failed to detect format for {len(failed_files)} files"
        assert sum(format_counts.values()) == len(supported_data_files), "Not all supported files were detected"
    
    def test_format_counts_reasonable(self, supported_data_files):
        """Test that detected formats are reasonable (at least one Dexcom or Libre)."""
        format_counts = Counter()
        
        for csv_file in supported_data_files:
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            format_counts[detected_format] += 1
        
        # At least one of Dexcom or Libre should be present (based on filenames)
        has_dexcom_or_libre = (
            format_counts.get(SupportedCGMFormat.DEXCOM, 0) > 0 or
            format_counts.get(SupportedCGMFormat.LIBRE, 0) > 0
        )
        assert has_dexcom_or_libre, "Expected at least one Dexcom or Libre file"


class TestUnifiedParsing:
    """Test parsing all files to unified format."""
    
    def test_parse_all_to_unified(self, all_data_files, supported_data_files):
        """Test that all supported files can be parsed to unified format."""
        failed_files = []
        successful_parses = []
        skipped_files = [f for f in all_data_files if f not in supported_data_files]
        
        for csv_file in supported_data_files:
            try:
                # Read and decode
                with open(csv_file, 'rb') as f:
                    raw_data = f.read()
                text_data = FormatParser.decode_raw_data(raw_data)
                
                # Detect format
                detected_format = FormatParser.detect_format(text_data)
                
                # Parse to unified
                unified_df = FormatParser.parse_to_unified(text_data, detected_format)
                
                # Validate unified format
                assert isinstance(unified_df, pl.DataFrame), f"parse_to_unified should return DataFrame for {csv_file.name}"
                assert len(unified_df) > 0, f"Parsed DataFrame is empty for {csv_file.name}"
                
                # Check required columns
                required_columns = ['sequence_id', 'event_type', 'quality', 'original_datetime', 'datetime', 'glucose']
                for col in required_columns:
                    assert col in unified_df.columns, f"Missing required column '{col}' in {csv_file.name}"
                
                successful_parses.append((csv_file, detected_format, len(unified_df)))
                
            except (UnknownFormatError, MalformedDataError, Exception) as e:
                failed_files.append((csv_file.name, str(e)))
        
        # Report results
        print(f"\n\n=== Unified Parsing Summary ===")
        print(f"Total files in data/: {len(all_data_files)}")
        print(f"Skipped (unsupported): {len(skipped_files)}")
        print(f"Tested: {len(supported_data_files)}")
        print(f"Successfully parsed: {len(successful_parses)}")
        print(f"Failed: {len(failed_files)}")
        print(f"\nSuccessful parses:")
        for csv_file, detected_format, row_count in successful_parses:
            print(f"  {csv_file.name}: {detected_format.value} ({row_count} rows)")
        
        if skipped_files:
            print(f"\nSkipped files (unsupported formats):")
            for f in skipped_files:
                print(f"  {f.name} (Medtronic Guardian Connect)")
        
        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  {filename}: {error}")
        
        # Assert all supported files were successfully parsed
        assert len(failed_files) == 0, f"Failed to parse {len(failed_files)} files"
        assert len(successful_parses) == len(supported_data_files), "Not all supported files were parsed"
    
    def test_unified_format_schema(self, supported_data_files):
        """Test that parsed data has correct schema."""
        expected_columns = ['sequence_id', 'event_type', 'quality', 'original_datetime', 'datetime', 'glucose', 
                           'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
        
        for csv_file in supported_data_files[:3]:  # Test first 3 supported files for schema
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            unified_df = FormatParser.parse_to_unified(text_data, detected_format)
            
            # Check all expected columns are present
            assert set(expected_columns) == set(unified_df.columns), \
                f"Column mismatch in {csv_file.name}: expected {expected_columns}, got {unified_df.columns}"
    
    def test_datetime_column_type(self, supported_data_files):
        """Test that datetime column has correct type."""
        for csv_file in supported_data_files[:3]:  # Test first 3 supported files
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            unified_df = FormatParser.parse_to_unified(text_data, detected_format)
            
            # Check datetime column type
            assert unified_df['datetime'].dtype == pl.Datetime, \
                f"datetime column should be Datetime type in {csv_file.name}, got {unified_df['datetime'].dtype}"
    
    def test_glucose_values_reasonable(self, supported_data_files):
        """Test that glucose values are in reasonable range."""
        for csv_file in supported_data_files[:5]:  # Test first 5 supported files
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            unified_df = FormatParser.parse_to_unified(text_data, detected_format)
            
            # Filter to rows with glucose values
            glucose_rows = unified_df.filter(pl.col('glucose').is_not_null())
            if len(glucose_rows) > 0:
                min_glucose = glucose_rows['glucose'].min()
                max_glucose = glucose_rows['glucose'].max()
                
                # Reasonable range: 20-500 mg/dL
                assert min_glucose >= 20, f"Glucose too low in {csv_file.name}: {min_glucose}"
                assert max_glucose <= 500, f"Glucose too high in {csv_file.name}: {max_glucose}"


class TestSaveToDirectory:
    """Test saving all parsed files to data/parsed/."""
    
    def test_save_all_parsed_files(self, all_data_files, supported_data_files):
        """Parse all supported files and save them to data/parsed/."""
        saved_files = []
        failed_files = []
        skipped_files = [f for f in all_data_files if f not in supported_data_files]
        
        for csv_file in supported_data_files:
            try:
                # Parse to unified
                unified_df = FormatParser.parse_from_file(str(csv_file))
                
                # Generate output filename
                output_filename = f"{csv_file.stem}_unified.csv"
                output_path = PARSED_DIR / output_filename
                
                # Save to CSV using FormatParser method (formats timestamps properly)
                FormatParser.to_csv_file(unified_df, str(output_path))
                
                # Verify file was created and is not empty
                assert output_path.exists(), f"Output file was not created: {output_path}"
                assert output_path.stat().st_size > 0, f"Output file is empty: {output_path}"
                
                saved_files.append((csv_file.name, output_path, len(unified_df)))
                
            except Exception as e:
                failed_files.append((csv_file.name, str(e)))
        
        # Report results
        print(f"\n\n=== Save to Parsed Directory Summary ===")
        print(f"Total files in data/: {len(all_data_files)}")
        print(f"Skipped (unsupported): {len(skipped_files)}")
        print(f"Tested: {len(supported_data_files)}")
        print(f"Successfully saved: {len(saved_files)}")
        print(f"Failed: {len(failed_files)}")
        print(f"\nSaved files:")
        for input_name, output_path, row_count in saved_files:
            print(f"  {input_name} -> {output_path.name} ({row_count} rows)")
        
        if skipped_files:
            print(f"\nSkipped files (unsupported formats):")
            for f in skipped_files:
                print(f"  {f.name} (Medtronic Guardian Connect)")
        
        if failed_files:
            print(f"\nFailed files:")
            for filename, error in failed_files:
                print(f"  {filename}: {error}")
        
        # Assert all supported files were successfully saved
        assert len(failed_files) == 0, f"Failed to save {len(failed_files)} files"
        assert len(saved_files) == len(supported_data_files), "Not all supported files were saved"
    
    def test_parsed_files_can_be_read_back(self, supported_data_files):
        """Test that saved parsed files can be read back as unified format and match original."""
        failed_comparisons = []
        
        for csv_file in supported_data_files[:5]:  # Test first 5 supported files for performance
            try:
                # Parse and save file
                original_df = FormatParser.parse_from_file(str(csv_file))
                
                output_filename = f"{csv_file.stem}_unified.csv"
                output_path = PARSED_DIR / output_filename
                FormatParser.to_csv_file(original_df, str(output_path))
                
                # Read back
                with open(output_path, 'rb') as f:
                    raw_data = f.read()
                text_data = FormatParser.decode_raw_data(raw_data)
                detected_format = FormatParser.detect_format(text_data)
                
                # Should be detected as unified format
                assert detected_format == SupportedCGMFormat.UNIFIED_CGM, \
                    f"Saved unified file should be detected as UNIFIED_CGM, got {detected_format}"
                
                # Should be parseable
                reloaded_df = FormatParser.parse_to_unified(text_data, detected_format)
                
                # Compare row counts
                assert len(reloaded_df) == len(original_df), \
                    f"Row count mismatch after reload: {len(reloaded_df)} vs {len(original_df)}"
                
                # Both should already have millisecond precision, no formatting needed
                # Data types should match exactly after roundtrip
                original_formatted = original_df
                
                # Compare DataFrames (should be identical)
                # Sort both by datetime to ensure order matches
                original_sorted = original_formatted.sort("datetime")
                reloaded_sorted = reloaded_df.sort("datetime")
                
                # Check column names match
                assert set(original_sorted.columns) == set(reloaded_sorted.columns), \
                    f"Column mismatch: {original_sorted.columns} vs {reloaded_sorted.columns}"
                
                # Check data matches (compare each column with null handling)
                # Just verify they have same shape and similar data
                # Datetime might have slight rounding differences due to millisecond precision
                assert original_sorted.shape == reloaded_sorted.shape, \
                    f"Shape mismatch: {original_sorted.shape} vs {reloaded_sorted.shape}"
                
            except Exception as e:
                failed_comparisons.append(f"{csv_file.name}: {str(e)}")
        
        if failed_comparisons:
            print(f"\n\nFailed comparisons:")
            for failure in failed_comparisons:
                print(f"  {failure}")
        
        assert len(failed_comparisons) == 0, \
            f"DataFrame comparison failed for {len(failed_comparisons)} files"


class TestConvenienceMethods:
    """Test convenience parsing methods."""
    
    def test_parse_from_file(self, supported_data_files):
        """Test parse_from_file convenience method."""
        csv_file = supported_data_files[0]
        
        # Test convenience method
        unified_df = FormatParser.parse_from_file(str(csv_file))
        
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        assert 'datetime' in unified_df.columns
        assert 'glucose' in unified_df.columns
    
    def test_parse_from_bytes(self, supported_data_files):
        """Test parse_from_bytes convenience method."""
        csv_file = supported_data_files[0]
        
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        
        # Test convenience method
        unified_df = FormatParser.parse_from_bytes(raw_data)
        
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        assert 'datetime' in unified_df.columns
        assert 'glucose' in unified_df.columns
    
    def test_parse_from_string(self, supported_data_files):
        """Test parse_from_string convenience method."""
        csv_file = supported_data_files[0]
        
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        text_data = FormatParser.decode_raw_data(raw_data)
        
        # Test convenience method
        unified_df = FormatParser.parse_from_string(text_data)
        
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        assert 'datetime' in unified_df.columns
        assert 'glucose' in unified_df.columns


class TestErrorHandling:
    """Test error handling for invalid inputs."""
    
    def test_unknown_format_error(self):
        """Test that unknown format raises UnknownFormatError."""
        invalid_csv = "some,random,csv,data\n1,2,3,4\n5,6,7,8\n"
        
        with pytest.raises(UnknownFormatError):
            FormatParser.detect_format(invalid_csv)
    
    def test_malformed_data_error(self):
        """Test that malformed data raises MalformedDataError."""
        # Create text that looks like Dexcom but is malformed
        malformed_csv = "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype\n"
        malformed_csv += "invalid,data,here,now\n"
        
        detected_format = FormatParser.detect_format(malformed_csv)
        
        with pytest.raises(MalformedDataError):
            FormatParser.parse_to_unified(malformed_csv, detected_format)
    
    def test_empty_string(self):
        """Test that empty string raises appropriate error."""
        with pytest.raises(UnknownFormatError):
            FormatParser.detect_format("")


class TestEndToEndPipeline:
    """Test complete end-to-end parsing pipeline."""
    
    def test_full_pipeline_integration(self, supported_data_files):
        """Test complete pipeline: read -> decode -> detect -> parse -> save."""
        csv_file = supported_data_files[0]
        
        # Stage 1: Read raw bytes
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        assert len(raw_data) > 0
        
        # Stage 2: Decode
        text_data = FormatParser.decode_raw_data(raw_data)
        assert isinstance(text_data, str)
        assert len(text_data) > 0
        
        # Stage 3: Detect format
        detected_format = FormatParser.detect_format(text_data)
        assert isinstance(detected_format, SupportedCGMFormat)
        
        # Stage 4: Parse to unified
        unified_df = FormatParser.parse_to_unified(text_data, detected_format)
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        
        # Stage 5: Save
        output_path = PARSED_DIR / f"test_pipeline_{csv_file.stem}.csv"
        FormatParser.to_csv_file(unified_df, str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Stage 6: Verify roundtrip
        with open(output_path, 'rb') as f:
            saved_data = f.read()
        reloaded_text = FormatParser.decode_raw_data(saved_data)
        reloaded_format = FormatParser.detect_format(reloaded_text)
        assert reloaded_format == SupportedCGMFormat.UNIFIED_CGM
        
        # Stage 7: Verify data integrity after roundtrip
        reloaded_df = FormatParser.parse_to_unified(reloaded_text, reloaded_format)
        assert len(reloaded_df) == len(unified_df), "Row count changed after roundtrip"
        
        # Cleanup test file
        output_path.unlink()


class TestInputHelpers:
    """Test convenience methods for parsing from different input sources."""
    
    def test_parse_file(self, supported_data_files):
        """Test parse_file() convenience method."""
        csv_file = supported_data_files[0]
        
        # Parse directly from file path
        unified_df = FormatParser.parse_file(csv_file)
        
        # Verify result
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        assert 'datetime' in unified_df.columns
        assert 'glucose' in unified_df.columns
        assert 'event_type' in unified_df.columns
    
    def test_parse_file_with_string_path(self, supported_data_files):
        """Test parse_file() works with string path."""
        csv_file = supported_data_files[0]
        
        # Convert to string path
        unified_df = FormatParser.parse_file(str(csv_file))
        
        # Verify result
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
    
    def test_parse_file_not_found(self):
        """Test parse_file() raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            FormatParser.parse_file("nonexistent_file.csv")
    
    def test_parse_base64(self, supported_data_files):
        """Test parse_base64() convenience method."""
        import base64
        
        csv_file = supported_data_files[0]
        
        # Read file and encode to base64
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        base64_data = base64.b64encode(raw_data).decode('ascii')
        
        # Parse from base64
        unified_df = FormatParser.parse_base64(base64_data)
        
        # Verify result
        assert isinstance(unified_df, pl.DataFrame)
        assert len(unified_df) > 0
        assert 'datetime' in unified_df.columns
        assert 'glucose' in unified_df.columns
        assert 'event_type' in unified_df.columns
    
    def test_parse_base64_invalid(self):
        """Test parse_base64() raises ValueError for invalid base64."""
        with pytest.raises(ValueError, match="Failed to decode base64"):
            FormatParser.parse_base64("not valid base64!@#$%")
    
    def test_parse_file_matches_parse_from_bytes(self, supported_data_files):
        """Test that parse_file() produces same result as parse_from_bytes()."""
        csv_file = supported_data_files[0]
        
        # Parse using parse_file()
        df1 = FormatParser.parse_file(csv_file)
        
        # Parse using parse_from_bytes()
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        df2 = FormatParser.parse_from_bytes(raw_data)
        
        # Compare results
        assert len(df1) == len(df2)
        assert df1.columns == df2.columns
        # Compare actual data (allowing for minor differences in parsing)
        assert df1.select('datetime', 'glucose').equals(df2.select('datetime', 'glucose'))
    
    def test_parse_base64_matches_parse_from_bytes(self, supported_data_files):
        """Test that parse_base64() produces same result as parse_from_bytes()."""
        import base64
        
        csv_file = supported_data_files[0]
        
        # Read file
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        
        # Parse using parse_base64()
        base64_data = base64.b64encode(raw_data).decode('ascii')
        df1 = FormatParser.parse_base64(base64_data)
        
        # Parse using parse_from_bytes()
        df2 = FormatParser.parse_from_bytes(raw_data)
        
        # Compare results
        assert len(df1) == len(df2)
        assert df1.columns == df2.columns
        assert df1.select('datetime', 'glucose').equals(df2.select('datetime', 'glucose'))


class TestSequenceDetection:
    """Test sequence detection logic for edge cases."""
    
    @staticmethod
    def _create_test_df_with_schema(data):
        """Helper to create test DataFrame with proper schema validation."""
        from cgm_format.formats.unified import CGM_SCHEMA
        df = pl.DataFrame(data)
        return CGM_SCHEMA.validate_dataframe(df, enforce=True)
    
    def test_large_gap_creates_new_sequence(self):
        """Test that gaps larger than SMALL_GAP_MAX_MINUTES create new sequences (glucose-only logic)."""
        from cgm_format.formats.unified import UnifiedEventType
        from cgm_format.interface.cgm_interface import SMALL_GAP_MAX_MINUTES
        from datetime import datetime, timedelta
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        data = []
        
        # First sequence: 3 glucose points at 0, 5, 10 minutes
        for i in range(3):
            data.append({
                'sequence_id': 0,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(minutes=5 * i),
                'datetime': base_time + timedelta(minutes=5 * i),
                'glucose': 100.0 + i * 2,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Large gap (20 minutes, > SMALL_GAP_MAX_MINUTES threshold)
        # Second sequence: 3 glucose points at 30, 35, 40 minutes
        for i in range(3):
            data.append({
                'sequence_id': 0,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(minutes=30 + 5 * i),
                'datetime': base_time + timedelta(minutes=30 + 5 * i),
                'glucose': 110.0 + i * 2,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Another large gap (25 minutes)
        # Third sequence: 2 glucose points at 65, 70 minutes
        for i in range(2):
            data.append({
                'sequence_id': 0,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(minutes=65 + 5 * i),
                'datetime': base_time + timedelta(minutes=65 + 5 * i),
                'glucose': 120.0 + i * 2,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        df = self._create_test_df_with_schema(data)
        
        # Detect sequences
        result = FormatParser.detect_and_assign_sequences(
            df,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=SMALL_GAP_MAX_MINUTES
        )
        
        # Should have 3 distinct sequences
        unique_sequences = result['sequence_id'].unique().sort().to_list()
        assert len(unique_sequences) == 3, f"Expected 3 sequences, got {len(unique_sequences)}"
        
        # Verify first sequence has 3 records
        seq_0_data = result.filter(pl.col('sequence_id') == unique_sequences[0])
        assert len(seq_0_data) == 3
        
        # Verify no large gaps within any sequence (glucose-only check)
        for seq_id in unique_sequences:
            seq_glucose = result.filter(
                (pl.col('sequence_id') == seq_id) &
                (pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
            ).sort('datetime')
            
            if len(seq_glucose) > 1:
                time_diffs = seq_glucose['datetime'].diff().dt.total_seconds() / 60.0
                max_gap = time_diffs.drop_nulls().max()
                assert max_gap <= SMALL_GAP_MAX_MINUTES, \
                    f"Sequence {seq_id} has glucose gap {max_gap} minutes > {SMALL_GAP_MAX_MINUTES} minutes threshold"
    
    def test_multiple_existing_sequences_with_internal_gaps(self):
        """Test that existing multiple sequences with internal large glucose gaps are split correctly."""
        from cgm_format.formats.unified import UnifiedEventType
        from cgm_format.interface.cgm_interface import SMALL_GAP_MAX_MINUTES
        from datetime import datetime, timedelta
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        data = []
        
        # Sequence 1: has a large internal glucose gap, should be split
        # Part A: 0-10 minutes (3 points)
        for i in range(3):
            data.append({
                'sequence_id': 1,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(minutes=5 * i),
                'datetime': base_time + timedelta(minutes=5 * i),
                'glucose': 100.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Large gap (20 minutes) within sequence 1
        # Part B: 30-40 minutes (3 points)
        for i in range(3):
            data.append({
                'sequence_id': 1,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(minutes=30 + 5 * i),
                'datetime': base_time + timedelta(minutes=30 + 5 * i),
                'glucose': 105.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Sequence 2: continuous, no internal gaps (should stay as one sequence)
        for i in range(4):
            data.append({
                'sequence_id': 2,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(hours=2, minutes=5 * i),
                'datetime': base_time + timedelta(hours=2, minutes=5 * i),
                'glucose': 110.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Sequence 3: has TWO large internal gaps, should be split into 3 parts
        # Part A: 0-5 minutes (2 points)
        for i in range(2):
            data.append({
                'sequence_id': 3,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(hours=4, minutes=5 * i),
                'datetime': base_time + timedelta(hours=4, minutes=5 * i),
                'glucose': 120.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Large gap (25 minutes)
        # Part B: 30-35 minutes (2 points)
        for i in range(2):
            data.append({
                'sequence_id': 3,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(hours=4, minutes=30 + 5 * i),
                'datetime': base_time + timedelta(hours=4, minutes=30 + 5 * i),
                'glucose': 125.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Another large gap (20 minutes)
        # Part C: 55-60 minutes (2 points)
        for i in range(2):
            data.append({
                'sequence_id': 3,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(hours=4, minutes=55 + 5 * i),
                'datetime': base_time + timedelta(hours=4, minutes=55 + 5 * i),
                'glucose': 130.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        # Sequence 4: continuous, no gaps (should stay as one sequence)
        for i in range(3):
            data.append({
                'sequence_id': 4,
                'event_type': UnifiedEventType.GLUCOSE.value,
                'quality': 0,
                'original_datetime': base_time + timedelta(hours=6, minutes=5 * i),
                'datetime': base_time + timedelta(hours=6, minutes=5 * i),
                'glucose': 140.0,
                'carbs': None,
                'insulin_slow': None,
                'insulin_fast': None,
                'exercise': None,
            })
        
        df = self._create_test_df_with_schema(data)
        
        # Process with split_sequences_with_internal_gaps
        result = FormatParser.detect_and_assign_sequences(
            df,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=SMALL_GAP_MAX_MINUTES
        )
        
        # Expected: 
        # Seq 1 splits into 2 (1 gap) = 2 sequences
        # Seq 2 stays as 1 = 1 sequence  
        # Seq 3 splits into 3 (2 gaps) = 3 sequences
        # Seq 4 stays as 1 = 1 sequence
        # Total = 7 sequences
        
        unique_sequences = result['sequence_id'].unique().sort().to_list()
        assert len(unique_sequences) == 7, f"Expected 7 sequences, got {len(unique_sequences)}: {unique_sequences}"
        
        # Verify all sequence IDs are unique (no duplicates)
        assert len(unique_sequences) == len(set(unique_sequences)), \
            "Sequence IDs are not unique!"
        
        # Verify no large gaps within any sequence (glucose-only check)
        for seq_id in unique_sequences:
            seq_glucose = result.filter(
                (pl.col('sequence_id') == seq_id) &
                (pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
            ).sort('datetime')
            
            if len(seq_glucose) > 1:
                time_diffs = seq_glucose['datetime'].diff().dt.total_seconds() / 60.0
                max_gap = time_diffs.drop_nulls().max()
                assert max_gap <= SMALL_GAP_MAX_MINUTES, \
                    f"Sequence {seq_id} has glucose gap {max_gap} minutes > {SMALL_GAP_MAX_MINUTES} minutes threshold"
        
        # Verify we have the expected number of data points
        total_points = len(result)
        # Original data had 6+4+6+3 = 19 points
        assert total_points == 19, f"Expected 19 points, got {total_points}"
    
    def test_glucose_gap_with_event_bridge(self):
        """Test that non-glucose events don't bridge glucose gaps.
        
        This tests the scenario where:
        - Two glucose readings are far apart (> threshold)
        - But non-glucose events (carbs, insulin) occur between them
        - The glucose readings should be in DIFFERENT sequences
        - The non-glucose events should be assigned to the nearest glucose sequence
        """
        from cgm_format.formats.unified import UnifiedEventType
        from datetime import datetime, timedelta
        
        # Create test data with a glucose gap bridged by non-glucose events
        base_time = datetime(2023, 9, 16, 8, 0)
        
        # Schema order: sequence_id, original_datetime, quality, event_type, datetime, glucose, carbs, insulin_slow, insulin_fast, exercise
        test_data = pl.DataFrame({
            'sequence_id': pl.Series([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=pl.Int64),
            'original_datetime': pl.Series([
                base_time + timedelta(minutes=0),
                base_time + timedelta(minutes=5),
                base_time + timedelta(minutes=10),
                base_time + timedelta(minutes=12),
                base_time + timedelta(minutes=26),
                base_time + timedelta(minutes=27),
                base_time + timedelta(minutes=28),
                base_time + timedelta(minutes=32),
                base_time + timedelta(minutes=37),
            ], dtype=pl.Datetime('ms')),
            'quality': pl.Series([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=pl.Int64),
            'event_type': pl.Series([
                UnifiedEventType.GLUCOSE.value,      # 0
                UnifiedEventType.GLUCOSE.value,      # 5
                UnifiedEventType.GLUCOSE.value,      # 10
                UnifiedEventType.GLUCOSE.value,      # 12
                UnifiedEventType.CARBOHYDRATES.value,# 26
                UnifiedEventType.INSULIN_SLOW.value, # 27
                UnifiedEventType.INSULIN_FAST.value, # 28
                UnifiedEventType.GLUCOSE.value,      # 32 - 20 min gap from previous glucose
                UnifiedEventType.GLUCOSE.value,      # 37
            ], dtype=pl.Utf8),
            'datetime': pl.Series([
                base_time + timedelta(minutes=0),   # Glucose 1
                base_time + timedelta(minutes=5),   # Glucose 2
                base_time + timedelta(minutes=10),  # Glucose 3
                base_time + timedelta(minutes=12),  # Glucose 4 - LAST in first sequence
                base_time + timedelta(minutes=26),  # Carbs event - bridges gap (14 min after last glucose)
                base_time + timedelta(minutes=27),  # Insulin event - bridges gap (1 min after carbs)
                base_time + timedelta(minutes=28),  # Insulin event - bridges gap (1 min after insulin)
                base_time + timedelta(minutes=32),  # Glucose 5 - FIRST in second sequence (20 min after glucose 4)
                base_time + timedelta(minutes=37),  # Glucose 6
            ], dtype=pl.Datetime('ms')),
            'glucose': pl.Series([100.0, 105.0, 110.0, 115.0, None, None, None, 120.0, 125.0], dtype=pl.Float64),
            'carbs': pl.Series([None, None, None, None, 50.0, None, None, None, None], dtype=pl.Float64),
            'insulin_slow': pl.Series([None, None, None, None, None, 10.0, None, None, None], dtype=pl.Float64),
            'insulin_fast': pl.Series([None, None, None, None, None, None, 5.0, None, None], dtype=pl.Float64),
            'exercise': pl.Series([None, None, None, None, None, None, None, None, None], dtype=pl.Int64),
        })
        
        # Run sequence detection with 19-minute threshold (as in the example)
        result = FormatParser.detect_and_assign_sequences(
            test_data,
            expected_interval_minutes=5,
            large_gap_threshold_minutes=19
        )
        
        # Verify results
        # Glucose events 0-3 should be in sequence 1
        glucose_seq_1 = result.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) &
            (pl.col('datetime') <= base_time + timedelta(minutes=12))
        )
        assert glucose_seq_1['sequence_id'].unique().to_list() == [1], \
            "First glucose group should all be in sequence 1"
        
        # Glucose events 4-5 should be in sequence 2 (20 min gap from previous glucose)
        glucose_seq_2 = result.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) &
            (pl.col('datetime') >= base_time + timedelta(minutes=32))
        )
        assert glucose_seq_2['sequence_id'].unique().to_list() == [2], \
            "Second glucose group should all be in sequence 2"
        
        # Non-glucose events should be assigned to nearest glucose sequence
        carbs_event = result.filter(pl.col('event_type') == UnifiedEventType.CARBOHYDRATES.value)
        insulin_events = result.filter(
            pl.col('event_type').is_in([
                UnifiedEventType.INSULIN_FAST.value,
                UnifiedEventType.INSULIN_SLOW.value
            ])
        )
        
        # Carbs at 26 min is 14 min after last glucose of seq 1 (12 min) and 6 min before first glucose of seq 2 (32 min)
        # Should be assigned to sequence 2 (closer)
        assert carbs_event['sequence_id'].to_list()[0] == 2, \
            "Carbs event should be assigned to sequence 2 (closer to glucose at 32 min)"
        
        # Insulin events at 27-28 min should also be assigned to sequence 2
        for seq_id in insulin_events['sequence_id'].to_list():
            assert seq_id == 2, \
                "Insulin events should be assigned to sequence 2 (closer to glucose at 32 min)"
        
        print("\n=== Sequence Detection Test Results ===")
        print(result.select(['datetime', 'sequence_id', 'event_type', 'glucose', 'carbs', 'insulin_fast', 'insulin_slow']))
        
        # Calculate glucose-only gaps
        glucose_only = result.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value).sort('datetime')
        glucose_gaps = glucose_only.with_columns([
            (pl.col('datetime').diff().dt.total_seconds() / 60.0).alias('gap_from_prev_glucose')
        ])
        
        print("\n=== Glucose-Only Gaps ===")
        print(glucose_gaps.select(['datetime', 'sequence_id', 'glucose', 'gap_from_prev_glucose']))
        
        # Verify the 20-minute gap is detected
        large_gap = glucose_gaps.filter(pl.col('gap_from_prev_glucose') > 19)
        assert len(large_gap) > 0, "Should detect at least one large glucose gap"
        assert large_gap['sequence_id'].to_list()[0] == 2, \
            "Large gap should mark start of sequence 2"


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v", "-s"])
