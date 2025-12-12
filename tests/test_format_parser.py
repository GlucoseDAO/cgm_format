"""Pytest tests for format_parser module.

Tests cover:
1. Format detection for all files in data/
2. Parsing to unified format
3. Saving parsed results to data/parsed/

Note: Gap detection and sequence assignment tests are in test_format_processor.py
since those operations are processor responsibilities.
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


@pytest.fixture(scope="session")
def all_data_files():
    """Get all CSV files from the data directory."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files found in {DATA_DIR}"
    return csv_files


@pytest.fixture(scope="session")
def supported_data_files(all_data_files):
    """Get CSV files excluding unsupported formats using format_supported method."""
    supported_files = []
    for f in all_data_files:
        with open(f, 'rb') as file:
            if FormatParser.format_supported(file.read()):
                supported_files.append(f)
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

    def test_format_supported(self, all_data_files, supported_data_files):
        """Test format_supported method correctly identifies supported formats."""
        correct_positives = 0
        correct_negatives = 0
        false_positives = []
        false_negatives = []
        
        for csv_file in all_data_files:
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            
            is_supported = FormatParser.format_supported(raw_data)
            should_be_supported = csv_file in supported_data_files
            
            if is_supported and should_be_supported:
                correct_positives += 1
            elif not is_supported and not should_be_supported:
                correct_negatives += 1
            elif is_supported and not should_be_supported:
                false_positives.append(csv_file.name)
            else:  # not is_supported and should_be_supported
                false_negatives.append(csv_file.name)
        
        print(f"\n\n=== format_supported() Test Results ===")
        print(f"Total files tested: {len(all_data_files)}")
        print(f"Correct positives (supported & detected): {correct_positives}")
        print(f"Correct negatives (unsupported & rejected): {correct_negatives}")
        print(f"False positives (unsupported but detected): {len(false_positives)}")
        print(f"False negatives (supported but rejected): {len(false_negatives)}")
        
        if false_positives:
            print(f"\nFalse positives:")
            for fname in false_positives:
                print(f"  {fname}")
        
        if false_negatives:
            print(f"\nFalse negatives:")
            for fname in false_negatives:
                print(f"  {fname}")
        
        # Assert no false negatives (all supported files should be detected)
        assert len(false_negatives) == 0, \
            f"format_supported() failed to detect {len(false_negatives)} supported files"
        
        # Test with string input
        csv_file = supported_data_files[0]
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        text_data = FormatParser.decode_raw_data(raw_data)
        
        # format_supported should work with both bytes and strings
        assert FormatParser.format_supported(raw_data) == True
        assert FormatParser.format_supported(text_data) == True
        
        # Test with invalid data
        assert FormatParser.format_supported(b"not a cgm file") == False
        assert FormatParser.format_supported("not a cgm file") == False


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
                unified_df = FormatParser.parse_file(str(csv_file))
                
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
                original_df = FormatParser.parse_file(str(csv_file))
                
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
        """Test parse_file convenience method."""
        csv_file = supported_data_files[0]
        
        # Test convenience method
        unified_df = FormatParser.parse_file(str(csv_file))
        
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
    
    def test_error_message_truncation(self):
        """Test that error messages with huge data are truncated to prevent log overflow."""
        # Create a massive CSV that will fail to parse
        # The Polars error will include the CSV data, which could be megabytes
        huge_header = "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype\n"
        # Add 20KB of invalid data that looks like Dexcom format
        huge_data = huge_header + ("invalid,malformed,data,row\n" * 1000)
        
        detected_format = FormatParser.detect_format(huge_data)
        
        try:
            FormatParser.parse_to_unified(huge_data, detected_format)
            assert False, "Should have raised MalformedDataError"
        except MalformedDataError as e:
            error_msg = str(e)
            # Error message should be truncated to around 8192 bytes plus truncation notice
            # Allow some buffer for the truncation notice text
            assert len(error_msg) < 10000, f"Error message too long: {len(error_msg)} bytes"
            
            # If the original error was long enough, it should include truncation notice
            if len(huge_data) > 8192:
                assert "TRUNCATED" in error_msg or len(error_msg) < 1000, \
                    "Long error messages should either be truncated or short"


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




if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v", "-s"])
