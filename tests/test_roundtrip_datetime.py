"""Test roundtrip datetime type preservation for all CGM formats.

This test ensures that datetime columns preserve their type through the roundtrip:
vendor CSV -> unified DataFrame -> unified CSV -> unified DataFrame

Tests parametrized for:
- Dexcom format
- Libre format  
- Unified format (direct roundtrip)

Also tests DataFrame equality (both Polars and Pandas) to ensure data is preserved exactly.
"""

import pytest
from pathlib import Path
from io import StringIO
import polars as pl
import pandas as pd

from cgm_format import FormatParser
from cgm_format.interface.cgm_interface import SupportedCGMFormat


# Test data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def is_medtronic_file(file_path: Path) -> bool:
    """Check if a file is a Medtronic Guardian Connect file."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(500).decode('utf-8', errors='ignore')
            return "Guardian Connect" in header
    except Exception:
        return False


def get_test_files_by_format():
    """Get test files grouped by format.
    
    Returns:
        List of tuples (file_path, expected_format)
    """
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if "parsed" not in str(f) and not is_medtronic_file(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {DATA_DIR}")
    
    # Detect format for each file
    files_by_format = []
    for csv_file in csv_files:
        try:
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            files_by_format.append((csv_file, detected_format))
        except Exception:
            continue
    
    return files_by_format


def create_minimal_unified_csv() -> str:
    """Create minimal unified CSV for testing."""
    return """sequence_id,event_type,quality,datetime,glucose,carbs,insulin_slow,insulin_fast,exercise
0,EGV_READ,0,2019-10-14T16:42:37.000,55.0,,,,
0,EGV_READ,0,2019-10-14T16:47:37.000,55.0,,,,
0,EGV_READ,0,2019-10-14T16:52:37.000,60.0,,,,"""


class TestDatetimeRoundtrip:
    """Test datetime type preservation through roundtrip."""
    
    def test_dataframe_equality_polars(self):
        """Test that roundtrip preserves DataFrame equality in Polars."""
        csv_data = create_minimal_unified_csv()
        
        # Parse to DataFrame
        df_original = FormatParser.parse_from_string(csv_data)
        
        # Roundtrip through CSV
        csv_string = FormatParser.to_csv_string(df_original)
        df_roundtrip = FormatParser.parse_from_string(csv_string)
        
        print(f"\n1. Original DataFrame:")
        print(df_original)
        print(f"\n   Schema: {df_original.schema}")
        
        print(f"\n2. Roundtrip DataFrame:")
        print(df_roundtrip)
        print(f"\n   Schema: {df_roundtrip.schema}")
        
        # Check schemas match
        assert df_original.schema == df_roundtrip.schema, \
            f"Schema mismatch:\nOriginal: {df_original.schema}\nRoundtrip: {df_roundtrip.schema}"
        
        # Check DataFrames are equal (Polars)
        # Sort both by datetime to ensure order matches
        df_original_sorted = df_original.sort('datetime')
        df_roundtrip_sorted = df_roundtrip.sort('datetime')
        
        try:
            assert df_original_sorted.equals(df_roundtrip_sorted), \
                "Polars DataFrames should be equal after roundtrip"
        except AssertionError as e:
            # Show differences
            print(f"\n❌ Polars DataFrames not equal!")
            print(f"\nOriginal:\n{df_original_sorted}")
            print(f"\nRoundtrip:\n{df_roundtrip_sorted}")
            
            # Check each column
            for col in df_original_sorted.columns:
                if not df_original_sorted[col].equals(df_roundtrip_sorted[col]):
                    print(f"\n❌ Column '{col}' differs:")
                    print(f"   Original dtype: {df_original_sorted[col].dtype}")
                    print(f"   Roundtrip dtype: {df_roundtrip_sorted[col].dtype}")
                    print(f"   Original values: {df_original_sorted[col].to_list()}")
                    print(f"   Roundtrip values: {df_roundtrip_sorted[col].to_list()}")
            raise
    
    def test_dataframe_equality_pandas(self):
        """Test that roundtrip preserves DataFrame equality when converted to Pandas."""
        csv_data = create_minimal_unified_csv()
        
        # Parse to Polars DataFrame
        df_original_pl = FormatParser.parse_from_string(csv_data)
        
        # Roundtrip through CSV
        csv_string = FormatParser.to_csv_string(df_original_pl)
        df_roundtrip_pl = FormatParser.parse_from_string(csv_string)
        
        # Convert to Pandas
        df_original_pd = df_original_pl.to_pandas()
        df_roundtrip_pd = df_roundtrip_pl.to_pandas()
        
        print(f"\n1. Original Pandas DataFrame:")
        print(df_original_pd)
        print(f"\n   Dtypes:\n{df_original_pd.dtypes}")
        
        print(f"\n2. Roundtrip Pandas DataFrame:")
        print(df_roundtrip_pd)
        print(f"\n   Dtypes:\n{df_roundtrip_pd.dtypes}")
        
        # Check dtypes match
        for col in df_original_pd.columns:
            assert df_original_pd[col].dtype == df_roundtrip_pd[col].dtype, \
                f"Pandas dtype mismatch for column '{col}': {df_original_pd[col].dtype} != {df_roundtrip_pd[col].dtype}"
        
        # Sort both by datetime
        df_original_pd_sorted = df_original_pd.sort_values('datetime').reset_index(drop=True)
        df_roundtrip_pd_sorted = df_roundtrip_pd.sort_values('datetime').reset_index(drop=True)
        
        # Check DataFrames are equal (Pandas)
        try:
            pd.testing.assert_frame_equal(df_original_pd_sorted, df_roundtrip_pd_sorted)
            print(f"\n✅ Pandas DataFrames are equal after roundtrip")
        except AssertionError as e:
            print(f"\n❌ Pandas DataFrames not equal!")
            print(f"\nDifference:\n{e}")
            
            # Show differences for each column
            for col in df_original_pd_sorted.columns:
                try:
                    pd.testing.assert_series_equal(df_original_pd_sorted[col], df_roundtrip_pd_sorted[col])
                except AssertionError:
                    print(f"\n❌ Column '{col}' differs:")
                    print(f"   Original: {df_original_pd_sorted[col].tolist()}")
                    print(f"   Roundtrip: {df_roundtrip_pd_sorted[col].tolist()}")
            raise
    
    def test_unified_format_roundtrip_datetime_type(self):
        """Test that unified format preserves datetime type in roundtrip.
        
        This is the bug: unified CSV -> DataFrame -> CSV -> DataFrame should preserve datetime type.
        """
        # Create minimal unified CSV
        csv_data = create_minimal_unified_csv()
        
        # Parse to DataFrame
        df1 = FormatParser.parse_from_string(csv_data)
        
        print(f"\n1. Initial parse from unified CSV:")
        print(f"   datetime dtype: {df1['datetime'].dtype}")
        
        # Check datetime column type (this should be Datetime, but it's String - BUG!)
        assert str(df1['datetime'].dtype).startswith('Datetime'), \
            f"BUG: First parse should have Datetime type, got {df1['datetime'].dtype}"
        
        # Convert back to CSV
        csv_string = FormatParser.to_csv_string(df1)
        
        # Parse again from CSV
        df2 = FormatParser.parse_from_string(csv_string)
        
        print(f"\n2. After roundtrip:")
        print(f"   datetime dtype: {df2['datetime'].dtype}")
        
        # Check datetime column type is preserved
        assert str(df2['datetime'].dtype).startswith('Datetime'), \
            f"BUG: Roundtrip should preserve Datetime type, got {df2['datetime'].dtype}"
        
        # Verify both have same dtype
        assert df1['datetime'].dtype == df2['datetime'].dtype, \
            f"Datetime dtype changed after roundtrip: {df1['datetime'].dtype} -> {df2['datetime'].dtype}"
    
    @pytest.mark.parametrize("file_path,format_type", get_test_files_by_format(), 
                            ids=lambda x: x.name if isinstance(x, Path) else str(x))
    def test_real_file_roundtrip_datetime_type(self, file_path: Path, format_type: SupportedCGMFormat):
        """Test that real vendor files preserve datetime type through roundtrip.
        
        Test flow:
        1. Parse vendor CSV (Dexcom/Libre/Unified) -> unified DataFrame
        2. Convert unified DataFrame -> unified CSV string
        3. Parse unified CSV string -> unified DataFrame
        4. Check datetime column has Datetime type (not String)
        
        Args:
            file_path: Path to test file
            format_type: Detected format type
        """
        print(f"\n{'='*70}")
        print(f"Testing: {file_path.name}")
        print(f"Format: {format_type.name}")
        print(f"{'='*70}")
        
        # Step 1: Parse original vendor file
        df_original = FormatParser.parse_file(file_path)
        
        print(f"\n1. Original parse ({format_type.name} -> unified):")
        print(f"   Rows: {len(df_original)}")
        print(f"   datetime dtype: {df_original['datetime'].dtype}")
        print(f"   datetime range: {df_original['datetime'].min()} to {df_original['datetime'].max()}")
        
        # Verify original parse has Datetime type
        assert str(df_original['datetime'].dtype).startswith('Datetime'), \
            f"Original parse should have Datetime type, got {df_original['datetime'].dtype}"
        
        # Step 2: Convert to unified CSV
        unified_csv = FormatParser.to_csv_string(df_original)
        
        print(f"\n2. Converted to unified CSV:")
        print(f"   CSV length: {len(unified_csv)} bytes")
        print(f"   Sample (first 200 chars):\n{unified_csv[:200]}")
        
        # Step 3: Parse unified CSV back to DataFrame
        df_roundtrip = FormatParser.parse_from_string(unified_csv)
        
        print(f"\n3. Roundtrip parse (unified CSV -> unified DataFrame):")
        print(f"   Rows: {len(df_roundtrip)}")
        print(f"   datetime dtype: {df_roundtrip['datetime'].dtype}")
        
        # **THE KEY TEST**: datetime column should be Datetime type, not String
        assert str(df_roundtrip['datetime'].dtype).startswith('Datetime'), \
            f"ROUNDTRIP BUG: datetime should be Datetime type after roundtrip, got {df_roundtrip['datetime'].dtype}"
        
        # Verify dtypes match
        assert df_original['datetime'].dtype == df_roundtrip['datetime'].dtype, \
            f"Datetime dtype changed: {df_original['datetime'].dtype} -> {df_roundtrip['datetime'].dtype}"
        
        # Verify row count matches
        assert len(df_original) == len(df_roundtrip), \
            f"Row count mismatch: {len(df_original)} -> {len(df_roundtrip)}"
        
        # Verify DataFrame equality (Polars)
        # After the fix, schemas must match perfectly for roundtrips
        print(f"\n4. Testing Polars DataFrame equality...")
        df_original_sorted = df_original.sort('datetime')
        df_roundtrip_sorted = df_roundtrip.sort('datetime')
        
        # Check schema equality
        print(f"   Original schema: {df_original_sorted.schema}")
        print(f"   Roundtrip schema: {df_roundtrip_sorted.schema}")
        
        assert df_original_sorted.schema == df_roundtrip_sorted.schema, \
            f"Schema mismatch after roundtrip:\nOriginal: {df_original_sorted.schema}\nRoundtrip: {df_roundtrip_sorted.schema}"
        
        # Check DataFrame equality
        assert df_original_sorted.equals(df_roundtrip_sorted), \
            "Polars DataFrames should be equal after roundtrip"
        
        print(f"   ✅ Polars DataFrames are equal")
        
        # Verify DataFrame equality (Pandas)
        print(f"\n5. Testing Pandas DataFrame equality...")
        df_original_pd = df_original_sorted.to_pandas()
        df_roundtrip_pd = df_roundtrip_sorted.to_pandas()
        
        pd.testing.assert_frame_equal(df_original_pd, df_roundtrip_pd)
        print(f"   ✅ Pandas DataFrames are equal")
        
        print(f"\n✅ SUCCESS: datetime type and data preserved perfectly through roundtrip")
    
    def test_unified_format_all_datetime_formats(self):
        """Test that unified format parser handles various ISO 8601 datetime formats."""
        test_cases = [
            # ISO 8601 with milliseconds (standard)
            "2019-10-14T16:42:37.000",
            # ISO 8601 without milliseconds
            "2019-10-14T16:42:37",
            # ISO 8601 with microseconds
            "2019-10-14T16:42:37.123456",
        ]
        
        for datetime_str in test_cases:
            csv_data = f"""sequence_id,event_type,quality,datetime,glucose,carbs,insulin_slow,insulin_fast,exercise
0,EGV_READ,0,{datetime_str},55.0,,,,"""
            
            print(f"\nTesting datetime format: {datetime_str}")
            
            # This might fail for some formats if UNIFIED_TIMESTAMP_FORMATS doesn't cover them
            try:
                df = FormatParser.parse_from_string(csv_data)
                print(f"  ✅ Parsed successfully, dtype: {df['datetime'].dtype}")
                assert str(df['datetime'].dtype).startswith('Datetime'), \
                    f"Should parse as Datetime, got {df['datetime'].dtype}"
            except Exception as e:
                pytest.fail(f"Failed to parse datetime format '{datetime_str}': {e}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

