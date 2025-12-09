"""Pytest tests for FormatProcessor utility methods.

Tests cover:
1. split_glucose_events() static method
"""

import pytest
from pathlib import Path
from datetime import datetime
import polars as pl

from cgm_format import FormatParser, FormatProcessor
from cgm_format.formats.unified import UnifiedEventType, CGM_SCHEMA


# Constants - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def sample_unified_df():
    """Get a sample unified format DataFrame for testing."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        pytest.skip("No CSV files found in data directory")
    
    # Parse first file
    unified_df = FormatParser.parse_file(csv_files[0])
    return unified_df


class TestSplitGlucoseEvents:
    """Test split_glucose_events() static method."""
    
    def test_split_basic(self, sample_unified_df):
        """Test basic split functionality."""
        glucose_df, events_df = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Both should be DataFrames
        assert isinstance(glucose_df, pl.DataFrame)
        assert isinstance(events_df, pl.DataFrame)
        
        # Verify columns are preserved (UnifiedFormat schema)
        expected_columns = sample_unified_df.columns
        assert glucose_df.columns == expected_columns
        assert events_df.columns == expected_columns
    
    def test_split_glucose_only_contains_egv(self, sample_unified_df):
        """Test that glucose DataFrame contains only EGV_READ events."""
        glucose_df, _ = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Get unique event types in glucose DataFrame
        event_types = glucose_df['event_type'].unique().to_list()
        
        # Should only contain EGV_READ (imputation is now a quality flag, not an event type)
        for event_type in event_types:
            assert event_type == 'EGV_READ', \
                f"Unexpected event type in glucose_df: {event_type}"
    
    def test_split_events_excludes_glucose(self, sample_unified_df):
        """Test that events DataFrame excludes EGV_READ."""
        _, events_df = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Get unique event types in events DataFrame
        event_types = events_df['event_type'].unique().to_list()
        
        # Should not contain EGV_READ (imputation is now a quality flag, not an event type)
        assert 'EGV_READ' not in event_types, "EGV_READ found in events_df"
    
    def test_split_no_data_loss(self, sample_unified_df):
        """Test that split doesn't lose any rows."""
        glucose_df, events_df = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Total rows should equal original
        total_rows = len(glucose_df) + len(events_df)
        assert total_rows == len(sample_unified_df), \
            f"Data loss detected: {len(sample_unified_df)} -> {total_rows}"
    
    def test_split_empty_events(self):
        """Test split when DataFrame has no non-glucose events."""
        # Create a DataFrame with only glucose events (use Python datetime for correct dtype inference)
        df = pl.DataFrame({
            'sequence_id': [1, 1, 1],
            'original_datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 10)
            ],
            'quality': [0, 0, 0],
            'event_type': ['EGV_READ', 'EGV_READ', 'EGV_READ'],
            'datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 10)
            ],
            'glucose': [100.0, 105.0, 110.0],
            'carbs': [None, None, None],
            'insulin_slow': [None, None, None],
            'insulin_fast': [None, None, None],
            'exercise': [None, None, None]
        })
        
        # Enforce schema to cast types properly
        df = CGM_SCHEMA.validate_dataframe(df, enforce=True)
        
        glucose_df, events_df = FormatProcessor.split_glucose_events(df)
        
        # Glucose should have all rows
        assert len(glucose_df) == 3
        # Events should be empty
        assert len(events_df) == 0
    
    def test_split_empty_glucose(self):
        """Test split when DataFrame has no glucose events."""
        # Create a DataFrame with only non-glucose events (use Python datetime for correct dtype inference)
        df = pl.DataFrame({
            'sequence_id': [1, 1],
            'original_datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5)
            ],
            'quality': [0, 0],
            'event_type': ['CARBS_IN', 'INS_FAST'],
            'datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5)
            ],
            'glucose': [None, None],
            'carbs': [30.0, None],
            'insulin_slow': [None, None],
            'insulin_fast': [None, 5.0],
            'exercise': [None, None]
        })
        
        # Enforce schema to cast types properly
        df = CGM_SCHEMA.validate_dataframe(df, enforce=True)
        
        glucose_df, events_df = FormatProcessor.split_glucose_events(df)
        
        # Glucose should be empty
        assert len(glucose_df) == 0
        # Events should have all rows
        assert len(events_df) == 2
    
    def test_split_preserves_data_integrity(self, sample_unified_df):
        """Test that split preserves all column data."""
        glucose_df, events_df = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Combine back together
        combined = pl.concat([glucose_df, events_df]).sort('datetime')
        original = sample_unified_df.sort('datetime')
        
        # Should match original (row order might differ)
        assert len(combined) == len(original)
        
        # Check that all datetimes are preserved
        assert set(combined['datetime'].to_list()) == set(original['datetime'].to_list())
    
    def test_split_with_imputation_events(self):
        """Test that only EGV_READ and CARBS events are split correctly."""
        # Create a DataFrame with glucose and other events (use Python datetime for correct dtype inference)
        df = pl.DataFrame({
            'sequence_id': [1, 1, 1, 1],
            'original_datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 10),
                datetime(2024, 1, 1, 12, 15)
            ],
            'quality': [0, 0, 0, 0],
            'event_type': ['EGV_READ', 'EGV_READ', 'EGV_READ', 'CARBS_IN'],
            'datetime': [
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 12, 5),
                datetime(2024, 1, 1, 12, 10),
                datetime(2024, 1, 1, 12, 15)
            ],
            'glucose': [100.0, 102.5, 105.0, None],
            'carbs': [None, None, None, 30.0],
            'insulin_slow': [None, None, None, None],
            'insulin_fast': [None, None, None, None],
            'exercise': [None, None, None, None]
        })
        
        # Enforce schema to cast types properly
        df = CGM_SCHEMA.validate_dataframe(df, enforce=True)
        
        glucose_df, events_df = FormatProcessor.split_glucose_events(df)
        
        # Glucose should have only EGV_READ (3 rows)
        assert len(glucose_df) == 3
        assert all(e == 'EGV_READ' for e in glucose_df['event_type'].to_list())
        
        # Events should only have CARBS_IN (1 row)
        assert len(events_df) == 1
        assert events_df['event_type'][0] == 'CARBS_IN'
    
    def test_split_chainable(self, sample_unified_df):
        """Test that split can be chained with other operations."""
        # Split
        glucose_df, events_df = FormatProcessor.split_glucose_events(sample_unified_df)
        
        # Chain with Polars operations
        filtered_glucose = glucose_df.filter(pl.col('glucose').is_not_null())
        
        # Should still work
        assert isinstance(filtered_glucose, pl.DataFrame)
        assert len(filtered_glucose) > 0


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v", "-s"])
