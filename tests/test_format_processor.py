"""Tests for FormatProcessor implementation."""

import pytest
import polars as pl
from datetime import datetime, timedelta
from cgm_format import FormatProcessor
from cgm_format.interface.cgm_interface import (
    ProcessingWarning,
    ZeroValidInputError,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
)
from cgm_format.formats.unified import UnifiedEventType, Quality


@pytest.fixture
def sample_unified_data() -> pl.DataFrame:
    """Create sample unified format data for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    # Create 10 data points with 5-minute intervals
    data = []
    for i in range(10):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_data_with_gaps() -> pl.DataFrame:
    """Create sample data with gaps for interpolation testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    data = []
    # First segment: 0, 5, 10 minutes
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Gap of 15 minutes (10 -> 25)
    # Next point at 25 minutes
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=25),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    return pl.DataFrame(data)


@pytest.fixture
def sample_data_with_quality_issues() -> pl.DataFrame:
    """Create sample data with quality issues."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    data = []
    for i in range(5):
        quality = Quality.ILL.value if i == 2 else Quality.GOOD.value
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': quality,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    return pl.DataFrame(data)


def test_processor_initialization():
    """Test processor initialization."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    assert processor.expected_interval_minutes == 5
    assert processor.small_gap_max_minutes == 15
    assert processor.expected_interval_seconds == 300
    assert processor.small_gap_max_seconds == 900
    assert not processor.has_warnings()
    assert processor.get_warnings() == []


def test_constants_match_documentation():
    """Test that constants match documented values in PIPELINE.md."""
    # From PIPELINE.md:
    # CALIBRATION_GAP_THRESHOLD = 9900 seconds (2:45:00)
    # MINIMUM_DURATION_MINUTES = 60
    # MAXIMUM_WANTED_DURATION_MINUTES = 480
    
    assert CALIBRATION_GAP_THRESHOLD == 9900, "CALIBRATION_GAP_THRESHOLD should be 9900 seconds (2:45:00)"
    assert MINIMUM_DURATION_MINUTES == 60, "MINIMUM_DURATION_MINUTES should be 60"
    assert MAXIMUM_WANTED_DURATION_MINUTES == 480, "MAXIMUM_WANTED_DURATION_MINUTES should be 480"


def test_synchronize_timestamps_basic(sample_unified_data):
    """Test basic timestamp synchronization."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # First interpolate gaps to create sequences
    interpolated = processor.interpolate_gaps(sample_unified_data)
    
    # Then synchronize timestamps
    result = processor.synchronize_timestamps(interpolated)
    
    # Check that timestamps are rounded to minutes (seconds = 0)
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0, f"Timestamp {timestamp} has non-zero seconds"
        assert timestamp.microsecond == 0, f"Timestamp {timestamp} has non-zero microseconds"
    
    # Check that intervals are consistent (5 minutes)
    if len(result) > 1:
        time_diffs = result['datetime'].diff().dt.total_seconds() / 60.0
        time_diffs_list = time_diffs.drop_nulls().to_list()
        
        # All intervals should be 5 minutes (or very close due to rounding)
        for diff in time_diffs_list:
            assert abs(diff - 5.0) < 0.1, f"Time interval {diff} minutes is not close to expected 5 minutes"


def test_synchronize_timestamps_empty_dataframe():
    """Test synchronization with empty DataFrame."""
    processor = FormatProcessor()
    
    empty_df = pl.DataFrame({
        'sequence_id': [],
        'event_type': [],
        'quality': [],
        'datetime': [],
        'glucose': [],
        'carbs': [],
        'insulin_slow': [],
        'insulin_fast': [],
        'exercise': [],
    })
    
    with pytest.raises(ZeroValidInputError):
        processor.synchronize_timestamps(empty_df)


def test_synchronize_timestamps_no_sequence_id():
    """Test that synchronize_timestamps raises error without sequence_id."""
    processor = FormatProcessor()
    
    # Create data without sequence_id
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(3):
        data.append({
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    with pytest.raises(ValueError, match="must have sequence_id column"):
        processor.synchronize_timestamps(df)


def test_synchronize_timestamps_with_large_gap():
    """Test that synchronize_timestamps raises error if large gaps exist."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with a large gap (20 minutes)
    for i in range(2):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Large gap
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=30),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    with pytest.raises(ValueError, match="has gap of.*minutes"):
        processor.synchronize_timestamps(df)


def test_synchronize_timestamps_glucose_interpolation():
    """Test that glucose values are interpolated during synchronization."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # Create data with irregular timestamps
    base_time = datetime(2024, 1, 1, 12, 0, 10)  # Start with 10 seconds
    data = []
    
    # Points at 0s, 5m12s, 10m8s
    for i, seconds_offset in enumerate([10, 312, 608]):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(seconds=seconds_offset),
            'glucose': 100.0 + i * 10,  # 100, 110, 120
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Synchronize
    result = processor.synchronize_timestamps(df)
    
    # Should have glucose values (may be interpolated)
    assert result['glucose'].null_count() < len(result), "Should have glucose values"
    
    # All timestamps should have seconds=0
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0


def test_synchronize_timestamps_discrete_events_shifted():
    """Test that discrete events (carbs, insulin) are shifted to nearest timestamp."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # Create data with carbs and insulin at specific times
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Regular glucose readings
    for i in range(5):
        carbs_value = 30.0 if i == 2 else None
        insulin_value = 5.0 if i == 3 else None
        
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': carbs_value,
            'insulin_slow': None,
            'insulin_fast': insulin_value,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Synchronize
    result = processor.synchronize_timestamps(df)
    
    # Should have at least one carbs entry
    carbs_count = result.filter(pl.col('carbs').is_not_null()).height
    assert carbs_count > 0, "Should have carbs data"
    
    # Should have at least one insulin entry
    insulin_count = result.filter(pl.col('insulin_fast').is_not_null()).height
    assert insulin_count > 0, "Should have insulin data"


def test_synchronize_timestamps_single_point_sequence():
    """Test synchronization with single-point sequence."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 15)  # 15 seconds
    data = [{
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time,
        'glucose': 100.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    }]
    
    df = pl.DataFrame(data)
    
    # Synchronize
    result = processor.synchronize_timestamps(df)
    
    # Should have 1 record with rounded timestamp
    assert len(result) == 1
    assert result['datetime'][0].second == 0


def test_synchronize_timestamps_multiple_sequences():
    """Test synchronization with multiple sequences."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Second sequence (different time range)
    for i in range(3):
        data.append({
            'sequence_id': 1,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(hours=2, minutes=5 * i),
            'glucose': 110.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Synchronize
    result = processor.synchronize_timestamps(df)
    
    # Should have 2 sequences
    assert result['sequence_id'].n_unique() == 2
    
    # All timestamps should be rounded
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0


def test_interpolate_gaps_no_gaps(sample_unified_data):
    """Test interpolation when there are no gaps."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    result = processor.interpolate_gaps(sample_unified_data)
    
    # Should have same number of records (no interpolation needed)
    assert len(result) == len(sample_unified_data)
    assert not processor.has_warnings()


def test_interpolate_gaps_with_small_gap(sample_data_with_gaps):
    """Test interpolation with small gaps."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    result = processor.interpolate_gaps(sample_data_with_gaps)
    
    # Should have more records due to interpolation
    assert len(result) > len(sample_data_with_gaps)
    
    # Should have imputation warning
    assert processor.has_warnings()
    warnings = processor.get_warnings()
    assert ProcessingWarning.IMPUTATION in warnings
    
    # Check that imputed events were created
    imputed_count = result.filter(
        pl.col('event_type') == UnifiedEventType.IMPUTATION.value
    ).height
    assert imputed_count > 0


def test_interpolate_gaps_empty_dataframe():
    """Test interpolation with empty DataFrame."""
    processor = FormatProcessor()
    empty_df = pl.DataFrame({
        'sequence_id': [],
        'event_type': [],
        'quality': [],
        'datetime': [],
        'glucose': [],
        'carbs': [],
        'insulin_slow': [],
        'insulin_fast': [],
        'exercise': [],
    })
    
    result = processor.interpolate_gaps(empty_df)
    assert len(result) == 0


def test_prepare_for_inference_keeps_only_latest_sequence():
    """Test that prepare_for_inference keeps only the last (latest) sequence."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence (older) - 3 points
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Second sequence (middle) - 4 points
    for i in range(4):
        data.append({
            'sequence_id': 1,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(hours=2, minutes=5 * i),
            'glucose': 110.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Third sequence (latest) - 5 points
    for i in range(5):
        data.append({
            'sequence_id': 2,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(hours=4, minutes=5 * i),
            'glucose': 120.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Prepare for inference
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have only the latest sequence (5 points from sequence_id=2)
    assert len(unified_df) == 5, f"Expected 5 points from latest sequence, got {len(unified_df)}"
    
    # Check that glucose values are from the latest sequence (120.0 range)
    glucose_values = unified_df['glucose'].to_list()
    assert all(g >= 120.0 for g in glucose_values if g is not None), \
        "Should only have glucose values from latest sequence (>= 120.0)"


def test_prepare_for_inference_single_sequence_unchanged():
    """Test that single sequence data is not affected by latest sequence filtering."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Single sequence - 5 points
    for i in range(5):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Prepare for inference
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have all 5 points
    assert len(unified_df) == 5, f"Expected 5 points, got {len(unified_df)}"


def test_prepare_for_inference_latest_sequence_identification():
    """Test that latest sequence is correctly identified by most recent timestamp."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Sequence 0: starts at 12:00, ends at 12:10 (3 points)
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Sequence 1: starts at 11:00, ends at 11:20 (5 points) - earlier start but also earlier end
    for i in range(5):
        data.append({
            'sequence_id': 1,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time - timedelta(hours=1) + timedelta(minutes=5 * i),
            'glucose': 110.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Sequence 2: starts at 10:00, ends at 14:00 (huge duration) - earliest start but LATEST end
    data.append({
        'sequence_id': 2,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time - timedelta(hours=2),
        'glucose': 120.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    data.append({
        'sequence_id': 2,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(hours=2),  # Latest timestamp overall
        'glucose': 125.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    # Prepare for inference
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=300,
    )
    
    # Should keep sequence 2 (has the most recent timestamp at 14:00)
    # After truncation to 300 minutes, should have 2 points
    assert len(unified_df) == 2, f"Expected 2 points from sequence 2, got {len(unified_df)}"
    
    # Check glucose values are from sequence 2
    glucose_values = unified_df['glucose'].to_list()
    assert 120.0 in glucose_values or 125.0 in glucose_values, \
        "Should have glucose values from sequence 2"


def test_prepare_for_inference_success(sample_unified_data):
    """Test successful inference preparation."""
    processor = FormatProcessor()
    
    unified_df, warnings = processor.prepare_for_inference(
        sample_unified_data,
        minimum_duration_minutes=30,
        maximum_wanted_duration=120,
    )
    
    # Should return full UnifiedFormat with all columns
    expected_columns = ['sequence_id', 'event_type', 'quality', 'datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
    assert all(col in unified_df.columns for col in expected_columns), \
        f"Missing columns. Expected all of {expected_columns}, got {unified_df.columns}"
    
    # Should have same number of records
    assert len(unified_df) == len(sample_unified_data)
    
    # Should NOT have TOO_SHORT warning (45 minutes of data >= 30 minute minimum)
    assert ProcessingWarning.TOO_SHORT not in warnings
    
    # Test that to_data_only_df() works correctly
    data_only_df = FormatProcessor.to_data_only_df(unified_df)
    expected_data_columns = ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
    assert data_only_df.columns == expected_data_columns


def test_prepare_for_inference_with_quality_issues(sample_data_with_quality_issues):
    """Test inference preparation with quality issues."""
    processor = FormatProcessor()
    
    unified_df, warnings = processor.prepare_for_inference(
        sample_data_with_quality_issues,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have QUALITY warning
    assert ProcessingWarning.QUALITY in warnings


def test_prepare_for_inference_zero_valid_input():
    """Test inference preparation with no valid data."""
    processor = FormatProcessor()
    
    # Create data with no glucose values
    empty_glucose_data = pl.DataFrame({
        'sequence_id': [0, 0],
        'event_type': [UnifiedEventType.GLUCOSE.value] * 2,
        'quality': [Quality.GOOD.value] * 2,
        'datetime': [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 5)],
        'glucose': [None, None],
        'carbs': [None, None],
        'insulin_slow': [None, None],
        'insulin_fast': [None, None],
        'exercise': [None, None],
    })
    
    with pytest.raises(ZeroValidInputError):
        processor.prepare_for_inference(empty_glucose_data)


def test_prepare_for_inference_empty_dataframe():
    """Test inference preparation with empty DataFrame."""
    processor = FormatProcessor()
    
    empty_df = pl.DataFrame({
        'sequence_id': [],
        'event_type': [],
        'quality': [],
        'datetime': [],
        'glucose': [],
        'carbs': [],
        'insulin_slow': [],
        'insulin_fast': [],
        'exercise': [],
    })
    
    with pytest.raises(ZeroValidInputError):
        processor.prepare_for_inference(empty_df)


def test_prepare_for_inference_truncation(sample_unified_data):
    """Test that sequences are truncated to maximum duration, keeping latest data."""
    processor = FormatProcessor()
    
    # sample_unified_data has 10 records from 12:00 to 12:45 (5-minute intervals)
    # Set maximum duration to 20 minutes - should keep LATEST 20 minutes
    unified_df, warnings = processor.prepare_for_inference(
        sample_unified_data,
        minimum_duration_minutes=10,
        maximum_wanted_duration=20,
    )
    
    # Should have fewer records due to truncation
    assert len(unified_df) <= 5, f"Expected at most 5 records, got {len(unified_df)}"
    
    # Verify that LATEST data is preserved (not oldest)
    # The latest timestamps should be present
    timestamps = sorted(unified_df['datetime'].to_list())
    if len(timestamps) > 0:
        # Latest timestamp should be close to 12:45 (the end of the data)
        latest = timestamps[-1]
        # Should be from the last 20 minutes of data
        assert latest.hour == 12 and latest.minute >= 25, \
            f"Expected latest data to be preserved, but got timestamp {latest}"


def test_prepare_for_inference_truncation_keeps_latest():
    """Test that truncation keeps the latest (most recent) data, not the oldest."""
    processor = FormatProcessor()
    
    # Create data spanning 60 minutes (0 to 60 minutes, 13 points)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(13):  # 0, 5, 10, ..., 60 minutes
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 10,  # 100, 110, 120, ..., 220
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Truncate to 30 minutes - should keep LATEST 30 minutes (30-60 min range)
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=30,
    )
    
    # Should have approximately 7 records (30, 35, 40, 45, 50, 55, 60 minutes)
    assert len(unified_df) >= 6 and len(unified_df) <= 8, \
        f"Expected 6-8 records for 30-minute window, got {len(unified_df)}"
    
    # Verify glucose values are from the LATEST part (should be 160-220, not 100-160)
    glucose_values = sorted([g for g in unified_df['glucose'].to_list() if g is not None])
    min_glucose = min(glucose_values) if glucose_values else 0
    max_glucose = max(glucose_values) if glucose_values else 0
    
    # Latest data should have glucose >= 160 (from last 30 minutes)
    assert min_glucose >= 150, \
        f"Expected latest data (glucose >= 160), but got min glucose {min_glucose}"
    assert max_glucose >= 200, \
        f"Expected latest data to include highest values, but got max glucose {max_glucose}"
    
    # Verify timestamps are from the latest 30 minutes
    timestamps = sorted(unified_df['datetime'].to_list())
    earliest = timestamps[0]
    latest = timestamps[-1]
    
    # Should span approximately 30 minutes
    duration = (latest - earliest).total_seconds() / 60.0
    assert duration <= 30, f"Duration {duration} should be <= 30 minutes"
    
    # Latest timestamp should be at or near 60 minutes (13:00)
    assert latest >= base_time + timedelta(minutes=55), \
        f"Expected latest timestamp >= 12:55, got {latest}"


def test_prepare_for_inference_with_calibration_events():
    """Test inference preparation with calibration events."""
    processor = FormatProcessor()
    
    # Create data with calibration event
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(5):
        event_type = UnifiedEventType.CALIBRATION.value if i == 2 else UnifiedEventType.GLUCOSE.value
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    calibration_data = pl.DataFrame(data)
    
    unified_df, warnings = processor.prepare_for_inference(
        calibration_data,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have CALIBRATION warning
    assert ProcessingWarning.CALIBRATION in processor.get_warnings()
    assert (warnings & ProcessingWarning.CALIBRATION) == ProcessingWarning.CALIBRATION


def test_prepare_for_inference_with_sensor_calibration_quality():
    """Test inference preparation with SENSOR_CALIBRATION quality flag."""
    processor = FormatProcessor()
    
    # Create data with sensor calibration quality
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(5):
        quality = Quality.SENSOR_CALIBRATION.value if i == 2 else Quality.GOOD.value
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': quality,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    sensor_calibration_data = pl.DataFrame(data)
    
    unified_df, warnings = processor.prepare_for_inference(
        sensor_calibration_data,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have QUALITY warning (SENSOR_CALIBRATION triggers QUALITY warning)
    assert ProcessingWarning.QUALITY in processor.get_warnings()
    assert (warnings & ProcessingWarning.QUALITY) == ProcessingWarning.QUALITY


def test_warnings_accumulation():
    """Test that warnings accumulate correctly in a list."""
    processor = FormatProcessor()
    
    # Manually add warnings
    processor._add_warning(ProcessingWarning.IMPUTATION)
    assert processor.has_warnings()
    warnings = processor.get_warnings()
    assert ProcessingWarning.IMPUTATION in warnings
    assert len(warnings) == 1
    
    processor._add_warning(ProcessingWarning.QUALITY)
    warnings = processor.get_warnings()
    assert ProcessingWarning.IMPUTATION in warnings
    assert ProcessingWarning.QUALITY in warnings
    assert len(warnings) == 2
    
    # Can add same warning multiple times
    processor._add_warning(ProcessingWarning.IMPUTATION)
    warnings = processor.get_warnings()
    assert len(warnings) == 3
    assert warnings.count(ProcessingWarning.IMPUTATION) == 2


def test_sequence_creation_with_no_sequence_id():
    """Test that sequence_id is created when not present."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # Create data without sequence_id
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First segment: 0, 5, 10 minutes
    for i in range(3):
        data.append({
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Large gap (20 minutes) - should create new sequence
    # Next point at 30 minutes
    data.append({
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=30),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    # Process - should create sequence_id
    result = processor.interpolate_gaps(df)
    
    # Should have sequence_id column
    assert 'sequence_id' in result.columns
    
    # Should have 2 sequences (split by large gap)
    assert result['sequence_id'].n_unique() == 2


def test_large_gap_creates_new_sequence():
    """Test that gaps larger than small_gap_max_minutes create new sequences."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence: 3 points at 0, 5, 10 minutes
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Large gap (20 minutes, > 15 minutes threshold)
    # Second sequence: 3 points at 30, 35, 40 minutes
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=30 + 5 * i),
            'glucose': 110.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Another large gap (25 minutes)
    # Third sequence: 2 points at 65, 70 minutes
    for i in range(2):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=65 + 5 * i),
            'glucose': 120.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Process - should split into 3 sequences
    result = processor.interpolate_gaps(df)
    
    # Should have 3 distinct sequences
    unique_sequences = result['sequence_id'].unique().sort().to_list()
    assert len(unique_sequences) == 3, f"Expected 3 sequences, got {len(unique_sequences)}"
    
    # Verify first sequence has 3 records (no interpolation, all within expected interval)
    seq_0_data = result.filter(pl.col('sequence_id') == unique_sequences[0])
    # Could have interpolation if there was a small gap
    assert len(seq_0_data) >= 3
    
    # Verify no interpolation across large gaps (check event types don't span sequences)
    for seq_id in unique_sequences:
        seq_data = result.filter(pl.col('sequence_id') == seq_id).sort('datetime')
        # Check that sequence is continuous (no large gaps)
        if len(seq_data) > 1:
            time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
            max_gap = time_diffs.drop_nulls().max()
            assert max_gap <= 15, f"Sequence {seq_id} has gap {max_gap} minutes > 15 minutes threshold"


def test_multiple_existing_sequences_with_internal_gaps():
    """Test that existing multiple sequences with internal large gaps are split correctly.
    
    This tests the scenario where we have sequences 1, 2, 3, 4 already, and some of them
    have large gaps internally that need to be split into new sub-sequences.
    Ensures sequence IDs remain unique and don't conflict.
    """
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Sequence 1: has a large internal gap, should be split
    # Part A: 0-10 minutes (3 points)
    for i in range(3):
        data.append({
            'sequence_id': 1,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
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
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(hours=6, minutes=5 * i),
            'glucose': 140.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Process
    result = processor.interpolate_gaps(df)
    
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
    
    # Verify no large gaps within any sequence
    for seq_id in unique_sequences:
        seq_data = result.filter(pl.col('sequence_id') == seq_id).sort('datetime')
        if len(seq_data) > 1:
            time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
            max_gap = time_diffs.drop_nulls().max()
            assert max_gap <= 15, \
                f"Sequence {seq_id} has gap {max_gap} minutes > 15 minutes threshold"
    
    # Verify we have the expected number of data points
    total_points = sum(len(result.filter(pl.col('sequence_id') == seq_id)) 
                      for seq_id in unique_sequences)
    # Original data had 6+4+6+3 = 19 points, may have interpolation
    assert total_points >= 19, f"Expected at least 19 points, got {total_points}"


def test_small_vs_large_gap_handling():
    """Test that small gaps are interpolated but large gaps create new sequences."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Point at 0 minutes
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time,
        'glucose': 100.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    # Small gap (12 minutes, < 15 minutes threshold) - should be interpolated
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=12),
        'glucose': 105.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    # Large gap (20 minutes, > 15 minutes threshold) - should create new sequence
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=32),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    # Process
    result = processor.interpolate_gaps(df)
    
    # Should have 2 sequences (split by large gap)
    unique_sequences = result['sequence_id'].unique().sort().to_list()
    assert len(unique_sequences) == 2
    
    # First sequence should have interpolated points
    seq_0_data = result.filter(pl.col('sequence_id') == unique_sequences[0])
    # Original 2 points + interpolated points (12 min gap with 5 min interval = 1 missing point)
    assert len(seq_0_data) > 2, "Expected interpolation in first sequence"
    
    # Check for imputation events in first sequence
    imputed_in_seq_0 = seq_0_data.filter(
        pl.col('event_type') == UnifiedEventType.IMPUTATION.value
    ).height
    assert imputed_in_seq_0 > 0, "Expected imputation events in first sequence"
    
    # Second sequence should only have 1 point (no interpolation)
    seq_1_data = result.filter(pl.col('sequence_id') == unique_sequences[1])
    assert len(seq_1_data) == 1
    
    # Should have IMPUTATION warning
    assert processor.has_warnings()
    assert ProcessingWarning.IMPUTATION in processor.get_warnings()


def test_calibration_gap_marks_next_24_hours_as_sensor_calibration():
    """Test that gaps >= CALIBRATION_GAP_THRESHOLD (2:45:00) mark next 24 hours as SENSOR_CALIBRATION quality.
    
    According to PIPELINE.md: "In case of large gap more than 2 hours 45 minutes 
    mark next 24 hours as ill quality."
    """
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First segment: 3 points before the gap
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Large gap >= CALIBRATION_GAP_THRESHOLD (2:45:00 = 165 minutes)
    # Gap of 3 hours (180 minutes) to ensure it exceeds threshold
    gap_start_time = base_time + timedelta(minutes=10)
    gap_end_time = gap_start_time + timedelta(hours=3)
    
    # After gap: create data points for next 25 hours (to test 24-hour window)
    # Points every 5 minutes for 25 hours = 300 points
    for i in range(300):  # 25 hours * 12 points/hour = 300 points
        point_time = gap_end_time + timedelta(minutes=5 * i)
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,  # Will be changed by interpolate_gaps
            'datetime': point_time,
            'glucose': 110.0 + i * 0.1,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Process with interpolate_gaps
    result = processor.interpolate_gaps(df)
    
    # Find the gap position (between last point before gap and first point after gap)
    # The gap is between index 2 (10 minutes) and index 3 (3 hours later)
    
    # Get all timestamps after the gap
    gap_start_idx = 2  # Last point before gap
    gap_start_datetime = result['datetime'][gap_start_idx]
    
    # All points after gap_end_time should be marked as SENSOR_CALIBRATION for 24 hours
    # Calculate 24 hours after gap_end_time
    gap_end_datetime = gap_start_datetime + timedelta(hours=3)  # 3 hour gap
    calibration_period_end = gap_end_datetime + timedelta(hours=24)
    
    # Check points within 24 hours after gap (including first point after gap)
    points_in_calibration_period = result.filter(
        (pl.col('datetime') >= gap_end_datetime) &
        (pl.col('datetime') <= calibration_period_end)
    )
    
    assert len(points_in_calibration_period) > 0, \
        "Should have points in the 24-hour calibration period"
    
    # All points in the 24-hour period should be marked as SENSOR_CALIBRATION
    sensor_calibration_count = points_in_calibration_period.filter(
        pl.col('quality') == Quality.SENSOR_CALIBRATION.value
    ).height
    
    assert sensor_calibration_count == len(points_in_calibration_period), \
        f"All {len(points_in_calibration_period)} points in 24-hour period should be SENSOR_CALIBRATION, " \
        f"but only {sensor_calibration_count} are marked"
    
    # Points after 24 hours should be GOOD quality
    points_after_calibration_period = result.filter(
        pl.col('datetime') > calibration_period_end
    )
    
    if len(points_after_calibration_period) > 0:
        good_quality_count = points_after_calibration_period.filter(
            pl.col('quality') == Quality.GOOD.value
        ).height
        
        assert good_quality_count == len(points_after_calibration_period), \
            f"Points after 24-hour period should be GOOD quality, " \
            f"but {len(points_after_calibration_period) - good_quality_count} are not"


def test_calibration_gap_exactly_at_threshold():
    """Test that gap exactly at CALIBRATION_GAP_THRESHOLD (2:45:00) triggers calibration marking."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First point
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time,
        'glucose': 100.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    # Gap exactly at CALIBRATION_GAP_THRESHOLD (2:45:00 = 165 minutes)
    gap_start = base_time
    gap_end = gap_start + timedelta(seconds=CALIBRATION_GAP_THRESHOLD)
    
    # Point immediately after gap
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': gap_end,
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    # A few more points within 24 hours
    for i in range(1, 5):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': gap_end + timedelta(minutes=5 * i),
            'glucose': 110.0 + i * 0.1,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Process
    result = processor.interpolate_gaps(df)
    
    # Points after gap (within 24 hours) should be SENSOR_CALIBRATION
    calibration_period_end = gap_end + timedelta(hours=24)
    points_after_gap = result.filter(
        (pl.col('datetime') >= gap_end) &
        (pl.col('datetime') <= calibration_period_end)
    )
    
    if len(points_after_gap) > 0:
        sensor_calibration_count = points_after_gap.filter(
            pl.col('quality') == Quality.SENSOR_CALIBRATION.value
        ).height
        
        assert sensor_calibration_count == len(points_after_gap), \
            f"All points after gap (within 24h) should be SENSOR_CALIBRATION, " \
            f"but only {sensor_calibration_count}/{len(points_after_gap)} are marked"


def test_calibration_gap_below_threshold_no_marking():
    """Test that gaps below CALIBRATION_GAP_THRESHOLD do not trigger calibration marking."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First point
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time,
        'glucose': 100.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    # Gap just below threshold (2:44:59 = 9899 seconds, threshold is 9900)
    gap_below_threshold = timedelta(seconds=CALIBRATION_GAP_THRESHOLD - 1)
    gap_end = base_time + gap_below_threshold
    
    # Point after gap
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': gap_end,
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    # Process
    result = processor.interpolate_gaps(df)
    
    # Point after gap should remain GOOD quality (not marked as SENSOR_CALIBRATION)
    point_after_gap = result.filter(pl.col('datetime') > base_time)
    
    if len(point_after_gap) > 0:
        good_quality_count = point_after_gap.filter(
            pl.col('quality') == Quality.GOOD.value
        ).height
        
        assert good_quality_count == len(point_after_gap), \
            f"Points after gap below threshold should remain GOOD quality, " \
            f"but {len(point_after_gap) - good_quality_count} are marked differently"


def test_full_pipeline(sample_data_with_gaps):
    """Test full processing pipeline."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # Step 1: Interpolate gaps
    interpolated = processor.interpolate_gaps(sample_data_with_gaps)
    assert len(interpolated) > len(sample_data_with_gaps)
    
    # Step 2: Prepare for inference
    unified_df, warnings = processor.prepare_for_inference(
        interpolated,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Convert to data-only format
    data_df = FormatProcessor.to_data_only_df(unified_df)
    
    # Check results
    assert len(data_df) > 0
    assert data_df.columns == ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
    assert ProcessingWarning.IMPUTATION in warnings


def test_full_pipeline_with_synchronization(sample_data_with_gaps):
    """Test full processing pipeline including timestamp synchronization."""
    processor = FormatProcessor(
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    
    # Step 1: Interpolate gaps
    interpolated = processor.interpolate_gaps(sample_data_with_gaps)
    assert len(interpolated) > len(sample_data_with_gaps)
    
    # Step 2: Synchronize timestamps
    synchronized = processor.synchronize_timestamps(interpolated)
    
    # Verify all timestamps are rounded
    for timestamp in synchronized['datetime'].to_list():
        assert timestamp.second == 0, f"Timestamp {timestamp} should have seconds=0"
    
    # Verify fixed frequency (5 minutes)
    if len(synchronized) > 1:
        time_diffs = synchronized['datetime'].diff().dt.total_seconds() / 60.0
        time_diffs_list = time_diffs.drop_nulls().to_list()
        for diff in time_diffs_list:
            assert abs(diff - 5.0) < 0.1, f"Time interval {diff} should be ~5 minutes"
    
    # Step 3: Prepare for inference
    unified_df, warnings = processor.prepare_for_inference(
        synchronized,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Convert to data-only format
    data_df = FormatProcessor.to_data_only_df(unified_df)
    
    # Check results
    assert len(data_df) > 0
    assert data_df.columns == ['datetime', 'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
    assert ProcessingWarning.IMPUTATION in warnings


def test_prepare_for_inference_glucose_only():
    """Test glucose_only flag drops non-EGV events before truncation."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create mixed events: GLUCOSE, CALIBRATION, IMPUTATION
    for i in range(10):
        if i == 2:
            event_type = UnifiedEventType.CALIBRATION.value
        elif i == 5:
            event_type = UnifiedEventType.IMPUTATION.value
        else:
            event_type = UnifiedEventType.GLUCOSE.value
            
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Test without glucose_only (should keep all events)
    unified_df_all, warnings_all = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
        glucose_only=False,
    )
    
    # Should have 10 records
    assert len(unified_df_all) == 10
    
    # Test with glucose_only (should drop CALIBRATION but keep IMPUTATION)
    # Create a new processor to reset warnings
    processor2 = FormatProcessor()
    unified_df_glucose, warnings_glucose = processor2.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
        glucose_only=True,
    )
    
    # Should have 9 records (10 - 1 CALIBRATION)
    assert len(unified_df_glucose) == 9, f"Expected 9 records, got {len(unified_df_glucose)}"
    
    # No CALIBRATION warning should be present (it was filtered out)
    assert ProcessingWarning.CALIBRATION not in processor2.get_warnings()


def test_prepare_for_inference_drop_duplicates():
    """Test drop_duplicates flag removes duplicate timestamps."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with duplicate timestamps
    for i in range(8):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Add duplicates at index 3 and 5
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=15),  # Same as index 3
        'glucose': 999.0,  # Different value
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=25),  # Same as index 5
        'glucose': 888.0,  # Different value
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    # Test without drop_duplicates (should keep duplicates)
    unified_df_with_dups, warnings_with_dups = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
        drop_duplicates=False,
    )
    
    # Should have 10 records (with duplicates)
    assert len(unified_df_with_dups) == 10
    # Should have TIME_DUPLICATES warning
    assert ProcessingWarning.TIME_DUPLICATES in processor.get_warnings()
    
    # Test with drop_duplicates (should remove duplicates)
    processor2 = FormatProcessor()
    unified_df_no_dups, warnings_no_dups = processor2.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
        drop_duplicates=True,
    )
    
    # Should have 8 records (duplicates removed)
    assert len(unified_df_no_dups) == 8, f"Expected 8 records after dropping duplicates, got {len(unified_df_no_dups)}"
    # Should NOT have TIME_DUPLICATES warning (duplicates were removed)
    assert ProcessingWarning.TIME_DUPLICATES not in processor2.get_warnings()


def test_prepare_for_inference_time_duplicates_warning():
    """Test that TIME_DUPLICATES warning is raised for non-unique timestamps."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with duplicate timestamps
    for i in range(5):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Add a duplicate timestamp
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': Quality.GOOD.value,
        'datetime': base_time + timedelta(minutes=10),  # Same as index 2
        'glucose': 999.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = pl.DataFrame(data)
    
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have TIME_DUPLICATES warning
    assert ProcessingWarning.TIME_DUPLICATES in processor.get_warnings()
    assert (warnings & ProcessingWarning.TIME_DUPLICATES) == ProcessingWarning.TIME_DUPLICATES


def test_prepare_for_inference_warnings_after_truncation():
    """Test that warnings are calculated on truncated data, not before truncation.
    
    This is the key bug fix: warnings should only reflect the data that is actually
    output, not data that was truncated away.
    """
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create 60 minutes of data (13 points at 5-minute intervals)
    for i in range(13):
        # Add calibration event at the beginning (will be truncated)
        event_type = UnifiedEventType.CALIBRATION.value if i == 0 else UnifiedEventType.GLUCOSE.value
        # Add quality issue at the beginning (will be truncated)
        quality = Quality.ILL.value if i == 1 else Quality.GOOD.value
        
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': quality,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 10,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # Truncate to last 20 minutes - should remove the calibration and quality issue
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=20,  # Keep only last 20 minutes
    )
    
    # Should have truncated to ~20 minutes (5 points)
    assert len(unified_df) <= 5, f"Expected ~5 records for 20 minutes, got {len(unified_df)}"
    
    # Should NOT have CALIBRATION warning (it was truncated away)
    assert ProcessingWarning.CALIBRATION not in processor.get_warnings(), \
        "CALIBRATION warning should not be present after truncation"
    
    # Should NOT have QUALITY warning (it was truncated away)
    assert ProcessingWarning.QUALITY not in processor.get_warnings(), \
        "QUALITY warning should not be present after truncation"


def test_prepare_for_inference_glucose_only_with_truncation():
    """Test that glucose_only filtering happens before truncation."""
    processor = FormatProcessor()
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create 60 minutes of data
    for i in range(13):
        # Add calibration events in the middle and at the end
        if i in [5, 6, 12]:
            event_type = UnifiedEventType.CALIBRATION.value
        else:
            event_type = UnifiedEventType.GLUCOSE.value
            
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': Quality.GOOD.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 10,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = pl.DataFrame(data)
    
    # First filter to glucose_only, then truncate to last 30 minutes
    unified_df, warnings = processor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=30,
        glucose_only=True,
    )
    
    # After filtering out 3 calibration events and truncating to 30 minutes,
    # we should have no calibration warning
    assert ProcessingWarning.CALIBRATION not in processor.get_warnings(), \
        "No CALIBRATION warning should be present with glucose_only=True"


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v", "-s"])
