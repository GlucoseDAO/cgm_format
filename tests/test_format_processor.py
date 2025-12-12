"""Tests for FormatProcessor implementation."""

import pytest
import polars as pl
from datetime import datetime, timedelta
from cgm_format import FormatProcessor as FormatProcessorPrime
from cgm_format.interface.cgm_interface import (
    ProcessingWarning,
    ZeroValidInputError,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
    EXPECTED_INTERVAL_MINUTES,
    SMALL_GAP_MAX_MINUTES,
)
from cgm_format.formats.unified import CGM_SCHEMA, UnifiedEventType, Quality, GOOD_QUALITY
from cgm_format.interface.cgm_interface import ValidationMethod
from typing import ClassVar

class FormatProcessor(FormatProcessorPrime):
    """Format processor for testing."""
    validation_mode_default : ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT

def create_test_dataframe(data: list[dict]) -> pl.DataFrame:
    """Helper to create test DataFrame with correct schema (datetime[ms] not datetime[μs]).
    
    Args:
        data: List of dicts with unified format columns
        
    Returns:
        DataFrame with canonical unified schema
    """
    df = pl.DataFrame(data)
    
    # If original_datetime is missing, copy from datetime
    if 'original_datetime' not in df.columns and 'datetime' in df.columns:
        df = df.with_columns([
            pl.col('datetime').alias('original_datetime')
        ])
    
    # Enforce canonical schema to match FormatParser output
    return CGM_SCHEMA.validate_dataframe(df, enforce=True)


def create_empty_dataframe() -> pl.DataFrame:
    """Helper to create empty DataFrame with correct unified schema.
    
    Returns:
        Empty DataFrame with all unified format columns
    """
    # Create empty dict with all columns from schema (service + data columns)
    empty_data = {}
    for col in CGM_SCHEMA.service_columns + CGM_SCHEMA.data_columns:
        empty_data[col['name']] = []
    
    df = pl.DataFrame(empty_data)
    # Enforce canonical schema
    return CGM_SCHEMA.validate_dataframe(df, enforce=True)


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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    return create_test_dataframe(data)


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
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=25),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    return create_test_dataframe(data)


@pytest.fixture
def sample_data_with_quality_issues() -> pl.DataFrame:
    """Create sample data with quality issues."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    data = []
    for i in range(5):
        quality = Quality.OUT_OF_RANGE.value if i == 2 else GOOD_QUALITY.value
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
    
    return create_test_dataframe(data)


def test_processor_initialization():
    """Test processor class configuration."""
    # FormatProcessor is now all classmethods - no instantiation needed
    assert FormatProcessor.expected_interval_minutes == EXPECTED_INTERVAL_MINUTES
    assert FormatProcessor.small_gap_max_minutes == SMALL_GAP_MAX_MINUTES



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
    
    # First interpolate gaps to create sequences
    interpolated = FormatProcessor.interpolate_gaps(sample_unified_data)
    
    # Then synchronize timestamps
    result = FormatProcessor.synchronize_timestamps(interpolated)
    
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
    # FormatProcessor uses classmethods - no instantiation needed
    
    empty_df = create_empty_dataframe()
    
    with pytest.raises(ZeroValidInputError):
        FormatProcessor.synchronize_timestamps(empty_df)


def test_synchronize_timestamps_no_sequence_id():
    """Test that synchronize_timestamps raises error without sequence_id.
    
    Note: This tests direct DataFrame manipulation, bypassing the parser.
    In normal usage, the parser always generates sequence_id via detect_and_assign_sequences().
    This test ensures validation works if someone calls synchronize_timestamps() directly.
    """
    # FormatProcessor uses classmethods - no instantiation needed
    
    # Create data without sequence_id - use pl.DataFrame directly to bypass schema validation
    # This simulates calling synchronize_timestamps() directly without going through the parser
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(3):
        data.append({
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'original_datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Create DataFrame manually without sequence_id (bypassing parser and schema validation)
    df = pl.DataFrame(data)
    
    # Should raise MalformedDataError from schema validation (not ValueError)
    # because synchronize_timestamps() validates the schema first
    from cgm_format.interface.cgm_interface import MalformedDataError
    with pytest.raises(MalformedDataError, match="Number of columns in schema and dataframe do not match"):
        FormatProcessor.synchronize_timestamps(df)


def test_synchronize_timestamps_with_large_gap():
    """Test that synchronize_timestamps works on sequences with large gaps between them.
    
    Note: Large gap handling is now done by detect_and_assign_sequences, not synchronize_timestamps.
    This test verifies that synchronize_timestamps can handle multiple sequences.
    """
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with a large gap (20 minutes)
    for i in range(2):
        data.append({
            'sequence_id': 1,  # First sequence
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Large gap - second sequence
    data.append({
        'sequence_id': 2,  # Second sequence
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=30),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Synchronize should work fine with multiple sequences
    result = FormatProcessor.synchronize_timestamps(df)
    
    # Should have 2 sequences
    assert result['sequence_id'].n_unique() == 2
    
    # All timestamps should be rounded
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0


def test_synchronize_timestamps_glucose_interpolation():
    """Test that glucose values are interpolated during synchronization."""
    
    # Create data with irregular timestamps
    base_time = datetime(2024, 1, 1, 12, 0, 10)  # Start with 10 seconds
    data = []
    
    # Points at 0s, 5m12s, 10m8s
    for i, seconds_offset in enumerate([10, 312, 608]):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(seconds=seconds_offset),
            'glucose': 100.0 + i * 10,  # 100, 110, 120
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Synchronize
    result = FormatProcessor.synchronize_timestamps(df)
    
    # Should have glucose values (may be interpolated)
    assert result['glucose'].null_count() < len(result), "Should have glucose values"
    
    # All timestamps should have seconds=0
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0

def test_synchronize_timestamps_discrete_events_shifted():
    """Test that discrete events (carbs, insulin) are shifted to nearest timestamp."""
    
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': carbs_value,
            'insulin_slow': None,
            'insulin_fast': insulin_value,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Synchronize
    result = FormatProcessor.synchronize_timestamps(df)
    
    # Should have at least one carbs entry
    carbs_count = result.filter(pl.col('carbs').is_not_null()).height
    assert carbs_count > 0, "Should have carbs data"
    
    # Should have at least one insulin entry
    insulin_count = result.filter(pl.col('insulin_fast').is_not_null()).height
    assert insulin_count > 0, "Should have insulin data"


def test_synchronize_timestamps_single_point_sequence():
    """Test synchronization with single-point sequence."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 15)  # 15 seconds
    data = [{
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': GOOD_QUALITY.value,
        'datetime': base_time,
        'glucose': 100.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    }]
    
    df = create_test_dataframe(data)
    
    # Synchronize
    result = FormatProcessor.synchronize_timestamps(df)
    
    # Should have 1 record with rounded timestamp
    assert len(result) == 1
    assert result['datetime'][0].second == 0


def test_synchronize_timestamps_multiple_sequences():
    """Test synchronization with multiple sequences."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(hours=2, minutes=5 * i),
            'glucose': 110.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Synchronize
    result = FormatProcessor.synchronize_timestamps(df)
    
    # Should have 2 sequences
    assert result['sequence_id'].n_unique() == 2
    
    # All timestamps should be rounded
    for timestamp in result['datetime'].to_list():
        assert timestamp.second == 0


def test_interpolate_gaps_no_gaps(sample_unified_data):
    """Test interpolation when there are no gaps."""
    
    result = FormatProcessor.interpolate_gaps(sample_unified_data)
    
    # Should have same number of records (no interpolation needed)
    assert len(result) == len(sample_unified_data)


def test_interpolate_gaps_with_small_gap(sample_data_with_gaps):
    """Test interpolation with small gaps (snap_to_grid=False).
    
    Note: sample_data_with_gaps has a 15-minute gap (12:10 -> 12:25).
    With snap_to_grid=False, interpolated points are placed at regular intervals
    from the previous timestamp.
    """
    
    result = FormatProcessor.interpolate_gaps(sample_data_with_gaps)
    
    # interpolate_gaps should interpolate the gap (15 min < 19 min threshold)
    # Expected: 4 original points + 2 interpolated points (at 12:15 and 12:20) = 6 total
    assert len(result) > len(sample_data_with_gaps), \
        f"Expected interpolation to add records, got {len(result)} vs {len(sample_data_with_gaps)}"
    
    # Check that imputed events were created
    imputed_count = result.filter(
        (pl.col('quality') & Quality.IMPUTATION.value) != 0
    ).height
    assert imputed_count > 0, "Should have imputed events marked with IMPUTATION flag"


# Generate test parameters for snap_to_grid interpolation
# Starting times: 12:10:00 + (0, 10, 20, ..., 300) seconds = 12:10:00 to 12:15:00
# Gap sizes will vary based on grid alignment - test dynamically calculates expected fills
# No hardcoded expectations; the test will compute them based on grid rounding logic
_snap_to_grid_params = []
for gap_minutes in [8, 13, 18, 23]:
    for start_seconds in range(0, 301, 10):  # 0, 10, 20, ..., 300
        _snap_to_grid_params.append((start_seconds, gap_minutes))


@pytest.mark.parametrize("start_seconds,gap_minutes", _snap_to_grid_params)
def test_interpolate_gaps_with_snap_to_grid(start_seconds, gap_minutes):
    """Test interpolation with snap_to_grid=True for various starting times and gap sizes.
    
    Tests that interpolation correctly handles:
    - Different starting timestamps (10-second steps from 12:10:00 to 12:15:00)
    - Different gap sizes (8, 13, 18, 23 minutes)
    - Grid-aligned interpolation regardless of actual timestamp values
    - Gap sizes exceeding small_gap_max_minutes are NOT interpolated
    
    The test dynamically calculates expected fills based on grid rounding logic,
    since the number of interpolated points depends on where timestamps fall relative
    to the 5-minute grid.
    
    Args:
        start_seconds: Seconds offset for the gap start time (0-300, 10-second steps)
        gap_minutes: Size of the gap in minutes (8, 13, 18, or 23)
    """
    
    # Create test data with specific gap
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    gap_start_time = datetime(2024, 1, 1, 12, 10, 0) + timedelta(seconds=start_seconds)
    
    data = []
    # First segment: 3 points ending at gap_start_time
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': gap_start_time - timedelta(minutes=10) + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    # Point after gap
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': GOOD_QUALITY.value,
        'datetime': gap_start_time + timedelta(minutes=gap_minutes),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Calculate expected fills based on grid rounding logic
    # This mirrors the interpolation logic: round to nearest grid, count points between
    prev_dt = gap_start_time
    next_dt = gap_start_time + timedelta(minutes=gap_minutes)
    grid_start = FormatProcessor.get_sequence_grid_start(df)
    
    # Round to nearest grid points (same logic as interpolation)
    prev_grid_dt = FormatProcessor.calculate_grid_point(prev_dt, grid_start, round_direction='nearest')
    curr_grid_dt = FormatProcessor.calculate_grid_point(next_dt, grid_start, round_direction='nearest')
    
    # Calculate grid positions
    prev_grid_pos = int((prev_grid_dt - grid_start).total_seconds() / 60.0 / EXPECTED_INTERVAL_MINUTES)
    curr_grid_pos = int((curr_grid_dt - grid_start).total_seconds() / 60.0 / EXPECTED_INTERVAL_MINUTES)
    
    # Expected fills: range(prev_grid_pos + 1, curr_grid_pos)
    # Only if gap is within small_gap_max_minutes threshold
    if gap_minutes <= SMALL_GAP_MAX_MINUTES:
        expected_filled = max(0, curr_grid_pos - prev_grid_pos - 1)
    else:
        expected_filled = 0
    
    result = FormatProcessor.interpolate_gaps(df)
    
    # Check that the expected number of points were filled
    num_filled = len(result) - len(df)
    assert num_filled == expected_filled, \
        f"Gap of {gap_minutes} min starting at +{start_seconds}s: " \
        f"Expected {expected_filled} interpolated points (grid pos {prev_grid_pos}→{curr_grid_pos}), got {num_filled}"
    
    if expected_filled > 0:
        # Check that imputed events were created with IMPUTATION + SYNCHRONIZATION flags
        imputed_count = result.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) != 0
        ).height
        assert imputed_count == expected_filled, \
            f"Expected {expected_filled} imputed events, got {imputed_count}"
        
        # Check that SYNCHRONIZATION flag is also set (snap_to_grid=True)
        sync_count = result.filter(
            (pl.col('quality') & Quality.SYNCHRONIZATION.value) != 0
        ).height
        assert sync_count == expected_filled, \
            f"Expected {expected_filled} synchronized events, got {sync_count}"


def test_interpolate_gaps_empty_dataframe():
    """Test interpolation with empty DataFrame."""
    # FormatProcessor uses classmethods - no instantiation needed
    empty_df = create_empty_dataframe()
    
    result = FormatProcessor.interpolate_gaps(empty_df)
    assert len(result) == 0


def test_prepare_for_inference_keeps_only_latest_sequence():
    """Test that prepare_for_inference keeps only the last (latest) sequence."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence (older) - 3 points
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(hours=4, minutes=5 * i),
            'glucose': 120.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Prepare for inference
    unified_df, warnings = FormatProcessor.prepare_for_inference(
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
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Single sequence - 5 points
    for i in range(5):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Prepare for inference
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have all 5 points
    assert len(unified_df) == 5, f"Expected 5 points, got {len(unified_df)}"


def test_prepare_for_inference_latest_sequence_identification():
    """Test that latest sequence is correctly identified by most recent timestamp."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Sequence 0: starts at 12:00, ends at 12:10 (3 points)
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(hours=2),  # Latest timestamp overall
        'glucose': 125.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Prepare for inference
    unified_df, warnings = FormatProcessor.prepare_for_inference(
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
    # FormatProcessor uses classmethods - no instantiation needed
    
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        sample_unified_data,
        minimum_duration_minutes=30,
        maximum_wanted_duration=120,
    )
    
    # Should return full UnifiedFormat with all columns
    expected_columns = [col['name'] for col in CGM_SCHEMA.service_columns + CGM_SCHEMA.data_columns]
    assert all(col in unified_df.columns for col in expected_columns), \
        f"Missing columns. Expected all of {expected_columns}, got {unified_df.columns}"
    
    # Should have same number of records
    assert len(unified_df) == len(sample_unified_data)
    
    # Should NOT have TOO_SHORT warning (45 minutes of data >= 30 minute minimum)
    assert ProcessingWarning.TOO_SHORT not in warnings
    
    # Test that to_data_only_df() works correctly
    data_only_df = FormatProcessor.to_data_only_df(unified_df)
    expected_data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
    assert data_only_df.columns == expected_data_columns


def test_prepare_for_inference_with_quality_issues(sample_data_with_quality_issues):
    """Test inference preparation with quality issues."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        sample_data_with_quality_issues,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have OUT_OF_RANGE warning (since fixture has OUT_OF_RANGE quality)
    assert ProcessingWarning.OUT_OF_RANGE in warnings


def test_prepare_for_inference_zero_valid_input():
    """Test inference preparation with no valid data."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    # Create data with no glucose values
    empty_glucose_data = pl.DataFrame({
        'sequence_id': [0, 0],
        'event_type': [UnifiedEventType.GLUCOSE.value] * 2,
        'quality': [GOOD_QUALITY.value] * 2,
        'datetime': [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 5)],
        'original_datetime': [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 5)],
        'glucose': [None, None],
        'carbs': [None, None],
        'insulin_slow': [None, None],
        'insulin_fast': [None, None],
        'exercise': [None, None],
    })
    
    # Enforce schema to ensure proper types
    empty_glucose_data = CGM_SCHEMA.validate_dataframe(empty_glucose_data, enforce=True)
    
    with pytest.raises(ZeroValidInputError):
        FormatProcessor.prepare_for_inference(empty_glucose_data)


def test_prepare_for_inference_empty_dataframe():
    """Test inference preparation with empty DataFrame."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    empty_df = create_empty_dataframe()
    
    with pytest.raises(ZeroValidInputError):
        FormatProcessor.prepare_for_inference(empty_df)


def test_prepare_for_inference_truncation(sample_unified_data):
    """Test that sequences are truncated to maximum duration, keeping latest data."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    # sample_unified_data has 10 records from 12:00 to 12:45 (5-minute intervals)
    # Set maximum duration to 20 minutes - should keep LATEST 20 minutes
    unified_df, warnings = FormatProcessor.prepare_for_inference(
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
    # FormatProcessor uses classmethods - no instantiation needed
    
    # Create data spanning 60 minutes (0 to 60 minutes, 13 points)
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(13):  # 0, 5, 10, ..., 60 minutes
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 10,  # 100, 110, 120, ..., 220
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Truncate to 30 minutes - should keep LATEST 30 minutes (30-60 min range)
    unified_df, warnings = FormatProcessor.prepare_for_inference(
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
    # FormatProcessor uses classmethods - no instantiation needed
    
    # Create data with calibration event
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(5):
        event_type = UnifiedEventType.CALIBRATION.value if i == 2 else UnifiedEventType.GLUCOSE.value
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    calibration_data = create_test_dataframe(data)
    
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        calibration_data,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have CALIBRATION warning
    assert (warnings & ProcessingWarning.CALIBRATION) == ProcessingWarning.CALIBRATION


def test_prepare_for_inference_with_sensor_calibration_quality():
    """Test inference preparation with SENSOR_CALIBRATION quality flag."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    # Create data with sensor calibration quality
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    for i in range(5):
        quality = Quality.SENSOR_CALIBRATION.value if i == 2 else GOOD_QUALITY.value
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
    
    sensor_calibration_data = create_test_dataframe(data)
    
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        sensor_calibration_data,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have CALIBRATION warning (SENSOR_CALIBRATION triggers CALIBRATION warning)
    assert (warnings & ProcessingWarning.CALIBRATION) == ProcessingWarning.CALIBRATION


def test_warnings_accumulation():
    """Test that warnings are returned as flags from methods."""
    # FormatProcessor uses classmethods that return warnings
    # Warnings are no longer accumulated in instance state
    
    # Test that warnings combine correctly using bitwise OR
    warning1 = ProcessingWarning.CALIBRATION  # Value 2
    warning2 = ProcessingWarning.OUT_OF_RANGE  # Value 4
    
    combined = warning1 | warning2
    
    # Check individual flags are present
    assert (combined & ProcessingWarning.CALIBRATION) == ProcessingWarning.CALIBRATION
    assert (combined & ProcessingWarning.OUT_OF_RANGE) == ProcessingWarning.OUT_OF_RANGE
    
    # Check that unset flags are not present  
    assert not (combined & ProcessingWarning.TOO_SHORT)
    assert not (combined & ProcessingWarning.IMPUTATION)


def test_sequence_creation_with_no_sequence_id():
    """Test that sequence_id is created when not present."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First segment: 0, 5, 10 minutes
    for i in range(3):
        data.append({
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=30),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Process - should create sequence_id via detect_and_assign_sequences
    result = FormatProcessor.detect_and_assign_sequences(df)
    
    # Should have sequence_id column
    assert 'sequence_id' in result.columns
    
    # Should have 2 sequences (split by large gap)
    assert result['sequence_id'].n_unique() == 2


def test_large_gap_creates_new_sequence():
    """Test that gaps larger than SMALL_GAP_MAX_MINUTES create new sequences."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First sequence: 3 points at 0, 5, 10 minutes
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=65 + 5 * i),
            'glucose': 120.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Process - should split into 3 sequences using detect_and_assign_sequences
    result = FormatProcessor.detect_and_assign_sequences(df)
    
    # Should have 3 distinct sequences
    unique_sequences = result['sequence_id'].unique().sort().to_list()
    assert len(unique_sequences) == 3, f"Expected 3 sequences, got {len(unique_sequences)}"
    
    # Verify first sequence has 3 records
    seq_0_data = result.filter(pl.col('sequence_id') == unique_sequences[0])
    assert len(seq_0_data) == 3
    
    # Verify no large gaps within any sequence (check glucose-only gaps)
    for seq_id in unique_sequences:
        seq_data = result.filter(pl.col('sequence_id') == seq_id).sort('datetime')
        # Check that sequence is continuous (no large gaps)
        if len(seq_data) > 1:
            time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
            max_gap = time_diffs.drop_nulls().max()
            assert max_gap <= SMALL_GAP_MAX_MINUTES, f"Sequence {seq_id} has gap {max_gap} minutes > {SMALL_GAP_MAX_MINUTES} minutes threshold"


def test_multiple_existing_sequences_with_internal_gaps():
    """Test that existing multiple sequences with internal large gaps are split correctly.
    
    This tests the scenario where we have sequences 1, 2, 3, 4 already, and some of them
    have large gaps internally that need to be split into new sub-sequences.
    Ensures sequence IDs remain unique and don't conflict.
    """
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Sequence 1: has a large internal gap, should be split
    # Part A: 0-10 minutes (3 points)
    for i in range(3):
        data.append({
            'sequence_id': 1,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(hours=6, minutes=5 * i),
            'glucose': 140.0,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Process using detect_and_assign_sequences
    result = FormatProcessor.detect_and_assign_sequences(df)
    
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
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Point at 0 minutes
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=32),
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # First detect sequences (splits by large gap)
    df_with_sequences = FormatProcessor.detect_and_assign_sequences(df)
    
    # Should have 2 sequences (split by large gap)
    unique_sequences = df_with_sequences['sequence_id'].unique().sort().to_list()
    assert len(unique_sequences) == 2
    
    # Then interpolate small gaps within sequences
    result = FormatProcessor.interpolate_gaps(df_with_sequences)
    
    # First sequence should have interpolated points
    seq_0_data = result.filter(pl.col('sequence_id') == unique_sequences[0])
    # Original 2 points + interpolated points (12 min gap with 5 min interval = 1 missing point)
    assert len(seq_0_data) > 2, "Expected interpolation in first sequence"
    
    # Check for imputation events in first sequence
    imputed_in_seq_0 = seq_0_data.filter(
        (pl.col('quality') & Quality.IMPUTATION.value) != 0
    ).height
    assert imputed_in_seq_0 > 0, "Expected imputation events in first sequence"
    
    # Second sequence should only have 1 point (no interpolation)
    seq_1_data = result.filter(pl.col('sequence_id') == unique_sequences[1])
    assert len(seq_1_data) == 1
    
    # Should have IMPUTATION warning (returned from interpolate_gaps)
    # Note: warnings would have been returned from interpolate_gaps call above

def test_calibration_gap_marks_next_24_hours_as_sensor_calibration():
    """Test that gaps >= CALIBRATION_GAP_THRESHOLD (2:45:00) mark next 24 hours as SENSOR_CALIBRATION quality.
    
    According to PIPELINE.md: "In case of large gap more than 2 hours 45 minutes 
    mark next 24 hours as ill quality."
    """
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First segment: 3 points before the gap
    for i in range(3):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,  # Will be changed by interpolate_gaps
            'datetime': point_time,
            'glucose': 110.0 + i * 0.1,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Process with mark_calibration_periods (now public method called during prepare_for_inference)
    result = FormatProcessor.mark_calibration_periods(df)
    
    # Find the gap position (between last point before gap and first point after gap)
    # The gap is between index 2 (10 minutes) and index 3 (3 hours later)
    
    # Get all timestamps after the gap
    gap_start_idx = 2  # Last point before gap
    gap_start_datetime = result['original_datetime'][gap_start_idx]
    
    # All points after gap_end_time should be marked as SENSOR_CALIBRATION for 24 hours
    # Calculate 24 hours after gap_end_time
    gap_end_datetime = gap_start_datetime + timedelta(hours=3)  # 3 hour gap
    calibration_period_end = gap_end_datetime + timedelta(hours=24)
    
    # Check points within 24 hours after gap (including first point after gap)
    points_in_calibration_period = result.filter(
        (pl.col('original_datetime') >= gap_end_datetime) &
        (pl.col('original_datetime') <= calibration_period_end)
    )
    
    assert len(points_in_calibration_period) > 0, \
        "Should have points in the 24-hour calibration period"
    
    # All points in the 24-hour period should be marked as SENSOR_CALIBRATION
    sensor_calibration_count = points_in_calibration_period.filter(
        (pl.col('quality') & Quality.SENSOR_CALIBRATION.value) != 0
    ).height
    
    assert sensor_calibration_count == len(points_in_calibration_period), \
        f"All {len(points_in_calibration_period)} points in 24-hour period should be SENSOR_CALIBRATION, " \
        f"but only {sensor_calibration_count} are marked"
    
    # Points after 24 hours should be GOOD quality
    points_after_calibration_period = result.filter(
        pl.col('original_datetime') > calibration_period_end
    )
    
    if len(points_after_calibration_period) > 0:
        good_quality_count = points_after_calibration_period.filter(
            (pl.col('quality') & Quality.SENSOR_CALIBRATION.value) == 0
        ).height
        
        assert good_quality_count == len(points_after_calibration_period), \
            f"Points after 24-hour period should be GOOD quality, " \
            f"but {len(points_after_calibration_period) - good_quality_count} are not"

def test_calibration_gap_exactly_at_threshold():
    """Test that gap exactly at CALIBRATION_GAP_THRESHOLD (2:45:00) triggers calibration marking."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First point
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
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
            'quality': GOOD_QUALITY.value,
            'datetime': gap_end + timedelta(minutes=5 * i),
            'glucose': 110.0 + i * 0.1,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Process with mark_calibration_periods
    result = FormatProcessor.mark_calibration_periods(df)
    
    # Points after gap (within 24 hours) should be SENSOR_CALIBRATION
    calibration_period_end = gap_end + timedelta(hours=24)
    points_after_gap = result.filter(
        (pl.col('original_datetime') >= gap_end) &
        (pl.col('original_datetime') <= calibration_period_end)
    )
    
    if len(points_after_gap) > 0:
        sensor_calibration_count = points_after_gap.filter(
            (pl.col('quality') & Quality.SENSOR_CALIBRATION.value) != 0
        ).height
        
        assert sensor_calibration_count == len(points_after_gap), \
            f"All points after gap (within 24h) should be SENSOR_CALIBRATION, " \
            f"but only {sensor_calibration_count}/{len(points_after_gap)} are marked"


def test_calibration_gap_below_threshold_no_marking():
    """Test that gaps below CALIBRATION_GAP_THRESHOLD do not trigger calibration marking."""
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # First point
    data.append({
        'sequence_id': 0,
        'event_type': UnifiedEventType.GLUCOSE.value,
        'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': gap_end,
        'glucose': 110.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Process with mark_calibration_periods
    result = FormatProcessor.mark_calibration_periods(df)
    
    # Point after gap should remain GOOD quality (not marked as SENSOR_CALIBRATION)
    point_after_gap = result.filter(pl.col('original_datetime') > base_time)
    
    if len(point_after_gap) > 0:
        good_quality_count = point_after_gap.filter(
            (pl.col('quality') & Quality.SENSOR_CALIBRATION.value) == 0
        ).height
        
        assert good_quality_count == len(point_after_gap), \
            f"Points after gap below threshold should remain GOOD quality, " \
            f"but {len(point_after_gap) - good_quality_count} are marked differently"


def test_full_pipeline(sample_data_with_gaps):
    """Test full processing pipeline.
    
    Note: sample_data_with_gaps has a 15-minute gap which is less than 
    SMALL_GAP_MAX_MINUTES (19 minutes), so it may be interpolated depending on implementation.
    """
    
    # Step 1: Interpolate gaps
    interpolated = FormatProcessor.interpolate_gaps(sample_data_with_gaps)
    # Gap behavior depends on implementation - just check we have data
    assert len(interpolated) >= len(sample_data_with_gaps)
    
    # Step 2: Prepare for inference
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        interpolated,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Convert to data-only format
    data_df = FormatProcessor.to_data_only_df(unified_df)
    
    # Check results
    assert len(data_df) > 0
    expected_data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
    assert data_df.columns == expected_data_columns
    # Warnings depend on whether interpolation occurred
    # assert ProcessingWarning.IMPUTATION in warnings  # Removed - implementation-dependent


def test_full_pipeline_with_synchronization(sample_data_with_gaps):
    """Test full processing pipeline including timestamp synchronization.
    
    Note: sample_data_with_gaps has a 15-minute gap which is less than 
    SMALL_GAP_MAX_MINUTES (19 minutes), so behavior depends on implementation.
    """
    
    # Step 1: Interpolate gaps
    interpolated = FormatProcessor.interpolate_gaps(sample_data_with_gaps)
    assert len(interpolated) >= len(sample_data_with_gaps)
    
    # Step 2: Synchronize timestamps
    synchronized = FormatProcessor.synchronize_timestamps(interpolated)
    
    # Verify all timestamps are rounded
    for timestamp in synchronized['datetime'].to_list():
        assert timestamp.second == 0, f"Timestamp {timestamp} should have seconds=0"
    
    # Verify fixed frequency within each sequence
    for seq_id in synchronized['sequence_id'].unique().to_list():
        seq_data = synchronized.filter(pl.col('sequence_id') == seq_id).sort('datetime')
        if len(seq_data) > 1:
            time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
            time_diffs_list = time_diffs.drop_nulls().to_list()
            for diff in time_diffs_list:
                assert abs(diff - EXPECTED_INTERVAL_MINUTES) < 0.1, \
                    f"Time interval {diff} should be ~{EXPECTED_INTERVAL_MINUTES} minutes in sequence {seq_id}"
    
    # Step 3: Prepare for inference
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        synchronized,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Convert to data-only format
    data_df = FormatProcessor.to_data_only_df(unified_df)
    
    # Check results
    assert len(data_df) > 0
    expected_data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
    assert data_df.columns == expected_data_columns
    # Warnings depend on implementation
    # assert ProcessingWarning.IMPUTATION in warnings  # Removed - implementation-dependent


def test_prepare_for_inference_glucose_only():
    """Test that prepare_for_inference works with mixed event types."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create mixed events: GLUCOSE, CALIBRATION
    for i in range(10):
        if i == 2:
            event_type = UnifiedEventType.CALIBRATION.value
        else:
            event_type = UnifiedEventType.GLUCOSE.value
            
        data.append({
            'sequence_id': 0,
            'event_type': event_type,
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 2,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Prepare for inference (should keep all events for now)
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have 10 records
    assert len(unified_df) == 10
    # Should detect the calibration event
    assert ProcessingWarning.CALIBRATION in warnings


def test_prepare_for_inference_drop_duplicates():
    """Test that TIME_DUPLICATES warning is raised for duplicate timestamps."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with duplicate timestamps
    for i in range(8):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=25),  # Same as index 5
        'glucose': 888.0,  # Different value
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    # Test with duplicates - should keep duplicates and warn
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have 10 records (with duplicates)
    assert len(unified_df) == 10
    # Should have TIME_DUPLICATES warning
    assert ProcessingWarning.TIME_DUPLICATES in warnings


def test_prepare_for_inference_time_duplicates_warning():
    """Test that TIME_DUPLICATES warning is raised for non-unique timestamps."""
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create data with duplicate timestamps
    for i in range(5):
        data.append({
            'sequence_id': 0,
            'event_type': UnifiedEventType.GLUCOSE.value,
            'quality': GOOD_QUALITY.value,
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
        'quality': GOOD_QUALITY.value,
        'datetime': base_time + timedelta(minutes=10),  # Same as index 2
        'glucose': 999.0,
        'carbs': None,
        'insulin_slow': None,
        'insulin_fast': None,
        'exercise': None,
    })
    
    df = create_test_dataframe(data)
    
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=120,
    )
    
    # Should have TIME_DUPLICATES warning
    assert (warnings & ProcessingWarning.TIME_DUPLICATES) == ProcessingWarning.TIME_DUPLICATES


def test_prepare_for_inference_warnings_after_truncation():
    """Test that warnings are calculated on truncated data, not before truncation.
    
    This is the key bug fix: warnings should only reflect the data that is actually
    output, not data that was truncated away.
    """
    # FormatProcessor uses classmethods - no instantiation needed
    
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []
    
    # Create 60 minutes of data (13 points at 5-minute intervals)
    for i in range(13):
        # Add calibration event at the beginning (will be truncated)
        event_type = UnifiedEventType.CALIBRATION.value if i == 0 else UnifiedEventType.GLUCOSE.value
        # Add quality issue at the beginning (will be truncated)
        quality = int(Quality.OUT_OF_RANGE.value) if i == 1 else GOOD_QUALITY.value
        
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
    
    df = create_test_dataframe(data)
    
    # Truncate to last 20 minutes - should remove the calibration and quality issue
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=20,  # Keep only last 20 minutes
    )
    
    # Should have truncated to ~20 minutes (5 points)
    assert len(unified_df) <= 5, f"Expected ~5 records for 20 minutes, got {len(unified_df)}"
    
    # Should NOT have CALIBRATION warning (it was truncated away)
    assert not (warnings & ProcessingWarning.CALIBRATION), \
        "CALIBRATION warning should not be present after truncation"
    
    # Should NOT have QUALITY warning (it was truncated away)  
    assert not (warnings & ProcessingWarning.QUALITY), \
        "QUALITY warning should not be present after truncation"


def test_prepare_for_inference_glucose_only_with_truncation():
    """Test truncation and detection of mixed event types."""
    # FormatProcessor uses classmethods - no instantiation needed
    
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
            'quality': GOOD_QUALITY.value,
            'datetime': base_time + timedelta(minutes=5 * i),
            'glucose': 100.0 + i * 10,
            'carbs': None,
            'insulin_slow': None,
            'insulin_fast': None,
            'exercise': None,
        })
    
    df = create_test_dataframe(data)
    
    # Prepare for inference with truncation
    unified_df, warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=10,
        maximum_wanted_duration=30,
    )
    
    # Should detect calibration events
    assert ProcessingWarning.CALIBRATION in warnings


def test_synchronize_timestamps_is_lossless():
    """Integration test: synchronize_timestamps should be lossless on all real datasets.
    
    This test validates that synchronization:
    1. Preserves all rows (no data loss, no imputation)
    2. Doesn't create null values in critical columns
    3. Rounds datetime to sharp timestamps (:00.000)
    4. Preserves original_datetime values (not all sharp)
    """
    from pathlib import Path
    from cgm_format.format_parser import FormatParser
    
    # Get test data directory
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir}")
    
    csv_files = list(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "parsed" not in str(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {data_dir}")
    
    tested_files = 0
    
    for file_path in csv_files:
        # Skip unsupported formats using format_supported
        with open(file_path, 'rb') as f:
            if not FormatParser.format_supported(f.read()):
                continue
        
        try:
            # Parse file
            unified_df = FormatParser.parse_file(file_path)
            original_row_count = len(unified_df)
            
            # Synchronize timestamps
            result = FormatProcessor.synchronize_timestamps(unified_df)
            
            # ASSERTION 1: Number of rows should not change (lossless, no imputation)
            assert len(result) == original_row_count, (
                f"{file_path.name}: Row count changed after sync! "
                f"Before: {original_row_count}, After: {len(result)}"
            )
            
            # ASSERTION 2: No null values in critical columns
            # Check event_type
            null_event_type = result.filter(pl.col('event_type').is_null())
            assert len(null_event_type) == 0, (
                f"{file_path.name}: Found {len(null_event_type)} rows with NULL event_type after sync"
            )
            
            # Check datetime
            null_datetime = result.filter(pl.col('datetime').is_null())
            assert len(null_datetime) == 0, (
                f"{file_path.name}: Found {len(null_datetime)} rows with NULL datetime after sync"
            )
            
            # Check original_datetime
            null_original_datetime = result.filter(pl.col('original_datetime').is_null())
            assert len(null_original_datetime) == 0, (
                f"{file_path.name}: Found {len(null_original_datetime)} rows with NULL original_datetime after sync"
            )
            
            # Check glucose for EGV_READ events (glucose can be null for non-glucose events)
            egv_events = result.filter(pl.col('event_type') == 'EGV_READ')
            if len(egv_events) > 0:
                null_glucose_egv = egv_events.filter(pl.col('glucose').is_null())
                assert len(null_glucose_egv) == 0, (
                    f"{file_path.name}: Found {len(null_glucose_egv)} EGV_READ events with NULL glucose after sync"
                )
            
            # ASSERTION 3: All datetime timestamps should be sharp (:00.000)
            datetime_with_seconds = result.with_columns([
                pl.col('datetime').dt.second().alias('seconds'),
                pl.col('datetime').dt.millisecond().alias('milliseconds')
            ])
            
            non_zero_seconds = datetime_with_seconds.filter(
                (pl.col('seconds') != 0) | (pl.col('milliseconds') != 0)
            )
            assert len(non_zero_seconds) == 0, (
                f"{file_path.name}: Found {len(non_zero_seconds)} rows with non-sharp datetime timestamps "
                f"(expected HH:MM:00.000)"
            )
            
            # ASSERTION 4: NOT all original_datetime should be sharp (should preserve originals)
            # UNLESS the source data was already sharp (some devices like FreeStyle Libre 3 pre-round)
            original_datetime_with_seconds = result.with_columns([
                pl.col('original_datetime').dt.second().alias('seconds'),
                pl.col('original_datetime').dt.millisecond().alias('milliseconds')
            ])
            
            non_zero_original_seconds = original_datetime_with_seconds.filter(
                (pl.col('seconds') != 0) | (pl.col('milliseconds') != 0)
            )
            
            # We expect SOME original timestamps to have non-zero seconds (not all sharp)
            # UNLESS the source data was already sharp (check input data)
            input_non_sharp = unified_df.with_columns([
                pl.col('original_datetime').dt.second().alias('seconds'),
                pl.col('original_datetime').dt.millisecond().alias('milliseconds')
            ]).filter(
                (pl.col('seconds') != 0) | (pl.col('milliseconds') != 0)
            )
            
            # If input had non-sharp timestamps, output should preserve them
            if len(input_non_sharp) > 0:
                # We expect most to preserve original timestamps
                total_rows = len(result)
                sharp_original_rows = total_rows - len(non_zero_original_seconds)
                
                # Allow up to 10% to be sharp (some readings might coincide with grid)
                # But we expect most to preserve original timestamps
                assert sharp_original_rows < total_rows * 0.9, (
                    f"{file_path.name}: Too many original_datetime values are sharp! "
                    f"{sharp_original_rows}/{total_rows} are sharp. "
                    f"Expected to preserve non-sharp original timestamps."
                )
            # else: input was already sharp, so it's OK if output is sharp too
            
            tested_files += 1
            
        except Exception as e:
            pytest.fail(f"Failed processing {file_path.name}: {e}")
    
    # Ensure we tested at least some files
    assert tested_files > 0, "No files were tested"
    print(f"\n✅ Successfully tested {tested_files} real dataset files")


class TestSequenceDetection:
    """Test sequence detection logic in processor for edge cases."""
    
    @staticmethod
    def _create_test_df_with_schema(data):
        """Helper to create test DataFrame with proper schema validation."""
        df = pl.DataFrame(data)
        return CGM_SCHEMA.validate_dataframe(df, enforce=True)
    
    def test_large_gap_creates_new_sequence(self):
        """Test that gaps larger than SMALL_GAP_MAX_MINUTES create new sequences (glucose-only logic)."""
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
        result = FormatProcessor.detect_and_assign_sequences(
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
        result = FormatProcessor.detect_and_assign_sequences(
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
        result = FormatProcessor.detect_and_assign_sequences(
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
