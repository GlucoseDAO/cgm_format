"""Integration tests for idempotency and commutativity of processing operations.

This test suite verifies that:
1. Multiple applications of interpolate_gaps and synchronize_timestamps are idempotent
2. The order of operations doesn't matter (commutative)
3. Strict output data constraints:
   - No gaps > TOLERANCE_INTERVAL_MINUTES within sequences (gaps between sequence_ids are allowed)
   - No NULL values in glucose column for EGV events
   - No values with non-00 seconds in datetime column
"""

from pathlib import Path
from typing import Dict, List
import polars as pl
import pytest

from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor
from cgm_format.interface.cgm_interface import (
    EXPECTED_INTERVAL_MINUTES,
    SMALL_GAP_MAX_MINUTES,
    TOLERANCE_INTERVAL_MINUTES,
)
from cgm_format.formats.unified import CGM_SCHEMA, Quality, UnifiedEventType
from cgm_format.interface.cgm_interface import SupportedCGMFormat


DATA_DIR = Path(__file__).parent.parent / "data" / "input"

# Every chain that calls synchronize_timestamps runs twice: once re-timing
# glucose onto the grid (the 0.12.0 default) and once keeping each reading's
# measured value under its new timestamp (what sync did before). Both have to
# converge, and for different reasons — the first because it recomputes from
# columns nothing writes, the second because it writes no values at all.
RETIME_MODES = [True, False]
RETIME_IDS = ["retimed", "measured"]




def get_supported_test_files() -> List[Path]:
    """Get only supported CSV files from the data directory.

    Uses format_supported to filter out unsupported formats. Discovers every
    ``*.csv`` under data/input/, so the committed synthetic Libre (now with
    scan rows) always participates, and the local mmol/L ``LIBRE_EU`` fixture
    participates when present.
    """
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    # Exclude the parsed subdirectory
    csv_files = [f for f in csv_files if "parsed" not in str(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {DATA_DIR}")
    
    supported_files = []
    
    for file_path in csv_files:
        with open(file_path, 'rb') as f:
            if FormatParser.format_supported(f.read()):
                supported_files.append(file_path)
    
    if not supported_files:
        pytest.skip("No supported CSV files found")
    
    return supported_files


@pytest.fixture(scope="session")
def parsed_files_cache() -> Dict[str, pl.DataFrame]:
    """Parse all test files once and cache them for reuse across tests.
    
    This dramatically speeds up tests by avoiding repeated parsing of the same files.
    Each test gets a clone of the cached dataframe.
    """
    cache = {}
    test_files = get_supported_test_files()

    for file_path in test_files:
        try:
            unified_df = FormatParser.parse_file(file_path)
            cache[file_path.name] = unified_df
        except Exception as e:
            print(f"Warning: Failed to parse {file_path.name}: {e}")
    
    return cache


def check_no_large_gaps(df: pl.DataFrame, expected_interval: int, label: str, max_gap_minutes: float | None = None, glucose_only: bool = True) -> None:
    """Verify that no gaps larger than the specified threshold exist within sequences.
    
    By default uses TOLERANCE_INTERVAL_MINUTES from interface (typically 6 minutes for 5-minute data).
    Gaps are only checked within the same sequence_id - gaps between sequences are ignored.
    
    **IMPORTANT**: By default (glucose_only=True), only checks gaps between glucose readings,
    as interpolation only fills gaps between glucose events, not between heterogeneous events.
    
    Args:
        df: DataFrame with 'datetime' and 'sequence_id' columns
        expected_interval: Expected interval in minutes (e.g., 5) - used for display only
        label: Label for error messages
        max_gap_minutes: Maximum acceptable gap in minutes (default: TOLERANCE_INTERVAL_MINUTES)
        glucose_only: If True (default), only check gaps between GLUCOSE events (ignore other event types)
    """
    if len(df) == 0:
        return
    
    # Use custom threshold or default to TOLERANCE_INTERVAL_MINUTES
    acceptable_gap = max_gap_minutes if max_gap_minutes is not None else TOLERANCE_INTERVAL_MINUTES
    
    # If glucose_only, split glucose events using FormatProcessor.split_glucose_events
    # This ensures we only check gaps between actual glucose readings
    if glucose_only:
        glucose_df, _ = FormatProcessor.split_glucose_events(df)
        check_df = glucose_df
    else:
        check_df = df
    
    # Check each sequence separately
    sequence_ids = check_df['sequence_id'].unique().sort()
    
    for seq_id in sequence_ids:
        seq_df = check_df.filter(pl.col('sequence_id') == seq_id).sort('datetime')
        
        if len(seq_df) <= 1:
            continue
        
        # Calculate time differences
        time_diffs = seq_df.select([
            (pl.col('datetime').diff().dt.total_seconds() / 60.0).alias('gap_minutes')
        ])
        
        # Skip the first row (which will be null from diff)
        gaps = time_diffs['gap_minutes'][1:]
        
        if len(gaps) > 0:
            max_gap = gaps.max()
            if max_gap > acceptable_gap:
                # Find the problematic rows
                gap_df = seq_df.with_columns([
                    (pl.col('datetime').diff().dt.total_seconds() / 60.0).alias('gap_minutes')
                ])
                large_gaps = gap_df.filter(pl.col('gap_minutes') > acceptable_gap)
                
                raise AssertionError(
                    f"{label}: Found gap of {max_gap:.1f} minutes in sequence {seq_id} "
                    f"(acceptable limit: {acceptable_gap:.1f} minutes)\n"
                    f"Problematic timestamps:\n{large_gaps.select(['datetime', 'gap_minutes'])}"
                )


def check_no_null_glucose_egv(df: pl.DataFrame, label: str) -> None:
    """Verify that no EGV events have NULL glucose values.

    The event type is read from the schema rather than spelled here. Until
    2026-08-17 this filtered the literal ``'EGV'``, which no unified frame has
    ever contained — the code is ``'EGV_READ'`` — so the filter matched nothing,
    the early return below always fired, and the null check never ran on any of
    its ten call sites. The column it then read, ``glucose_value_mg_dl``, had
    likewise been renamed to ``glucose`` and would have raised on contact.

    Args:
        df: DataFrame with 'event_type' and 'glucose' columns
        label: Label for error messages
    """
    if len(df) == 0:
        return

    # Filter for EGV events
    egv_df = df.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value)

    if len(egv_df) == 0:
        return

    # Check for NULL glucose values
    null_glucose = egv_df.filter(pl.col('glucose').is_null())

    if len(null_glucose) > 0:
        raise AssertionError(
            f"{label}: Found {len(null_glucose)} EGV events with NULL glucose values\n"
            f"Sample rows:\n{null_glucose.head(10).select(['datetime', 'event_type', 'glucose', 'sequence_id'])}"
        )


def check_seconds_are_zero(df: pl.DataFrame, label: str) -> None:
    """Verify that all datetime values have 00 seconds.
    
    Args:
        df: DataFrame with 'datetime' column
        label: Label for error messages
    """
    if len(df) == 0:
        return
    
    # Extract seconds from datetime
    df_with_seconds = df.with_columns([
        pl.col('datetime').dt.second().alias('seconds')
    ])
    
    # Find rows with non-zero seconds
    non_zero_seconds = df_with_seconds.filter(pl.col('seconds') != 0)
    
    if len(non_zero_seconds) > 0:
        raise AssertionError(
            f"{label}: Found {len(non_zero_seconds)} rows with non-zero seconds in datetime\n"
            f"Sample rows:\n{non_zero_seconds.head(10).select(['datetime', 'seconds', 'sequence_id'])}"
        )


def assert_dataframes_equal(df1: pl.DataFrame, df2: pl.DataFrame, label: str) -> None:
    """Assert two dataframes are identical.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        label: Label for error messages
    """
    # Check shape
    assert df1.shape == df2.shape, (
        f"{label}: Shape mismatch - {df1.shape} vs {df2.shape}"
    )
    
    # Check column names
    assert df1.columns == df2.columns, (
        f"{label}: Column mismatch - {df1.columns} vs {df2.columns}"
    )
    
    # Sort both with stable keys from schema definition
    stable_sort_keys = CGM_SCHEMA.get_stable_sort_keys()
    df1_sorted = df1.sort(stable_sort_keys)
    df2_sorted = df2.sort(stable_sort_keys)
    
    # Check if they're equal
    try:
        assert df1_sorted.equals(df2_sorted), f"{label}: DataFrames not equal"
    except AssertionError:
        # If not equal, find differences
        for col in df1.columns:
            col1 = df1_sorted[col]
            col2 = df2_sorted[col]
            if not col1.equals(col2):
                diff_mask = col1 != col2
                diff_rows = df1_sorted.filter(diff_mask)
                print(f"\n{label}: Differences in column '{col}':")
                print(f"DF1:\n{diff_rows}")
                diff_rows2 = df2_sorted.filter(diff_mask)
                print(f"DF2:\n{diff_rows2}")
        raise


class TestProcessingIdempotency:
    """Test idempotency and commutativity of processing operations."""
    
    @pytest.mark.parametrize("retime", RETIME_MODES, ids=RETIME_IDS)
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_triple_sync_idempotency(self, file_path: Path, retime: bool, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test: synchronize_timestamps → synchronize_timestamps → synchronize_timestamps

        Applying sync three times should produce the same result as the first application.

        Run with re-timing both on and off. On is the interesting case: it is
        the only stage that rewrites `glucose`, so a version that read the value
        it had just written would compound here and nowhere else.
        """

        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")

        print(f"\n{'='*70}")
        print(f"Testing triple sync (retime_glucose={retime}): {file_path.name}")
        print(f"{'='*70}\n")

        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()

        # Apply sync three times
        df1 = FormatProcessor.synchronize_timestamps(unified_df, retime_glucose=retime)
        print(f"After 1st sync: {len(df1)} rows")

        df2 = FormatProcessor.synchronize_timestamps(df1, retime_glucose=retime)
        print(f"After 2nd sync: {len(df2)} rows")

        df3 = FormatProcessor.synchronize_timestamps(df2, retime_glucose=retime)
        print(f"After 3rd sync: {len(df3)} rows")

        # Assert 1st and 3rd are identical
        assert_dataframes_equal(df1, df3, "Triple sync idempotency (1st vs 3rd)")
        
        # Validate final output: sync doesn't fill gaps, only aligns timestamps
        # So check: seconds are zero, no null glucose
        # Don't check: gaps (sync doesn't fill them)
        check_no_null_glucose_egv(df3, "Final output")
        check_seconds_are_zero(df3, "Final output")
        
        print("✅ PASSED: Triple sync is idempotent")
    
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_triple_interpolate_idempotency(self, file_path: Path, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test: interpolate_gaps → interpolate_gaps → interpolate_gaps
        
        Applying interpolate three times should produce the same result as the first application.
        """
        
        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")
        
        print(f"\n{'='*70}")
        print(f"Testing triple interpolate: {file_path.name}")
        print(f"{'='*70}\n")
        
        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()
        
        # Apply interpolate three times
        df1 = FormatProcessor.interpolate_gaps(unified_df)
        print(f"After 1st interpolate: {len(df1)} rows")
        
        df2 = FormatProcessor.interpolate_gaps(df1)
        print(f"After 2nd interpolate: {len(df2)} rows")
        
        df3 = FormatProcessor.interpolate_gaps(df2)
        print(f"After 3rd interpolate: {len(df3)} rows")
        
        # Assert 1st and 3rd are identical
        assert_dataframes_equal(df1, df3, "Triple interpolate idempotency (1st vs 3rd)")
        
        # Validate final output: interpolate fills gaps up to small_gap_max_minutes
        # between GLUCOSE events only. Check glucose-to-glucose gaps, not all events.
        # Check: no glucose-to-glucose gaps ≥ 10 minutes (2 * expected_interval)
        # Don't check: seconds are zero (interpolate doesn't sync timestamps)
        # Don't check: TOLERANCE_INTERVAL_MINUTES (only applies after sync)
        check_no_large_gaps(df3, EXPECTED_INTERVAL_MINUTES, "Final output", max_gap_minutes=10.0, glucose_only=True)
        check_no_null_glucose_egv(df3, "Final output")
        
        print("✅ PASSED: Triple interpolate is idempotent")
    
    @pytest.mark.parametrize("retime", RETIME_MODES, ids=RETIME_IDS)
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_interpolate_sync_interpolate_idempotency(self, file_path: Path, retime: bool, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test: interpolate_gaps → synchronize_timestamps → interpolate_gaps

        The second interpolate_gaps should not change anything.
        """

        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")

        print(f"\n{'='*70}")
        print(f"Testing interpolate→sync→interpolate (retime_glucose={retime}): {file_path.name}")
        print(f"{'='*70}\n")

        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()

        # Step 1: interpolate_gaps
        df1 = FormatProcessor.interpolate_gaps(unified_df)
        print(f"After 1st interpolate_gaps: {len(df1)} rows")

        # Step 2: synchronize_timestamps
        df2 = FormatProcessor.synchronize_timestamps(df1, retime_glucose=retime)
        print(f"After synchronize_timestamps: {len(df2)} rows")

        # Step 3: interpolate_gaps again (should be idempotent)
        df3 = FormatProcessor.interpolate_gaps(df2)
        print(f"After 2nd interpolate_gaps: {len(df3)} rows")
        
        # Assert df2 and df3 are identical
        assert_dataframes_equal(df2, df3, "interpolate→sync→interpolate")
        
        # Validate final output meets all constraints
        check_no_large_gaps(df3, EXPECTED_INTERVAL_MINUTES, "Final output")
        check_no_null_glucose_egv(df3, "Final output")
        check_seconds_are_zero(df3, "Final output")
        
        print("✅ PASSED: Second interpolate_gaps was idempotent")
    
    @pytest.mark.parametrize("retime", RETIME_MODES, ids=RETIME_IDS)
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_sync_interpolate_sync_idempotency(self, file_path: Path, retime: bool, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test: synchronize_timestamps → interpolate_gaps → synchronize_timestamps

        The second synchronize_timestamps should not change anything.

        This is the chain that forces the two stages to share one interpolant.
        With re-timing on, the interpolate in the middle runs over neighbours
        whose `glucose` has already been rewritten; if it derived its created
        points from those values instead of from the measured anchors, the sync
        that follows would recompute them and this assertion would fail.
        """

        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")

        print(f"\n{'='*70}")
        print(f"Testing sync→interpolate→sync (retime_glucose={retime}): {file_path.name}")
        print(f"{'='*70}\n")

        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()

        # Step 1: synchronize_timestamps
        df1 = FormatProcessor.synchronize_timestamps(unified_df, retime_glucose=retime)
        print(f"After 1st synchronize_timestamps: {len(df1)} rows")

        # Step 2: interpolate_gaps
        df2 = FormatProcessor.interpolate_gaps(df1)
        print(f"After interpolate_gaps: {len(df2)} rows")

        # Step 3: synchronize_timestamps again (should be idempotent)
        df3 = FormatProcessor.synchronize_timestamps(df2, retime_glucose=retime)
        print(f"After 2nd synchronize_timestamps: {len(df3)} rows")

        # Assert df2 and df3 are identical
        assert_dataframes_equal(df2, df3, "sync→interpolate→sync")
        
        # Validate final output meets all constraints
        check_no_large_gaps(df3, EXPECTED_INTERVAL_MINUTES, "Final output")
        check_no_null_glucose_egv(df3, "Final output")
        check_seconds_are_zero(df3, "Final output")
        
        print("✅ PASSED: Second synchronize_timestamps was idempotent")
    
    @pytest.mark.parametrize("retime", RETIME_MODES, ids=RETIME_IDS)
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_processing_commutativity(self, file_path: Path, retime: bool, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test that both processing chains produce identical results.
        
        Chain A: interpolate_gaps → synchronize_timestamps → interpolate_gaps
        Chain B: synchronize_timestamps → interpolate_gaps → synchronize_timestamps
        
        Both should produce the same final result (commutativity).
        """
        
        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")
        
        print(f"\n{'='*70}")
        print(f"Testing commutativity: {file_path.name}")
        print(f"{'='*70}\n")
        
        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()
        
        # Chain A: interpolate → sync → interpolate
        df_a1 = FormatProcessor.interpolate_gaps(unified_df)
        df_a2 = FormatProcessor.synchronize_timestamps(df_a1, retime_glucose=retime)
        df_a3 = FormatProcessor.interpolate_gaps(df_a2)
        print(f"Chain A final: {len(df_a3)} rows")
        check_no_large_gaps(df_a3, EXPECTED_INTERVAL_MINUTES, "Chain A final")
        check_no_null_glucose_egv(df_a3, "Chain A final")
        check_seconds_are_zero(df_a3, "Chain A final")
        
        # Chain B: sync → interpolate → sync
        df_b1 = FormatProcessor.synchronize_timestamps(unified_df, retime_glucose=retime)
        df_b2 = FormatProcessor.interpolate_gaps(df_b1)
        df_b3 = FormatProcessor.synchronize_timestamps(df_b2, retime_glucose=retime)
        print(f"Chain B final: {len(df_b3)} rows")
        check_no_large_gaps(df_b3, EXPECTED_INTERVAL_MINUTES, "Chain B final")
        check_no_null_glucose_egv(df_b3, "Chain B final")
        check_seconds_are_zero(df_b3, "Chain B final")
        
        # Assert both chains produce identical results
        assert_dataframes_equal(df_a3, df_b3, "Commutativity check")
        print("✅ PASSED: Both processing chains produced identical results")
    
    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_triple_sequence_detection_idempotency(self, file_path: Path, parsed_files_cache: Dict[str, pl.DataFrame]) -> None:
        """Test: detect_and_assign_sequences → detect_and_assign_sequences → detect_and_assign_sequences
        
        Applying sequence detection three times should produce the same result as the first application.
        All sequences should have sequence_id >= 1 (no unassigned sequences with id 0).
        """
        
        # Get cached parsed data
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")
        
        print(f"\n{'='*70}")
        print(f"Testing triple sequence detection: {file_path.name}")
        print(f"{'='*70}\n")
        
        # Clone the cached dataframe for this test
        unified_df = parsed_files_cache[file_path.name].clone()
        
        # Apply detect_and_assign_sequences three times
        result1 = FormatProcessor.detect_and_assign_sequences(unified_df)
        print(f"After 1st detect_and_assign_sequences: {len(result1)} rows")
        
        result2 = FormatProcessor.detect_and_assign_sequences(result1)
        print(f"After 2nd detect_and_assign_sequences: {len(result2)} rows")
        
        result3 = FormatProcessor.detect_and_assign_sequences(result2)
        print(f"After 3rd detect_and_assign_sequences: {len(result3)} rows")
        
        # Check that ALL GLUCOSE events have sequence_id >= 1 (assigned)
        # Non-glucose events should be assigned to nearest glucose sequence if within gap distance
        # If no glucose sequence is nearby (> gap distance), they remain unassigned (sequence_id = 0)
        from cgm_format.formats.unified import UnifiedEventType
        
        glucose_unassigned_1 = result1.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) & 
            (pl.col('sequence_id') == 0)
        ).height
        glucose_unassigned_2 = result2.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) & 
            (pl.col('sequence_id') == 0)
        ).height
        glucose_unassigned_3 = result3.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) & 
            (pl.col('sequence_id') == 0)
        ).height
        
        total_unassigned_1 = result1.filter(pl.col('sequence_id') == 0).height
        total_unassigned_2 = result2.filter(pl.col('sequence_id') == 0).height
        total_unassigned_3 = result3.filter(pl.col('sequence_id') == 0).height
        
        print(f"Run 1: {glucose_unassigned_1} unassigned glucose events, {total_unassigned_1} total unassigned")
        print(f"Run 2: {glucose_unassigned_2} unassigned glucose events, {total_unassigned_2} total unassigned")
        print(f"Run 3: {glucose_unassigned_3} unassigned glucose events, {total_unassigned_3} total unassigned")
        
        # All glucose events MUST be assigned
        assert glucose_unassigned_1 == 0, f"Result 1 has {glucose_unassigned_1} unassigned glucose events (should be 0)"
        assert glucose_unassigned_2 == 0, f"Result 2 has {glucose_unassigned_2} unassigned glucose events (should be 0)"
        assert glucose_unassigned_3 == 0, f"Result 3 has {glucose_unassigned_3} unassigned glucose events (should be 0)"
        
        # Unassigned events should be the same across runs (idempotency)
        assert total_unassigned_1 == total_unassigned_2 == total_unassigned_3, \
            f"Unassigned event count differs: run1={total_unassigned_1}, run2={total_unassigned_2}, run3={total_unassigned_3}"
        
        # Assert all three runs produce identical results
        assert_dataframes_equal(result1, result2, "Triple sequence detection idempotency (1st vs 2nd)")
        assert_dataframes_equal(result2, result3, "Triple sequence detection idempotency (2nd vs 3rd)")
        assert_dataframes_equal(result1, result3, "Triple sequence detection idempotency (1st vs 3rd)")
        
        # Display sequence statistics
        unique_sequences = result1['sequence_id'].unique().sort()
        assigned_sequences = result1.filter(pl.col('sequence_id') > 0)['sequence_id'].unique().sort()
        
        print(f"Total unique sequence IDs: {len(unique_sequences)} (including 0 for unassigned)")
        print(f"Total assigned sequences: {len(assigned_sequences)}")
        
        if len(assigned_sequences) > 0:
            print(f"Assigned sequence IDs: {assigned_sequences.to_list()[:10]}{'...' if len(assigned_sequences) > 10 else ''}")
            print(f"Assigned sequence ID range: {assigned_sequences.min()} to {assigned_sequences.max()}")
            
            # Verify assigned sequences start from 1
            assert assigned_sequences.min() >= 1, "Assigned sequence IDs should start at 1 or higher"
        
        # Verify no data loss
        assert len(result1) == len(unified_df), \
            f"Row count changed from {len(unified_df)} to {len(result1)}"
        
        print("✅ PASSED: Triple sequence detection is idempotent")


class TestGridRetiming:
    """The value anchor, and the number the anchor is supposed to produce.

    The chains above prove re-timing converges. These prove it converges on the
    *right* value and for the right reason — a stage that simply refused to
    recompute after the first pass would satisfy every chain above and be wrong.
    """

    COMPARISON_DIR = Path(__file__).parent.parent / "data" / "comparison"

    @pytest.mark.parametrize("file_path", get_supported_test_files(), ids=lambda p: p.name)
    def test_retiming_never_consumes_its_own_output(
        self, file_path: Path, parsed_files_cache: Dict[str, pl.DataFrame]
    ) -> None:
        """`original_glucose` still holds the device's reading after three passes.

        This is the assertion the whole design exists to satisfy. Re-timing
        rewrites `glucose`, so if it read `glucose` back on the next pass each
        run would interpolate from the previous run's approximation and drift —
        silently, because the timestamps stop moving after pass one. Comparing
        the anchors against a *fresh parse* is what catches that: the chains
        cannot, since a frame drifting by a fixed rule still equals itself.
        """
        if file_path.name not in parsed_files_cache:
            pytest.skip(f"File not in cache: {file_path.name}")

        parsed = parsed_files_cache[file_path.name].clone()

        df = FormatProcessor.synchronize_timestamps(parsed, retime_glucose=True)
        df = FormatProcessor.synchronize_timestamps(df, retime_glucose=True)
        df = FormatProcessor.synchronize_timestamps(df, retime_glucose=True)

        # Sync adds no rows, so the anchors are the parsed readings, one for one.
        measured = parsed.sort('original_datetime')['original_glucose']
        after_three_passes = df.sort('original_datetime')['original_glucose']

        assert after_three_passes.equals(measured), (
            "original_glucose changed under re-timing — the anchor is being "
            "written by the stage that reads it, so every pass compounds"
        )

        glucose_rows = df.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
        if glucose_rows.height > 0:
            unflagged = glucose_rows.filter(
                (pl.col('quality') & Quality.GRID_RETIMED.value) == 0
            )
            assert unflagged.height == 0, (
                f"{unflagged.height} re-timed glucose rows are not flagged "
                f"GRID_RETIMED, so a reader cannot tell derived from measured"
            )

    def test_retimed_value_matches_the_raw_readings_that_bracket_it(self) -> None:
        """A re-timed value is the raw series evaluated at the grid instant.

        Ground truth comes from the Dexcom export's own rows, read here as text,
        never from the parser under test. Every grid point in the window is
        checked, not one worked example — a single point can agree by accident,
        a whole day cannot.
        """
        raw_csv = self.COMPARISON_DIR / "livia_test.csv"
        if not raw_csv.exists():
            pytest.skip(f"Fixture not found: {raw_csv}")

        # Ground truth: the raw EGV rows, straight out of the vendor CSV.
        raw = pl.read_csv(raw_csv, infer_schema_length=0, encoding="utf8-lossy")
        timestamp_col = next(c for c in raw.columns if "Timestamp" in c)
        event_col = next(c for c in raw.columns if c.strip() == "Event Type")
        glucose_col = next(c for c in raw.columns if "Glucose Value" in c)

        readings = (
            raw.filter(pl.col(event_col) == "EGV")
            .select(
                pl.col(timestamp_col)
                .str.strptime(pl.Datetime("ms"), "%Y-%m-%dT%H:%M:%S", strict=False)
                .alias("t"),
                # Dexcom's out-of-range markers, replaced the way the parser does.
                pl.when(pl.col(glucose_col) == "High").then(pl.lit(401.0))
                .when(pl.col(glucose_col) == "Low").then(pl.lit(39.0))
                .otherwise(pl.col(glucose_col).cast(pl.Float64, strict=False))
                .alias("g"),
            )
            .drop_nulls(["t", "g"])
            .sort("t")
            .unique(subset=["t"], keep="first")
            .sort("t")
        )

        processed = FormatProcessor.synchronize_timestamps(
            FormatProcessor.interpolate_gaps(
                FormatProcessor.detect_and_assign_sequences(
                    FormatParser.parse_file(raw_csv)
                )
            ),
            retime_glucose=True,
        )

        # One long sequence dominates this export; check its grid points.
        newest_sequence = (
            processed.group_by('sequence_id')
            .agg(pl.col('datetime').max().alias('_last'))
            .sort('_last')['sequence_id'][-1]
        )
        window = processed.filter(
            (pl.col('sequence_id') == newest_sequence)
            & (pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
        ).sort('datetime')

        # Ground truth has to respect the sequence boundary the processor does.
        # A sequence exists precisely because the gap on either side of it is
        # too long to interpolate across, so the raw reading beyond that gap is
        # not an anchor for these grid points — the processor clamps to the
        # sequence's own first and last readings instead, and a check that
        # interpolated across the boundary would be marking correct behaviour
        # wrong.
        measured_span = window.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) == 0
        )['original_datetime']
        in_sequence = readings.filter(
            pl.col('t').is_between(measured_span.min(), measured_span.max())
        )

        joined = (
            window.select('datetime', 'glucose')
            # Grid points outside the sequence's own readings are clamped by
            # design; they are checked separately below.
            .filter(pl.col('datetime').is_between(in_sequence['t'].min(),
                                                  in_sequence['t'].max()))
            .join_asof(in_sequence.rename({'t': 't0', 'g': 'g0'}),
                       left_on='datetime', right_on='t0', strategy='backward')
            .join_asof(in_sequence.rename({'t': 't1', 'g': 'g1'}),
                       left_on='datetime', right_on='t1', strategy='forward')
            .drop_nulls(['g0', 'g1'])
            .filter(pl.col('t1') > pl.col('t0'))
        )
        assert joined.height > 100, "window too small to be a meaningful check"

        alpha = (
            (pl.col('datetime') - pl.col('t0')).dt.total_nanoseconds()
            / (pl.col('t1') - pl.col('t0')).dt.total_nanoseconds()
        )
        expected = pl.col('g0') + alpha * (pl.col('g1') - pl.col('g0'))
        deviation = joined.select((pl.col('glucose') - expected).abs().alias('d'))['d']

        assert deviation.max() < 1e-6, (
            f"re-timed values deviate from the raw readings that bracket them "
            f"by up to {deviation.max()} mg/dL over {joined.height} grid points"
        )

        # Outside the sequence's own readings there is nothing to interpolate
        # between, so the value must be the nearest reading itself — clamped,
        # never extrapolated past it. Extrapolating is what makes the reference
        # implementation emit values tens of mg/dL from anything measured.
        outside = window.filter(
            ~pl.col('datetime').is_between(in_sequence['t'].min(),
                                           in_sequence['t'].max())
        )
        if outside.height > 0:
            lo, hi = in_sequence['g'].min(), in_sequence['g'].max()
            assert outside['glucose'].min() >= lo and outside['glucose'].max() <= hi, (
                "a grid point outside the sequence's readings was extrapolated "
                "beyond the range of anything the sensor reported"
            )

        # The gap this closes: keeping the measured value under the new
        # timestamp is a materially different number, so the check above is
        # not trivially satisfied by doing nothing.
        untouched = FormatProcessor.synchronize_timestamps(
            FormatProcessor.interpolate_gaps(
                FormatProcessor.detect_and_assign_sequences(
                    FormatParser.parse_file(raw_csv)
                )
            ),
            retime_glucose=False,
        ).filter(
            (pl.col('sequence_id') == newest_sequence)
            & (pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
        ).sort('datetime')

        assert not untouched['glucose'].equals(window['glucose']), (
            "re-timing produced the same values as leaving them alone, so this "
            "test would pass against a no-op implementation"
        )


class TestLibreCoverageInGlob:
    """The glob must see scan-row Libre (CI) and LIBRE_EU when the local fixture exists."""

    def test_synthetic_libre_is_discovered(self) -> None:
        names = {p.name for p in get_supported_test_files()}
        assert "FreeStyle_Libre_3_synthetic.csv" in names

    def test_libre_eu_fixture_discovered_when_present(self) -> None:
        jongrove = DATA_DIR / "JonGrove_glucose_13-8-2026.csv"
        if not jongrove.exists():
            pytest.skip(f"Fixture not found: {jongrove}")
        files = get_supported_test_files()
        assert jongrove in files
        raw = jongrove.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt == SupportedCGMFormat.LIBRE_EU


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

