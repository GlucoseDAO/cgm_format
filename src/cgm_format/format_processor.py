"""CGM Data Processor Implementation.

Implements vendor-agnostic processing operations on UnifiedFormat data (Stages 4-5).
Adapted from glucose_ml_preprocessor.py for single-user unified format processing.
"""

import polars as pl
from typing import Dict, Any, List, Tuple, ClassVar, Optional
from datetime import timedelta, datetime
from cgm_format.interface.cgm_interface import (
    CGMProcessor,
    UnifiedFormat,
    InferenceResult,
    ProcessingWarning,
    ZeroValidInputError,
    MalformedDataError,
    ValidationMethod,
    EXPECTED_INTERVAL_MINUTES,
    SMALL_GAP_MAX_MINUTES,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
    CALIBRATION_PERIOD_HOURS,
)
from cgm_format.formats.unified import UnifiedEventType, Quality, CGM_SCHEMA


class FormatProcessor(CGMProcessor):
    """Implementation of CGMProcessor for unified format data processing.
    
    This processor handles single-user unified format data and provides:
    - Gap detection and sequence creation
    - Gap interpolation with imputation tracking
    - Inference preparation with duration checks and truncation
    - Warning collection in prepare_for_inference
    
    All methods are classmethods - no need to instantiate.
    Configuration constants can be overridden via optional method parameters.
    """
    
    # Configuration constants as ClassVars
    expected_interval_minutes: ClassVar[int] = EXPECTED_INTERVAL_MINUTES
    small_gap_max_minutes: ClassVar[int] = SMALL_GAP_MAX_MINUTES
    minimum_duration_minutes: ClassVar[int] = MINIMUM_DURATION_MINUTES
    maximum_wanted_duration_minutes: ClassVar[int] = MAXIMUM_WANTED_DURATION_MINUTES
    calibration_gap_threshold: ClassVar[int] = CALIBRATION_GAP_THRESHOLD
    calibration_period_hours: ClassVar[int] = CALIBRATION_PERIOD_HOURS
    snap_to_grid: ClassVar[bool] = True
    validation_mode_default: ClassVar[ValidationMethod] = ValidationMethod.INPUT


    @classmethod
    def mark_time_duplicates(
        cls,
        df: UnifiedFormat,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Mark events with duplicate timestamps (keeping first occurrence).
        
        Uses keepfirst logic: the first event at a timestamp is kept clean,
        subsequent events with the same timestamp are marked with TIME_DUPLICATE flag.
        
        Args:
            df: DataFrame in unified format (must have 'datetime' and 'quality' columns)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            DataFrame with TIME_DUPLICATE flag added to quality column for duplicate timestamps
        """
        if len(df) == 0:
            return df

        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Validate input if validation mode includes INPUT
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
        # For each datetime, mark which rows are duplicates (all but the first)
        # is_duplicated() returns True for ALL occurrences including the first
        # We use is_first_distinct() to find the first occurrence
        df_marked = df.with_columns([
            pl.when(
                pl.col("datetime").is_duplicated() & 
                ~pl.col("datetime").is_first_distinct()
            )
            .then(pl.col("quality") | Quality.TIME_DUPLICATE.value)
            .otherwise(pl.col("quality"))
            .alias("quality")
        ])
        
        # Validate output if validation mode includes OUTPUT
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df_marked, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df_marked
        
    @classmethod
    def synchronize_timestamps(
        cls,
        dataframe: UnifiedFormat,
        expected_interval_minutes: Optional[int] = None,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Align timestamps to minute boundaries and create fixed-frequency data.
        
        This method should be called after interpolate_gaps() when sequences are already
        created and small gaps are filled. It performs:
        1. Rounds timestamps to nearest minute using built-in rounding
        2. Creates fixed-frequency timestamps with expected_interval_minutes
        3. Linearly interpolates glucose values (time-weighted)
        4. Shifts discrete events (carbs, insulin, exercise) to nearest timestamps
        
        Args:
            dataframe: DataFrame in unified format (should already have sequences created)
            expected_interval_minutes: Expected interval in minutes (defaults to cls.expected_interval_minutes)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            DataFrame with synchronized timestamps at fixed intervals
            
        Raises:
            ZeroValidInputError: If dataframe is empty or has no data
            ValueError: If data has gaps larger than small_gap_max_minutes (not preprocessed)
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("Cannot synchronize timestamps on empty dataframe")

        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes
        if validation_mode is None:
            validation_mode = cls.validation_mode_default
        
        # Ensure sequences are assigned (auto-detect if missing)
        if not cls.has_sequences(dataframe):
            dataframe = cls.detect_and_assign_sequences(
                dataframe,
                expected_interval_minutes=expected_interval_minutes,
                validation_mode=validation_mode
            )
        
        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # Process each sequence separately
        unique_sequences = dataframe['sequence_id'].unique().to_list()
        synchronized_sequences = []
        
        for seq_id in unique_sequences:
            # Sort by original_datetime for idempotent processing
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort(['sequence_id', 'original_datetime', 'quality'])
            
            if len(seq_data) < 2:
                # Keep single-point sequences as-is, just round the timestamp using Polars rounding
                seq_data = seq_data.with_columns([
                    pl.col('datetime').dt.round('1m').alias('datetime')
                ])
                synchronized_sequences.append(seq_data)
                continue
            
            # Synchronize this sequence
            synced_seq = cls._synchronize_sequence(seq_data, seq_id, expected_interval_minutes)
            synchronized_sequences.append(synced_seq)
        
        # Combine all sequences with stable sorting from schema definition
        result_df = pl.concat(synchronized_sequences).sort(CGM_SCHEMA.get_stable_sort_keys())
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
    
    @classmethod
    def get_sequence_grid_start(cls, seq_data: UnifiedFormat, expected_interval_minutes: Optional[int] = None) -> datetime:
        """Determine the grid start time for a sequence.
        
        The grid start is based on the first original_datetime in the sequence,
        rounded to the nearest minute. This ensures both synchronize_timestamps
        and interpolate_gaps use the same grid alignment.
        
        Uses original_datetime (not datetime) to preserve the original grid alignment
        even after synchronization or other timestamp modifications.
        
        Args:
            seq_data: Sequence data
            expected_interval_minutes: Expected interval in minutes (defaults to cls.expected_interval_minutes)
            
        Returns:
            Grid start timestamp (rounded to nearest minute)
        """
        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes

        first_timestamp = seq_data['original_datetime'].min()
        
        # Round to nearest minute (same logic as synchronize_timestamps)
        if first_timestamp.second >= 30:
            grid_start = first_timestamp.replace(second=0, microsecond=0) + timedelta(minutes=1)
        else:
            grid_start = first_timestamp.replace(second=0, microsecond=0)
        
        return grid_start
    
    @classmethod
    def calculate_grid_point(
        cls,
        timestamp: datetime, 
        grid_start: datetime,
        expected_interval_minutes: Optional[int] = None,
        round_direction: str = 'nearest'
    ) -> datetime:
        """Calculate the nearest grid point for a given timestamp.
        
        Args:
            timestamp: Timestamp to align to grid
            grid_start: Start of the grid
            expected_interval_minutes: Expected interval in minutes (defaults to cls.expected_interval_minutes)
            round_direction: 'nearest', 'up', or 'down'
            
        Returns:
            Timestamp aligned to grid
        """
        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes

        elapsed_seconds = (timestamp - grid_start).total_seconds()
        interval_seconds = expected_interval_minutes * 60
        
        if round_direction == 'down':
            intervals = int(elapsed_seconds // interval_seconds)
        elif round_direction == 'up':
            intervals = int((elapsed_seconds + interval_seconds - 1) // interval_seconds)
        else:  # nearest
            intervals = int((elapsed_seconds + interval_seconds / 2) // interval_seconds)
        
        return grid_start + timedelta(minutes=intervals * expected_interval_minutes)
    
    @classmethod
    def _interpolate_glucose_value(
        cls,
        target_time: datetime,
        prev_time: datetime,
        next_time: datetime,
        prev_glucose: float,
        next_glucose: float
    ) -> float:
        """Calculate interpolated glucose value using time-weighted linear interpolation.
        
        Uses the time positions of the boundary points to calculate the interpolation weight.
        When snap_to_grid=True, pass grid-aligned timestamps for prev_time and next_time
        to ensure idempotency with synchronize_timestamps.
        
        Formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        where x = target_time, x0 = prev_time, x1 = next_time
        
        Args:
            target_time: Time point to interpolate for (grid-aligned or original)
            prev_time: Time of previous reading (grid-aligned if snap_to_grid, else original_datetime)
            next_time: Time of next reading (grid-aligned if snap_to_grid, else original_datetime)
            prev_glucose: Glucose value at prev_time
            next_glucose: Glucose value at next_time
            
        Returns:
            Interpolated glucose value
        """
        total_seconds = (next_time - prev_time).total_seconds()
        if total_seconds <= 0:
            return prev_glucose
        
        elapsed_seconds = (target_time - prev_time).total_seconds()
        alpha = elapsed_seconds / total_seconds
        
        return prev_glucose + alpha * (next_glucose - prev_glucose)
    
    @classmethod
    def _synchronize_sequence(
        cls,
        seq_data: pl.DataFrame, 
        seq_id: int,
        expected_interval_minutes: int
    ) -> pl.DataFrame:
        """Synchronize timestamps for a single sequence to fixed frequency.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            expected_interval_minutes: Expected interval in minutes
            
        Returns:
            Sequence with synchronized timestamps at fixed intervals
        """
        # Get grid start using common logic
        grid_start = cls.get_sequence_grid_start(seq_data, expected_interval_minutes)
        
        # For idempotency: determine grid extent based ONLY on non-interpolated data
        # Filter out interpolated points (those with IMPUTATION flag)
        non_interpolated = seq_data.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) == 0
        )

        # Use max datetime from non-interpolated data to determine grid extent
        # This ensures the grid is stable even after interpolation
        last_timestamp = non_interpolated['datetime'].max()
        
        # Calculate duration and number of intervals
        total_duration = (last_timestamp - grid_start).total_seconds()
        
        if total_duration < 0:
            num_intervals = 0
        else:
            num_intervals = int(total_duration / (expected_interval_minutes * 60)) + 1
        
        # Create fixed-frequency timestamps using the grid
        fixed_timestamps_list = [
            grid_start + timedelta(minutes=i * expected_interval_minutes)
            for i in range(num_intervals)
        ]
        
        # Filter to strictly <= last_timestamp
        fixed_timestamps_list = [
            ts for ts in fixed_timestamps_list if ts <= last_timestamp
        ]
        
        # If list ended up empty, at least include grid start
        if not fixed_timestamps_list:
            fixed_timestamps_list = [grid_start]

        # Create fixed-frequency DataFrame with proper dtypes matching unified schema
        fixed_df = pl.DataFrame({
            'datetime': fixed_timestamps_list,
            'sequence_id': [seq_id] * len(fixed_timestamps_list)
        })
        
        # DON'T enforce full schema here - we'll get the data columns from the join
        # Just ensure datetime and sequence_id have the correct dtypes
        fixed_df = fixed_df.with_columns([
            pl.col('datetime').cast(pl.Datetime('ms')),
            pl.col('sequence_id').cast(pl.Int64)
        ])
        
        # Join with original data to get nearest values
        result_df = cls._join_and_interpolate_values(fixed_df, seq_data, expected_interval_minutes)
        
        return result_df
    
    @classmethod
    def _join_and_interpolate_values(
        cls,
        fixed_df: pl.DataFrame,
        seq_data: pl.DataFrame,
        expected_interval_minutes: int
    ) -> pl.DataFrame:
        """Map original data to fixed grid timestamps.
        
        Synchronization is LOSSLESS - it keeps ALL source rows, just rounds their datetime to the grid.
        Each source row is independently mapped to its nearest grid point.
        
        The only exception: if an IMPUTED+SYNCED row and a real SYNCED row map to the same
        grid point with the same event_type, keep only the real one (replace imputed).
        
        Args:
            fixed_df: DataFrame with fixed timestamps (not used in new implementation)
            seq_data: Original sequence data
            expected_interval_minutes: Expected interval in minutes
            
        Returns:
            DataFrame with datetime values rounded to grid timestamps
        """
        if len(seq_data) == 0:
            return seq_data
        
        seq_data_prep = seq_data.sort(['sequence_id', 'original_datetime', 'quality'])
        
        # Get grid start for this sequence
        grid_start = cls.get_sequence_grid_start(seq_data, expected_interval_minutes)
        
        # For each source row, calculate its nearest grid point
        # CRITICAL: Use the same rounding logic as interpolate (round half UP)
        # to ensure sync and interpolate are consistent
        result = seq_data_prep.with_columns([
            # Calculate which grid point each row should map to
            # Add 0.5 before floor to get "round half up" behavior (same as interpolate)
            ((pl.col('original_datetime') - pl.lit(grid_start)).dt.total_seconds() / 60.0 / expected_interval_minutes + 0.5)
            .floor()
            .cast(pl.Int64)
            .alias('_grid_offset')
        ]).with_columns([
            # Calculate the grid datetime (cast to ms to match schema)
            (pl.lit(grid_start) + pl.duration(minutes=pl.col('_grid_offset') * expected_interval_minutes))
            .cast(pl.Datetime('ms'))
            .alias('datetime')
        ])
        
        # Add SYNCHRONIZATION flag to quality
        result = result.with_columns([
            (pl.col('quality') | pl.lit(Quality.SYNCHRONIZATION.value)).alias('quality')
        ])
        
        # Drop temporary column
        result = result.drop('_grid_offset')
        
        # Sync is lossless - keep ALL rows, no deduplication
        # The only exception would be replacing imputed rows with real ones,
        # but that's handled during interpolation, not here
        
        # Ensure column order matches unified format
        result = CGM_SCHEMA.validate_columns(result, enforce=True)
        
        return result
    
    @classmethod
    def interpolate_gaps(
        cls,
        dataframe: UnifiedFormat,
        expected_interval_minutes: Optional[int] = None,
        small_gap_max_minutes: Optional[int] = None,
        snap_to_grid: Optional[bool] = None,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Fill gaps in continuous data with imputed values.
        
        This method interpolates small gaps (<= small_gap_max_minutes) within existing sequences
        and marks imputed values with the Quality.IMPUTATION flag.
        
        **Important**: This method expects sequence_id to already exist in the dataframe.
        
        Args:
            dataframe: DataFrame with sequence_id column indicating continuous sequences
            expected_interval_minutes: Expected interval in minutes (defaults to cls.expected_interval_minutes)
            small_gap_max_minutes: Maximum gap size to interpolate (defaults to cls.small_gap_max_minutes)
            snap_to_grid: If True, snap to grid (defaults to cls.snap_to_grid)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            DataFrame with interpolated values
        """
        if len(dataframe) == 0:
            return dataframe

        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes
        if small_gap_max_minutes is None:
            small_gap_max_minutes = cls.small_gap_max_minutes
        if snap_to_grid is None:
            snap_to_grid = cls.snap_to_grid
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Ensure sequences are assigned (auto-detect if missing)
        # Use small_gap_max_minutes as the gap threshold for sequence detection
        if not cls.has_sequences(dataframe):
            dataframe = cls.detect_and_assign_sequences(
                dataframe,
                expected_interval_minutes=expected_interval_minutes,
                large_gap_threshold_minutes=small_gap_max_minutes,
                validation_mode=validation_mode
            )
        
        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
        # Process each sequence separately for interpolation
        unique_sequences = dataframe['sequence_id'].unique().to_list()
        processed_sequences = []
        
        for seq_id in unique_sequences:
            # Sort by original_datetime for idempotent processing
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort(['sequence_id', 'original_datetime', 'quality'])
            
            if len(seq_data) < 2:
                processed_sequences.append(seq_data)
                continue
            
            # Interpolate gaps within this sequence
            interpolated_seq = cls._interpolate_sequence(seq_data, seq_id, expected_interval_minutes, small_gap_max_minutes, snap_to_grid)
            processed_sequences.append(interpolated_seq)
        
        # Combine all sequences with stable sorting from schema definition
        result_df = pl.concat(processed_sequences).sort(CGM_SCHEMA.get_stable_sort_keys())
        
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
 
    
    @classmethod
    def _interpolate_sequence(
        cls,
        seq_data: pl.DataFrame, 
        seq_id: int,
        expected_interval_minutes: int,
        small_gap_max_minutes: int,
        snap_to_grid: bool
    ) -> pl.DataFrame:
        """Interpolate missing values for a single sequence.
        
        Only interpolates between EGV_READ events with valid glucose values.
        Non-glucose events (INS_FAST, CARBS_IN, etc.) are not used as interpolation endpoints.
        
        Strategy: Split glucose and non-glucose events, interpolate only glucose, then merge back.
        This ensures non-glucose events don't interfere with gap detection.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            expected_interval_minutes: Expected interval in minutes
            small_gap_max_minutes: Maximum gap size to interpolate
            snap_to_grid: If True, snap to grid
            
        Returns:
            Sequence with interpolated values
        """
        # Split into glucose and non-glucose events
        glucose_events = seq_data.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
        non_glucose_events = seq_data.filter(pl.col('event_type') != UnifiedEventType.GLUCOSE.value)
        
        # If no glucose events or only 1, nothing to interpolate
        if len(glucose_events) < 2:
            return seq_data
        
        # Get common grid start for this sequence
        grid_start = cls.get_sequence_grid_start(seq_data, expected_interval_minutes)
        
        # Detect if data is already synchronized (has SYNCHRONIZATION flag)
        # If synchronized, use datetime for gap detection; otherwise use original_datetime
        has_sync_flag = glucose_events.filter(
            (pl.col('quality') & Quality.SYNCHRONIZATION.value) != 0
        ).height > 0
        
        # Use datetime if already synchronized, original_datetime otherwise
        # This ensures interpolation aligns with the existing grid after synchronization
        timestamp_col = 'datetime' if has_sync_flag else 'original_datetime'
        
        # Sort glucose events by the appropriate timestamp column
        glucose_events_sorted = glucose_events.sort(timestamp_col)
        
        # Calculate time differences using appropriate timestamp column
        time_diffs = glucose_events_sorted[timestamp_col].diff().dt.total_seconds() / 60.0
        time_diffs_list = time_diffs.to_list()
        
        # Convert to list of dicts for easier row creation
        glucose_list = glucose_events_sorted.to_dicts()
        
        # Find small gaps to interpolate (now we know consecutive rows are all glucose events)
        small_gaps = []
        for i, diff in enumerate(time_diffs_list):
            if i > 0 and expected_interval_minutes < diff <= small_gap_max_minutes:
                prev_row = glucose_list[i - 1]
                current_row = glucose_list[i]
                
                # Check that both have valid glucose values
                if (prev_row.get('glucose') is not None and
                    current_row.get('glucose') is not None):
                    small_gaps.append((i, diff))
        
        if not small_gaps:
            # No gaps to fill, return original data
            return seq_data
        
        new_rows = []
        
        for gap_idx, time_diff_minutes in small_gaps:
            prev_row = glucose_list[gap_idx - 1]
            current_row = glucose_list[gap_idx]
            
            # Use the appropriate timestamp column (datetime if synchronized, original_datetime otherwise)
            prev_dt = prev_row[timestamp_col]
            current_dt = current_row[timestamp_col]
            
            if snap_to_grid:
                    # Snap to sequence grid: determine ALL grid points that should exist in the gap
                    # CRITICAL: Use the ROUNDED grid positions, not the original timestamps
                    # This ensures we fill gaps between where timestamps WILL BE after rounding
                    
                    # Round both timestamps to their nearest grid points
                    prev_grid_dt = cls.calculate_grid_point(prev_dt, grid_start, expected_interval_minutes, 'nearest')
                    curr_grid_dt = cls.calculate_grid_point(current_dt, grid_start, expected_interval_minutes, 'nearest')
                    
                    # Calculate grid positions from rounded timestamps
                    prev_grid_pos = int((prev_grid_dt - grid_start).total_seconds() / 60.0 / expected_interval_minutes)
                    curr_grid_pos = int((curr_grid_dt - grid_start).total_seconds() / 60.0 / expected_interval_minutes)
                    
                    # Fill all grid points BETWEEN the rounded positions (exclusive on both ends)
                    first_grid_pos = prev_grid_pos + 1
                    last_grid_pos = curr_grid_pos
                    
                    # Generate ALL missing grid points in the gap
                    for grid_pos in range(first_grid_pos, last_grid_pos):
                        interpolated_time = grid_start + timedelta(minutes=grid_pos * expected_interval_minutes)
                        
                        # Interpolate glucose using grid-aligned timestamps for idempotency with sync
                        prev_glucose = prev_row['glucose']
                        curr_glucose = current_row['glucose']
                        interpolated_glucose = cls._interpolate_glucose_value(
                            target_time=interpolated_time,
                            prev_time=prev_grid_dt,
                            next_time=curr_grid_dt,
                            prev_glucose=prev_glucose,
                            next_glucose=curr_glucose
                        )
                        
                        # Create new row with GLUCOSE event type
                        # Quality combines flags from both neighbors + IMPUTATION + SYNCHRONIZATION
                        prev_quality = prev_row.get('quality', 0) or 0
                        curr_quality = current_row.get('quality', 0) or 0
                        combined_quality = (prev_quality | curr_quality | 
                                          Quality.IMPUTATION.value | 
                                          Quality.SYNCHRONIZATION.value)
                        
                        new_row = {
                            'sequence_id': seq_id,
                            'event_type': UnifiedEventType.GLUCOSE.value,
                            'quality': combined_quality,
                            'original_datetime': interpolated_time,  # Grid-aligned position
                            'datetime': interpolated_time,  # Both are the same for new interpolated points
                            'glucose': interpolated_glucose,
                            'carbs': None,
                            'insulin_slow': None,
                            'insulin_fast': None,
                            'exercise': None,
                        }
                        new_rows.append(new_row)
            else:
                # Non-grid logic: place points at regular intervals from previous timestamp
                # Calculate number of missing points
                missing_points = int(time_diff_minutes / expected_interval_minutes) - 1
                
                if missing_points > 0:
                    prev_glucose = prev_row['glucose']
                    curr_glucose = current_row['glucose']
                    
                    # Use the appropriate timestamp column
                    for j in range(1, missing_points + 1):
                        interpolated_time = prev_dt + timedelta(
                            minutes=expected_interval_minutes * j
                        )
                        
                        # Interpolate glucose using original timestamps (not grid-aligned)
                        interpolated_glucose = cls._interpolate_glucose_value(
                            target_time=interpolated_time,
                            prev_time=prev_dt,
                            next_time=current_dt,
                            prev_glucose=prev_glucose,
                            next_glucose=curr_glucose
                        )
                        
                        # Create new row with GLUCOSE event type
                        # Quality combines flags from both neighbors + IMPUTATION flag
                        prev_quality = prev_row.get('quality', 0) or 0
                        curr_quality = current_row.get('quality', 0) or 0
                        combined_quality = prev_quality | curr_quality | Quality.IMPUTATION.value
                        
                        new_row = {
                            'sequence_id': seq_id,
                            'event_type': UnifiedEventType.GLUCOSE.value,
                            'quality': combined_quality,
                            'original_datetime': interpolated_time,  # Set original to interpolated position
                            'datetime': interpolated_time,  # Both are the same for new interpolated points
                            'glucose': interpolated_glucose,
                            'carbs': None,
                            'insulin_slow': None,
                            'insulin_fast': None,
                            'exercise': None,
                        }
                        new_rows.append(new_row)
        
        # Add interpolated rows to glucose events
        if new_rows:
            interpolated_df = pl.DataFrame(new_rows, schema=glucose_events_sorted.schema)
            # Combine glucose events with interpolated points
            # Use stable sort: original_datetime, quality, then glucose (event_type is always GLUCOSE here)
            glucose_with_interpolation = pl.concat([glucose_events_sorted, interpolated_df]).sort([
                'original_datetime', 'quality', 'glucose'
            ])
        else:
            glucose_with_interpolation = glucose_events_sorted
        
        # Merge glucose events (with interpolation) back with non-glucose events
        # Use schema-defined stable sort, but skip sequence_id (already within same sequence)
        if len(non_glucose_events) > 0:
            sort_keys = [k for k in CGM_SCHEMA.get_stable_sort_keys() if k != 'sequence_id']
            result = pl.concat([glucose_with_interpolation, non_glucose_events]).sort(sort_keys)
        else:
            result = glucose_with_interpolation
        
        # Assert we didn't lose or duplicate rows
        expected_length = len(seq_data) + len(new_rows)
        actual_length = len(result)
        assert actual_length == expected_length, (
            f"Interpolation merge error: expected {expected_length} rows "
            f"(original {len(seq_data)} + interpolated {len(new_rows)}), "
            f"but got {actual_length} rows. "
            f"Glucose events: {len(glucose_events)}, Non-glucose: {len(non_glucose_events)}"
        )
        
        return result
    
    @classmethod
    def mark_calibration_periods(
        cls,
        dataframe: UnifiedFormat,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Mark 24-hour periods after calibration gaps as SENSOR_CALIBRATION quality.
        
        According to PIPELINE.md: "In case of large gap more than 2 hours 45 minutes
        mark next 24 hours as ill quality."
        
        This method detects gaps >= calibration_gap_threshold (2:45:00) using original_datetime
        and marks all data points within 24 hours after the gap end as Quality.SENSOR_CALIBRATION.
        
        Uses original_datetime for gap detection to ensure idempotent behavior regardless of
        whether synchronize_timestamps has been applied.
        
        Args:
            dataframe: DataFrame with sequences and original_datetime column
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            DataFrame with quality flags updated for calibration periods
        """
        if len(dataframe) == 0:
            return dataframe

        if validation_mode is None:
            validation_mode = cls.validation_mode_default
        
        # Validate input if validation mode includes INPUT
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
        # Use original_datetime for gap detection (idempotent regardless of sync)
        timestamp_col = 'original_datetime' #if 'original_datetime' in dataframe.columns else 'datetime'
        
        # Sort by timestamp to process chronologically
        df = dataframe.sort(timestamp_col)
        
        # Calculate time differences between consecutive rows using original_datetime
        df = df.with_columns([
            pl.col(timestamp_col).diff().dt.total_seconds().alias('time_diff_seconds'),
        ])
        
        # Identify calibration gaps (>= CALIBRATION_GAP_THRESHOLD)
        df = df.with_columns([
            pl.when(pl.col('time_diff_seconds').is_null())
            .then(pl.lit(False))
            .otherwise(pl.col('time_diff_seconds') >= cls.calibration_gap_threshold)
            .alias('is_calibration_gap'),
        ])
        
        # Extract timestamp values and gap flags before modifying DataFrame
        timestamp_values = df[timestamp_col].to_list()
        calibration_gap_mask = df['is_calibration_gap'].to_list()
        
        # Collect calibration period start times (rows after calibration gaps)
        calibration_period_starts = []
        for i in range(len(calibration_gap_mask)):
            if calibration_gap_mask[i]:  # This row is after a calibration gap
                gap_end_time = timestamp_values[i]
                calibration_period_starts.append(gap_end_time)
        
        # Create a column to mark rows that should be SENSOR_CALIBRATION
        df = df.with_columns([
            pl.lit(False).alias('in_calibration_period')
        ])
        
        # Mark all rows within 24 hours after each calibration gap (using original_datetime)
        if calibration_period_starts:
            # Create conditions for each calibration period
            conditions = []
            for gap_end_time in calibration_period_starts:
                calibration_period_end = gap_end_time + timedelta(hours=cls.calibration_period_hours)
                # Mark all points from gap_end_time (inclusive) for 24 hours
                conditions.append(
                    (pl.col(timestamp_col) >= gap_end_time) &
                    (pl.col(timestamp_col) <= calibration_period_end)
                )
            
            # Combine all conditions with OR
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            
            # Mark rows in calibration periods
            df = df.with_columns([
                combined_condition.alias('in_calibration_period')
            ])
        
        # Update quality column for rows in calibration periods
        # Use bitwise OR to add SENSOR_CALIBRATION flag on top of existing flags
        df = df.with_columns([
            pl.when(pl.col('in_calibration_period'))
            .then(pl.col('quality') | Quality.SENSOR_CALIBRATION.value)
            .otherwise(pl.col('quality'))
            .alias('quality')
        ])
        
        # Remove temporary columns
        df = df.drop(['time_diff_seconds', 'is_calibration_gap', 'in_calibration_period'])
        
        # Validate output if validation mode includes OUTPUT
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df
    
    @classmethod
    def prepare_for_inference(
        cls,
        dataframe: UnifiedFormat,
        minimum_duration_minutes: Optional[int] = None,
        maximum_wanted_duration: Optional[int] = None,
        validation_mode: Optional[ValidationMethod] = None
    ) -> InferenceResult:
        """Prepare data for inference with full UnifiedFormat and warning flags.
        
        Operations performed:
        1. Check for zero valid data points (raises ZeroValidInputError)
        2. Keep only the last (latest) sequence based on most recent timestamps
        3. Filter to glucose-only events if requested (drops non-EGV events before truncation)
        4. Truncate sequences exceeding maximum_wanted_duration
        5. Drop duplicate timestamps if requested
        6. Collect warnings based on truncated data quality:
           - TOO_SHORT: sequence duration < minimum_duration_minutes
           - CALIBRATION: contains calibration events
           - OUT_OF_RANGE: contains OUT_OF_RANGE quality flags
           - IMPUTATION: contains imputed data (IMPUTATION quality flag, tracked in interpolate_gaps)
           - TIME_DUPLICATES: contains non-unique time entries
        
        Returns full UnifiedFormat with all columns (sequence_id, event_type, quality, etc).
        Use to_data_only_df() to strip service columns if needed for ML models.
        
        Args:
            dataframe: Fully processed DataFrame in unified format
            minimum_duration_minutes: Minimum required sequence duration (defaults to MINIMUM_DURATION_MINUTES)
            maximum_wanted_duration: Maximum desired sequence duration (defaults to MAXIMUM_WANTED_DURATION_MINUTES)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            Tuple of (unified_format_dataframe, warnings)
            
        Raises:
            ZeroValidInputError: If there are no valid data points
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("No data points in the sequence")

        if minimum_duration_minutes is None:
            minimum_duration_minutes = cls.minimum_duration_minutes
        if maximum_wanted_duration is None:
            maximum_wanted_duration = cls.maximum_wanted_duration_minutes
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Local warning collection
        warnings: List[ProcessingWarning] = []
        
        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
        # Check for valid glucose readings
        valid_glucose_count = dataframe.filter(
            pl.col('glucose').is_not_null()
        ).height
        
        if valid_glucose_count == 0:
            raise ZeroValidInputError("No valid glucose data points in the sequence")
        
        # Keep only the last (latest) valid sequence
        # Try sequences starting from the most recent, fallback to previous ones if invalid
        if 'sequence_id' in dataframe.columns:
            # Get the maximum datetime for each sequence, sorted by recency
            seq_max_times = dataframe.group_by('sequence_id').agg([
                pl.col('datetime').max().alias('max_time'),
                pl.col('glucose').count().alias('glucose_count')
            ]).sort('max_time', descending=True)
            
            # Try sequences starting from the most recent
            df_truncated = None
            for seq_idx in range(len(seq_max_times)):
                candidate_seq_id = seq_max_times['sequence_id'][seq_idx]
                candidate_df = dataframe.filter(pl.col('sequence_id') == candidate_seq_id)
                
                # Check if this sequence has glucose data
                if candidate_df.filter(pl.col('glucose').is_not_null()).height == 0:
                    continue  # Skip sequences with no glucose data
                
                # Try truncating this sequence
                candidate_truncated = cls._truncate_by_duration(
                    candidate_df, 
                    maximum_wanted_duration
                )
                
                # Check if truncated sequence meets minimum duration
                if len(candidate_truncated) > 0:
                    duration_minutes = cls._calculate_duration_minutes(candidate_truncated)
                    if duration_minutes >= minimum_duration_minutes:
                        # Found a valid sequence!
                        df_truncated = candidate_truncated
                        break
            
            # If no valid sequence found, raise error
            if df_truncated is None:
                raise ZeroValidInputError(
                    f"No valid sequences found. Tried {len(seq_max_times)} sequences, "
                    f"none met minimum duration of {minimum_duration_minutes} minutes with glucose data."
                )
        else:
            # No sequence_id column, process entire dataframe
            df_truncated = cls._truncate_by_duration(
                dataframe, 
                maximum_wanted_duration
            )
        
        # NOW calculate warnings on the truncated data
        df_truncated = cls.mark_time_duplicates(df_truncated, validation_mode) #mark time duplicates
        df_truncated = cls.mark_calibration_periods(df_truncated, validation_mode) #mark calibration periods
        
        # Check duration (already verified above, but add warning if close to minimum)
        if len(df_truncated) > 0:
            duration_minutes = cls._calculate_duration_minutes(df_truncated)
            if duration_minutes < minimum_duration_minutes:
                warnings.append(ProcessingWarning.TOO_SHORT)
        
        # Check for calibration events or SENSOR_CALIBRATION flag
        calibration_count = df_truncated.filter(
            (pl.col('event_type') == UnifiedEventType.CALIBRATION.value) |
            ((pl.col('quality') & Quality.SENSOR_CALIBRATION.value) != 0)
        ).height
        if calibration_count > 0:
            warnings.append(ProcessingWarning.CALIBRATION)
        
        # Check for out-of-range values (OUT_OF_RANGE flag)
        out_of_range_count = df_truncated.filter(
            (pl.col('quality') & Quality.OUT_OF_RANGE.value) != 0
        ).height

        if out_of_range_count > 0:
            warnings.append(ProcessingWarning.OUT_OF_RANGE)
        
        # Check for IMPUTATION flag (may have already been added in interpolate_gaps)
        imputed_count = df_truncated.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) != 0
        ).height
        if imputed_count > 0 and ProcessingWarning.IMPUTATION not in warnings:
            warnings.append(ProcessingWarning.IMPUTATION)

        # Check for time duplicates in the final sequence or TIME_DUPLICATE flag
        has_time_duplicates = False
        if len(df_truncated) > 0:
            unique_time_count = df_truncated['datetime'].n_unique()
            total_count = len(df_truncated)
            if unique_time_count < total_count:
                has_time_duplicates = True
        
        # Also check for TIME_DUPLICATE flag in quality column
        time_duplicate_flag_count = df_truncated.filter(
            (pl.col('quality') & Quality.TIME_DUPLICATE.value) != 0
        ).height
        
        if has_time_duplicates or time_duplicate_flag_count > 0:
            warnings.append(ProcessingWarning.TIME_DUPLICATES)
        
        # Return full UnifiedFormat (keep all columns including service columns)
        # Combine warnings into flags for return value (for interface compatibility)
        combined_warnings = ProcessingWarning(0)
        for warning in warnings:
            combined_warnings |= warning
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df_truncated, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df_truncated, combined_warnings
    
    @classmethod
    def _calculate_duration_minutes(cls, dataframe: pl.DataFrame) -> float:
        """Calculate duration of sequence in minutes.
        
        Args:
            dataframe: DataFrame with datetime column
            
        Returns:
            Duration in minutes
        """
        if len(dataframe) == 0:
            return 0.0
        
        min_time = dataframe['datetime'].min()
        max_time = dataframe['datetime'].max()
        
        if min_time is None or max_time is None:
            return 0.0
        
        duration_seconds = (max_time - min_time).total_seconds()
        return duration_seconds / 60.0
    
    @classmethod
    def _truncate_by_duration(
        cls,
        dataframe: pl.DataFrame, 
        max_duration_minutes: int
    ) -> pl.DataFrame:
        """Truncate sequence to maximum duration, keeping the latest (most recent) data.
        
        Truncates from the beginning, preserving the most recent data points.
        
        Args:
            dataframe: DataFrame to truncate
            max_duration_minutes: Maximum duration in minutes
            
        Returns:
            Truncated DataFrame with latest data preserved
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Get end time (most recent)
        end_time = dataframe['datetime'].max()
        if end_time is None:
            return dataframe
        
        # Calculate cutoff time (truncate from beginning)
        cutoff_time = end_time - timedelta(minutes=max_duration_minutes)
        
        # Filter to keep only records after cutoff (latest data)
        truncated_df = dataframe.filter(pl.col('datetime') >= cutoff_time)
        
        return truncated_df
    
    @classmethod
    def to_data_only_df(
            cls,
            unified_df: UnifiedFormat,
            drop_service_columns: bool = True,
            drop_duplicates: bool = False, 
            glucose_only: bool = False,
            validation_mode: Optional[ValidationMethod] = None
        ) -> pl.DataFrame:
        """Strip service columns from UnifiedFormat, keeping only data columns for ML models.
        
        This is a small optional pipeline-terminating function that removes metadata columns
        (sequence_id, event_type, quality) and keeps only the data columns needed for inference.
        
        Data columns are computed from the unified format schema definition.
        Currently includes:
        - datetime: Timestamp of the reading
        - glucose: Blood glucose value (mg/dL)
        - carbs: Carbohydrate intake (grams)
        - insulin_slow: Slow-acting insulin dose (units)
        - insulin_fast: Fast-acting insulin dose (units)
        - exercise: Exercise indicator/intensity
        
        Args:
            unified_df: DataFrame in UnifiedFormat with all columns
            drop_service_columns: If True, drop service columns (sequence_id, event_type, quality)
            drop_duplicates: If True, drop duplicate timestamps (keeps first occurrence)
            glucose_only: If True, drop non-EGV events before truncation (keeps only GLUCOSE)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)

        Returns:
            DataFrame with only data columns (no service/metadata columns)
            
        """
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(unified_df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # Filter to glucose-only events if requested (before truncation)
        if glucose_only:
            unified_df, _ = cls.split_glucose_events(unified_df, validation_mode)

        # Drop duplicate timestamps if requested
        if drop_duplicates:
            unified_df = unified_df.unique(subset=['datetime'], keep='first')

        if drop_service_columns:
            data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
            unified_df = unified_df.select(data_columns)
        #no Output validation - is not unified format
        return unified_df
    
    @classmethod
    def split_glucose_events(
        cls,
        unified_df: UnifiedFormat,
        validation_mode: Optional[ValidationMethod] = None
    ) -> Tuple[UnifiedFormat, UnifiedFormat]:
        """Split UnifiedFormat DataFrame into glucose readings and other events.
        
        Divides a single UnifiedFormat DataFrame into two separate UnifiedFormat DataFrames:
        - Glucose DataFrame: Contains only GLUCOSE events (including imputed ones marked with quality flag)
        - Events DataFrame: Contains all other event types (insulin, carbs, exercise, calibration, etc.)
        
        Both output DataFrames maintain the full UnifiedFormat schema with all columns.
        This is a non-destructive split operation - no data transformation or column coalescing.
        
        Args:
            unified_df: DataFrame in UnifiedFormat with mixed event types
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            Tuple of (glucose_df, events_df) where:
            - glucose_df: UnifiedFormat DataFrame with GLUCOSE events
            - events_df: UnifiedFormat DataFrame with all other events
            
        Examples:
            >>> # Split mixed data into glucose and events
            >>> glucose, events = FormatProcessor.split_glucose_events(unified_df)
            >>> 
            >>> # Can be chained with other operations
            >>> unified_df = FormatParser.parse_file("data.csv")
            >>> glucose, events = FormatProcessor.split_glucose_events(unified_df)
            >>> glucose, warnings = FormatProcessor.interpolate_gaps(glucose)
        """
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(unified_df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
        # Filter for glucose events (GLUCOSE event type)
        glucose_df = unified_df.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        
        # Filter for all other events
        events_df = unified_df.filter(
            pl.col("event_type") != UnifiedEventType.GLUCOSE.value
        )
        
        # Verify output dataframes match schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(glucose_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
            CGM_SCHEMA.validate_dataframe(events_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return glucose_df, events_df
    
    @classmethod
    def has_sequences(cls, dataframe: UnifiedFormat) -> bool:
        """Check if the dataframe has valid sequence_id assignments.
        
        A dataframe has sequences if:
        1. sequence_id column exists
        2. sequence_id column has no null values
        3. At least one sequence_id is non-zero (0 means unassigned)
        
        Args:
            dataframe: DataFrame in unified format
            
        Returns:
            True if sequences are present and valid, False otherwise
        """
        if len(dataframe) == 0:
            return False
        
        if 'sequence_id' not in dataframe.columns:
            return False
        
        # Check for null values in sequence_id
        null_count = dataframe['sequence_id'].null_count()
        if null_count > 0:
            return False
        
        # Check if all values are 0 (unassigned)
        non_zero_count = dataframe.filter(pl.col('sequence_id') != 0).height
        if non_zero_count == 0:
            return False
        
        return True
    
    @classmethod
    def detect_and_assign_sequences(
        cls, 
        dataframe: UnifiedFormat,
        expected_interval_minutes: Optional[int] = None,
        large_gap_threshold_minutes: Optional[int] = None,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Detect large gaps and assign sequence_id column (lossless annotation).
        
        This method splits data into continuous sequences based on time gaps IN GLUCOSE EVENTS ONLY.
        Non-glucose events are then assigned to the nearest glucose sequence by time.
        
        Large gaps (> large_gap_threshold_minutes) between glucose readings create new sequences.
        sequence_id = 0 means unassigned (no glucose events available for assignment).
        sequence_id >= 1 means assigned to a glucose sequence.
        
        Two-pass approach:
        1. Detect sequences based on glucose event gaps only
        2. Assign non-glucose events to nearest glucose sequence by time
        
        This prevents non-glucose events from "bridging" glucose gaps and incorrectly
        keeping discontinuous glucose data in the same sequence.
        
        **Idempotency**: This method nullifies any existing sequence_id column at the start,
        ensuring consistent results regardless of whether sequences were previously assigned.
        
        Args:
            dataframe: DataFrame in unified format (may or may not have sequence_id)
            expected_interval_minutes: Expected data collection interval (defaults to cls.expected_interval_minutes)
            large_gap_threshold_minutes: Threshold for creating new sequences (defaults to cls.small_gap_max_minutes)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)
            
        Returns:
            DataFrame with sequence_id column assigned
        """
        if len(dataframe) == 0:
            return dataframe
        
        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes
        if large_gap_threshold_minutes is None:
            large_gap_threshold_minutes = cls.small_gap_max_minutes
        if validation_mode is None:
            validation_mode = cls.validation_mode_default
        
        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # IDEMPOTENCY: Reset existing sequence_id to 0 to ensure consistent results
        # This allows re-running sequence detection with different gap thresholds
        # Get canonical sequence_id dtype from schema
        sequence_id_dtype = next(
            col['dtype'] for col in CGM_SCHEMA.service_columns if col['name'] == 'sequence_id'
        )
        
        df = dataframe.with_columns([
            pl.lit(0).cast(sequence_id_dtype).alias('sequence_id')
        ]).sort('original_datetime')     

        large_gap_threshold_seconds: int = large_gap_threshold_minutes * 60
        
        # Single sequence or no sequence_id: create sequences based on GLUCOSE GAPS ONLY
        # Pass 1: Filter to glucose events only
        glucose_events = df.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value).sort('datetime')
        
        if len(glucose_events) == 0:
            # No glucose events - all events get sequence_id = 0 (unassigned)
            return df.with_columns([
                pl.lit(0).cast(sequence_id_dtype).alias('sequence_id')
            ])
        
        # Calculate time differences between glucose events only using original_datetime
        glucose_events = glucose_events.with_columns([
            pl.col('original_datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
        ])
        
        # Mark large gaps (> large_gap_threshold_minutes)
        # Fill None (first row) with False to avoid issues
        glucose_events = glucose_events.with_columns([
            pl.when(pl.col('time_diff_seconds').is_null())
            .then(pl.lit(False))
            .otherwise(pl.col('time_diff_seconds') > large_gap_threshold_seconds)
            .alias('is_gap'),
        ])
        
        # Create sequence IDs based on gaps (starts at 1, not 0)
        # sequence_id = 0 is reserved for unassigned events
        glucose_events = glucose_events.with_columns([
            (pl.col('is_gap').cum_sum() + 1).cast(sequence_id_dtype).alias('sequence_id')
        ])
        
        # Remove temporary columns
        glucose_events = glucose_events.drop(['time_diff_seconds', 'is_gap'])
        
        # Pass 2: Assign non-glucose events to nearest glucose sequence
        non_glucose_events = df.filter(pl.col('event_type') != UnifiedEventType.GLUCOSE.value)
        
        if len(non_glucose_events) == 0:
            # Only glucose events - we're done
            result_df = glucose_events
        else:
            # For each non-glucose event, find the closest glucose sequence by time using original_datetime
            # Drop old sequence_id before joining to avoid conflicts
            non_glucose_no_seq = non_glucose_events.drop('sequence_id')
            
            # Use join_asof to find nearest glucose event
            sequence_info = glucose_events.select(['original_datetime', 'sequence_id'])
            
            # Join non-glucose events to nearest glucose event (by time)
            non_glucose_with_seq = non_glucose_no_seq.join_asof(
                sequence_info,
                on='original_datetime',
                strategy='nearest'
            )
            
            # If join_asof couldn't find a match (shouldn't happen), set to 0
            non_glucose_with_seq = non_glucose_with_seq.with_columns([
                pl.col('sequence_id').fill_null(0).cast(sequence_id_dtype)
            ])
            
            # Combine glucose and non-glucose events
            result_df = pl.concat([glucose_events, non_glucose_with_seq], how='diagonal')
            
            # Reorder columns to match schema (use existing validation method)
            result_df = CGM_SCHEMA.validate_dataframe(result_df, enforce=True)
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
    
    @classmethod
    def _split_sequences_with_internal_gaps(
        cls,
        dataframe: pl.DataFrame,
        large_gap_threshold_seconds: float,
        sequence_id_dtype: pl.DataType
    ) -> pl.DataFrame:
        """Split existing sequences that have internal large gaps (glucose-only logic).
        
        Processes each existing sequence separately and splits it if it contains
        large gaps IN GLUCOSE EVENTS. Non-glucose events are then reassigned to
        the nearest glucose sequence.
        
        Args:
            dataframe: DataFrame with existing sequence_id column
            large_gap_threshold_seconds: Threshold in seconds for splitting sequences
            sequence_id_dtype: Target dtype for sequence_id column
            
        Returns:
            DataFrame with updated sequence_id column (some sequences may be split)
        """
        unique_sequences = dataframe['sequence_id'].unique().sort().to_list()
        processed_glucose_sequences = []
        next_sequence_id = max(unique_sequences) + 1
        
        # Process each existing sequence, checking for internal glucose gaps
        for seq_id in unique_sequences:
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort('original_datetime')
            
            # Filter to glucose events only for gap detection
            glucose_seq = seq_data.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value).sort('original_datetime')
            
            if len(glucose_seq) < 2:
                # Single glucose point or no glucose points, keep as is
                processed_glucose_sequences.append(glucose_seq)
                continue
            
            # Check for internal large gaps in glucose events using original_datetime
            glucose_seq = glucose_seq.with_columns([
                pl.col('original_datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
            ])
            
            # Mark large gaps within this sequence's glucose events
            glucose_seq = glucose_seq.with_columns([
                pl.when(pl.col('time_diff_seconds').is_null())
                .then(pl.lit(False))
                .otherwise(pl.col('time_diff_seconds') > large_gap_threshold_seconds)
                .alias('is_gap'),
            ])
            
            # Check if this sequence has any large gaps
            has_gaps = glucose_seq['is_gap'].sum() > 0
            
            if not has_gaps:
                # No internal gaps, keep sequence as is
                glucose_seq = glucose_seq.drop(['time_diff_seconds', 'is_gap'])
                processed_glucose_sequences.append(glucose_seq)
            else:
                # Split this sequence based on internal glucose gaps
                # Create sub-sequence IDs
                glucose_seq = glucose_seq.with_columns([
                    (pl.col('is_gap').cum_sum() + next_sequence_id).cast(sequence_id_dtype).alias('sequence_id')
                ])
                
                # Remove temporary columns
                glucose_seq = glucose_seq.drop(['time_diff_seconds', 'is_gap'])
                
                # Update next_sequence_id for next iteration
                next_sequence_id = glucose_seq['sequence_id'].max() + 1
                
                processed_glucose_sequences.append(glucose_seq)
        
        # Combine all processed glucose sequences
        if len(processed_glucose_sequences) == 0:
            # No glucose events at all - return original with sequence_id = 0
            return dataframe.with_columns([
                pl.lit(0).cast(sequence_id_dtype).alias('sequence_id')
            ])
        
        all_glucose = pl.concat(processed_glucose_sequences).sort('original_datetime')
        
        # Now reassign non-glucose events to nearest glucose sequence
        non_glucose_events = dataframe.filter(pl.col('event_type') != UnifiedEventType.GLUCOSE.value)
        
        if len(non_glucose_events) == 0:
            # Only glucose events
            result_df = all_glucose
        else:
            # Join non-glucose events to nearest glucose sequence using original_datetime
            # Drop old sequence_id before joining to avoid conflicts
            non_glucose_no_seq = non_glucose_events.drop('sequence_id')
            sequence_info = all_glucose.select(['original_datetime', 'sequence_id'])
            
            non_glucose_with_seq = non_glucose_no_seq.join_asof(
                sequence_info,
                on='original_datetime',
                strategy='nearest'
            )
            
            # If join_asof couldn't find a match, set to 0
            non_glucose_with_seq = non_glucose_with_seq.with_columns([
                pl.col('sequence_id').fill_null(0).cast(sequence_id_dtype)
            ])
            
            # Combine glucose and non-glucose events
            result_df = pl.concat([all_glucose, non_glucose_with_seq], how='diagonal')
            
            # Reorder columns to match schema (use existing validation method)
            result_df = CGM_SCHEMA.validate_dataframe(result_df, enforce=True)
        
        return result_df

        