"""CGM Data Processor Implementation.

Implements vendor-agnostic processing operations on UnifiedFormat data (Stages 4-5).
Adapted from glucose_ml_preprocessor.py for single-user unified format processing.
"""

import polars as pl
from typing import Dict, Any, List, Tuple, ClassVar
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
    - Warning collection throughout processing pipeline
    
    Processing warnings are collected in a list during operations and can be retrieved
    via get_warnings() or checked via has_warnings().
    """
    
    validation_mode_default : ClassVar[ValidationMethod] = ValidationMethod.INPUT

    def __init__(
        self,
        expected_interval_minutes: int = EXPECTED_INTERVAL_MINUTES,
        small_gap_max_minutes: int = SMALL_GAP_MAX_MINUTES,
        snap_to_grid: bool = True,
        validation_mode: ValidationMethod = validation_mode_default,
    ):
        """Initialize the processor.
        
        Args:
            expected_interval_minutes: Expected data collection interval (default: 5 minutes)
            small_gap_max_minutes: Maximum gap size to interpolate (default: 15 minutes)
            snap_to_grid: If True, interpolated points are placed on the sequence grid
                         with SYNCHRONIZATION flag (ensures idempotency with sync).
                         If False, interpolated points are placed at regular intervals
                         from the previous timestamp (may not align with grid).
        """
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.snap_to_grid = snap_to_grid
        self.expected_interval_seconds = expected_interval_minutes * 60
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        self.validation_mode = validation_mode
        # Warning collection (list to track multiple instances)
        self._warnings: List[ProcessingWarning] = []
    
    def get_warnings(self) -> List[ProcessingWarning]:
        """Get collected processing warnings.
        
        Returns:
            List of ProcessingWarning flags collected during processing
        """
        return self._warnings.copy()
    
    def has_warnings(self) -> bool:
        """Check if any warnings were collected during processing.
        
        Returns:
            True if any warnings were raised, False otherwise
        """
        return len(self._warnings) > 0
    
    def _add_warning(self, warning: ProcessingWarning) -> None:
        """Add a warning to the collected warnings list.
        
        Args:
            warning: ProcessingWarning flag to add
        """
        self._warnings.append(warning)


    def mark_time_duplicates(self, df: UnifiedFormat) -> UnifiedFormat:
        """Mark events with duplicate timestamps (keeping first occurrence).
        
        Uses keepfirst logic: the first event at a timestamp is kept clean,
        subsequent events with the same timestamp are marked with TIME_DUPLICATE flag.
        
        Args:
            df: DataFrame in unified format (must have 'datetime' and 'quality' columns)
            
        Returns:
            DataFrame with TIME_DUPLICATE flag added to quality column for duplicate timestamps
        """
        if len(df) == 0:
            return df

        # Validate input if validation mode includes INPUT
        if self.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df, enforce=self.validation_mode & ValidationMethod.INPUT_FORCED)
        
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
        if self.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df_marked, enforce=self.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df_marked
        
    def synchronize_timestamps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Align timestamps to minute boundaries and create fixed-frequency data.
        
        This method should be called after interpolate_gaps() when sequences are already
        created and small gaps are filled. It performs:
        1. Rounds timestamps to nearest minute using built-in rounding
        2. Creates fixed-frequency timestamps with expected_interval_minutes
        3. Linearly interpolates glucose values (time-weighted)
        4. Shifts discrete events (carbs, insulin, exercise) to nearest timestamps
        
        Args:
            dataframe: DataFrame in unified format (should already have sequences created)
            
        Returns:
            DataFrame with synchronized timestamps at fixed intervals
            
        Raises:
            ZeroValidInputError: If dataframe is empty or has no data
            ValueError: If data has gaps larger than small_gap_max_minutes (not preprocessed)
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("Cannot synchronize timestamps on empty dataframe")
        
        # Verify input dataframe matches schema
        if self.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=self.validation_mode & ValidationMethod.INPUT_FORCED)

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
            synced_seq = self._synchronize_sequence(seq_data, seq_id)
            synchronized_sequences.append(synced_seq)
        
        # Combine all sequences with stable sorting from schema definition
        result_df = pl.concat(synchronized_sequences).sort(CGM_SCHEMA.get_stable_sort_keys())
        
        # Verify output dataframe matches schema
        if self.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=self.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
    
    def get_sequence_grid_start(self, seq_data: UnifiedFormat) -> datetime:
        """Determine the grid start time for a sequence.
        
        The grid start is based on the first original_datetime in the sequence,
        rounded to the nearest minute. This ensures both synchronize_timestamps
        and interpolate_gaps use the same grid alignment.
        
        Uses original_datetime (not datetime) to preserve the original grid alignment
        even after synchronization or other timestamp modifications.
        
        Args:
            seq_data: Sequence data
            
        Returns:
            Grid start timestamp (rounded to nearest minute)
        """
        first_timestamp = seq_data['original_datetime'].min()
        
        # Round to nearest minute (same logic as synchronize_timestamps)
        if first_timestamp.second >= 30:
            grid_start = first_timestamp.replace(second=0, microsecond=0) + timedelta(minutes=1)
        else:
            grid_start = first_timestamp.replace(second=0, microsecond=0)
        
        return grid_start
    
    def calculate_grid_point(
        self, 
        timestamp: datetime, 
        grid_start: datetime, 
        round_direction: str = 'nearest'
    ) -> datetime:
        """Calculate the nearest grid point for a given timestamp.
        
        Args:
            timestamp: Timestamp to align to grid
            grid_start: Start of the grid
            round_direction: 'nearest', 'up', or 'down'
            
        Returns:
            Timestamp aligned to grid
        """
        elapsed_seconds = (timestamp - grid_start).total_seconds()
        interval_seconds = self.expected_interval_minutes * 60
        
        if round_direction == 'down':
            intervals = int(elapsed_seconds // interval_seconds)
        elif round_direction == 'up':
            intervals = int((elapsed_seconds + interval_seconds - 1) // interval_seconds)
        else:  # nearest
            intervals = int((elapsed_seconds + interval_seconds / 2) // interval_seconds)
        
        return grid_start + timedelta(minutes=intervals * self.expected_interval_minutes)
    
    def _interpolate_glucose_value(
        self,
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
    
    def _synchronize_sequence(
        self, 
        seq_data: pl.DataFrame, 
        seq_id: int
    ) -> pl.DataFrame:
        """Synchronize timestamps for a single sequence to fixed frequency.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            
        Returns:
            Sequence with synchronized timestamps at fixed intervals
        """
        # Get grid start using common logic
        grid_start = self.get_sequence_grid_start(seq_data)
        
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
            num_intervals = int(total_duration / (self.expected_interval_minutes * 60)) + 1
        
        # Create fixed-frequency timestamps using the grid
        fixed_timestamps_list = [
            grid_start + timedelta(minutes=i * self.expected_interval_minutes)
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
        result_df = self._join_and_interpolate_values(fixed_df, seq_data)
        
        return result_df
    
    def _join_and_interpolate_values(
        self,
        fixed_df: pl.DataFrame,
        seq_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Map original data to fixed grid timestamps.
        
        Synchronization is LOSSLESS - it keeps ALL source rows, just rounds their datetime to the grid.
        Each source row is independently mapped to its nearest grid point.
        
        The only exception: if an IMPUTED+SYNCED row and a real SYNCED row map to the same
        grid point with the same event_type, keep only the real one (replace imputed).
        
        Args:
            fixed_df: DataFrame with fixed timestamps (not used in new implementation)
            seq_data: Original sequence data
            
        Returns:
            DataFrame with datetime values rounded to grid timestamps
        """
        if len(seq_data) == 0:
            return seq_data
        
        seq_data_prep = seq_data.sort(['sequence_id', 'original_datetime', 'quality'])
        
        # Get grid start for this sequence
        grid_start = self.get_sequence_grid_start(seq_data)
        
        # For each source row, calculate its nearest grid point
        # CRITICAL: Use the same rounding logic as interpolate (round half UP)
        # to ensure sync and interpolate are consistent
        result = seq_data_prep.with_columns([
            # Calculate which grid point each row should map to
            # Add 0.5 before floor to get "round half up" behavior (same as interpolate)
            ((pl.col('original_datetime') - pl.lit(grid_start)).dt.total_seconds() / 60.0 / self.expected_interval_minutes + 0.5)
            .floor()
            .cast(pl.Int64)
            .alias('_grid_offset')
        ]).with_columns([
            # Calculate the grid datetime (cast to ms to match schema)
            (pl.lit(grid_start) + pl.duration(minutes=pl.col('_grid_offset') * self.expected_interval_minutes))
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
    
    def interpolate_gaps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Fill gaps in continuous data with imputed values.
        
        This method interpolates small gaps (<= small_gap_max_minutes) within existing sequences
        and marks imputed values with the Quality.IMPUTATION flag.
        
        **Important**: This method expects sequence_id to already exist in the dataframe.
        
        Args:
            dataframe: DataFrame with sequence_id column indicating continuous sequences
            
        Returns:
            DataFrame with interpolated values marked with IMPUTATION flag
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Verify input dataframe matches schema
        if self.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=self.validation_mode & ValidationMethod.INPUT_FORCED)
        
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
            interpolated_seq = self._interpolate_sequence(seq_data, seq_id)
            processed_sequences.append(interpolated_seq)
        
        # Combine all sequences with stable sorting from schema definition
        result_df = pl.concat(processed_sequences).sort(CGM_SCHEMA.get_stable_sort_keys())
        
        # Check if any imputation was done (check for IMPUTATION flag)
        imputed_count = result_df.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) != 0
        ).height
        
        if imputed_count > 0:
            self._add_warning(ProcessingWarning.IMPUTATION)
        
        # Verify output dataframe matches schema
        if self.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=self.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
 
    
    def _interpolate_sequence(
        self, 
        seq_data: pl.DataFrame, 
        seq_id: int
    ) -> pl.DataFrame:
        """Interpolate missing values for a single sequence.
        
        Only interpolates between EGV_READ events with valid glucose values.
        Non-glucose events (INS_FAST, CARBS_IN, etc.) are not used as interpolation endpoints.
        
        Strategy: Split glucose and non-glucose events, interpolate only glucose, then merge back.
        This ensures non-glucose events don't interfere with gap detection.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            
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
        grid_start = self.get_sequence_grid_start(seq_data)
        
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
            if i > 0 and self.expected_interval_minutes < diff <= self.small_gap_max_minutes:
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
            
            if self.snap_to_grid:
                    # Snap to sequence grid: determine ALL grid points that should exist in the gap
                    # CRITICAL: Use the ROUNDED grid positions, not the original timestamps
                    # This ensures we fill gaps between where timestamps WILL BE after rounding
                    
                    # Round both timestamps to their nearest grid points
                    prev_grid_dt = self.calculate_grid_point(prev_dt, grid_start, 'nearest')
                    curr_grid_dt = self.calculate_grid_point(current_dt, grid_start, 'nearest')
                    
                    # Calculate grid positions from rounded timestamps
                    prev_grid_pos = int((prev_grid_dt - grid_start).total_seconds() / 60.0 / self.expected_interval_minutes)
                    curr_grid_pos = int((curr_grid_dt - grid_start).total_seconds() / 60.0 / self.expected_interval_minutes)
                    
                    # Fill all grid points BETWEEN the rounded positions (exclusive on both ends)
                    first_grid_pos = prev_grid_pos + 1
                    last_grid_pos = curr_grid_pos
                    
                    # Generate ALL missing grid points in the gap
                    for grid_pos in range(first_grid_pos, last_grid_pos):
                        interpolated_time = grid_start + timedelta(minutes=grid_pos * self.expected_interval_minutes)
                        
                        # Interpolate glucose using grid-aligned timestamps for idempotency with sync
                        prev_glucose = prev_row['glucose']
                        curr_glucose = current_row['glucose']
                        interpolated_glucose = self._interpolate_glucose_value(
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
                missing_points = int(time_diff_minutes / self.expected_interval_minutes) - 1
                
                if missing_points > 0:
                    prev_glucose = prev_row['glucose']
                    curr_glucose = current_row['glucose']
                    
                    # Use the appropriate timestamp column
                    for j in range(1, missing_points + 1):
                        interpolated_time = prev_dt + timedelta(
                            minutes=self.expected_interval_minutes * j
                        )
                        
                        # Interpolate glucose using original timestamps (not grid-aligned)
                        interpolated_glucose = self._interpolate_glucose_value(
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
    
    def mark_calibration_periods(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Mark 24-hour periods after calibration gaps as SENSOR_CALIBRATION quality.
        
        According to PIPELINE.md: "In case of large gap more than 2 hours 45 minutes
        mark next 24 hours as ill quality."
        
        This method detects gaps >= CALIBRATION_GAP_THRESHOLD (2:45:00) using original_datetime
        and marks all data points within 24 hours after the gap end as Quality.SENSOR_CALIBRATION.
        
        Uses original_datetime for gap detection to ensure idempotent behavior regardless of
        whether synchronize_timestamps has been applied.
        
        Args:
            dataframe: DataFrame with sequences and original_datetime column
            
        Returns:
            DataFrame with quality flags updated for calibration periods
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Validate input if validation mode includes INPUT
        if self.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=self.validation_mode & ValidationMethod.INPUT_FORCED)
        
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
            .otherwise(pl.col('time_diff_seconds') >= CALIBRATION_GAP_THRESHOLD)
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
                calibration_period_end = gap_end_time + timedelta(hours=CALIBRATION_PERIOD_HOURS)
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
        if self.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df, enforce=self.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df
    
    def prepare_for_inference(
        self,
        dataframe: UnifiedFormat,
        minimum_duration_minutes: int = MINIMUM_DURATION_MINUTES,
        maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION_MINUTES,
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
            minimum_duration_minutes: Minimum required sequence duration
            maximum_wanted_duration: Maximum desired sequence duration (truncates if exceeded)
            
        Returns:
            Tuple of (unified_format_dataframe, warnings)
            
        Raises:
            ZeroValidInputError: If there are no valid data points
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("No data points in the sequence")
        
        # Verify input dataframe matches schema
        if self.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=self.validation_mode & ValidationMethod.INPUT_FORCED)
        
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
                candidate_truncated = self._truncate_by_duration(
                    candidate_df, 
                    maximum_wanted_duration
                )
                
                # Check if truncated sequence meets minimum duration
                if len(candidate_truncated) > 0:
                    duration_minutes = self._calculate_duration_minutes(candidate_truncated)
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
            df_truncated = self._truncate_by_duration(
                dataframe, 
                maximum_wanted_duration
            )
        
        # NOW calculate warnings on the truncated data
        df_truncated = self.mark_time_duplicates(df_truncated) #mark time duplicates
        df_truncated = self.mark_calibration_periods(df_truncated) #mark calibration periods
        
        # Check duration (already verified above, but add warning if close to minimum)
        if len(df_truncated) > 0:
            duration_minutes = self._calculate_duration_minutes(df_truncated)
            if duration_minutes < minimum_duration_minutes:
                self._add_warning(ProcessingWarning.TOO_SHORT)
        
        # Check for calibration events or SENSOR_CALIBRATION flag
        calibration_count = df_truncated.filter(
            (pl.col('event_type') == UnifiedEventType.CALIBRATION.value) |
            ((pl.col('quality') & Quality.SENSOR_CALIBRATION.value) != 0)
        ).height
        if calibration_count > 0:
            self._add_warning(ProcessingWarning.CALIBRATION)
        
        # Check for out-of-range values (OUT_OF_RANGE flag)
        out_of_range_count = df_truncated.filter(
            (pl.col('quality') & Quality.OUT_OF_RANGE.value) != 0
        ).height

        if out_of_range_count > 0:
            self._add_warning(ProcessingWarning.OUT_OF_RANGE)
        
        # Check for IMPUTATION flag (may have already been added in interpolate_gaps)
        imputed_count = df_truncated.filter(
            (pl.col('quality') & Quality.IMPUTATION.value) != 0
        ).height
        if imputed_count > 0 and ProcessingWarning.IMPUTATION not in self._warnings:
            self._add_warning(ProcessingWarning.IMPUTATION)
        
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
            self._add_warning(ProcessingWarning.TIME_DUPLICATES)
        
        # Return full UnifiedFormat (keep all columns including service columns)
        # Combine warnings into flags for return value (for interface compatibility)
        combined_warnings = ProcessingWarning(0)
        for warning in self._warnings:
            combined_warnings |= warning
        
        # Verify output dataframe matches schema
        if self.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(df_truncated, enforce=self.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df_truncated, combined_warnings
    
    def _calculate_duration_minutes(self, dataframe: pl.DataFrame) -> float:
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
    
    def _truncate_by_duration(
        self, 
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
    
    @staticmethod
    def to_data_only_df(
            unified_df: UnifiedFormat,
            drop_service_columns: bool = True,
            drop_duplicates: bool = False, 
            glucose_only: bool = False
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

        Returns:
            DataFrame with only data columns (no service/metadata columns)
            
        """
        # Verify input dataframe matches schema
        CGM_SCHEMA.validate_dataframe(unified_df, enforce=False)

        # Filter to glucose-only events if requested (before truncation)
        if glucose_only:
            unified_df, _ = FormatProcessor.split_glucose_events(unified_df)

        # Drop duplicate timestamps if requested
        if drop_duplicates:
            unified_df = unified_df.unique(subset=['datetime'], keep='first')

        if drop_service_columns:
            data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
            unified_df = unified_df.select(data_columns)

        return unified_df
    
    @staticmethod
    def split_glucose_events(unified_df: UnifiedFormat) -> Tuple[UnifiedFormat, UnifiedFormat]:
        """Split UnifiedFormat DataFrame into glucose readings and other events.
        
        Divides a single UnifiedFormat DataFrame into two separate UnifiedFormat DataFrames:
        - Glucose DataFrame: Contains only GLUCOSE events (including imputed ones marked with quality flag)
        - Events DataFrame: Contains all other event types (insulin, carbs, exercise, calibration, etc.)
        
        Both output DataFrames maintain the full UnifiedFormat schema with all columns.
        This is a non-destructive split operation - no data transformation or column coalescing.
        
        Args:
            unified_df: DataFrame in UnifiedFormat with mixed event types
            
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
            >>> glucose = processor.interpolate_gaps(glucose)
        """
        # Verify input dataframe matches schema
        CGM_SCHEMA.validate_dataframe(unified_df, enforce=False)
        
        # Filter for glucose events (GLUCOSE event type)
        glucose_df = unified_df.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        
        # Filter for all other events
        events_df = unified_df.filter(
            pl.col("event_type") != UnifiedEventType.GLUCOSE.value
        )
        
        # Verify output dataframes match schema
        CGM_SCHEMA.validate_dataframe(glucose_df, enforce=False)
        CGM_SCHEMA.validate_dataframe(events_df, enforce=False)
        
        return glucose_df, events_df

