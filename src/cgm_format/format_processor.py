"""CGM Data Processor Implementation.

Implements vendor-agnostic processing operations on UnifiedFormat data (Stages 4-5).
Adapted from glucose_ml_preprocessor.py for single-user unified format processing.
"""

import polars as pl
from typing import Dict, Any, List
from datetime import timedelta
from cgm_format.interface.cgm_interface import (
    CGMProcessor,
    UnifiedFormat,
    InferenceResult,
    ProcessingWarning,
    ZeroValidInputError,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
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
    
    def __init__(
        self,
        expected_interval_minutes: int = 5,
        small_gap_max_minutes: int = 15,
    ):
        """Initialize the processor.
        
        Args:
            expected_interval_minutes: Expected data collection interval (default: 5 minutes)
            small_gap_max_minutes: Maximum gap size to interpolate (default: 15 minutes)
        """
        self.expected_interval_minutes = expected_interval_minutes
        self.small_gap_max_minutes = small_gap_max_minutes
        self.expected_interval_seconds = expected_interval_minutes * 60
        self.small_gap_max_seconds = small_gap_max_minutes * 60
        
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
    
    def synchronize_timestamps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Align timestamps to minute boundaries and create fixed-frequency data.
        
        This method should be called after interpolate_gaps() when sequences are already
        created and small gaps are filled. It performs:
        1. Rounds timestamps to nearest minute (seconds = 00)
        2. Creates fixed-frequency timestamps with expected_interval_minutes
        3. Linearly interpolates glucose values
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
        
        # Verify data has sequence_id (should be preprocessed)
        if 'sequence_id' not in dataframe.columns:
            raise ValueError(
                "Data must have sequence_id column. "
                "Run interpolate_gaps() first to create sequences."
            )
        
        # Process each sequence separately
        unique_sequences = dataframe['sequence_id'].unique().to_list()
        synchronized_sequences = []
        
        for seq_id in unique_sequences:
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort('datetime')
            
            if len(seq_data) < 2:
                # Keep single-point sequences as-is, just round the timestamp
                seq_data = seq_data.with_columns([
                    pl.col('datetime').dt.truncate('1m').alias('datetime')
                ])
                synchronized_sequences.append(seq_data)
                continue
            
            # Check for large gaps (data should already be preprocessed)
            time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
            max_gap = time_diffs.drop_nulls().max()
            
            if max_gap > self.small_gap_max_minutes:
                raise ValueError(
                    f"Sequence {seq_id} has gap of {max_gap:.1f} minutes "
                    f"(> {self.small_gap_max_minutes} minutes). "
                    f"Run interpolate_gaps() first to fill gaps and split sequences."
                )
            
            # Synchronize this sequence
            synced_seq = self._synchronize_sequence(seq_data, seq_id)
            synchronized_sequences.append(synced_seq)
        
        # Combine all sequences
        result_df = pl.concat(synchronized_sequences).sort(['sequence_id', 'datetime'])
        
        return result_df
    
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
        # Get first and last timestamps
        first_timestamp = seq_data['datetime'].min()
        last_timestamp = seq_data['datetime'].max()
        
        # Round to nearest minute (remove seconds)
        # If seconds >= 30, round up; otherwise round down
        first_second = first_timestamp.second
        if first_second >= 30:
            aligned_start = first_timestamp.replace(second=0, microsecond=0) + timedelta(minutes=1)
        else:
            aligned_start = first_timestamp.replace(second=0, microsecond=0)
        
        # Calculate duration and number of intervals
        total_duration = (last_timestamp - aligned_start).total_seconds()
        num_intervals = int(total_duration / (self.expected_interval_minutes * 60)) + 1
        
        # Create fixed-frequency timestamps
        fixed_timestamps_list = [
            aligned_start + timedelta(minutes=i * self.expected_interval_minutes)
            for i in range(num_intervals)
            if aligned_start + timedelta(minutes=i * self.expected_interval_minutes) <= last_timestamp
        ]
        
        # Handle edge case: if no timestamps generated, use aligned_start
        if not fixed_timestamps_list:
            fixed_timestamps_list = [aligned_start]
        
        # Create DataFrame with fixed timestamps
        fixed_df = pl.DataFrame({
            'datetime': fixed_timestamps_list,
            'sequence_id': [seq_id] * len(fixed_timestamps_list)
        })
        
        # Join with original data to get nearest values
        result_df = self._join_and_interpolate_values(fixed_df, seq_data)
        
        return result_df
    
    def _join_and_interpolate_values(
        self,
        fixed_df: pl.DataFrame,
        seq_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Join fixed timestamps with original data and interpolate/shift values.
        
        Uses asof_join to find nearest neighbors efficiently.
        - Glucose: linear interpolation between prev and next
        - Carbs, insulin, exercise: use nearest value (prefer backward/previous)
        - Event type, quality: use from nearest neighbor
        
        Args:
            fixed_df: DataFrame with fixed timestamps and sequence_id
            seq_data: Original sequence data
            
        Returns:
            DataFrame with interpolated/shifted values
        """
        # Join backward (get previous values)
        backward_join = fixed_df.join_asof(
            seq_data,
            left_on='datetime',
            right_on='datetime',
            strategy='backward',
            suffix='_prev'
        )
        
        # Join forward (get next values) for glucose interpolation
        forward_join = fixed_df.join_asof(
            seq_data.select(['datetime', 'glucose']),
            left_on='datetime',
            right_on='datetime',
            strategy='forward',
            suffix='_next'
        )
        
        # Combine joins
        combined = backward_join.join(
            forward_join.select(['datetime', pl.col('glucose').alias('glucose_next')]),
            on='datetime',
            how='left'
        )
        
        # Handle sequence_id conflict (keep the one from fixed_df)
        if 'sequence_id_prev' in combined.columns:
            combined = combined.drop('sequence_id_prev')
        
        # Interpolate glucose linearly between prev and next values
        combined = combined.with_columns([
            pl.when(pl.col('glucose').is_not_null())
            .then(pl.col('glucose'))
            .when(
                pl.col('glucose').is_null() &
                pl.col('glucose_next').is_not_null() &
                (pl.col('glucose').shift(1).over('sequence_id').is_not_null())
            )
            .then(
                # Linear interpolation: average of prev and next
                (pl.col('glucose').shift(1).over('sequence_id') + pl.col('glucose_next')) / 2.0
            )
            .otherwise(pl.col('glucose'))
            .alias('glucose')
        ])
        
        # For discrete events (carbs, insulin, exercise), use previous value or None
        # These are point events that should not be interpolated
        event_columns = ['carbs', 'insulin_slow', 'insulin_fast', 'exercise']
        
        for col in event_columns:
            if col in combined.columns:
                # Keep the value from backward join (previous value)
                # This is already the desired behavior from asof_join backward
                pass
        
        # For event_type and quality, use from previous value or default
        if 'event_type' in combined.columns:
            combined = combined.with_columns([
                pl.when(pl.col('event_type').is_null())
                .then(pl.lit(UnifiedEventType.GLUCOSE.value))
                .otherwise(pl.col('event_type'))
                .alias('event_type')
            ])
        
        if 'quality' in combined.columns:
            combined = combined.with_columns([
                pl.when(pl.col('quality').is_null())
                .then(pl.lit(Quality.GOOD.value))
                .otherwise(pl.col('quality'))
                .alias('quality')
            ])
        
        # Clean up temporary columns
        temp_cols = [col for col in combined.columns if col.endswith('_next') or col.endswith('_prev')]
        result = combined.drop(temp_cols)
        
        # Ensure column order matches unified format
        expected_columns = ['sequence_id', 'event_type', 'quality', 'datetime', 
                           'glucose', 'carbs', 'insulin_slow', 'insulin_fast', 'exercise']
        result = result.select([col for col in expected_columns if col in result.columns])
        
        return result
    
    def interpolate_gaps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Fill gaps in continuous data with imputed values.
        
        This method performs two key operations:
        1. Detects large gaps and creates/updates sequence_id to split data into sequences
        2. Interpolates small gaps with imputed values
        
        Adds rows with event_type='IMPUTATN' for missing data points.
        Only interpolates small gaps (<= small_gap_max_minutes).
        Large gaps (> small_gap_max_minutes) create new sequences.
        
        Based on detect_gaps_and_sequences() and interpolate_missing_values()
        from glucose_ml_preprocessor.py, adapted for unified format.
        
        Args:
            dataframe: DataFrame with potential gaps (may or may not have sequence_id)
            
        Returns:
            DataFrame with sequence_id, interpolated values and impute events
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Step 1: Detect gaps and create/update sequence_id
        df = self._detect_gaps_and_create_sequences(dataframe)
        
        # Step 2: Process each sequence separately for interpolation
        unique_sequences = df['sequence_id'].unique().to_list()
        processed_sequences = []
        
        for seq_id in unique_sequences:
            seq_data = df.filter(pl.col('sequence_id') == seq_id).sort('datetime')
            
            if len(seq_data) < 2:
                processed_sequences.append(seq_data)
                continue
            
            # Detect gaps and interpolate
            interpolated_seq = self._interpolate_sequence(seq_data, seq_id)
            processed_sequences.append(interpolated_seq)
        
        # Combine all sequences
        result_df = pl.concat(processed_sequences).sort(['sequence_id', 'datetime'])
        
        # Step 3: Mark calibration periods (24 hours after gaps >= CALIBRATION_GAP_THRESHOLD)
        result_df = self._mark_calibration_periods(result_df)
        
        # Check if any imputation was done
        imputed_count = result_df.filter(
            pl.col('event_type') == UnifiedEventType.IMPUTATION.value
        ).height
        
        if imputed_count > 0:
            self._add_warning(ProcessingWarning.IMPUTATION)
        
        return result_df
    
    def _detect_gaps_and_create_sequences(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Detect large gaps and create/update sequence_id column.
        
        Large gaps (> small_gap_max_minutes) create new sequences.
        If sequence_id doesn't exist or all rows have the same ID, this method
        will create/update it based on gap detection.
        
        If multiple sequence_ids already exist, processes each sequence separately
        and splits them if they contain internal large gaps, ensuring unique sequence IDs.
        
        Args:
            dataframe: DataFrame sorted by datetime
            
        Returns:
            DataFrame with sequence_id column
        """
        # Sort by datetime
        df = dataframe.sort('datetime')
        
        if len(df) == 0:
            return df
        
        # Check if sequence_id exists and has variation
        has_sequence_id = 'sequence_id' in df.columns
        
        if has_sequence_id:
            unique_seq_ids = df['sequence_id'].n_unique()
            
            # If there are multiple sequences, process each separately to detect internal gaps
            if unique_seq_ids > 1:
                return self._split_sequences_with_internal_gaps(df)
        
        # Single sequence or no sequence_id: create sequences based on gaps
        # Calculate time differences
        df = df.with_columns([
            pl.col('datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
        ])
        
        # Mark large gaps (> small_gap_max_minutes)
        # Fill None (first row) with False to avoid issues
        df = df.with_columns([
            pl.when(pl.col('time_diff_seconds').is_null())
            .then(pl.lit(False))
            .otherwise(pl.col('time_diff_seconds') > self.small_gap_max_seconds)
            .alias('is_gap'),
        ])
        
        # Create sequence IDs based on gaps
        df = df.with_columns([
            pl.col('is_gap').cum_sum().alias('sequence_id')
        ])
        
        # Remove temporary columns
        df = df.drop(['time_diff_seconds', 'is_gap'])
        
        return df
    
    def _split_sequences_with_internal_gaps(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Split existing sequences that have internal large gaps.
        
        Processes each existing sequence separately and splits it if it contains
        large gaps. Creates new unique sequence IDs for split sequences.
        
        Args:
            dataframe: DataFrame with existing sequence_id column
            
        Returns:
            DataFrame with updated sequence_id column (some sequences may be split)
        """
        unique_sequences = dataframe['sequence_id'].unique().sort().to_list()
        processed_sequences = []
        next_sequence_id = max(unique_sequences) + 1
        
        for seq_id in unique_sequences:
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort('datetime')
            
            if len(seq_data) < 2:
                # Single point, keep as is
                processed_sequences.append(seq_data)
                continue
            
            # Check for internal large gaps
            seq_data = seq_data.with_columns([
                pl.col('datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
            ])
            
            # Mark large gaps within this sequence
            seq_data = seq_data.with_columns([
                pl.when(pl.col('time_diff_seconds').is_null())
                .then(pl.lit(False))
                .otherwise(pl.col('time_diff_seconds') > self.small_gap_max_seconds)
                .alias('is_gap'),
            ])
            
            # Check if this sequence has any large gaps
            has_gaps = seq_data['is_gap'].sum() > 0
            
            if not has_gaps:
                # No internal gaps, keep sequence as is
                seq_data = seq_data.drop(['time_diff_seconds', 'is_gap'])
                processed_sequences.append(seq_data)
            else:
                # Split this sequence based on internal gaps
                # Create sub-sequence IDs
                seq_data = seq_data.with_columns([
                    pl.col('is_gap').cum_sum().alias('sub_seq_id')
                ])
                
                # Get unique sub-sequences
                unique_sub_seqs = seq_data['sub_seq_id'].unique().sort().to_list()
                
                for sub_seq_idx in unique_sub_seqs:
                    sub_seq_data = seq_data.filter(pl.col('sub_seq_id') == sub_seq_idx)
                    
                    # Assign new unique sequence_id (cast to match original dtype)
                    original_dtype = dataframe['sequence_id'].dtype
                    sub_seq_data = sub_seq_data.with_columns([
                        pl.lit(next_sequence_id).cast(original_dtype).alias('sequence_id')
                    ])
                    
                    # Remove temporary columns
                    sub_seq_data = sub_seq_data.drop(['time_diff_seconds', 'is_gap', 'sub_seq_id'])
                    
                    processed_sequences.append(sub_seq_data)
                    next_sequence_id += 1
        
        # Combine all processed sequences
        result_df = pl.concat(processed_sequences).sort(['sequence_id', 'datetime'])
        
        return result_df
    
    def _interpolate_sequence(
        self, 
        seq_data: pl.DataFrame, 
        seq_id: int
    ) -> pl.DataFrame:
        """Interpolate missing values for a single sequence.
        
        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID
            
        Returns:
            Sequence with interpolated values
        """
        # Calculate time differences
        time_diffs = seq_data['datetime'].diff().dt.total_seconds() / 60.0
        time_diffs_list = time_diffs.to_list()
        
        # Find small gaps to interpolate
        small_gaps = [
            (i, diff) for i, diff in enumerate(time_diffs_list)
            if i > 0 and self.expected_interval_minutes < diff <= self.small_gap_max_minutes
        ]
        
        if not small_gaps:
            return seq_data
        
        # Convert to list of dicts for easier row creation
        seq_list = seq_data.to_dicts()
        new_rows = []
        
        for gap_idx, time_diff_minutes in small_gaps:
            prev_row = seq_list[gap_idx - 1]
            current_row = seq_list[gap_idx]
            
            # Calculate number of missing points
            missing_points = int(time_diff_minutes / self.expected_interval_minutes) - 1
            
            if missing_points > 0:
                # Create interpolated points
                for j in range(1, missing_points + 1):
                    interpolated_time = prev_row['datetime'] + timedelta(
                        minutes=self.expected_interval_minutes * j
                    )
                    
                    # Create new row with imputation event type
                    new_row = {
                        'sequence_id': seq_id,
                        'event_type': UnifiedEventType.IMPUTATION.value,
                        'quality': Quality.GOOD.value,  # Imputed data is considered good quality
                        'datetime': interpolated_time,
                        'glucose': None,
                        'carbs': None,
                        'insulin_slow': None,
                        'insulin_fast': None,
                        'exercise': None,
                    }
                    
                    # Linear interpolation for glucose if both values exist
                    prev_glucose = prev_row.get('glucose')
                    curr_glucose = current_row.get('glucose')
                    
                    if prev_glucose is not None and curr_glucose is not None:
                        alpha = j / (missing_points + 1)
                        new_row['glucose'] = prev_glucose + alpha * (curr_glucose - prev_glucose)
                    
                    # For insulin and carbs, use previous value (or None)
                    # This follows the principle that discrete events persist
                    new_row['insulin_slow'] = None  # Don't interpolate insulin
                    new_row['insulin_fast'] = None  # Don't interpolate insulin
                    new_row['carbs'] = None  # Don't interpolate carbs
                    new_row['exercise'] = None  # Don't interpolate exercise
                    
                    new_rows.append(new_row)
        
        # Add interpolated rows to sequence
        if new_rows:
            interpolated_df = pl.DataFrame(new_rows, schema=seq_data.schema)
            seq_data = pl.concat([seq_data, interpolated_df]).sort('datetime')
        
        return seq_data
    
    def _mark_calibration_periods(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Mark 24-hour periods after calibration gaps as SENSOR_CALIBRATION quality.
        
        According to PIPELINE.md: "In case of large gap more than 2 hours 45 minutes
        mark next 24 hours as ill quality."
        
        This method detects gaps >= CALIBRATION_GAP_THRESHOLD (2:45:00) and marks
        all data points within 24 hours after the gap end as Quality.SENSOR_CALIBRATION.
        
        Args:
            dataframe: DataFrame with sequences and interpolated gaps
            
        Returns:
            DataFrame with quality flags updated for calibration periods
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Sort by datetime to process chronologically
        df = dataframe.sort('datetime')
        
        # Calculate time differences between consecutive rows
        df = df.with_columns([
            pl.col('datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
        ])
        
        # Identify calibration gaps (>= CALIBRATION_GAP_THRESHOLD)
        df = df.with_columns([
            pl.when(pl.col('time_diff_seconds').is_null())
            .then(pl.lit(False))
            .otherwise(pl.col('time_diff_seconds') >= CALIBRATION_GAP_THRESHOLD)
            .alias('is_calibration_gap'),
        ])
        
        # Extract datetime values and gap flags before modifying DataFrame
        datetime_values = df['datetime'].to_list()
        calibration_gap_mask = df['is_calibration_gap'].to_list()
        
        # Collect calibration period start times (rows after calibration gaps)
        calibration_period_starts = []
        for i in range(len(calibration_gap_mask)):
            if calibration_gap_mask[i]:  # This row is after a calibration gap
                gap_end_time = datetime_values[i]
                calibration_period_starts.append(gap_end_time)
        
        # Create a column to mark rows that should be SENSOR_CALIBRATION
        df = df.with_columns([
            pl.lit(False).alias('in_calibration_period')
        ])
        
        # Mark all rows within 24 hours after each calibration gap
        if calibration_period_starts:
            # Create conditions for each calibration period
            conditions = []
            for gap_end_time in calibration_period_starts:
                calibration_period_end = gap_end_time + timedelta(hours=24)
                # Mark all points from gap_end_time (inclusive) for 24 hours
                conditions.append(
                    (pl.col('datetime') >= gap_end_time) &
                    (pl.col('datetime') <= calibration_period_end)
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
        df = df.with_columns([
            pl.when(pl.col('in_calibration_period'))
            .then(pl.lit(Quality.SENSOR_CALIBRATION.value))
            .otherwise(pl.col('quality'))
            .alias('quality')
        ])
        
        # Remove temporary columns
        df = df.drop(['time_diff_seconds', 'is_calibration_gap', 'in_calibration_period'])
        
        return df
    
    def prepare_for_inference(
        self,
        dataframe: UnifiedFormat,
        minimum_duration_minutes: int = MINIMUM_DURATION_MINUTES,
        maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION_MINUTES,
        glucose_only: bool = False,
        drop_duplicates: bool = False
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
           - QUALITY: contains ILL or SENSOR_CALIBRATION quality flags
           - IMPUTATION: contains imputed events (already tracked in interpolate_gaps)
           - TIME_DUPLICATES: contains non-unique time entries
        
        Returns full UnifiedFormat with all columns (sequence_id, event_type, quality, etc).
        Use to_data_only_df() to strip service columns if needed for ML models.
        
        Args:
            dataframe: Fully processed DataFrame in unified format
            minimum_duration_minutes: Minimum required sequence duration
            maximum_wanted_duration: Maximum desired sequence duration (truncates if exceeded)
            glucose_only: If True, drop non-EGV events before truncation (keeps only GLUCOSE and IMPUTATION)
            drop_duplicates: If True, drop duplicate timestamps (keeps first occurrence)
            
        Returns:
            Tuple of (unified_format_dataframe, warnings)
            
        Raises:
            ZeroValidInputError: If there are no valid data points
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("No data points in the sequence")
        
        # Check for valid glucose readings
        valid_glucose_count = dataframe.filter(
            pl.col('glucose').is_not_null()
        ).height
        
        if valid_glucose_count == 0:
            raise ZeroValidInputError("No valid glucose data points in the sequence")
        
        # Keep only the last (latest) sequence
        # Find the sequence with the most recent (maximum) timestamp
        if 'sequence_id' in dataframe.columns:
            # Get the maximum datetime for each sequence
            seq_max_times = dataframe.group_by('sequence_id').agg(
                pl.col('datetime').max().alias('max_time')
            ).sort('max_time', descending=True)
            
            # Get the sequence_id with the latest timestamp
            if len(seq_max_times) > 0:
                latest_sequence_id = seq_max_times['sequence_id'][0]
                dataframe = dataframe.filter(pl.col('sequence_id') == latest_sequence_id)
        
        # Filter to glucose-only events if requested (before truncation)
        if glucose_only:
            dataframe = dataframe.filter(
                (pl.col('event_type') == UnifiedEventType.GLUCOSE.value) |
                (pl.col('event_type') == UnifiedEventType.IMPUTATION.value)
            )
            
            # Check if we still have data after filtering
            if len(dataframe) == 0:
                raise ZeroValidInputError("No glucose data points after filtering to glucose-only events")
        
        # Truncate if exceeding maximum duration (before warning calculations)
        df_truncated = self._truncate_by_duration(
            dataframe, 
            maximum_wanted_duration
        )
        
        # Drop duplicate timestamps if requested
        if drop_duplicates:
            df_truncated = df_truncated.unique(subset=['datetime'], keep='first')
        
        # NOW calculate warnings on the truncated data
        
        # Check duration
        if len(df_truncated) > 0:
            duration_minutes = self._calculate_duration_minutes(df_truncated)
            if duration_minutes < minimum_duration_minutes:
                self._add_warning(ProcessingWarning.TOO_SHORT)
        else:
            raise ZeroValidInputError("No data points in the sequence after truncation")
        # Check for calibration events
        calibration_count = df_truncated.filter(
            pl.col('event_type') == UnifiedEventType.CALIBRATION.value
        ).height
        if calibration_count > 0:
            self._add_warning(ProcessingWarning.CALIBRATION)
        
        # Check for quality issues (ILL or SENSOR_CALIBRATION)
        quality_issues = df_truncated.filter(
            (pl.col('quality') == Quality.ILL.value) |
            (pl.col('quality') == Quality.SENSOR_CALIBRATION.value)
        ).height
        if quality_issues > 0:
            self._add_warning(ProcessingWarning.QUALITY)
        
        # Check for imputed events (may have already been added in interpolate_gaps)
        imputed_count = df_truncated.filter(
            pl.col('event_type') == UnifiedEventType.IMPUTATION.value
        ).height
        if imputed_count > 0 and ProcessingWarning.IMPUTATION not in self._warnings:
            self._add_warning(ProcessingWarning.IMPUTATION)
        
        # Check for time duplicates in the final sequence
        if len(df_truncated) > 0:
            unique_time_count = df_truncated['datetime'].n_unique()
            total_count = len(df_truncated)
            if unique_time_count < total_count:
                self._add_warning(ProcessingWarning.TIME_DUPLICATES)
        
        # Return full UnifiedFormat (keep all columns including service columns)
        # Combine warnings into flags for return value (for interface compatibility)
        combined_warnings = ProcessingWarning(0)
        for warning in self._warnings:
            combined_warnings |= warning
        
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
    def to_data_only_df(unified_df: UnifiedFormat) -> pl.DataFrame:
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
            
        Returns:
            DataFrame with only data columns (no service/metadata columns)
            
        Examples:
            >>> # After processing, strip service columns for ML model
            >>> unified_df, warnings = processor.prepare_for_inference(processed_df)
            >>> data_only_df = FormatProcessor.to_data_only_df(unified_df)
            >>> model.predict(data_only_df)
            >>> 
            >>> # Or keep full format for further analysis
            >>> unified_df, warnings = processor.prepare_for_inference(processed_df)
            >>> # Analyze quality flags, event types, etc.
            >>> quality_issues = unified_df.filter(pl.col('quality') == 'ILL')
        """
        # Extract data column names from schema definition
        data_columns = [col['name'] for col in CGM_SCHEMA.data_columns]
        return unified_df.select(data_columns)
    
    @staticmethod
    def split_glucose_events(unified_df: UnifiedFormat) -> tuple[UnifiedFormat, UnifiedFormat]:
        """Split UnifiedFormat DataFrame into glucose readings and other events.
        
        Divides a single UnifiedFormat DataFrame into two separate UnifiedFormat DataFrames:
        - Glucose DataFrame: Contains only EGV_READ and IMPUTATION events (glucose readings)
        - Events DataFrame: Contains all other event types (insulin, carbs, exercise, calibration, etc.)
        
        Both output DataFrames maintain the full UnifiedFormat schema with all columns.
        This is a non-destructive split operation - no data transformation or column coalescing.
        
        Args:
            unified_df: DataFrame in UnifiedFormat with mixed event types
            
        Returns:
            Tuple of (glucose_df, events_df) where:
            - glucose_df: UnifiedFormat DataFrame with EGV_READ and IMPUTATION events
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
        # Filter for glucose events (EGV_READ and IMPUTATION)
        glucose_df = unified_df.filter(
            (pl.col("event_type") == "EGV_READ") |
            (pl.col("event_type") == "IMPUTATION")
        )
        
        # Filter for all other events
        events_df = unified_df.filter(
            (pl.col("event_type") != "EGV_READ") &
            (pl.col("event_type") != "IMPUTATION")
        )
        
        return glucose_df, events_df

