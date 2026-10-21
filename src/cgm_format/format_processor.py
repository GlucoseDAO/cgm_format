"""CGM Data Processor Implementation.

Implements vendor-agnostic processing operations on UnifiedFormat data (Stages 4-5).
Adapted from glucose_ml_preprocessor.py for single-user unified format processing.
"""

import logging
import polars as pl
from typing import Dict, List, Tuple, ClassVar, Optional, Sequence, Union
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
    MINUTES_PER_HOUR,
    MERGE_PAIR_SLOTS,
    FOLD_MERGE_INSULIN_TOLERANCE_U,
    FOLD_MERGE_CARBS_TOLERANCE_G,
)
from cgm_format.interface.schema import CGMSchemaDefinition, EnumLiteral
from cgm_format.formats.unified import (
    UnifiedEventType,
    Quality,
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
)


# One imputed row, keyed by whatever columns the active schema declares. The
# value union spans every dtype the unified schemas use; a key is always present
# and maps to None when the row has nothing to say for that column.
InterpolatedRow = Dict[str, Union[int, float, str, datetime, None]]

logger = logging.getLogger(__name__)


class FoldSource(EnumLiteral):
    """Where `FormatProcessor.fold_insulin_onto_grid` takes boluses and carbs from.

    Basal always comes from the pump: a CGM export has no rate.
    """
    PUMP = "pump"    # the device that delivered the dose and timed the meal
    CGM = "cgm"      # what was logged in the CGM's app
    MERGE = "merge"  # both, paired within a slot threshold (see the method)


class FormatProcessor(CGMProcessor):
    """Implementation of CGMProcessor for unified format data processing.
    
    This processor handles single-user unified format data and provides:
    - Gap detection and sequence creation
    - Gap interpolation with imputation tracking
    - Inference preparation with duration checks and truncation
    - Warning collection in prepare_for_inference
    
    All methods are classmethods - no need to instantiate.
    Configuration constants can be overridden via optional method parameters.

    The processor is **schema-parameterized, never vendor-aware**: `schema`
    names the unified schema every stage validates, enforces, sorts and narrows
    against. It is read as `cls.schema` at every site, including the ones that
    enforce unconditionally regardless of `validation_mode` — those are the ones
    that silently drop columns, so a subclass that overrode only the gated ones
    would still lose data. Use `ExtendedFormatProcessor` for frames carrying the
    extended schema's columns.
    """

    # Configuration constants as ClassVars
    schema: ClassVar[CGMSchemaDefinition] = CGM_SCHEMA
    expected_interval_minutes: ClassVar[int] = EXPECTED_INTERVAL_MINUTES
    small_gap_max_minutes: ClassVar[int] = SMALL_GAP_MAX_MINUTES
    minimum_duration_minutes: ClassVar[int] = MINIMUM_DURATION_MINUTES
    maximum_wanted_duration_minutes: ClassVar[int] = MAXIMUM_WANTED_DURATION_MINUTES
    calibration_gap_threshold: ClassVar[int] = CALIBRATION_GAP_THRESHOLD
    calibration_period_hours: ClassVar[int] = CALIBRATION_PERIOD_HOURS
    snap_to_grid: ClassVar[bool] = True
    retime_glucose: ClassVar[bool] = True
    validation_mode_default: ClassVar[ValidationMethod] = ValidationMethod.INPUT
    fold_source: ClassVar[FoldSource] = FoldSource.PUMP
    fold_basal: ClassVar[bool] = False
    merge_pair_slots: ClassVar[int] = MERGE_PAIR_SLOTS


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
            cls.schema.validate_dataframe(df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
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
            cls.schema.validate_dataframe(df_marked, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return df_marked
        
    @classmethod
    def synchronize_timestamps(
        cls,
        dataframe: UnifiedFormat,
        expected_interval_minutes: Optional[int] = None,
        retime_glucose: Optional[bool] = None,
        validation_mode: Optional[ValidationMethod] = None
    ) -> UnifiedFormat:
        """Align timestamps to the sequence grid, re-timing glucose onto it.

        Every source row keeps its place — this maps each row's `datetime` to
        its nearest grid point and never adds or drops one. What it also does,
        when `retime_glucose` is on, is answer the question the timestamp change
        raises: a reading taken at 14:11:45 and re-labelled 14:11:00 is no
        longer a reading *of* 14:11:00. So `glucose` is recomputed as the
        measured series evaluated at the grid instant, and the row is flagged
        `Quality.GRID_RETIMED`. The device's own number stays in
        `original_glucose`, which is what the next pass reads — re-timing an
        already re-timed frame is a no-op rather than a drift.

        Call after `interpolate_gaps()`, or before it; the two share one grid
        and one interpolant, so the order does not matter.

        Args:
            dataframe: DataFrame in unified format (should already have sequences created)
            expected_interval_minutes: Expected interval in minutes (defaults to cls.expected_interval_minutes)
            retime_glucose: Recompute glucose at the grid instant (defaults to
                cls.retime_glucose, True). Pass False to keep each reading's
                measured value under its new timestamp, which is what this
                method did before 0.12.0.
            validation_mode: Validation mode (defaults to cls.validation_mode_default)

        Returns:
            DataFrame with synchronized timestamps at fixed intervals

        Raises:
            ZeroValidInputError: If dataframe is empty or has no data
        """
        if len(dataframe) == 0:
            raise ZeroValidInputError("Cannot synchronize timestamps on empty dataframe")

        if expected_interval_minutes is None:
            expected_interval_minutes = cls.expected_interval_minutes
        if retime_glucose is None:
            retime_glucose = cls.retime_glucose
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
            cls.schema.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # Process each sequence separately
        unique_sequences = dataframe['sequence_id'].unique().to_list()
        synchronized_sequences = []
        
        for seq_id in unique_sequences:
            # Sort by original_datetime for idempotent processing
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort(['sequence_id', 'original_datetime', 'quality'])
            
            # A one-row sequence used to take a shortcut here — round the
            # timestamp with `dt.round('1m')` and skip the rest — which quietly
            # exempted it from the SYNCHRONIZATION flag as well, and from
            # GRID_RETIMED once that existed. The general path handles it
            # correctly and identically: the grid starts at that row's own
            # rounded timestamp, so it lands on offset 0, and a series of one
            # anchor evaluates to that anchor everywhere. There is nothing left
            # for the special case to buy.
            synced_seq = cls._synchronize_sequence(
                seq_data, seq_id, expected_interval_minutes, retime_glucose
            )
            synchronized_sequences.append(synced_seq)
        
        # Combine all sequences with stable sorting from schema definition
        result_df = pl.concat(synchronized_sequences).sort(cls.schema.get_stable_sort_keys())
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            cls.schema.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
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
    def _measured_glucose_anchors(cls, seq_data: pl.DataFrame) -> pl.DataFrame:
        """The measured glucose series of one sequence, as interpolation anchors.

        Two columns — `original_datetime` and `original_glucose` — because those
        are the only two things no processing stage ever writes after parse.
        Anchoring on them is what makes re-timing a pure function of the source
        data instead of a function of the previous pass's output.

        Membership is decided by the IMPUTATION flag, never by whether
        `original_glucose` happens to be null: an imputed row carries its own
        invented value there (see `_build_interpolated_row`), so null-testing
        would silently promote invented points to anchors.

        Duplicate `original_datetime` is normal — a Libre scan lands on a
        historic minute several hundred times per export — so the survivor is
        first-occurrence after an explicit sort rather than whatever `unique()`
        felt like returning.
        """
        anchors = seq_data.filter(
            (pl.col('event_type') == UnifiedEventType.GLUCOSE.value)
            & ((pl.col('quality') & Quality.IMPUTATION.value) == 0)
            & pl.col('original_glucose').is_not_null()
            & pl.col('original_datetime').is_not_null()
        )

        return (
            anchors
            .select(['original_datetime', 'original_glucose'])
            .sort(['original_datetime', 'original_glucose'])
            .unique(subset=['original_datetime'], keep='first')
            .sort('original_datetime')
        )

    @classmethod
    def _glucose_at(cls, anchors: pl.DataFrame, targets: pl.Series) -> pl.Series:
        """Evaluate the measured glucose series at arbitrary instants.

        Piecewise-linear between the two anchors bracketing each target, and
        **clamped** outside the anchor range — the value at either end is that
        end's reading, never an extrapolation. This is the one place a glucose
        number is derived for a grid instant; `interpolate_gaps` and the
        re-timing half of `synchronize_timestamps` both come here, so the value
        at a grid point cannot depend on which stage produced its row.

        Clamping is not a detail. `glucose_data_processing` picks its two points
        by absolute time distance without requiring them to bracket the target,
        so where both fall on the same side its weights sum to more than one and
        it emits a mirrored pseudo-extrapolation. That is a defect, not a
        convention worth matching: measured against the committed GDP reference,
        clamped interpolation reproduces the inference window to within
        0.0005 mg/dL and only diverges where GDP produced that artifact.

        Args:
            anchors: Output of `_measured_glucose_anchors` — sorted, deduped.
            targets: Instants to evaluate at. Order is preserved.

        Returns:
            One Float64 value per target. All-null when there are no anchors.
        """
        if anchors.height == 0 or len(targets) == 0:
            return pl.Series('glucose', [None] * len(targets), dtype=pl.Float64)

        probe = pl.DataFrame({'_t': targets}).with_columns(
            pl.col('_t').cast(anchors['original_datetime'].dtype)
        ).with_row_index('_order')

        # join_asof needs sorted keys on both sides; _order restores the caller's.
        probe_sorted = probe.sort('_t')

        bracketed = (
            probe_sorted
            .join_asof(
                anchors.rename({'original_datetime': '_t0', 'original_glucose': '_g0'}),
                left_on='_t', right_on='_t0', strategy='backward',
            )
            .join_asof(
                anchors.rename({'original_datetime': '_t1', 'original_glucose': '_g1'}),
                left_on='_t', right_on='_t1', strategy='forward',
            )
        )

        span = (pl.col('_t1') - pl.col('_t0')).dt.total_nanoseconds()
        elapsed = (pl.col('_t') - pl.col('_t0')).dt.total_nanoseconds()

        value = (
            pl.when(pl.col('_g0').is_null()).then(pl.col('_g1'))       # before the first anchor
            .when(pl.col('_g1').is_null()).then(pl.col('_g0'))         # after the last anchor
            .when(span <= 0).then(pl.col('_g0'))                       # target sits on an anchor
            .otherwise(pl.col('_g0') + (elapsed / span) * (pl.col('_g1') - pl.col('_g0')))
            .cast(pl.Float64)
            .alias('glucose')
        )

        return bracketed.with_columns(value).sort('_order')['glucose']

    @classmethod
    def _synchronize_sequence(
        cls,
        seq_data: pl.DataFrame,
        seq_id: int,
        expected_interval_minutes: int,
        retime_glucose: bool
    ) -> pl.DataFrame:
        """Map one sequence's rows onto its grid, optionally re-timing glucose.

        Lossless in rows: every source row survives with a new `datetime`. The
        grid itself is not materialized — each row computes its own grid offset
        from `original_datetime`, which is why a second pass lands on exactly
        the same instants.

        Args:
            seq_data: Sequence data as Polars DataFrame
            seq_id: Sequence ID (carried by the rows; kept for call-site clarity)
            expected_interval_minutes: Expected interval in minutes
            retime_glucose: Recompute glucose at each row's grid instant

        Returns:
            Sequence with grid-aligned timestamps
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

        if retime_glucose:
            result = cls._retime_glucose_onto_grid(result)

        # Sync is lossless - keep ALL rows, no deduplication
        # The only exception would be replacing imputed rows with real ones,
        # but that's handled during interpolation, not here

        # Ensure column order matches unified format
        result = cls.schema.validate_columns(result, enforce=True)

        return result

    @classmethod
    def _retime_glucose_onto_grid(cls, seq_data: pl.DataFrame) -> pl.DataFrame:
        """Rewrite each glucose row's value as the measured series at its new instant.

        Applies to every `EGV_READ` row, imputed ones included. That is what
        makes the two grid stages agree: a point invented by `interpolate_gaps`
        and a measured reading snapped onto the neighbouring grid point are both
        just "the series, evaluated here", so neither stage can disagree with
        the other about what belongs at a grid instant.

        A no-op when the sequence has no measured anchors — a `sequence_id = 0`
        group of insulin and carb events with no glucose to attach to, or a
        sequence whose glucose rows are entirely imputed. Withholding is the
        honest answer there; inventing one from imputed rows is not.
        """
        anchors = cls._measured_glucose_anchors(seq_data)
        if anchors.height == 0:
            return seq_data

        is_glucose = pl.col('event_type') == UnifiedEventType.GLUCOSE.value
        retimed = cls._glucose_at(anchors, seq_data['datetime'])

        return seq_data.with_columns([
            pl.when(is_glucose).then(retimed).otherwise(pl.col('glucose')).alias('glucose'),
            pl.when(is_glucose)
            .then(pl.col('quality') | pl.lit(Quality.GRID_RETIMED.value))
            .otherwise(pl.col('quality'))
            .alias('quality'),
        ])
    
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
            cls.schema.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
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
        result_df = pl.concat(processed_sequences).sort(cls.schema.get_stable_sort_keys())
        
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            cls.schema.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
 
    
    @classmethod
    def _build_interpolated_row(
        cls,
        seq_id: int,
        interpolated_time: datetime,
        glucose: float,
        quality: int,
    ) -> InterpolatedRow:
        """Build one imputed glucose row shaped by `cls.schema`.

        Derived from the schema rather than restated beside it: a literal dict
        of the six core columns would silently drop every extended column when
        the frame is wider than the core schema, and `pl.DataFrame(rows,
        schema=...)` would then be handed rows whose keys do not cover the
        frame. Every column the row does not carry is null — an interpolated
        point says nothing about carbs, macros or heart rate, and null is how
        the schema spells "did not say".
        """
        row: InterpolatedRow = {name: None for name in cls.schema.get_column_names()}
        row['sequence_id'] = seq_id
        row['event_type'] = UnifiedEventType.GLUCOSE.value
        row['quality'] = quality
        row['original_datetime'] = interpolated_time
        row['datetime'] = interpolated_time
        row['glucose'] = glucose
        # A created point is its own origin, so its anchors are itself — the same
        # reasoning that sets original_datetime above. Leaving this null instead
        # would let the parser's coalesce fill it from `glucose` on a CSV
        # round-trip, which changes the frame a round-trip is supposed to preserve.
        # It is never read as an anchor regardless: the IMPUTATION flag excludes it.
        row['original_glucose'] = glucose
        return row

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
        
        # Always use original_datetime for gap detection.
        # Grid snapping can both inflate and compress gaps (e.g. 19.98 min raw → 15 min
        # on grid), so only the absolute reference frame gives correct threshold decisions.
        # The snap_to_grid branch below handles WHERE to place interpolated points.
        glucose_events_sorted = glucose_events.sort('original_datetime')

        time_diffs = glucose_events_sorted['original_datetime'].diff().dt.total_seconds() / 60.0
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

        # WHERE each created point goes, and what quality it inherits. WHAT
        # value it carries is decided afterwards, in one pass over the measured
        # series — see below.
        placements: List[Tuple[datetime, int]] = []

        for gap_idx, time_diff_minutes in small_gaps:
            prev_row = glucose_list[gap_idx - 1]
            current_row = glucose_list[gap_idx]

            # Always use original_datetime as absolute reference
            prev_dt = prev_row['original_datetime']

            # Quality combines flags from both neighbours + IMPUTATION, plus
            # SYNCHRONIZATION when the point is placed on the grid.
            prev_quality = prev_row.get('quality', 0) or 0
            curr_quality = current_row.get('quality', 0) or 0
            combined_quality = prev_quality | curr_quality | Quality.IMPUTATION.value

            if snap_to_grid:
                # Snap to sequence grid: determine ALL grid points that should exist in the gap
                # CRITICAL: Use the ROUNDED grid positions, not the original timestamps
                # This ensures we fill gaps between where timestamps WILL BE after rounding
                current_dt = current_row['original_datetime']
                prev_grid_dt = cls.calculate_grid_point(prev_dt, grid_start, expected_interval_minutes, 'nearest')
                curr_grid_dt = cls.calculate_grid_point(current_dt, grid_start, expected_interval_minutes, 'nearest')

                prev_grid_pos = int((prev_grid_dt - grid_start).total_seconds() / 60.0 / expected_interval_minutes)
                curr_grid_pos = int((curr_grid_dt - grid_start).total_seconds() / 60.0 / expected_interval_minutes)

                # Fill all grid points BETWEEN the rounded positions (exclusive on both ends)
                for grid_pos in range(prev_grid_pos + 1, curr_grid_pos):
                    placements.append((
                        grid_start + timedelta(minutes=grid_pos * expected_interval_minutes),
                        combined_quality | Quality.SYNCHRONIZATION.value,
                    ))
            else:
                # Non-grid logic: place points at regular intervals from previous timestamp
                missing_points = int(time_diff_minutes / expected_interval_minutes) - 1

                for j in range(1, missing_points + 1):
                    placements.append((
                        prev_dt + timedelta(minutes=expected_interval_minutes * j),
                        combined_quality,
                    ))

        # One evaluation of the measured series for every created point, through
        # the same helper `synchronize_timestamps` uses. Sharing it is what makes
        # the two stages commute: whichever runs first, a given instant gets the
        # same number, so the second stage has nothing left to disagree with.
        anchors = cls._measured_glucose_anchors(seq_data)
        target_times = pl.Series(
            '_target',
            [placement for placement, _ in placements],
            dtype=seq_data['original_datetime'].dtype,
        )
        interpolated_values = cls._glucose_at(anchors, target_times).to_list()

        new_rows = [
            cls._build_interpolated_row(
                seq_id=seq_id,
                # datetime and original_datetime are the same for a newly
                # created point — it has no earlier position to preserve.
                interpolated_time=placement,
                glucose=value,
                quality=quality,
            )
            for (placement, quality), value in zip(placements, interpolated_values)
        ]

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
            sort_keys = [k for k in cls.schema.get_stable_sort_keys() if k != 'sequence_id']
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
            cls.schema.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
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
            cls.schema.validate_dataframe(df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
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
            cls.schema.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
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

        # Timestamps moved onto the grid, and/or glucose re-timed onto it. Both
        # mean the number under a given instant is not the number the device
        # stamped with that instant, which is a thing the caller is entitled to
        # know before feeding it to a model.
        synchronized_count = df_truncated.filter(
            (pl.col('quality') & (Quality.SYNCHRONIZATION.value | Quality.GRID_RETIMED.value)) != 0
        ).height
        if synchronized_count > 0:
            warnings.append(ProcessingWarning.SYNCHRONIZATION)

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
            cls.schema.validate_dataframe(df_truncated, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
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
        
        Data columns are computed from `cls.schema`, so an ExtendedFormatProcessor
        keeps the extended data columns (including `annotations`). Use
        `to_core_df()` first if you want the core six regardless of schema.
        For the core schema this is:
        - datetime: Timestamp of the reading
        - glucose: Blood glucose value (mg/dL)
        - carbs: Carbohydrate intake (grams)
        - insulin_slow: Slow-acting insulin dose (units)
        - insulin_fast: Fast-acting insulin dose (units)
        - exercise: Exercise indicator/intensity
        
        Args:
            unified_df: DataFrame in UnifiedFormat with all columns
            drop_service_columns: If True, drop service columns (sequence_id, event_type, quality)
            drop_duplicates: If True, collapse rows sharing a timestamp into one row,
                keeping the first non-null value of each column. A glucose reading and an
                insulin dose that land on the same grid point merge into a single wide row
                instead of one discarding the other.
            glucose_only: If True, drop non-EGV events before truncation (keeps only GLUCOSE)
            validation_mode: Validation mode (defaults to cls.validation_mode_default)

        Returns:
            DataFrame with only data columns (no service/metadata columns)
            
        """
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        # Verify input dataframe matches schema
        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            cls.schema.validate_dataframe(unified_df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # Filter to glucose-only events if requested (before truncation)
        if glucose_only:
            unified_df, _ = cls.split_glucose_events(unified_df, validation_mode)

        # Collapse rows sharing a timestamp if requested
        if drop_duplicates:
            unified_df = cls._coalesce_duplicate_timestamps(unified_df)

        if drop_service_columns:
            data_columns = [col['name'] for col in cls.schema.data_columns]
            unified_df = unified_df.select(data_columns)
        #no Output validation - is not unified format
        return unified_df

    # Columns holding an amount: two entries on one grid point add up. Every other column keeps its
    # first non-null value.
    COALESCE_SUMMED_COLUMNS: ClassVar[Tuple[str, ...]] = ("carbs", "insulin_slow", "insulin_fast")

    @classmethod
    def _coalesce_duplicate_timestamps(cls, unified_df: pl.DataFrame) -> pl.DataFrame:
        """Merge rows sharing a datetime into one row.

        Events are stored one per row, so a glucose reading and an insulin dose that fall on
        the same grid point occupy two rows. Dropping one of them silently discarded whichever
        sorted second; merging keeps both in the single wide row that ML models expect. An
        amount (`COALESCE_SUMMED_COLUMNS`) is summed: two boluses on one grid point are both
        given. A sum over nothing recorded stays null, never 0. Every other column keeps its
        first non-null value.
        """
        if unified_df.height == 0:
            return unified_df

        def merged(col: str) -> pl.Expr:
            if col in cls.COALESCE_SUMMED_COLUMNS:
                return pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).otherwise(None).alias(col)
            return pl.col(col).drop_nulls().first().alias(col)

        other_columns = [col for col in unified_df.columns if col != 'datetime']
        return (
            unified_df
            .group_by('datetime', maintain_order=True)
            .agg([merged(col) for col in other_columns])
            .select(unified_df.columns)
            .sort('datetime')
        )

    @classmethod
    def to_core_df(cls, unified_df: UnifiedFormat) -> UnifiedFormat:
        """Narrow a frame to the core CGM_SCHEMA shape.

        The escape hatch for consumers that only speak the six core data
        columns. An extended frame carries columns they have no code for, and
        `ExtraColumnError` deliberately refuses to pretend otherwise — so the
        way across the seam is an explicit, named narrowing rather than a
        relaxed check.

        This is **lossy by design and by name**: every extended column is
        dropped. Rows are not: enforcement adds, casts and reorders columns and
        stable-sorts, it never filters. A row whose only content was a
        macronutrient or an annotation survives as a row with null data.

        Targets CGM_SCHEMA unconditionally, *not* `cls.schema` — "narrow to
        core" is the whole meaning of the method, so following an overridden
        schema would make it a no-op on exactly the frames it exists for.

        Args:
            unified_df: DataFrame in any unified shape (core or extended)

        Returns:
            DataFrame conforming to CGM_SCHEMA
        """
        return CGM_SCHEMA.validate_dataframe(unified_df, enforce=True)

    @classmethod
    def to_ml_ready_df(
            cls,
            unified_df: UnifiedFormat,
            user_id: str = "Subject 000",
            round_precision: int = 3,
            basal_rate_from_insulin_slow: bool = False,
            validation_mode: Optional[ValidationMethod] = None
        ) -> pl.DataFrame:
        """Render a processed frame in the SugarOne model's input shape.

        SugarOne (`GlucoseDAO/glucose-forecasting`) consumes a fixed-frequency wide grid
        with display column names, one row per (sequence_id, timestamp):

            sequence_id, Timestamp, Event Type, User ID, Glucose (mg/dL),
            Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g),
            Recommended Split, Study Group

        `Recommended Split` and `Study Group` are training-time split bookkeeping; they are
        emitted empty because inference has no split to recommend.

        **`Basal Rate (U/h)` is left empty by default.** SugarOne's basal is a continuous pump
        rate in units per hour; `insulin_slow` is a discrete long-acting injection in units.
        They are not the same quantity, and writing one into the other would feed a 26 U
        injection to the model as a 26 U/h infusion rate. Pass
        `basal_rate_from_insulin_slow=True` only if the deployment has established that its
        `insulin_slow` really is a rate. `evaluate_model.py` fills missing covariates with 0.0,
        so an empty column degrades to a covariate-free prediction rather than an error.

        Args:
            unified_df: Processed DataFrame in unified format (post-`prepare_for_inference`)
            user_id: Value for the `User ID` column
            round_precision: Decimal places for glucose, matching training's `round_precision`
            basal_rate_from_insulin_slow: Map `insulin_slow` into `Basal Rate (U/h)` despite
                the unit mismatch described above
            validation_mode: Validation mode (defaults to cls.validation_mode_default)

        Returns:
            DataFrame with SugarOne display columns, one row per grid timestamp
        """
        if validation_mode is None:
            validation_mode = cls.validation_mode_default

        if validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            cls.schema.validate_dataframe(unified_df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # One row per grid point, covariates merged onto it.
        collapsed = cls._coalesce_duplicate_timestamps(unified_df)

        basal = (
            pl.col('insulin_slow') if basal_rate_from_insulin_slow
            else pl.lit(None, dtype=pl.Float64)
        )

        return collapsed.select([
            pl.col('sequence_id'),
            pl.col('datetime').dt.strftime('%Y-%m-%dT%H:%M:%S').alias('Timestamp'),
            cls._ml_event_type_expr().alias('Event Type'),
            pl.lit(user_id).alias('User ID'),
            pl.col('glucose').round(round_precision).alias('Glucose (mg/dL)'),
            basal.alias('Basal Rate (U/h)'),
            pl.col('insulin_fast').alias('Bolus Insulin (U)'),
            pl.col('carbs').alias('Carbohydrates (g)'),
            pl.lit('').alias('Recommended Split'),
            pl.lit('').alias('Study Group'),
        ])

    # Unified event codes -> the Event Type labels the training exports carry.
    ML_EVENT_TYPE_LABELS: ClassVar[Dict[str, str]] = {
        UnifiedEventType.GLUCOSE.value: 'EGV',
        UnifiedEventType.INSULIN_FAST.value: 'Insulin',
        UnifiedEventType.INSULIN_SLOW.value: 'Insulin',
        UnifiedEventType.CARBOHYDRATES.value: 'Carbs',
        UnifiedEventType.CALIBRATION.value: 'Calibration',
        UnifiedEventType.IMPUTATION.value: 'Interpolated',
        UnifiedEventType.HEALTH_ILLNESS.value: 'Health',
        UnifiedEventType.HEALTH_STRESS.value: 'Health',
        UnifiedEventType.HEALTH_LOW_SYMPTOMS.value: 'Health',
        UnifiedEventType.HEALTH_CYCLE.value: 'Health',
        UnifiedEventType.HEALTH_ALCOHOL.value: 'Health',
        UnifiedEventType.EXERCISE_LIGHT.value: 'Exercise',
        UnifiedEventType.EXERCISE_MEDIUM.value: 'Exercise',
        UnifiedEventType.EXERCISE_HEAVY.value: 'Exercise',
    }

    @classmethod
    def _ml_event_type_expr(cls) -> pl.Expr:
        """Translate unified event codes to training-export Event Type labels."""
        expr = pl.col('event_type')
        mapped = pl.when(expr == UnifiedEventType.GLUCOSE.value).then(pl.lit('EGV'))
        for code, label in cls.ML_EVENT_TYPE_LABELS.items():
            if code == UnifiedEventType.GLUCOSE.value:
                continue
            mapped = mapped.when(expr == code).then(pl.lit(label))
        # Anything else keeps its unified code rather than being silently blanked.
        return mapped.otherwise(expr)

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
            cls.schema.validate_dataframe(unified_df, enforce=validation_mode & ValidationMethod.INPUT_FORCED)
        
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
            cls.schema.validate_dataframe(glucose_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
            cls.schema.validate_dataframe(events_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
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
            cls.schema.validate_dataframe(dataframe, enforce=validation_mode & ValidationMethod.INPUT_FORCED)

        # IDEMPOTENCY: Reset existing sequence_id to 0 to ensure consistent results
        # This allows re-running sequence detection with different gap thresholds
        # Get canonical sequence_id dtype from schema
        sequence_id_dtype = next(
            col['dtype'] for col in cls.schema.service_columns if col['name'] == 'sequence_id'
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
            result_df = cls.schema.validate_dataframe(result_df, enforce=True)
        
        # Verify output dataframe matches schema
        if validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            cls.schema.validate_dataframe(result_df, enforce=validation_mode & ValidationMethod.OUTPUT_FORCED)
        
        return result_df
    
    @classmethod
    def _split_sequences_with_internal_gaps(
        cls,
        dataframe: pl.DataFrame,
        large_gap_threshold_seconds: float,
        sequence_id_dtype: pl.DataType
    ) -> pl.DataFrame:
        """DEPRECATED: marked for deletion.

        Gap-aware splitting is handled by detect_and_assign_sequences.
        This stub exists so that any unexpected caller surfaces loudly.
        """
        raise NotImplementedError(
            "_split_sequences_with_internal_gaps is deprecated. "
            "Use detect_and_assign_sequences instead."
        )


    # ===== Folding a pump onto a CGM's grid =====

    @classmethod
    def fold_insulin_onto_grid(
        cls,
        cgm: UnifiedFormat,
        pump: Union[UnifiedFormat, Sequence[UnifiedFormat]],
        source: Optional[Union[FoldSource, str]] = None,
        merge_pair_slots: Optional[int] = None,
        fold_basal: Optional[bool] = None,
        expected_interval_minutes: Optional[int] = None,
    ) -> pl.DataFrame:
        """Give each grid point of a CGM frame the insulin and carbs a pump recorded around it.

        `cgm` is a synchronized unified frame (`synchronize_timestamps`), whose
        glucose grid this reads and never changes; `pump` is a parsed pump frame
        (`MEDTRONIC_PUMP`), or several (monthly exports, which may overlap), which
        this integrates. The pump is never bundled with the CGM or synchronized:
        its rows would move the CGM's grid (F19 in docs/dogfooding.md).

        One output row per grid point of each CGM sequence, first to last, every
        point included: `sequence_id`, `datetime`, `quality` and `glucose` from
        the CGM's glucose row there (null where it has none; nothing is
        invented), then per bin `[t - I/2, t + I/2)`, `I` the interval:

        - `insulin_fast`: boluses starting in the bin, plus each spread bolus's
          share (`insulin_duration`) of the bin it overlaps. 0 where the pump
          covered the bin and delivered none; null where no source covers it.
        - `basal_rate`: the pump's basal rate averaged over the bin, U/h. Null
          unless the pump's rate is known across the whole bin; 0 during a
          suspend. With `fold_basal`, the bin's basal dose, `rate x I /
          MINUTES_PER_HOUR` for a constant rate, is added into `insulin_fast`
          instead of being kept apart (the models read the two apart).
        - `carbs`: summed in the bin; null where no entry was recorded.
        - `insulin_slow`: the CGM frame's long-acting doses in the bin.

        `source` picks where boluses and carbs come from (`FoldSource`): the pump
        (default), the CGM frame's own entries, or both merged. Under `MERGE`,
        each CGM entry pairs with at most one pump entry of the same kind at
        most `merge_pair_slots` grid slots away, nearest in time first; a pair
        lands on the pump entry's slot, and beyond the threshold both are kept.
        A 0 or null on either side means that source did not say, so the other
        side is taken; two non-zero values are averaged, and named in a warning
        when they differ by more than a tolerance.

        What is not attached to any row is said in one warning with its total:
        entries outside every sequence, the source not chosen, pump rows
        repeated across overlapping exports, basal in bins the pump only partly
        covered. The output is a terminal view without `event_type`, so it
        cannot be folded again; the inputs are left untouched.

        Args:
            cgm: Synchronized unified frame with glucose readings
            pump: Parsed pump frame, or a sequence of them
            source: Where boluses and carbs come from (defaults to cls.fold_source)
            merge_pair_slots: Pairing threshold under MERGE (defaults to cls.merge_pair_slots)
            fold_basal: Add the basal dose into insulin_fast (defaults to cls.fold_basal)
            expected_interval_minutes: Grid interval (defaults to cls.expected_interval_minutes)

        Returns:
            One row per grid point: sequence_id, datetime, quality, glucose, carbs,
            insulin_slow, insulin_fast, basal_rate

        Raises:
            ValueError: If `cgm` is not a synchronized unified frame, or no pump frame is given
            ZeroValidInputError: If `cgm` has no glucose readings
        """
        source = FoldSource(source if source is not None else cls.fold_source)
        pair_slots = cls.merge_pair_slots if merge_pair_slots is None else merge_pair_slots
        fold_basal = cls.fold_basal if fold_basal is None else fold_basal
        interval = expected_interval_minutes or cls.expected_interval_minutes
        pumps = [pump] if isinstance(pump, pl.DataFrame) else list(pump)
        if not pumps:
            raise ValueError("fold_insulin_onto_grid needs at least one pump frame")
        if "event_type" not in cgm.columns or "original_datetime" not in cgm.columns:
            raise ValueError(
                "fold_insulin_onto_grid needs a unified CGM frame (with event_type and "
                "original_datetime); a folded frame cannot be folded again"
            )

        bins = cls._fold_bins(cgm, interval)
        notes: List[str] = []

        pump_rows = cls._fold_pump_rows(pumps, notes)
        coverage = cls._fold_coverage(pumps)

        def entries(frame: pl.DataFrame, event: UnifiedEventType, value: str, at: str) -> pl.DataFrame:
            out = frame.filter((pl.col("event_type") == event.value) & pl.col(value).is_not_null())
            duration = (
                pl.col("insulin_duration") if "insulin_duration" in out.columns
                else pl.lit(None, dtype=pl.Float64)
            )
            return out.select(
                pl.col(at).alias("at"),
                pl.col(value).cast(pl.Float64).alias("amount"),
                duration.cast(pl.Float64).alias("minutes"),
            ).sort("at", "amount")

        pump_bolus = entries(pump_rows, UnifiedEventType.INSULIN_FAST, "insulin_fast", "datetime")
        pump_carbs = entries(pump_rows, UnifiedEventType.CARBOHYDRATES, "carbs", "datetime")
        cgm_bolus = entries(cgm, UnifiedEventType.INSULIN_FAST, "insulin_fast", "original_datetime")
        cgm_carbs = entries(cgm, UnifiedEventType.CARBOHYDRATES, "carbs", "original_datetime")
        cgm_slow = entries(cgm, UnifiedEventType.INSULIN_SLOW, "insulin_slow", "original_datetime")

        if source == FoldSource.PUMP:
            bolus, carbs = pump_bolus, pump_carbs
            cls._fold_note_unused("the CGM frame's", cgm_bolus, cgm_carbs, notes)
        elif source == FoldSource.CGM:
            bolus, carbs = cgm_bolus, cgm_carbs
            cls._fold_note_unused("the pump's", pump_bolus, pump_carbs, notes)
        else:
            bolus = cls._fold_merge(
                bins, pump_bolus, cgm_bolus, pair_slots, FOLD_MERGE_INSULIN_TOLERANCE_U, "U", "insulin", interval, notes)
            carbs = cls._fold_merge(
                bins, pump_carbs, cgm_carbs, pair_slots, FOLD_MERGE_CARBS_TOLERANCE_G, "g", "carbs", interval, notes)

        bolus_per_bin = cls._fold_amounts(bins, bolus, interval, "insulin_fast", "U of bolus insulin", notes)
        carbs_per_bin = cls._fold_amounts(bins, carbs, interval, "carbs", "g of carbs", notes)
        slow_per_bin = cls._fold_amounts(bins, cgm_slow, interval, "insulin_slow", "U of long-acting insulin", notes)
        basal_per_bin = cls._fold_basal_doses(bins, pump_rows, coverage, interval, notes)

        bins = bins.with_columns(
            cls._fold_covered(bins, coverage, interval) if source != FoldSource.CGM
            else pl.lit(False).alias("_covered")
        )
        out = (
            bins.join(bolus_per_bin, on=["sequence_id", "datetime"], how="left")
            .join(carbs_per_bin, on=["sequence_id", "datetime"], how="left")
            .join(slow_per_bin, on=["sequence_id", "datetime"], how="left")
            .join(basal_per_bin, on=["sequence_id", "datetime"], how="left")
            .with_columns(
                pl.when(pl.col("insulin_fast").is_null() & pl.col("_covered"))
                .then(pl.lit(0.0))
                .otherwise(pl.col("insulin_fast"))
                .alias("insulin_fast")
            )
        )
        if fold_basal:
            lost = out.filter(pl.col("basal_dose").is_null() & (pl.col("insulin_fast") > 0))
            if lost.height:
                notes.append(
                    f"{lost['insulin_fast'].sum():g} U of bolus insulin in {lost.height} bin(s) where "
                    "basal is unknown, so their folded insulin_fast is null"
                )
            out = out.with_columns((pl.col("insulin_fast") + pl.col("basal_dose")).alias("insulin_fast"))
        out = out.with_columns(
            (pl.col("basal_dose") * MINUTES_PER_HOUR / interval).alias("basal_rate")
        ).select(
            "sequence_id", "datetime", "quality", "glucose",
            "carbs", "insulin_slow", "insulin_fast", "basal_rate",
        ).sort("sequence_id", "datetime")

        if notes:
            logger.warning("fold_insulin_onto_grid: not attached to any row: %s.", "; ".join(notes))
        return out

    # The input columns of the models trained in GlucoseDAO/glucose_forecasting_metabonet, in order, from
    # that repo's src/common/data/columns.py (COL_SEQ, COL_TS_SHORT, COL_EVENT, COL_USER, COL_GLU,
    # COL_BASAL, COL_BOLUS, COL_CARB, COL_SPLIT, COL_GROUP) as SugarOne's trainer reads them.
    SUGARONE_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "sequence_id", "Timestamp", "Event Type", "User ID", "Glucose (mg/dL)",
        "Basal Rate (U/h)", "Bolus Insulin (U)", "Carbohydrates (g)",
        "Recommended Split", "Study Group",
    )

    @classmethod
    def to_sugarone_ready_df(
        cls,
        folded: pl.DataFrame,
        user_id: str = "Subject 000",
        round_precision: int = 3,
    ) -> pl.DataFrame:
        """Render `fold_insulin_onto_grid` output in the models' input shape.

        The columns and their order are those the models trained in
        `glucose_forecasting_metabonet` read (`SUGARONE_COLUMNS`), with basal and
        bolus as separate channels: `Basal Rate (U/h)` from `basal_rate`, `Bolus
        Insulin (U)` from `insulin_fast`. Nulls stay null, because the training
        loader fills them itself (basal carried forward, bolus zero-filled as an
        event) and a 0 would claim none was given. `Event Type` is `EGV` for a
        reading, `Interpolated` for an imputed one (the trainer's
        `drop_interpolated` filters on it), and empty where the grid point has no
        glucose. `Recommended Split` and `Study Group` are training bookkeeping,
        emitted empty.

        `to_ml_ready_df` stays for frames without a pump: it reads a unified frame
        and cannot carry a basal rate.

        Args:
            folded: Output of `fold_insulin_onto_grid`
            user_id: Value for the `User ID` column
            round_precision: Decimal places for glucose

        Returns:
            DataFrame with `SUGARONE_COLUMNS`, one row per grid point

        Raises:
            ValueError: If `folded` is not a fold's output
        """
        needed = {"sequence_id", "datetime", "quality", "glucose", "carbs", "insulin_fast", "basal_rate"}
        missing = needed - set(folded.columns)
        if missing or "event_type" in folded.columns:
            raise ValueError(
                "to_sugarone_ready_df takes the output of fold_insulin_onto_grid; "
                + (f"missing {sorted(missing)}" if missing else "got a unified frame (it has event_type)")
            )
        imputed = (pl.col("quality").fill_null(0) & Quality.IMPUTATION.value) != 0
        event = (
            pl.when(pl.col("glucose").is_null()).then(pl.lit(""))
            .when(imputed).then(pl.lit("Interpolated"))
            .otherwise(pl.lit("EGV"))
        )
        names = cls.SUGARONE_COLUMNS
        return folded.sort("sequence_id", "datetime").select(
            pl.col("sequence_id").alias(names[0]),
            pl.col("datetime").dt.strftime("%Y-%m-%dT%H:%M:%S").alias(names[1]),
            event.alias(names[2]),
            pl.lit(user_id).alias(names[3]),
            pl.col("glucose").round(round_precision).alias(names[4]),
            pl.col("basal_rate").alias(names[5]),
            pl.col("insulin_fast").alias(names[6]),
            pl.col("carbs").alias(names[7]),
            pl.lit("").alias(names[8]),
            pl.lit("").alias(names[9]),
        )

    @classmethod
    def _fold_bins(cls, cgm: UnifiedFormat, interval: int) -> pl.DataFrame:
        """Every grid point of every CGM sequence, with the glucose row there if any."""
        readings = (
            cgm.filter((pl.col("event_type") == UnifiedEventType.GLUCOSE.value) & pl.col("glucose").is_not_null())
            .sort("sequence_id", "datetime", "quality")
            .unique(subset=["sequence_id", "datetime"], keep="first", maintain_order=True)
            .select("sequence_id", "datetime", "quality", "glucose")
        )
        if readings.height == 0:
            raise ZeroValidInputError("fold_insulin_onto_grid: the CGM frame has no glucose readings")
        step_ms = interval * 60_000
        spans = (
            readings.group_by("sequence_id")
            .agg(pl.col("datetime").min().alias("first"), pl.col("datetime").max().alias("last"))
            .sort("first")
        )
        offgrid = readings.join(spans, on="sequence_id").filter(
            (pl.col("datetime") - pl.col("first")).dt.total_milliseconds() % step_ms != 0
        )
        if offgrid.height:
            raise ValueError(
                f"fold_insulin_onto_grid: {offgrid.height} glucose reading(s) are off the "
                f"{interval}-minute grid; synchronize_timestamps the CGM frame first"
            )
        grid = (
            spans.with_columns(
                pl.int_ranges(
                    0, (pl.col("last") - pl.col("first")).dt.total_milliseconds() // step_ms + 1
                ).alias("k")
            )
            .explode("k", empty_as_null=False)  # never empty: a sequence has at least its first point
            .select(
                "sequence_id",
                (pl.col("first") + pl.duration(milliseconds=pl.col("k") * step_ms)).alias("datetime"),
            )
        )
        return grid.join(readings, on=["sequence_id", "datetime"], how="left").sort("sequence_id", "datetime")

    @classmethod
    def _fold_pump_rows(cls, pumps: List[UnifiedFormat], notes: List[str]) -> pl.DataFrame:
        """All pump frames as one, rows repeated across overlapping exports dropped once."""
        wanted = ["datetime", "event_type", "insulin_fast", "insulin_duration", "basal_rate", "carbs"]
        frames = [
            frame.select(
                [pl.col(c) if c in frame.columns else pl.lit(None, dtype=pl.Float64).alias(c) for c in wanted]
            ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
            for frame in pumps
        ]
        rows = pl.concat(frames, how="vertical_relaxed")
        unique = rows.unique(maintain_order=True)
        if unique.height < rows.height:
            notes.append(f"{rows.height - unique.height} pump row(s) repeated across overlapping exports, counted once")
        return unique.sort("datetime", "event_type", maintain_order=True)

    @classmethod
    def _fold_coverage(cls, pumps: List[UnifiedFormat]) -> pl.DataFrame:
        """The spans the pump frames cover, overlapping ones merged: [start, end]."""
        spans = pl.DataFrame(
            {
                "start": [frame["datetime"].min() for frame in pumps],
                "end": [frame["datetime"].max() for frame in pumps],
            },
            schema={"start": pl.Datetime("ms"), "end": pl.Datetime("ms")},
        ).sort("start", "end")
        return (
            spans.with_columns(
                (pl.col("start") > pl.col("end").cum_max().shift(1)).fill_null(True).cum_sum().alias("_group")
            )
            .group_by("_group")
            .agg(pl.col("start").min(), pl.col("end").max())
            .sort("start")
            .select("start", "end")
        )

    @classmethod
    def _fold_covered(cls, bins: pl.DataFrame, coverage: pl.DataFrame, interval: int) -> pl.Series:
        """`_covered`: whether each bin lies wholly inside one pump span, in `bins` row order."""
        half = pl.duration(milliseconds=interval * 60_000 // 2)
        return (
            bins.select("datetime").with_row_index("_row")
            .with_columns((pl.col("datetime") - half).alias("_lo"), (pl.col("datetime") + half).alias("_hi"))
            .sort("_lo")
            .join_asof(coverage, left_on="_lo", right_on="start", strategy="backward")
            .with_columns(((pl.col("_lo") >= pl.col("start")) & (pl.col("_hi") <= pl.col("end"))).fill_null(False).alias("_covered"))
            .sort("_row")["_covered"]
        )

    @classmethod
    def _fold_place(cls, bins: pl.DataFrame, entries: pl.DataFrame, interval: int) -> pl.DataFrame:
        """Each entry's sequence and slot: the grid point whose bin holds its `at` (null when none does)."""
        step_ms = interval * 60_000
        spans = (
            bins.group_by("sequence_id")
            .agg(pl.col("datetime").min().alias("_first"), pl.col("datetime").max().alias("_last"))
            .with_columns((pl.col("_first") - pl.duration(milliseconds=step_ms // 2)).alias("_edge"))
            .sort("_edge")
        )
        placed = (
            entries.with_row_index("_entry")
            .sort("at")
            .join_asof(spans, left_on="at", right_on="_edge", strategy="backward")
            .with_columns(
                (((pl.col("at") - pl.col("_first")).dt.total_milliseconds() + step_ms // 2) // step_ms).alias("_k")
            )
        )
        inside = pl.col("_k").is_between(0, ((pl.col("_last") - pl.col("_first")).dt.total_milliseconds() // step_ms))
        return placed.with_columns(
            pl.when(inside).then(pl.col("sequence_id")).otherwise(None).alias("sequence_id"),
            pl.when(inside).then(pl.col("_k")).otherwise(None).alias("slot"),
            pl.when(inside)
            .then(pl.col("_first") + pl.duration(milliseconds=pl.col("_k") * step_ms))
            .otherwise(None)
            .alias("datetime"),
        ).sort("_entry").drop("_entry", "_first", "_last", "_edge", "_k")

    @classmethod
    def _fold_amounts(
        cls, bins: pl.DataFrame, entries: pl.DataFrame, interval: int, column: str, what: str, notes: List[str]
    ) -> pl.DataFrame:
        """Sum entries per bin. One with `minutes` is delivered evenly over them: a rate for that long,
        integrated over each bin it overlaps, exactly as basal is."""
        key = {"sequence_id": bins.schema["sequence_id"], "datetime": pl.Datetime("ms")}
        spread = pl.col("minutes").fill_null(0) > 0
        instant = cls._fold_place(bins, entries.filter(~spread).select("at", "amount"), interval)
        outside = instant.filter(pl.col("sequence_id").is_null())
        outside_units = outside["amount"].sum() if outside.height else 0.0
        per_bin = (
            instant.filter(pl.col("sequence_id").is_not_null())
            .group_by("sequence_id", "datetime")
            .agg(pl.col("amount").sum().alias(column))
            .select(pl.col("sequence_id").cast(key["sequence_id"]), "datetime", column)
        )
        spreading = entries.filter(spread)
        if spreading.height:
            ms = (pl.col("minutes") * 60_000).round(0).cast(pl.Int64)
            per_hour = pl.col("amount") / pl.col("minutes") * MINUTES_PER_HOUR
            steps = (
                pl.concat([
                    spreading.select(pl.col("at").alias("datetime"), per_hour.alias("_delta")),
                    spreading.select((pl.col("at") + pl.duration(milliseconds=ms)).alias("datetime"), (-per_hour).alias("_delta")),
                ])
                .group_by("datetime").agg(pl.col("_delta").sum())
                .sort("datetime")
                .with_columns(pl.col("_delta").cum_sum().alias("_rate"), pl.col("datetime").shift(-1).alias("_end"))
                .filter(pl.col("_end").is_not_null())
            )
            doses = cls._fold_integrate(bins, steps.select("datetime", "_end", "_rate"), interval)
            spread_bins = doses.select("sequence_id", "datetime", pl.col("_dose").alias(column)).filter(pl.col(column) > 0)
            outside_units += spreading["amount"].sum() - spread_bins[column].sum()
            per_bin = (
                pl.concat([per_bin, spread_bins], how="vertical_relaxed")
                .group_by("sequence_id", "datetime").agg(pl.col(column).sum())
            )
        if outside_units > 1e-9:
            notes.append(f"{outside_units:g} {what} outside every glucose sequence")
        return per_bin.select(pl.col("sequence_id").cast(key["sequence_id"]), pl.col("datetime").cast(key["datetime"]), column)

    @classmethod
    def _fold_integrate(cls, bins: pl.DataFrame, segments: pl.DataFrame, interval: int) -> pl.DataFrame:
        """Integrate a step rate over every bin.

        `segments` is `datetime`, `_end`, `_rate` (U/h, null = unknown), sorted and non-overlapping. The
        running integral F and the running known minutes K are piecewise linear between segment edges,
        so a bin `[lo, hi)` gets `_dose` = F(hi) - F(lo) units and `_known` = K(hi) - K(lo) minutes.
        """
        segments = (
            segments.with_columns(
                ((pl.col("_end") - pl.col("datetime")).dt.total_milliseconds().clip(lower_bound=0) / 60_000).alias("_minutes")
            )
            .with_columns(
                (pl.col("_rate").fill_null(0) * pl.col("_minutes") / MINUTES_PER_HOUR).alias("_units"),
                pl.when(pl.col("_rate").is_not_null()).then(pl.col("_minutes")).otherwise(0.0).alias("_known_minutes"),
            )
            .with_columns(
                (pl.col("_units").cum_sum() - pl.col("_units")).alias("_units_before"),
                (pl.col("_known_minutes").cum_sum() - pl.col("_known_minutes")).alias("_known_before"),
            )
            .select("datetime", "_end", "_rate", "_units_before", "_known_before")
        )
        half = pl.duration(milliseconds=interval * 60_000 // 2)
        edges = bins.select("sequence_id", "datetime").with_row_index("_row").with_columns(
            (pl.col("datetime") - half).alias("lo"), (pl.col("datetime") + half).alias("hi")
        )

        def running(edge: str) -> pl.DataFrame:
            within = (pl.min_horizontal(pl.col("x"), pl.col("_end")) - pl.col("datetime")).dt.total_milliseconds() / 60_000
            return (
                edges.select("_row", pl.col(edge).alias("x")).sort("x")
                .join_asof(segments, left_on="x", right_on="datetime", strategy="backward")
                .select(
                    "_row",
                    (pl.col("_units_before") + pl.col("_rate").fill_null(0) * within / MINUTES_PER_HOUR).fill_null(0.0).alias(f"F_{edge}"),
                    (pl.col("_known_before") + pl.when(pl.col("_rate").is_not_null()).then(within).otherwise(0.0)).fill_null(0.0).alias(f"K_{edge}"),
                )
            )

        return (
            edges.join(running("lo"), on="_row").join(running("hi"), on="_row")
            .sort("_row")
            .select(
                "sequence_id", "datetime",
                (pl.col("F_hi") - pl.col("F_lo")).alias("_dose"),
                (pl.col("K_hi") - pl.col("K_lo")).alias("_known"),
            )
        )

    @classmethod
    def _fold_basal_doses(
        cls, bins: pl.DataFrame, pump_rows: pl.DataFrame, coverage: pl.DataFrame, interval: int, notes: List[str]
    ) -> pl.DataFrame:
        """Each bin's basal dose, U: the rate integrated over it, null unless known throughout.

        A rate is in force until the next change or the end of the pump span it starts in, whichever
        comes first; a gap between spans, or a rate the pump logged as unknown, is not known.
        """
        changes = (
            pump_rows.filter(pl.col("event_type") == UnifiedEventType.BASAL_RATE.value)
            .select("datetime", "basal_rate")
            .unique(subset="datetime", keep="last", maintain_order=True)
            .sort("datetime")
        )
        if changes.height == 0:
            notes.append("no basal rate in any pump frame, so basal_rate is null throughout")
            return bins.select("sequence_id", "datetime", pl.lit(None, dtype=pl.Float64).alias("basal_dose"))
        segments = (
            changes.join_asof(coverage, left_on="datetime", right_on="start", strategy="backward")
            .with_columns(pl.min_horizontal(pl.col("datetime").shift(-1), pl.col("end")).fill_null(pl.col("end")).alias("_end"))
            .select("datetime", "_end", pl.col("basal_rate").alias("_rate"))
        )
        doses = cls._fold_integrate(bins, segments, interval)
        whole = pl.col("_known") >= interval - 1e-9
        partial = doses.filter(~whole & (pl.col("_known") > 1e-9))
        if partial.height:
            notes.append(
                f"{partial['_dose'].sum():g} U of basal in {partial.height} bin(s) the pump's rate covers "
                "only in part, so their basal_rate is null"
            )
        return doses.select(
            "sequence_id", "datetime", pl.when(whole).then(pl.col("_dose")).otherwise(None).alias("basal_dose")
        )

    @classmethod
    def _fold_note_unused(cls, whose: str, bolus: pl.DataFrame, carbs: pl.DataFrame, notes: List[str]) -> None:
        if bolus.height:
            notes.append(f"{bolus['amount'].sum():g} U in {bolus.height} of {whose} bolus entries (source not chosen)")
        if carbs.height:
            notes.append(f"{carbs['amount'].sum():g} g in {carbs.height} of {whose} carb entries (source not chosen)")

    @classmethod
    def _fold_merge(
        cls,
        bins: pl.DataFrame,
        pump: pl.DataFrame,
        cgm: pl.DataFrame,
        max_slots: int,
        tolerance: float,
        unit: str,
        kind: str,
        interval: int,
        notes: List[str],
    ) -> pl.DataFrame:
        """Pair CGM entries with pump entries at most `max_slots` apart, one to one, nearest first."""
        near_slots = SMALL_GAP_MAX_MINUTES // interval
        p = cls._fold_place(bins, pump, interval).with_row_index("_p")
        c = cls._fold_place(bins, cgm, interval).with_row_index("_c")
        candidates = (
            p.filter(pl.col("slot").is_not_null())
            .join(c.filter(pl.col("slot").is_not_null()), on="sequence_id", suffix="_cgm")
            .with_columns(
                (pl.col("slot") - pl.col("slot_cgm")).abs().alias("_slots"),
                (pl.col("at") - pl.col("at_cgm")).dt.total_milliseconds().abs().alias("_apart"),
            )
            .filter(pl.col("_slots") <= max(near_slots, max_slots))
            .sort("_apart", "at", "at_cgm", "_p", "_c")
        )
        # Greedy, nearest first: a pairing is inherently sequential.
        used_p: set[int] = set()
        used_c: set[int] = set()
        pairs: List[Tuple[int, int]] = []
        for row in candidates.filter(pl.col("_slots") <= max_slots).iter_rows(named=True):
            if row["_p"] in used_p or row["_c"] in used_c:
                continue
            used_p.add(row["_p"]); used_c.add(row["_c"])
            pairs.append((row["_p"], row["_c"]))
        near = candidates.filter(
            ~pl.col("_p").is_in(list(used_p)) & ~pl.col("_c").is_in(list(used_c))
        )["_c"].n_unique()
        if near:
            notes.append(
                f"{near} {kind} entr(ies) in the CGM frame unpaired but within {near_slots} slots of an "
                f"unpaired pump entry (merge_pair_slots={max_slots}): both kept"
            )
        resolved: List[Dict[str, Union[datetime, float, None]]] = []
        disagreements: List[str] = []
        for pi, ci in pairs:
            prow = p.row(pi, named=True); crow = c.row(ci, named=True)
            a, b = prow["amount"], crow["amount"]
            if not a:
                value = b
            elif not b:
                value = a
            else:
                value = (a + b) / 2
                if abs(a - b) > tolerance:
                    disagreements.append(f"{prow['at']} pump {a:g} {unit} / {crow['at']} cgm {b:g} {unit}")
            resolved.append({"at": prow["at"], "amount": value, "minutes": prow["minutes"]})
        if disagreements:
            logger.warning(
                "fold_insulin_onto_grid: %d merged %s pair(s) differ by more than %g %s, averaged: %s",
                len(disagreements), kind, tolerance, unit, "; ".join(disagreements),
            )
        keep_p = p.filter(~pl.col("_p").is_in([pi for pi, _ in pairs])).select("at", "amount", "minutes")
        keep_c = c.filter(~pl.col("_c").is_in([ci for _, ci in pairs])).select("at", "amount", "minutes")
        merged = pl.DataFrame(
            resolved, schema={"at": pl.Datetime("ms"), "amount": pl.Float64, "minutes": pl.Float64}
        )
        return pl.concat([merged, keep_p, keep_c], how="vertical_relaxed").sort("at", "amount")

class ExtendedFormatProcessor(FormatProcessor):
    """FormatProcessor targeting the extended unified schema.

    Identical behavior to FormatProcessor, with one thing changed: the schema.
    Every stage validates, enforces, sorts and narrows against
    CGM_SCHEMA_EXTENDED, so macronutrients, wearable streams, ketones and
    annotations survive sequence detection, interpolation and synchronization
    instead of being dropped by the unconditional enforcement inside them.

    Note the widened total ordering. `get_stable_sort_keys()` returns every
    column, so the extended schema's appended columns are appended tie-breakers.
    Row order is still fully deterministic, but a core frame and its extended
    counterpart are not guaranteed to order identically where earlier keys tie.

    `annotations` is a data column and therefore part of both the sort keys and
    the primary key. Serialize it only with
    `cgm_format.formats.unified.annotations_to_json`, whose deterministic output
    is what keeps that ordering stable across runs.
    """

    schema: ClassVar[CGMSchemaDefinition] = CGM_SCHEMA_EXTENDED
