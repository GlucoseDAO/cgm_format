"""Format converter for CGM vendor formats working on text data."""

import logging
from typing import Dict, List, Tuple, Union, ClassVar, Optional
from io import StringIO
from typing import Union
import polars as pl
from pathlib import Path
from base64 import b64decode

logger = logging.getLogger(__name__)

from cgm_format.formats.d1namo import (
    D1NAMO_DATE_FORMATS,
    D1NAMO_FOOD_DATETIME_FORMATS,
    D1NAMO_GLUCOSE_SCHEMA,
    D1NAMO_NULL_LITERALS,
    D1NAMO_SENSOR_TYPES,
    D1NAMO_TIME_FORMATS,
    D1namoFoodColumn,
    D1namoGlucoseColumn,
    D1namoHealthyFoodColumn,
    D1namoInsulinColumn,
)
from cgm_format.formats.cgmacros import (
    CGMACROS_IGNORED_COLUMNS,
    CGMACROS_MEAL_TYPE_NORMALIZATION,
    CGMACROS_MEAN_TRACK,
    CGMACROS_METS_SCALE,
    CGMACROS_OPTIONAL_COLUMNS,
    CGMACROS_SCHEMA,
    CGMACROS_TIMESTAMP_FORMATS,
    CGMACROS_TRACKS,
    CGMacrosColumn,
)
from cgm_format.formats.supported import (
    FORMAT_DETECTION_PATTERNS,
    DETECTION_LINE_COUNT,
    PATH_DETECTION_PROBES,
    UNIFIED_TARGET_SCHEMA,
)
from cgm_format.interface.schema import CGMSchemaDefinition
from cgm_format.interface.cgm_interface import (
    CGMParser,
    SupportedCGMFormat,
    UnifiedFormat,
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    MultiTrackSourceError,
    ValidationMethod,
    truncate_error_message,
)

# Import detection patterns from format modules
from cgm_format.formats.unified import (
    UnifiedEventType,
    Quality,
    UNIFIED_TIMESTAMP_FORMATS,
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    UNIT_CONVERSIONS,
    CANONICAL_GLUCOSE_UNIT,
)
from cgm_format.formats.dexcom import (
    DexcomColumn,
    DEXCOM_METADATA_LINES,
    DEXCOM_TIMESTAMP_FORMATS,
    DEXCOM_HIGH_GLUCOSE_DEFAULT,
    DEXCOM_LOW_GLUCOSE_DEFAULT,
    DEXCOM_SCHEMA,
)
from cgm_format.formats.dexcom_eu import (
    DexcomEUColumn,
    DEXCOM_EU_METADATA_LINES,
    DEXCOM_EU_SCHEMA,
)
from cgm_format.formats.libre import (
    LibreColumn,
    LibreRecordType,
    LIBRE_HEADER_LINE,
    LIBRE_TIMESTAMP_FORMATS,
    LIBRE_SCHEMA,
)
from cgm_format.formats.libre_eu import (
    LibreEUColumn,
    LIBRE_EU_SCHEMA,
)
from cgm_format.formats.medtronic import (
    MedtronicColumn,
    MEDTRONIC_TIMESTAMP_FORMATS,
    MEDTRONIC_REQUIRED_HEADERS,
    MEDTRONIC_CSV_SEPARATOR,
    MEDTRONIC_SCHEMA_OVERRIDES_UTF8,
)
from cgm_format.formats.nightscout import (
    NightscoutEntryColumn,
    NightscoutTreatmentColumn,
    NightscoutTreatmentEventType,
    NIGHTSCOUT_TIMESTAMP_FORMATS,
    NIGHTSCOUT_TREATMENTS_DETECTION_PATTERNS,
)

# Common encoding artifacts and their fixes 
UTF8_BOM = b'\xef\xbb\xbf'
ENCODING_ARTIFACTS = {
    # Double-encoded BOM in quotes: "ïººº¿"
    b'\x22\xc3\xaf\xc2\xbb\xc2\xbf\x22': UTF8_BOM,
    # Triple-encoded BOM (enterprise nightmare)
    b'\x22\xc3\x83\xc2\xaf\xc3\x82\xc2\xbb\xc3\x82\xc2\xbf\x22': UTF8_BOM,
    # Double-encoded BOM without quotes
    b'\xc3\xaf\xc2\xbb\xc2\xbf': UTF8_BOM,
    # Quoted BOM (some systems do this)
    b'\x22\xef\xbb\xbf\x22': UTF8_BOM,
}



class FormatParser(CGMParser):
    """Main format parser implementing the CGMParser interface.
    
    This class orchestrates the parsing pipeline from raw data to unified format:
    1. Decode raw data (remove BOM, fix encoding)
    2. Detect format (determine vendor)
    3. Parse to unified format (vendor-specific processing)
    """
    
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT
    detection_line_count: ClassVar[int] = DETECTION_LINE_COUNT
    cgm_detection_patterns: ClassVar[Dict[SupportedCGMFormat, List[str]]] = FORMAT_DETECTION_PATTERNS
    path_detection_probes: ClassVar[Dict[SupportedCGMFormat, Tuple[str, ...]]] = (
        PATH_DETECTION_PROBES
    )
    # Widest first: merge_bundle_frames canonicalizes a merged frame's column
    # order against these, so a bundle of a core and an extended member sorts
    # by the extended declaration order rather than by concat order.
    unified_schemas: ClassVar[Tuple[CGMSchemaDefinition, ...]] = (
        CGM_SCHEMA_EXTENDED,
        CGM_SCHEMA,
    )
    # ===== STAGE 1: Preprocess Raw Data =====
    
    @classmethod
    def decode_raw_data(cls, raw_data: Union[bytes, str]) -> str:
        """Remove BOM marks, encoding artifacts, and other junk from raw input.
       
        Args:
            raw_data: Raw file contents (bytes or string)
            
        Returns:
            Cleaned string data ready for format detection
        """
        # If already a string, return as-is
        if isinstance(raw_data, str):
            return raw_data
        
        # Normalize encoding artifacts
        normalized = raw_data
        for corrupted_pattern, proper_bom in ENCODING_ARTIFACTS.items():
            if normalized.startswith(corrupted_pattern):
                normalized = proper_bom + normalized[len(corrupted_pattern):]
                break
        
        # Decode with utf-8-sig to handle BOM
        text = normalized.decode('utf-8-sig', errors='replace')
        
        return text
    
    # ===== STAGE 2: Format Detection  =====
    
    @classmethod
    def detect_format(cls, text_data: str) -> SupportedCGMFormat:
        """Guess the vendor format based on header patterns in raw CSV string.
        
        This determines which vendor-specific processor to use.
        Works on string data before parsing to avoid vendor-specific CSV quirks.
        
        Detection strategy:
        1. Check for unified format patterns first (highest priority)
        2. Check for Dexcom patterns
        3. Check for Libre patterns
        4. Raise UnknownFormatError if no match
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            SupportedCGMFormat enum value 
            
        Raises:
            UnknownFormatError: If format cannot be determined
        """

        # Check first N lines for format indicators
        lines = text_data.split('\n',cls.detection_line_count+1)[:cls.detection_line_count]
        
        # Check each CGM type's patterns
        for cgm_type, patterns in cls.cgm_detection_patterns.items():
            if any(pattern in line for line in lines for pattern in patterns):
                return cgm_type
        
        error_msg = f"Unknown CGM data format. Sample lines: {lines[:3]}"
        raise UnknownFormatError(cls._truncate_error_message(error_msg))

    @classmethod
    def detect_path_format(cls, root: Union[str, Path]) -> SupportedCGMFormat:
        """Identify a directory-shaped source by the files it contains.

        The path-shaped counterpart to :meth:`detect_format`. A bundle or a
        corpus has no single text to sniff — what identifies it is the shape of
        the directory: whether a per-subject CSV sits beside its own folder,
        whether a named subset directory exists. Sniffing the first file found
        would be worse than useless, because a corpus member's *contents* often
        look like a plain vendor export.

        Contract mirrors `detect_format` deliberately: iterate the registry in
        insertion order, first match wins, raise `UnknownFormatError` on no
        match. A format matches when **every** one of its probes finds at least
        one path — probes are conjunctive, because a single glob is rarely
        specific enough to identify a corpus and an accidental match here sends
        a whole directory tree to the wrong parser.

        Args:
            root: Directory to identify. Not a file.

        Returns:
            The registered format whose probes all match.

        Raises:
            UnknownFormatError: If `root` is not a directory, or no registered
                format's probes all match.
        """
        root_path = Path(root)
        if not root_path.is_dir():
            raise UnknownFormatError(
                cls._truncate_error_message(
                    f"Path-shaped detection needs a directory, got: {root_path}"
                )
            )

        for cgm_type, probes in cls.path_detection_probes.items():
            if probes and all(
                any(root_path.glob(probe)) for probe in probes
            ):
                return cgm_type

        error_msg = (
            f"Unknown directory-shaped CGM source: {root_path}. "
            f"Checked {len(cls.path_detection_probes)} registered path format(s)."
        )
        raise UnknownFormatError(cls._truncate_error_message(error_msg))

    @classmethod
    def format_supported(cls, raw_data: Union[bytes, str]) -> bool:
        """Check if the library can parse the given data format.
        
        Uses the detector to determine if the format is supported without parsing the data.
        
        Args:
            raw_data: Raw file contents (bytes or string)
            
        Returns:
            True if format is supported and can be parsed, False otherwise
        """
        try:
            text_data = cls.decode_raw_data(raw_data)
            cls.detect_format(text_data)
            return True
        except (UnknownFormatError, MalformedDataError, Exception):
            return False


    # ===== STAGE 3: Device-Specific Parsing to Unified Format =====
    
    @classmethod
    def parse_to_unified(cls, text_data: str, format_type: SupportedCGMFormat) -> UnifiedFormat:
        """Parse vendor-specific CSV to unified format (device-specific parsing).
        
        This stage combines:
        - CSV validation and sanity checks
        - Vendor-specific quirk handling (High/Low values, timezone fixes, etc.)
        - Column mapping to unified schema
        - Populating service fields (sequence_id, event_type, quality)
        
        Delegates to format-specific parsers:
        - DexcomParser for DEXCOM format
        - LibreParser for LIBRE format
        - UnifiedParser for UNIFIED_CGM format (passthrough with validation)
        
        After this stage, processing flow converges to UnifiedFormat.
        
        Args:
            text_data: Preprocessed string data
            format_type: Detected vendor format
            
        Returns:
            DataFrame in unified format matching CGM_SCHEMA
            
        Raises:
            MalformedDataError: If CSV is unparseable, zero valid rows, or conversion fails
        """

        if format_type in (SupportedCGMFormat.UNIFIED_CGM, SupportedCGMFormat.UNIFIED_EXTENDED):
            # The unified family is the one place the target schema varies, so
            # it is resolved from the registry rather than defaulted (D2).
            unified_df = cls._process_unified(
                text_data, target_schema=UNIFIED_TARGET_SCHEMA[format_type]
            )
        elif format_type == SupportedCGMFormat.DEXCOM:
            unified_df = cls._process_dexcom(text_data)
        elif format_type == SupportedCGMFormat.DEXCOM_EU:
            unified_df = cls._process_dexcom(text_data, european=True)
        elif format_type == SupportedCGMFormat.LIBRE:
            unified_df = cls._process_libre(text_data)
        elif format_type == SupportedCGMFormat.LIBRE_EU:
            unified_df = cls._process_libre(text_data, european=True)
        elif format_type == SupportedCGMFormat.MEDTRONIC:
            unified_df = cls._process_medtronic(text_data)
        elif format_type == SupportedCGMFormat.NIGHTSCOUT:
            unified_df = cls._process_nightscout(text_data)
        elif format_type == SupportedCGMFormat.CGMACROS:
            # A CGMacros file carries two independent sensor series, so there
            # is no single frame to return. Refusing here rather than picking
            # one is the whole point of D5.
            raise MultiTrackSourceError(
                "CGMacros files carry two independent glucose series "
                f"({' and '.join(CGMACROS_TRACKS)}), so there is no single "
                "unified frame to return — picking one silently would hide "
                "which sensor the caller got. Use "
                "FormatParser.parse_tracks(path) for a frame per sensor, or "
                f"parse_tracks(path, track={CGMACROS_MEAN_TRACK!r}) for the "
                "opt-in synthetic mean."
            )
        else:
            raise UnknownFormatError(f"Unknown CGM data format: {format_type}")
        
        # Final validation before emitting is done by postprocessing step
        return unified_df

    
    # ===== Private: Format-Specific Processing Methods =====
    @classmethod
    def _truncate_error_message(cls, message: str, max_length: Optional[int] = None) -> str:
        """Truncate error message to prevent huge CSV dumps in logs.
        
        Args:
            message: Original error message
            max_length: Maximum length in bytes (default 8192)
            
        Returns:
            Truncated error message with indicator if truncated
        """
        if max_length is None:
            return truncate_error_message(message)
        else:
            return truncate_error_message(message, max_length)


    @classmethod
    def _postprocess_unified(
        cls,
        unified_df: UnifiedFormat,
        schema: Optional[CGMSchemaDefinition] = None,
    ) -> UnifiedFormat:
        """Postprocess the unified format dataframe.

        Every vendor parse path converges here, so this is where the unified
        contract is enforced — unconditionally, not under `validation_mode`.

        Args:
            unified_df: DataFrame in unified format
            schema: Target unified schema to enforce. Defaults to CGM_SCHEMA.
                A parser for a source carrying macronutrients, wearable streams
                or annotations passes CGM_SCHEMA_EXTENDED (looked up in
                UNIFIED_TARGET_SCHEMA); enforcing the core schema here would
                drop those columns before any caller saw them.
        """
        if schema is None:
            schema = CGM_SCHEMA

        if len(unified_df) == 0:
            raise ZeroValidInputError("No valid data rows found after processing")

        # Populate original_datetime from datetime (preserve original timestamps)
        # This must be done before detect_and_assign_sequences which uses original_datetime
        if 'original_datetime' not in unified_df.columns:
            unified_df = unified_df.with_columns([
                pl.col('datetime').alias('original_datetime')
            ])

        # Sort by datetime
        unified_df = unified_df.sort("datetime")
        
        # Enforce canonical unified schema for idempotent roundtrips
        # Part of the processing pipline, not affected by validation mode!!!!
        unified_df = schema.validate_dataframe(unified_df, enforce=True)
        
        # Mark duplicate timestamps - moved to processor
        # Detect and assign sequences - moved to processor (requires gap size knowledge)
        # Initialize sequence_id column to 0 (unassigned) for processor to fill in
        if 'sequence_id' not in unified_df.columns:
            unified_df = unified_df.with_columns([
                pl.lit(0).alias('sequence_id')
            ])

        return unified_df
    
    @classmethod
    def _probe_timestamp_format(cls, df: pl.DataFrame, column_name: str, formats: tuple) -> str:
        """Probe which timestamp format works for this file.
        
        Args:
            df: DataFrame with timestamp column
            column_name: Name of the timestamp column
            formats: Tuple of format strings to try
            
        Returns:
            The first format string that successfully parses
            
        Raises:
            MalformedDataError: If no format works
        """
        # Get first non-null, non-blank timestamp value for probing.
        # Blank strings ("") can survive filtering when a vendor leaves the
        # timestamp field empty (e.g. Dexcom metadata/alert rows), so exclude
        # them explicitly — probing an empty string would spuriously fail.
        sample = (
            df.filter(
                pl.col(column_name).is_not_null()
                & (pl.col(column_name).cast(pl.Utf8).str.strip_chars() != "")
            )
            .select(column_name)
            .head(1)
        )
        if len(sample) == 0:
            raise MalformedDataError("No timestamp values found for format probing")
        
        # Try each format
        for fmt in formats:
            try:
                sample.select(pl.col(column_name).str.strptime(pl.Datetime, fmt, strict=True))
                return fmt  # This format works!
            except:
                continue  # Try next format
        
        error_msg = f"Could not parse timestamps with any known format: {formats}"
        raise MalformedDataError(cls._truncate_error_message(error_msg))

    @classmethod
    def _to_canonical_unit(
        cls, col_expr: pl.Expr, source_unit: str | None, target_unit: str
    ) -> pl.Expr:
        """Scale a value expression from its source unit to the canonical unit.

        Declarative: the factor is looked up in UNIT_CONVERSIONS keyed by
        (source_unit, target_unit). Identity when the units are equal or when no
        factor is registered — so a mg/dL glucose column passes through untouched
        while a mmol/L one is multiplied, with no per-variant branch in the parser.
        """
        if source_unit is None or source_unit == target_unit:
            return col_expr
        factor = UNIT_CONVERSIONS.get((source_unit, target_unit))
        return col_expr * factor if factor is not None else col_expr

    @classmethod
    def _glucose_to_canonical(
        cls, schema: CGMSchemaDefinition, source_column: str, expr: pl.Expr
    ) -> pl.Expr:
        """Scale a vendor glucose column to mg/dL using the unit declared in its schema."""
        return cls._to_canonical_unit(
            expr, schema.get_unit(source_column), CANONICAL_GLUCOSE_UNIT
        )

    @classmethod
    def _process_unified(
        cls,
        text_data: str,
        target_schema: Optional[CGMSchemaDefinition] = None,
    ) -> UnifiedFormat:
        """Process data already in unified format (validation only).

        Serves both UNIFIED_CGM and UNIFIED_EXTENDED — they are the same CSV
        geometry, differing only in how many data columns the header carries,
        so the format identity is expressed purely as the target schema.

        Args:
            text_data: CSV string in unified format
            target_schema: Unified schema to enforce (defaults to CGM_SCHEMA)

        Returns:
            Validated DataFrame in unified format

        Raises:
            MalformedDataError: If validation fails
        """
        try:
            df = pl.read_csv(
                StringIO(text_data),
                truncate_ragged_lines=True,
                infer_schema_length=None,
                ignore_errors=False
            )
            
            # Clean column names
            df = df.rename({col: col.strip().replace('"', '').replace('"', '') for col in df.columns})
            
            # Validate we have some data
            if len(df) == 0:
                raise ZeroValidInputError("No valid data rows found")
            
            for column in ['datetime', 'original_datetime']:
                if column not in df.columns:
                    continue #original_datetime is not always present in old files
                # Parse datetime column if it's a string (applies to data loaded from CSV)
                if df[column].dtype == pl.Utf8 or df[column].dtype == pl.String:
                    timestamp_format = FormatParser._probe_timestamp_format(df, column, UNIFIED_TIMESTAMP_FORMATS)
                    df = df.with_columns([
                        pl.col(column).str.strptime(pl.Datetime("ms"), timestamp_format)
                    ])

            return cls._postprocess_unified(df, schema=target_schema)
            
        except pl.exceptions.PolarsError as e:
            error_msg = f"Failed to parse unified format CSV: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))
    
    @classmethod
    def _process_dexcom(
        cls,
        text_data: str,
        european: bool = False,
        high_glucose_value: int = DEXCOM_HIGH_GLUCOSE_DEFAULT,
        low_glucose_value: int = DEXCOM_LOW_GLUCOSE_DEFAULT
    ) -> UnifiedFormat:
        """Process Dexcom CSV to unified format.

        Args:
            text_data: Dexcom CSV string
            european: If True, glucose is in mmol/L and will be converted to mg/dL
            high_glucose_value: Value to replace 'High' readings (default 401 mg/dL)
            low_glucose_value: Value to replace 'Low' readings (default 39 mg/dL)

        Returns:
            DataFrame in unified format

        Raises:
            MalformedDataError: If parsing fails
        """
        Col = DexcomEUColumn if european else DexcomColumn
        eff_schema = DEXCOM_EU_SCHEMA if european else DEXCOM_SCHEMA
        metadata_lines = DEXCOM_EU_METADATA_LINES if european else DEXCOM_METADATA_LINES
        expected_metadata_rows = len(metadata_lines)
        try:
            # Dexcom/Clarity layout: Row 1 = column headers, then a number of
            # metadata rows (FirstName, LastName, Device, an optional Sensor
            # row, and one row per configured Alert), then the data rows.
            #
            # Static minimal skip: skip the known number of metadata rows for
            # this format variant (10 for standard mg/dL exports, 11 for the EU
            # mmol/L exports that include a "Sensor" row).
            df = pl.read_csv(
                StringIO(text_data),
                skip_rows_after_header=expected_metadata_rows,  # Skip known metadata rows
                truncate_ragged_lines=True,  # Handle Dexcom's variable-length rows
                infer_schema_length=None,
                ignore_errors=False
            )

            # Clean column names
            df = df.rename({col: col.strip().replace('“', '').replace('”', '').replace('"', '') for col in df.columns})

            # Absorb benign header drift (any registered alias -> canonical name)
            df = eff_schema.normalize_headers(df)

            # Dynamic post-handler: these proprietary Clarity exports are not
            # perfectly stable — newer G6/G7 exports emit an extra metadata row
            # (e.g. "Sensor") beyond the static count, which then survives the
            # skip above. Every real data row carries a Timestamp while metadata
            # rows leave it blank, so drop any blank-timestamp rows that slipped
            # through and warn if the real metadata length differs from expected.
            rows_before = len(df)
            df = df.filter(
                pl.col(Col.TIMESTAMP).is_not_null()
                & (pl.col(Col.TIMESTAMP).cast(pl.Utf8).str.strip_chars() != "")
            )
            extra_metadata_rows = rows_before - len(df)
            if extra_metadata_rows > 0:
                logger.warning(
                    "Dexcom%s export metadata length mismatch: expected %d metadata "
                    "row(s) after the header but found %d extra blank-timestamp "
                    "row(s); dropped them dynamically. The Clarity export format may "
                    "have changed.",
                    " EU" if european else "",
                    expected_metadata_rows,
                    extra_metadata_rows,
                )

            # Probe timestamp format once for this file. Supports both the older
            # space form ("2025-05-01 0:01:47") and the newer ISO "T" form
            # ("2026-07-13T00:01:37") emitted by recent Clarity exports.
            timestamp_format = FormatParser._probe_timestamp_format(df, Col.TIMESTAMP, DEXCOM_TIMESTAMP_FORMATS)

            # Process EGV (glucose) rows — includes "Fasting Glucose" (G7)
            egv_data = (df
                .filter(pl.col(Col.EVENT_TYPE).str.to_lowercase().is_in(["egv", "fasting glucose"]))
                .select([
                    pl.col(Col.TIMESTAMP).alias("datetime"),
                    pl.col(Col.GLUCOSE_VALUE).alias("glucose"),
                    pl.col(Col.EVENT_SUBTYPE).alias("subtype"),
                ])
                .with_columns([
                    # Track if glucose was High/Low BEFORE replacement (sensor out-of-range error)
                    # These are NOT real measurements - sensor couldn't measure the actual value
                    pl.col("glucose")
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(["high", "low"])
                    .alias("is_out_of_range"),
                ])
                .with_columns([
                    # Replace High/Low with numeric placeholders (mg/dL) for processing
                    # High = >400 mg/dL (sensor max), Low = <50 mg/dL (sensor min)
                    pl.col("glucose")
                    .cast(pl.Utf8)
                    .str.replace("High", str(high_glucose_value))
                    .str.replace("Low", str(low_glucose_value))
                    .cast(pl.Float64, strict=False)
                    .alias("glucose"),
                    # Mark out-of-range readings with OUT_OF_RANGE flag (sensor error, not real data)
                    pl.when(pl.col("is_out_of_range"))
                    .then(pl.lit(Quality.OUT_OF_RANGE.value))
                    .otherwise(pl.lit(0))  # 0 = GOOD (no flags)
                    .alias("quality"),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                ])
            )

            # Convert glucose to the canonical unit (mg/dL) based on the column's
            # declared unit in the effective schema — mmol/L exports are scaled,
            # mg/dL pass through, with no format-specific branch. High/Low markers
            # are already substituted with mg/dL placeholders, so leave them as-is.
            egv_data = egv_data.with_columns([
                pl.when(pl.col("is_out_of_range"))
                .then(pl.col("glucose"))
                .otherwise(cls._glucose_to_canonical(
                    eff_schema, Col.GLUCOSE_VALUE, pl.col("glucose")
                ))
                .alias("glucose"),
            ])

            egv_data = (egv_data
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                ])
                .drop(["subtype", "is_out_of_range"])
            )

            # Process insulin events
            insulin_data = (df
                .filter(pl.col(Col.EVENT_TYPE) == "Insulin")
                .select([
                    pl.col(Col.TIMESTAMP).alias("datetime"),
                    pl.col(Col.EVENT_SUBTYPE).alias("subtype"),
                    pl.col(Col.INSULIN_VALUE).alias("insulin_value"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.when(pl.col("subtype") == "Fast-Acting")
                    .then(pl.lit(UnifiedEventType.INSULIN_FAST.value))
                    .when(pl.col("subtype") == "Long-Acting")
                    .then(pl.lit(UnifiedEventType.INSULIN_SLOW.value))
                    .otherwise(pl.lit(UnifiedEventType.INSULIN_FAST.value))
                    .alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
                .with_columns([
                    pl.when(pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value)
                    .then(pl.col("insulin_value"))
                    .otherwise(pl.lit(None))
                    .alias("insulin_fast"),
                    pl.when(pl.col("event_type") == UnifiedEventType.INSULIN_SLOW.value)
                    .then(pl.col("insulin_value"))
                    .otherwise(pl.lit(None))
                    .alias("insulin_slow"),
                ])
                .drop(["subtype", "insulin_value"])
            )

            # Process carbohydrate events
            carb_data = (df
                .filter(pl.col(Col.EVENT_TYPE) == "Carbs")
                .select([
                    pl.col(Col.TIMESTAMP).alias("datetime"),
                    pl.col(Col.CARB_VALUE).alias("carbs"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )

            # Process exercise events — includes "Activity" (G7)
            exercise_data = (df
                .filter(pl.col(Col.EVENT_TYPE).is_in(["Exercise", "Activity"]))
                .select([
                    pl.col(Col.TIMESTAMP).alias("datetime"),
                    pl.col(Col.DURATION).alias("duration_str"),
                    pl.col(Col.EVENT_SUBTYPE).alias("subtype"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    # Convert duration HH:MM:SS to seconds
                    pl.col("duration_str").str.split(":").list.get(0).cast(pl.Int64) * 3600 +
                    pl.col("duration_str").str.split(":").list.get(1).cast(pl.Int64) * 60 +
                    pl.col("duration_str").str.split(":").list.get(2).cast(pl.Int64)
                    .alias("exercise"),
                    pl.when(pl.col("subtype") == "Light")
                    .then(pl.lit(UnifiedEventType.EXERCISE_LIGHT.value))
                    .when(pl.col("subtype") == "Medium")
                    .then(pl.lit(UnifiedEventType.EXERCISE_MEDIUM.value))
                    .when(pl.col("subtype") == "Heavy")
                    .then(pl.lit(UnifiedEventType.EXERCISE_HEAVY.value))
                    .otherwise(pl.lit(UnifiedEventType.EXERCISE_MEDIUM.value))
                    .alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
                .drop(["duration_str", "subtype"])
            )

            # Combine all data types
            all_data = [egv_data]
            if len(insulin_data) > 0:
                all_data.append(insulin_data)
            if len(carb_data) > 0:
                all_data.append(carb_data)
            if len(exercise_data) > 0:
                all_data.append(exercise_data)

            # Concatenate with alignment
            unified = pl.concat(all_data, how="diagonal")

            # Add sequence_id
            unified = unified.with_columns([
                pl.lit(0).alias("sequence_id")
            ])

            return cls._postprocess_unified(unified)

        except pl.exceptions.PolarsError as e:
            error_msg = f"Failed to parse Dexcom CSV: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))
    
    @classmethod
    def _process_libre(cls, text_data: str, european: bool = False) -> UnifiedFormat:
        """Process FreeStyle Libre CSV to unified format.

        Args:
            text_data: Libre CSV string
            european: If True, glucose is in mmol/L and will be converted to mg/dL

        Returns:
            DataFrame in unified format

        Raises:
            MalformedDataError: If parsing fails
        """
        Col = LibreEUColumn if european else LibreColumn
        eff_schema = LIBRE_EU_SCHEMA if european else LIBRE_SCHEMA
        try:
            # Libre has: Row 1 = metadata, Row 2 = columns, Row 3+ = data
            # Use skip_rows to skip the first metadata row
            df = pl.read_csv(
                StringIO(text_data),
                skip_rows=LIBRE_HEADER_LINE - 1,  # Skip metadata row, next row becomes header
                truncate_ragged_lines=True,
                infer_schema_length=None,
                ignore_errors=False
            )
            
            # Clean column names
            df = df.rename({col: col.strip().replace('"', '').replace('"', '') for col in df.columns})

            # Absorb benign header drift (e.g. newer LibreView renamed the
            # long-acting insulin column) by mapping any registered alias to its
            # canonical name, so the enum-driven selects below resolve.
            df = eff_schema.normalize_headers(df)

            # Probe timestamp format once for this file
            timestamp_format = FormatParser._probe_timestamp_format(df, Col.DEVICE_TIMESTAMP, LIBRE_TIMESTAMP_FORMATS)

            # Process historic glucose data (Record Type = 0) — automatic CGM interval
            glucose_data = (df
                .filter(pl.col(Col.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.HISTORIC_GLUCOSE.value)
                .select([
                    pl.col(Col.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(Col.HISTORIC_GLUCOSE).alias("glucose"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    cls._glucose_to_canonical(
                        eff_schema,
                        Col.HISTORIC_GLUCOSE,
                        pl.col("glucose").cast(pl.Float64, strict=False),
                    ).alias("glucose"),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )

            # Process scan glucose data (Record Type = 1) — user-initiated sensor scan.
            # Same sensor glucose as historic, so it joins the EGV_READ series.
            scan_data = (df
                .filter(pl.col(Col.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.SCAN_GLUCOSE.value)
                .select([
                    pl.col(Col.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(Col.SCAN_GLUCOSE).alias("glucose"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    cls._glucose_to_canonical(
                        eff_schema,
                        Col.SCAN_GLUCOSE,
                        pl.col("glucose").cast(pl.Float64, strict=False),
                    ).alias("glucose"),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )

            # Process strip glucose data (Record Type = 2) — finger-prick calibration
            strip_data = (df
                .filter(pl.col(Col.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.STRIP_GLUCOSE.value)
                .select([
                    pl.col(Col.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(Col.STRIP_GLUCOSE).alias("glucose"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    cls._glucose_to_canonical(
                        eff_schema,
                        Col.STRIP_GLUCOSE,
                        pl.col("glucose").cast(pl.Float64, strict=False),
                    ).alias("glucose"),
                    pl.lit(UnifiedEventType.CALIBRATION.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Process insulin events (Record Type = 4)
            insulin_data = (df
                .filter(pl.col(Col.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.INSULIN.value)
                .select([
                    pl.col(Col.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(Col.RAPID_INSULIN).alias("insulin_fast"),
                    pl.col(Col.LONG_INSULIN).alias("insulin_slow"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    # Determine event type based on which insulin column has a value
                    pl.when(pl.col("insulin_fast").is_not_null())
                    .then(pl.lit(UnifiedEventType.INSULIN_FAST.value))
                    .when(pl.col("insulin_slow").is_not_null())
                    .then(pl.lit(UnifiedEventType.INSULIN_SLOW.value))
                    .otherwise(pl.lit(UnifiedEventType.INSULIN_FAST.value))
                    .alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Process food/carb events (Record Type = 5)
            carb_data = (df
                .filter(pl.col(Col.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.FOOD.value)
                .select([
                    pl.col(Col.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(Col.CARBOHYDRATES_GRAMS).alias("carbs"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Combine all data types
            all_data = [glucose_data]
            if len(scan_data) > 0:
                all_data.append(scan_data)
            if len(strip_data) > 0:
                all_data.append(strip_data)
            if len(insulin_data) > 0:
                all_data.append(insulin_data)
            if len(carb_data) > 0:
                all_data.append(carb_data)
            
            # Concatenate with alignment
            unified = pl.concat(all_data, how="diagonal")
            
            # Add sequence_id
            unified = unified.with_columns([
                pl.lit(0).alias("sequence_id")
            ])
            
            return cls._postprocess_unified(unified)
            
        except pl.exceptions.PolarsError as e:
            error_msg = f"Failed to parse Libre CSV: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))
    
    @staticmethod
    def _euro_number_to_float(expr: pl.Expr) -> pl.Expr:
        """Convert European-format number expression to Float64.

        Replaces comma decimal separators with periods, then casts.
        Invalid values (e.g. "-------") become null via strict=False.
        """
        return (
            expr.cast(pl.Utf8, strict=False)
            .str.replace_all(",", ".")
            .cast(pl.Float64, strict=False)
        )

    @classmethod
    def _find_medtronic_header_line(cls, text_data: str, max_lines: int = 30) -> int:
        """Find the header row index in a Medtronic CareLink CSV.

        Scans the first *max_lines* lines for a semicolon-separated line
        containing all of MEDTRONIC_REQUIRED_HEADERS.

        Returns:
            0-based line index of the header row.

        Raises:
            MalformedDataError: If no header row is found.
        """
        lines = text_data.split("\n", max_lines + 1)[:max_lines]
        for idx, line in enumerate(lines):
            stripped = line.strip().lstrip("\ufeff")
            if not stripped:
                continue
            candidates: list[list[str]] = []
            if ";" in stripped:
                candidates.append([c.strip().strip('"') for c in stripped.split(";")])
            if "," in stripped:
                candidates.append([c.strip().strip('"') for c in stripped.split(",")])
            for headers in candidates:
                if all(req in headers for req in MEDTRONIC_REQUIRED_HEADERS):
                    return idx
        raise MalformedDataError(
            "Could not find Medtronic header row containing required columns: "
            + ", ".join(MEDTRONIC_REQUIRED_HEADERS)
        )

    @classmethod
    def _process_medtronic(cls, text_data: str) -> UnifiedFormat:
        """Process Medtronic Guardian Connect / CareLink CSV to unified format.

        Handles:
        - Variable metadata rows before the header
        - Semicolon delimiter with European decimal format
        - "-------" placeholders in numeric columns
        - Multiple device sections with repeated header rows
        - Event Marker free-text parsing for insulin and carbs

        Args:
            text_data: Medtronic CSV string

        Returns:
            DataFrame in unified format

        Raises:
            MalformedDataError: If parsing fails
        """
        try:
            header_line_idx = cls._find_medtronic_header_line(text_data)

            schema_overrides: dict[str, pl.DataType] = {
                col: pl.Utf8 for col in MEDTRONIC_SCHEMA_OVERRIDES_UTF8
            }

            df = pl.read_csv(
                StringIO(text_data),
                separator=MEDTRONIC_CSV_SEPARATOR,
                skip_lines=header_line_idx,
                truncate_ragged_lines=True,
                infer_schema_length=200,
                schema_overrides=schema_overrides,
            )

            # Clean column names (BOM, smart quotes)
            df = df.rename(
                {col: col.strip().lstrip("\ufeff").replace("\u201c", '"').replace("\u201d", '"')
                 for col in df.columns}
            )

            # Drop repeated header rows and "-------" separator lines
            df = df.filter(
                (pl.col(MedtronicColumn.DATE) != MedtronicColumn.DATE.value)
                & ~pl.col(MedtronicColumn.INDEX).cast(pl.Utf8, strict=False).str.starts_with("-------")
            )

            # Build combined timestamp column
            ts_raw = pl.concat_str(
                [pl.col(MedtronicColumn.DATE), pl.col(MedtronicColumn.TIME)],
                separator=" ",
            ).alias("_ts_raw")

            df = df.with_columns([ts_raw])

            timestamp_format = cls._probe_timestamp_format(df, "_ts_raw", MEDTRONIC_TIMESTAMP_FORMATS)

            # Parse Euro-decimal numeric columns
            sensor_gl = cls._euro_number_to_float(pl.col(MedtronicColumn.SENSOR_GLUCOSE)).alias("_sensor_gl")
            bg_gl = cls._euro_number_to_float(pl.col(MedtronicColumn.BG_READING)).alias("_bg_gl")
            bolus_u = cls._euro_number_to_float(pl.col(MedtronicColumn.BOLUS_VOLUME_DELIVERED)).alias("_bolus_u")
            basal_u = cls._euro_number_to_float(pl.col(MedtronicColumn.BASAL_RATE)).alias("_basal_u")
            bwz_carbs = cls._euro_number_to_float(pl.col(MedtronicColumn.BWZ_CARB_INPUT)).alias("_bwz_carbs")

            # Extract insulin/carbs from Event Marker as fallback
            event_marker_col = pl.col(MedtronicColumn.EVENT_MARKER).cast(pl.Utf8, strict=False).fill_null("")
            marker_insulin = cls._euro_number_to_float(
                event_marker_col.str.extract(r"Insulin:\s*([\d,\.]+)", 1)
            ).alias("_marker_insulin")
            marker_carbs = cls._euro_number_to_float(
                event_marker_col.str.extract(r"Meal:\s*([\d,\.]+)\s*grams?", 1)
            ).alias("_marker_carbs")

            df = df.with_columns([sensor_gl, bg_gl, bolus_u, basal_u, bwz_carbs, marker_insulin, marker_carbs])

            # Coalesce: structured columns take priority over Event Marker
            df = df.with_columns([
                pl.coalesce([pl.col("_sensor_gl"), pl.col("_bg_gl")]).alias("_glucose"),
                pl.coalesce([pl.col("_bolus_u"), pl.col("_marker_insulin")]).alias("_insulin_fast"),
                pl.col("_basal_u").alias("_insulin_slow"),
                pl.coalesce([pl.col("_bwz_carbs"), pl.col("_marker_carbs")]).alias("_carbs"),
            ])

            _ts = pl.col("_ts_raw").str.strptime(pl.Datetime("ms"), timestamp_format).alias("datetime")
            _quality = pl.lit(0).alias("quality")

            # --- Glucose rows ---
            glucose_data = (
                df.filter(pl.col("_glucose").is_not_null())
                .select([
                    _ts,
                    pl.col("_glucose").alias("glucose"),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                    _quality,
                ])
            )

            # --- Fast insulin rows (bolus / Event Marker insulin) ---
            insulin_fast_data = (
                df.filter(pl.col("_insulin_fast").is_not_null())
                .select([
                    _ts,
                    pl.col("_insulin_fast").alias("insulin_fast"),
                    pl.lit(UnifiedEventType.INSULIN_FAST.value).alias("event_type"),
                    _quality,
                ])
            )

            # --- Slow insulin rows (basal rate) ---
            insulin_slow_data = (
                df.filter(pl.col("_insulin_slow").is_not_null())
                .select([
                    _ts,
                    pl.col("_insulin_slow").alias("insulin_slow"),
                    pl.lit(UnifiedEventType.INSULIN_SLOW.value).alias("event_type"),
                    _quality,
                ])
            )

            # --- Carb rows ---
            carb_data = (
                df.filter(pl.col("_carbs").is_not_null())
                .select([
                    _ts,
                    pl.col("_carbs").alias("carbs"),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    _quality,
                ])
            )

            # Combine all event types
            all_data = [glucose_data]
            if len(insulin_fast_data) > 0:
                all_data.append(insulin_fast_data)
            if len(insulin_slow_data) > 0:
                all_data.append(insulin_slow_data)
            if len(carb_data) > 0:
                all_data.append(carb_data)

            unified = pl.concat(all_data, how="diagonal")

            unified = unified.with_columns([
                pl.lit(0).alias("sequence_id")
            ])

            return cls._postprocess_unified(unified)

        except pl.exceptions.PolarsError as e:
            error_msg = f"Failed to parse Medtronic CSV: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))

    # ===== Nightscout Processing Methods =====

    @classmethod
    def _is_nightscout_entries_json(cls, data: str) -> bool:
        """Check if data looks like Nightscout entries JSON (array with sgv keys)."""
        stripped = data.strip()
        return stripped.startswith("[") and '"sgv"' in stripped[:2000]

    @classmethod
    def _is_nightscout_treatments_json(cls, data: str) -> bool:
        """Check if data looks like Nightscout treatments JSON (array with eventType keys)."""
        stripped = data.strip()
        return stripped.startswith("[") and '"eventType"' in stripped[:2000]

    # ----- JSON parsing (Nightscout REST API) -----

    @classmethod
    def _parse_nightscout_entries_json(cls, json_data: str) -> pl.DataFrame:
        """Parse Nightscout entries JSON array to glucose DataFrame.

        Returns a DataFrame with columns ``dateString`` and ``sgv``.
        """
        import json as json_mod
        records = json_mod.loads(json_data)
        if not records:
            raise ZeroValidInputError("Nightscout entries JSON is empty")

        rows: list[dict] = []
        for entry in records:
            if entry.get("type") != "sgv":
                continue
            date_str = entry.get("dateString") or entry.get("sysTime")
            sgv = entry.get("sgv") or entry.get("glucose")
            if date_str is None or sgv is None:
                continue
            rows.append({"dateString": str(date_str), "sgv": sgv})

        if not rows:
            raise ZeroValidInputError("No SGV entries found in Nightscout JSON")
        return pl.DataFrame(rows)

    @classmethod
    def _parse_nightscout_treatments_json(cls, json_data: str) -> pl.DataFrame:
        """Parse Nightscout treatments JSON to a flat DataFrame.

        Returns a DataFrame with JSON API field names (``created_at``,
        ``eventType``, ``insulin``, ``carbs``, ``rate``, ``duration``).
        """
        import json as json_mod
        records = json_mod.loads(json_data)
        if not records:
            return pl.DataFrame()

        rows: list[dict] = []
        for t in records:
            event_type = t.get("eventType")
            created_at = t.get("created_at")
            if not event_type or not created_at:
                continue
            rows.append({
                "created_at": str(created_at),
                "eventType": str(event_type),
                "insulin": t.get("insulin"),
                "carbs": t.get("carbs"),
                "rate": t.get("rate"),
                "duration": t.get("duration"),
            })
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    @classmethod
    def _entries_json_to_unified(cls, entries_df: pl.DataFrame) -> pl.DataFrame:
        """Convert JSON-parsed entries DataFrame to unified glucose rows."""
        timestamp_format = cls._probe_timestamp_format(entries_df, "dateString", NIGHTSCOUT_TIMESTAMP_FORMATS)
        return (
            entries_df
            .select([
                pl.col("dateString").str.strptime(pl.Datetime("ms"), timestamp_format).alias("datetime"),
                pl.col("sgv").cast(pl.Float64, strict=False).alias("glucose"),
                pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                pl.lit(0).alias("quality"),
            ])
            .filter(pl.col("glucose").is_not_null())
        )

    @classmethod
    def _treatments_json_to_unified(cls, treatments_df: pl.DataFrame) -> pl.DataFrame:
        """Convert JSON-parsed treatments DataFrame to unified rows."""
        if len(treatments_df) == 0:
            return pl.DataFrame()

        timestamp_format = cls._probe_timestamp_format(treatments_df, "created_at", NIGHTSCOUT_TIMESTAMP_FORMATS)

        all_frames: list[pl.DataFrame] = []

        # Bolus / SMB -> insulin_fast
        bolus_types = [NightscoutTreatmentEventType.BOLUS.value, NightscoutTreatmentEventType.SMB.value,
                       NightscoutTreatmentEventType.MEAL_BOLUS.value, NightscoutTreatmentEventType.CORRECTION_BOLUS.value]
        if "insulin" in treatments_df.columns:
            bolus_df = (
                treatments_df
                .filter(
                    pl.col("eventType").is_in(bolus_types)
                    & pl.col("insulin").is_not_null()
                    & (pl.col("insulin").cast(pl.Float64, strict=False) > 0)
                )
                .select([
                    pl.col("created_at").str.strptime(pl.Datetime("ms"), timestamp_format).alias("datetime"),
                    pl.col("insulin").cast(pl.Float64, strict=False).alias("insulin_fast"),
                    pl.lit(UnifiedEventType.INSULIN_FAST.value).alias("event_type"),
                    pl.lit(0).alias("quality"),
                ])
            )
            if len(bolus_df) > 0:
                all_frames.append(bolus_df)

        # Temp Basal -> insulin_slow (only non-zero rates)
        if "rate" in treatments_df.columns:
            basal_df = (
                treatments_df
                .filter(
                    (pl.col("eventType") == NightscoutTreatmentEventType.TEMP_BASAL.value)
                    & pl.col("rate").is_not_null()
                    & (pl.col("rate").cast(pl.Float64, strict=False) > 0)
                )
                .select([
                    pl.col("created_at").str.strptime(pl.Datetime("ms"), timestamp_format).alias("datetime"),
                    pl.col("rate").cast(pl.Float64, strict=False).alias("insulin_slow"),
                    pl.lit(UnifiedEventType.INSULIN_SLOW.value).alias("event_type"),
                    pl.lit(0).alias("quality"),
                ])
            )
            if len(basal_df) > 0:
                all_frames.append(basal_df)

        # Any treatment with carbs -> carbs
        if "carbs" in treatments_df.columns:
            carb_df = (
                treatments_df
                .filter(
                    pl.col("carbs").is_not_null()
                    & (pl.col("carbs").cast(pl.Float64, strict=False) > 0)
                )
                .select([
                    pl.col("created_at").str.strptime(pl.Datetime("ms"), timestamp_format).alias("datetime"),
                    pl.col("carbs").cast(pl.Float64, strict=False).alias("carbs"),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),
                ])
            )
            if len(carb_df) > 0:
                all_frames.append(carb_df)

        if not all_frames:
            return pl.DataFrame()
        return pl.concat(all_frames, how="diagonal")

    # ----- CSV parsing (nightscout-exporter format) -----

    @classmethod
    def _split_nightscout_exporter_csv(cls, text_data: str) -> tuple[str, Optional[str]]:
        """Split a nightscout-exporter combined CSV into entries and treatments sections.

        The exporter format uses ``# CGM ENTRIES`` and ``# TREATMENTS ...``
        comment lines as section delimiters.  Returns (entries_csv, treatments_csv)
        where treatments_csv may be None if no treatments section is present.
        """
        entries_lines: list[str] = []
        treatments_lines: list[str] = []
        current: Optional[list[str]] = None

        for line in text_data.splitlines():
            stripped = line.strip()
            if stripped.startswith("# CGM ENTRIES"):
                current = entries_lines
                continue
            if stripped.startswith("# TREATMENTS"):
                current = treatments_lines
                continue
            if stripped.startswith("#"):
                continue
            if not stripped:
                continue
            if current is not None:
                current.append(line)
            else:
                # Before any section marker → assume entries
                entries_lines.append(line)

        entries_csv = "\n".join(entries_lines) if entries_lines else ""
        treatments_csv = "\n".join(treatments_lines) if treatments_lines else None
        return entries_csv, treatments_csv

    @classmethod
    def _parse_nightscout_entries_csv(cls, csv_data: str) -> pl.DataFrame:
        """Parse nightscout-exporter entries CSV section.

        Expected columns: Date, Time, Glucose (mg/dL), Type, Device, Trend, ID
        """
        df = pl.read_csv(
            StringIO(csv_data),
            truncate_ragged_lines=True,
            infer_schema_length=None,
            ignore_errors=False,
        )
        df = df.rename({col: col.strip() for col in df.columns})
        glucose_col = NightscoutEntryColumn.GLUCOSE_MGDL
        if glucose_col not in df.columns:
            raise MalformedDataError(
                f"Missing required column: '{glucose_col}'. "
                f"Got columns: {df.columns}"
            )
        return df

    @classmethod
    def _parse_nightscout_treatments_csv(cls, csv_data: str) -> pl.DataFrame:
        """Parse nightscout-exporter treatments CSV section.

        Expected columns: Date, Time, Event Type, Insulin (U), Carbs (g), Notes, ID
        """
        df = pl.read_csv(
            StringIO(csv_data),
            truncate_ragged_lines=True,
            infer_schema_length=None,
            ignore_errors=False,
        )
        df = df.rename({col: col.strip() for col in df.columns})
        return df

    # Datetime formats produced by JavaScript's toLocaleDateString() / toLocaleTimeString()
    # in the nightscout-exporter.  We try US locale first, then ISO-ish fallbacks.
    _EXPORTER_DATETIME_FORMATS: ClassVar[list[str]] = [
        "%m/%d/%Y %I:%M:%S %p",    # US locale: 3/31/2026 7:51:03 PM
        "%d/%m/%Y %I:%M:%S %p",    # UK locale: 31/3/2026 7:51:03 PM
        "%Y-%m-%d %H:%M:%S",       # ISO-ish: 2026-03-31 19:51:03
    ]

    @classmethod
    def _parse_exporter_datetime(cls, df: pl.DataFrame, date_col: str, time_col: str) -> pl.DataFrame:
        """Add a ``datetime`` column by combining Date + Time with format probing."""
        combined = df.with_columns(
            (pl.col(date_col).cast(pl.Utf8) + " " + pl.col(time_col).cast(pl.Utf8)).alias("_datetime_str")
        )
        for fmt in cls._EXPORTER_DATETIME_FORMATS:
            try:
                return combined.with_columns(
                    pl.col("_datetime_str").str.strptime(pl.Datetime("ms"), fmt, strict=False).alias("datetime")
                )
            except Exception:
                continue
        raise MalformedDataError(
            f"Cannot parse Date+Time columns with any known format. "
            f"Sample: {combined['_datetime_str'].head(3).to_list()}"
        )

    @classmethod
    def _entries_csv_to_unified(cls, entries_df: pl.DataFrame) -> pl.DataFrame:
        """Convert nightscout-exporter entries CSV to unified glucose rows.

        Combines locale Date + Time columns into a datetime, maps
        Glucose (mg/dL) to the unified glucose column.
        """
        date_col = NightscoutEntryColumn.DATE
        time_col = NightscoutEntryColumn.TIME
        glucose_col = NightscoutEntryColumn.GLUCOSE_MGDL

        entries_df = cls._parse_exporter_datetime(entries_df, date_col, time_col)

        return (
            entries_df
            .select([
                pl.col("datetime"),
                pl.col(glucose_col).cast(pl.Float64, strict=False).alias("glucose"),
                pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                pl.lit(0).alias("quality"),
            ])
            .filter(pl.col("glucose").is_not_null() & pl.col("datetime").is_not_null())
        )

    @classmethod
    def _treatments_csv_to_unified(cls, treatments_df: pl.DataFrame) -> pl.DataFrame:
        """Convert nightscout-exporter treatments CSV to unified rows."""
        if len(treatments_df) == 0:
            return pl.DataFrame()

        date_col = NightscoutTreatmentColumn.DATE
        time_col = NightscoutTreatmentColumn.TIME
        event_col = NightscoutTreatmentColumn.EVENT_TYPE
        insulin_col = NightscoutTreatmentColumn.INSULIN_U
        carbs_col = NightscoutTreatmentColumn.CARBS_G

        treatments_df = cls._parse_exporter_datetime(treatments_df, date_col, time_col)

        all_frames: list[pl.DataFrame] = []

        bolus_types = [NightscoutTreatmentEventType.BOLUS.value, NightscoutTreatmentEventType.SMB.value,
                       NightscoutTreatmentEventType.MEAL_BOLUS.value, NightscoutTreatmentEventType.CORRECTION_BOLUS.value]

        if insulin_col in treatments_df.columns:
            bolus_df = (
                treatments_df
                .filter(
                    pl.col(event_col).is_in(bolus_types)
                    & pl.col(insulin_col).is_not_null()
                    & (pl.col(insulin_col).cast(pl.Float64, strict=False) > 0)
                )
                .select([
                    pl.col("datetime"),
                    pl.col(insulin_col).cast(pl.Float64, strict=False).alias("insulin_fast"),
                    pl.lit(UnifiedEventType.INSULIN_FAST.value).alias("event_type"),
                    pl.lit(0).alias("quality"),
                ])
            )
            if len(bolus_df) > 0:
                all_frames.append(bolus_df)

        if carbs_col in treatments_df.columns:
            carb_df = (
                treatments_df
                .filter(
                    pl.col(carbs_col).is_not_null()
                    & (pl.col(carbs_col).cast(pl.Float64, strict=False) > 0)
                )
                .select([
                    pl.col("datetime"),
                    pl.col(carbs_col).cast(pl.Float64, strict=False).alias("carbs"),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),
                ])
            )
            if len(carb_df) > 0:
                all_frames.append(carb_df)

        if not all_frames:
            return pl.DataFrame()
        return pl.concat(all_frames, how="diagonal")

    # ----- Top-level Nightscout dispatch -----

    @classmethod
    def _process_nightscout(cls, text_data: str) -> UnifiedFormat:
        """Process Nightscout data (JSON or exporter CSV) to unified format.

        Dispatches to JSON or CSV parsing based on content sniffing.
        Called by ``parse_to_unified`` when format is NIGHTSCOUT, and also
        used internally by ``parse_nightscout`` for entries-only data.
        """
        try:
            # JSON entries (from API)
            if cls._is_nightscout_entries_json(text_data):
                entries_df = cls._parse_nightscout_entries_json(text_data)
                glucose_rows = cls._entries_json_to_unified(entries_df)
                unified = glucose_rows.with_columns([pl.lit(0).alias("sequence_id")])
                return cls._postprocess_unified(unified)

            # nightscout-exporter CSV (combined entries + optional treatments)
            entries_csv, treatments_csv = cls._split_nightscout_exporter_csv(text_data)

            if not entries_csv.strip():
                raise ZeroValidInputError("No entries data found in Nightscout data")

            entries_df = cls._parse_nightscout_entries_csv(entries_csv)
            glucose_rows = cls._entries_csv_to_unified(entries_df)
            all_frames: list[pl.DataFrame] = [glucose_rows]

            if treatments_csv:
                treatments_df = cls._parse_nightscout_treatments_csv(treatments_csv)
                treatment_rows = cls._treatments_csv_to_unified(treatments_df)
                if len(treatment_rows) > 0:
                    all_frames.append(treatment_rows)

            unified = pl.concat(all_frames, how="diagonal")
            unified = unified.with_columns([pl.lit(0).alias("sequence_id")])
            return cls._postprocess_unified(unified)

        except (ZeroValidInputError, MalformedDataError):
            raise
        except Exception as e:
            error_msg = f"Failed to parse Nightscout data: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))

    @classmethod
    def parse_nightscout(
        cls,
        entries_data: Union[bytes, str],
        treatments_data: Union[bytes, str, None] = None,
    ) -> UnifiedFormat:
        """Parse Nightscout entries and optional treatments to unified format.

        Accepts JSON (from the Nightscout REST API) or nightscout-exporter CSV
        for entries.  Treatments must be JSON (Nightscout doesn't serve
        treatments as CSV).  Merges glucose readings with insulin / carb
        treatments into a single unified DataFrame.

        Args:
            entries_data: Nightscout entries (JSON or exporter CSV), as bytes or string
            treatments_data: Optional Nightscout treatments JSON

        Returns:
            DataFrame in unified format matching CGM_SCHEMA
        """
        if isinstance(entries_data, bytes):
            entries_text = cls.decode_raw_data(entries_data)
        else:
            entries_text = entries_data

        if cls._is_nightscout_entries_json(entries_text):
            entries_df = cls._parse_nightscout_entries_json(entries_text)
            glucose_rows = cls._entries_json_to_unified(entries_df)
        else:
            entries_df = cls._parse_nightscout_entries_csv(entries_text)
            glucose_rows = cls._entries_csv_to_unified(entries_df)

        all_frames: list[pl.DataFrame] = [glucose_rows]

        if treatments_data is not None:
            if isinstance(treatments_data, bytes):
                treatments_text = cls.decode_raw_data(treatments_data)
            else:
                treatments_text = treatments_data

            if cls._is_nightscout_treatments_json(treatments_text):
                treatments_df = cls._parse_nightscout_treatments_json(treatments_text)
                treatment_rows = cls._treatments_json_to_unified(treatments_df)
            else:
                treatments_df = cls._parse_nightscout_treatments_csv(treatments_text)
                treatment_rows = cls._treatments_csv_to_unified(treatments_df)

            if len(treatment_rows) > 0:
                all_frames.append(treatment_rows)

        unified = pl.concat(all_frames, how="diagonal")
        unified = unified.with_columns([pl.lit(0).alias("sequence_id")])
        return cls._postprocess_unified(unified)

    @classmethod
    def from_nightscout_exports(
        cls,
        entries_path: Union[str, Path],
        treatments_path: Union[str, Path, None] = None,
        profile_path: Union[str, Path, None] = None,
    ) -> UnifiedFormat:
        """Parse Nightscout export files (JSON or CSV) to unified format.

        A named shortcut for the bundle shape (see :meth:`parse_bundle`):
        entries and treatments are two modalities of one person's record.
        Retained because it is public API and because naming Nightscout's two
        specific files is friendlier than an ordered sequence — but the
        general entry point is `parse_bundle`.

        It does not delegate to `parse_bundle`: Nightscout's members are JSON,
        which `detect_format` deliberately does not handle (it pattern-matches
        CSV headers), and the entries/treatments merge is a keyed join rather
        than a diagonal concat of two independently parsed frames. So this
        keeps its own path through `parse_nightscout`.

        Accepts both JSON and CSV for entries; treatments are expected to be
        JSON (the Nightscout API does not support CSV for treatments).

        Args:
            entries_path: Path to entries file (JSON or CSV)
            treatments_path: Optional path to treatments JSON file
            profile_path: **Accepted and ignored.** Nightscout's profile
                document carries settings — basal schedules, targets, the
                display unit — not glucose readings or events, so it has no
                rows to contribute to a unified frame and no column to land
                in. It stays in the signature because it is public API and
                because a caller holding all three exports naturally passes
                all three. Passing it logs a warning rather than failing:
                dropping it silently would be the "silent fallback" the
                charter forbids, and rejecting it would break existing callers
                for no gain.

        Returns:
            DataFrame in unified format matching CGM_SCHEMA
        """
        if profile_path is not None:
            logger.warning(
                "from_nightscout_exports received profile_path=%s and is ignoring "
                "it: the Nightscout profile document holds settings (basal "
                "schedules, targets, display unit), not readings or events, so "
                "it contributes no rows to a unified frame.",
                profile_path,
            )

        entries_data = Path(entries_path).read_bytes()
        treatments_data: Union[bytes, None] = None
        if treatments_path is not None:
            treatments_data = Path(treatments_path).read_bytes()
        return cls.parse_nightscout(entries_data, treatments_data)

    @classmethod
    def from_nightscout_url(
        cls,
        base_url: str,
        count: int = 10_000,
        token: Optional[str] = None,
        api_secret: Optional[str] = None,
        days: Optional[int] = None,
        timeout: float = 60.0,
        output_dir: Optional[Path] = None,
    ) -> UnifiedFormat:
        """Download Nightscout data and parse to unified format in one call.

        Fetches entries and treatments as JSON from the Nightscout REST API,
        optionally saves the raw files to *output_dir*, and returns a unified
        DataFrame.

        Requires ``httpx`` (optional dependency).

        Args:
            base_url: Nightscout base URL
            count: Maximum number of entries/treatments to fetch
            token: Optional readable access token
            api_secret: Optional API_SECRET (hashed and sent as header)
            days: If set, only fetch data from the last N days
            timeout: HTTP request timeout in seconds
            output_dir: Optional directory to persist raw JSON files

        Returns:
            Unified-format Polars DataFrame
        """
        from cgm_format.nightscout_downloader import (
            download_nightscout,
        )

        if output_dir is None:
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="nightscout_"))
        else:
            tmp = Path(output_dir)

        entries_path, treatments_path, _ = download_nightscout(
            base_url=base_url,
            output_dir=tmp,
            count=count,
            token=token,
            api_secret=api_secret,
            days=days,
            timeout=timeout,
        )

        return cls.from_nightscout_exports(entries_path, treatments_path)

    # ===== Serialization Methods =====
    
    # Unified schemas a serialized frame may legitimately conform to, most
    # specific first. Dispatch is by exact column-name tuple, so it recognizes a
    # frame rather than guessing at one.
    _SERIALIZABLE_SCHEMAS: ClassVar[tuple[CGMSchemaDefinition, ...]] = (
        CGM_SCHEMA_EXTENDED,
        CGM_SCHEMA,
    )

    @classmethod
    def _schema_for_serialization(cls, dataframe: UnifiedFormat) -> CGMSchemaDefinition:
        """Pick the unified schema a frame is to be validated against on output.

        Matches the frame's exact column list against the registered unified
        schemas. Anything that matches none falls back to CGM_SCHEMA, which is
        what validation then rejects — so a malformed frame still raises exactly
        as it does today, rather than being quietly accepted.
        """
        columns = tuple(dataframe.columns)
        for schema in cls._SERIALIZABLE_SCHEMAS:
            if columns == tuple(schema.get_column_names(data_only=False)):
                return schema
        return CGM_SCHEMA

    @classmethod
    def to_csv_string(cls, dataframe: UnifiedFormat) -> str:
        """Serialize unified format DataFrame to CSV string.

        A classmethod, not a staticmethod: it must read `cls.validation_mode`
        and the frame's target schema, and a staticmethod naming `FormatParser`
        literally would ignore both in a subclass.

        Args:
            dataframe: DataFrame in unified format (core or extended)

        Returns:
            CSV string representation
        """
        # Verify input dataframe matches schema
        if cls.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            schema = cls._schema_for_serialization(dataframe)
            schema.validate_dataframe(dataframe, enforce=cls.validation_mode & ValidationMethod.INPUT_FORCED)
        return dataframe.write_csv(separator=",")

    @classmethod
    def to_csv_file(cls, dataframe: UnifiedFormat, file_path: str) -> None:
        """Save unified format DataFrame to CSV file.

        Args:
            dataframe: DataFrame in unified format (core or extended)
            file_path: Path where to save the CSV file
        """
        # Verify input dataframe matches schema
        if cls.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            schema = cls._schema_for_serialization(dataframe)
            schema.validate_dataframe(dataframe, enforce=cls.validation_mode & ValidationMethod.INPUT_FORCED)
        dataframe.write_csv(file_path)
    


    # ===== CGMacros: a multi-track corpus =====

    @classmethod
    def _process_cgmacros(
        cls,
        text_data: str,
        track: str = CGMACROS_TRACKS[0],
    ) -> UnifiedFormat:
        """Parse one CGMacros subject CSV into one sensor's unified frame.

        The non-glucose rows — meals, macronutrients, heart rate, photo
        annotations — are **replicated into every track**, because each track is
        a complete self-contained view of those ten days as seen through one
        sensor. Tracks are alternatives, never shards; concatenating two of them
        double-counts every meal.

        Args:
            text_data: Decoded contents of one subject CSV.
            track: Which sensor's series becomes `glucose`. One of
                `CGMACROS_TRACKS`, or `CGMACROS_MEAN_TRACK` for the opt-in
                synthetic average.

        Returns:
            One extended-schema frame for the requested track.
        """
        if track not in CGMACROS_TRACKS and track != CGMACROS_MEAN_TRACK:
            raise ValueError(
                f"Unknown CGMacros track {track!r}; expected one of "
                f"{CGMACROS_TRACKS} or {CGMACROS_MEAN_TRACK!r}"
            )

        try:
            raw = pl.read_csv(
                StringIO(text_data),
                infer_schema_length=10000,
                truncate_ragged_lines=True,
            )
        except pl.exceptions.PolarsError as e:
            raise MalformedDataError(
                cls._truncate_error_message(f"Failed to read CGMacros CSV: {e}")
            )

        # Strip header whitespace before aliasing: one subject spells the
        # column "Amount Consumed " with a trailing space, and an alias cannot
        # match what the header never normalized.
        raw = raw.rename({c: c.strip() for c in raw.columns})
        raw = CGMACROS_SCHEMA.normalize_headers(raw)
        raw = raw.drop([c for c in CGMACROS_IGNORED_COLUMNS if c in raw.columns])

        # Typed nulls for columns this subject simply lacks. Done once, up
        # front: a `.select(pl.col(X))` is evaluated even when an upstream
        # filter leaves zero rows, so an absent column raises ColumnNotFound
        # regardless of whether any row would have used it.
        missing = [c for c in CGMACROS_OPTIONAL_COLUMNS if c not in raw.columns]
        if missing:
            logger.warning(
                "CGMacros subject file omits %d optional column(s): %s. "
                "Emitting typed nulls — the source did not say, which is not "
                "the same as zero.",
                len(missing),
                ", ".join(sorted(missing)),
            )
            raw = raw.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
            )

        ts_col = CGMacrosColumn.TIMESTAMP.value
        ts_format = cls._probe_timestamp_format(
            raw, ts_col, CGMACROS_TIMESTAMP_FORMATS
        )
        raw = raw.with_columns(
            pl.col(ts_col).str.strptime(pl.Datetime("ms"), ts_format).alias("datetime")
        ).drop_nulls("datetime")

        if len(raw) == 0:
            raise ZeroValidInputError(
                "No CGMacros rows carried a parseable timestamp"
            )

        libre = pl.col(CGMacrosColumn.LIBRE_GLUCOSE.value)
        dexcom = pl.col(CGMacrosColumn.DEXCOM_GLUCOSE.value)
        if track == CGMACROS_TRACKS[0]:
            glucose_expr = libre
            merged_expr = pl.lit(False)
        elif track == CGMACROS_TRACKS[1]:
            glucose_expr = dexcom
            merged_expr = pl.lit(False)
        else:
            # The synthetic mean. Polars' horizontal mean ignores nulls, so a
            # row with one sensor yields that sensor's reading unchanged —
            # which is why only rows with BOTH populated are flagged: a
            # single-sensor row is a real reading, not a synthesized one.
            glucose_expr = pl.mean_horizontal(libre, dexcom)
            merged_expr = libre.is_not_null() & dexcom.is_not_null()

        raw = raw.with_columns(
            glucose_expr.alias("_glucose"),
            merged_expr.alias("_merged"),
        )

        frames: List[pl.DataFrame] = []

        # --- Glucose readings, one row per populated sample ---
        glucose_rows = raw.filter(pl.col("_glucose").is_not_null()).with_columns(
            pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
            pl.when(pl.col("_merged"))
            .then(pl.lit(Quality.TRACK_MERGE.value))
            .otherwise(pl.lit(0))
            .cast(pl.Int64)
            .alias("quality"),
            pl.col("_glucose").alias("glucose"),
        )
        if len(glucose_rows) > 0:
            frames.append(
                glucose_rows.select(
                    "datetime", "event_type", "quality", "glucose",
                    *cls._cgmacros_wearable_columns(),
                )
            )

        # --- Meals: carbs plus the macronutrients the core schema cannot hold ---
        meal_rows = raw.filter(
            pl.col(CGMacrosColumn.MEAL_TYPE.value).is_not_null()
            & (pl.col(CGMacrosColumn.MEAL_TYPE.value).cast(pl.Utf8).str.strip_chars() != "")
        )
        if len(meal_rows) > 0:
            frames.append(cls._cgmacros_meal_frame(meal_rows))

        # --- Annotation-only rows: a photo with no meal attached ---
        # The meal-END photograph. 1,553 such rows across the corpus, against
        # 1,644 carrying both, so these are the MAJORITY of photo rows — which
        # is why annotations cannot simply hang off a CARBS_IN event.
        photo_only = raw.filter(
            pl.col(CGMacrosColumn.IMAGE_PATH.value).is_not_null()
            & (pl.col(CGMacrosColumn.IMAGE_PATH.value).cast(pl.Utf8).str.strip_chars() != "")
            & (
                pl.col(CGMacrosColumn.MEAL_TYPE.value).is_null()
                | (pl.col(CGMacrosColumn.MEAL_TYPE.value).cast(pl.Utf8).str.strip_chars() == "")
            )
        )
        if len(photo_only) > 0:
            frames.append(
                photo_only.with_columns(
                    # OTHEREVT, not an invented code: the row records that
                    # something was photographed, and the schema already has a
                    # member for "an event we cannot type more precisely".
                    pl.lit(UnifiedEventType.OTHER.value).alias("event_type"),
                    pl.lit(0, dtype=pl.Int64).alias("quality"),
                    cls._cgmacros_annotation_expr().alias("annotations"),
                ).select(
                    "datetime", "event_type", "quality", "annotations",
                    *cls._cgmacros_wearable_columns(),
                )
            )

        if not frames:
            raise ZeroValidInputError("No usable CGMacros rows found")

        unified = pl.concat(frames, how="diagonal")
        unified = unified.with_columns(pl.lit(0).alias("sequence_id"))
        return cls._postprocess_unified(
            unified, schema=UNIFIED_TARGET_SCHEMA[SupportedCGMFormat.CGMACROS]
        )

    @classmethod
    def _cgmacros_wearable_columns(cls) -> List[pl.Expr]:
        """Wearable channels carried on every row, whatever the event type.

        METs is stored multiplied by 10 (data dictionary, and the observed
        10-126 range against a physiological 1.0-12.6), so it is divided here.
        """
        return [
            pl.col(CGMacrosColumn.HEART_RATE.value).alias("heart_rate"),
            (pl.col(CGMacrosColumn.METS.value) / CGMACROS_METS_SCALE).alias("mets"),
            pl.col(CGMacrosColumn.ACTIVITY_CALORIES.value).alias("activity_calories"),
            (
                pl.col(CGMacrosColumn.STEPS.value).alias("steps")
                if CGMacrosColumn.STEPS.value
                else pl.lit(None).alias("steps")
            ),
        ]

    @classmethod
    def _cgmacros_annotation_expr(cls) -> pl.Expr:
        """Build the deterministic `annotations` JSON for a row.

        Keys are emitted in sorted order by construction, matching
        `annotations_to_json`, because the column participates in the sort keys
        and in the byte-level round-trip guarantee.
        """
        image = pl.col(CGMacrosColumn.IMAGE_PATH.value).cast(pl.Utf8)
        return (
            pl.when(image.is_not_null() & (image.str.strip_chars() != ""))
            .then(
                pl.format(
                    '{{"image_path":"{}"}}',
                    image.str.strip_chars(),
                )
            )
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )

    @classmethod
    def _cgmacros_meal_frame(cls, meal_rows: pl.DataFrame) -> pl.DataFrame:
        """Meal rows: carbs into the core column, macros into the extended ones.

        `Meal Type` carries ten raw spellings for four meals. The normalized
        label and the raw string both go into `annotations`, so the
        normalization stays inspectable instead of being a lossy rewrite. An
        unrecognized spelling is warned about once with a count, never
        silently coerced.
        """
        raw_type = pl.col(CGMacrosColumn.MEAL_TYPE.value).cast(pl.Utf8).str.strip_chars()
        normalized = raw_type.str.to_lowercase().replace_strict(
            CGMACROS_MEAL_TYPE_NORMALIZATION, default=None
        )

        unrecognized = (
            meal_rows.select(raw_type.alias("raw"))
            .filter(
                pl.col("raw").str.to_lowercase().is_in(
                    list(CGMACROS_MEAL_TYPE_NORMALIZATION)
                ).not_()
            )
            .get_column("raw")
            .value_counts()
        )
        if len(unrecognized) > 0:
            # Aggregated by reason with a count, never one warning per row.
            logger.warning(
                "CGMacros meal labels not in the known vocabulary: %s. "
                "Kept verbatim in annotations; no normalized label emitted.",
                ", ".join(
                    f"{row[0]!r} x{row[1]}" for row in unrecognized.iter_rows()
                ),
            )

        image = pl.col(CGMacrosColumn.IMAGE_PATH.value).cast(pl.Utf8).str.strip_chars()
        amount = pl.col(CGMacrosColumn.AMOUNT_CONSUMED.value)
        annotations = pl.format(
            '{{"amount_consumed":{},"image_path":{},"meal_type":{},"meal_type_raw":"{}"}}',
            pl.when(amount.is_not_null()).then(amount.cast(pl.Utf8)).otherwise(pl.lit("null")),
            pl.when(image.is_not_null() & (image != ""))
            .then(pl.format('"{}"', image))
            .otherwise(pl.lit("null")),
            pl.when(normalized.is_not_null())
            .then(pl.format('"{}"', normalized))
            .otherwise(pl.lit("null")),
            raw_type,
        )

        return meal_rows.with_columns(
            pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
            pl.lit(0, dtype=pl.Int64).alias("quality"),
            annotations.alias("annotations"),
        ).select(
            "datetime",
            "event_type",
            "quality",
            "annotations",
            pl.col(CGMacrosColumn.CARBS.value).alias("carbs"),
            pl.col(CGMacrosColumn.CALORIES.value).alias("calories"),
            pl.col(CGMacrosColumn.PROTEIN.value).alias("protein"),
            pl.col(CGMacrosColumn.FAT.value).alias("fat"),
            pl.col(CGMacrosColumn.FIBER.value).alias("fiber"),
            *cls._cgmacros_wearable_columns(),
        )

    # ===== Faceted output: tracks and corpora =====

    @classmethod
    def parse_tracks(
        cls,
        file_path: Union[str, Path],
        track: Optional[str] = None,
    ) -> Dict[str, UnifiedFormat]:
        """Parse a multi-track file into one frame per sensor.

        See `CGMParser.parse_tracks`. Tracks are alternative views, never
        shards: non-sensor rows are replicated into each frame, so
        concatenating two of them double-counts every meal.

        Args:
            file_path: A multi-track source file.
            track: Optionally restrict the result to one track. Accepts a real
                sensor name or the synthetic mean. The synthetic track is
                **only** available this way — it never appears in the default
                result, because it is a derived view rather than a member of
                the corpus.

        Returns:
            Track name → extended-schema frame.

        Raises:
            NotImplementedError: If the detected format is single-track.
        """
        path = Path(file_path)
        text_data = cls.decode_raw_data(path.read_bytes())
        format_type = cls.detect_format(text_data)

        if format_type != SupportedCGMFormat.CGMACROS:
            raise NotImplementedError(
                f"{format_type.value} is a single-track format; use "
                "parse_file(path). parse_tracks is for sources carrying more "
                "than one independent measurement of the same quantity."
            )

        if track is not None:
            if track not in CGMACROS_TRACKS and track != CGMACROS_MEAN_TRACK:
                raise ValueError(
                    f"Unknown track {track!r}; expected one of "
                    f"{CGMACROS_TRACKS} or {CGMACROS_MEAN_TRACK!r}"
                )
            return {track: cls._process_cgmacros(text_data, track=track)}

        return {
            name: cls._process_cgmacros(text_data, track=name)
            for name in CGMACROS_TRACKS
        }

    @classmethod
    def parse_corpus(
        cls,
        root: Union[str, Path],
        track: Optional[str] = None,
    ) -> Dict[str, UnifiedFormat]:
        """Parse a many-subject corpus into one frame per subject per track.

        See `CGMParser.parse_corpus`. Built out of `parse_tracks` / `parse_file`
        rather than reimplemented: composition is most of the value of naming
        the categories.

        Keys are `"<subject>/<track>"` for a multi-track corpus and `"<subject>"`
        for a single-track one. The `/` separator is public contract; subject
        ids may contain `_` but never `/`.

        Args:
            root: Corpus root directory.
            track: Optionally restrict every subject to one track.

        Returns:
            Subject (or `subject/track`) → frame, ordered by subject id so the
            mapping's iteration order is deterministic.
        """
        root_path = Path(root)
        format_type = cls.detect_path_format(root_path)

        if format_type in (
            SupportedCGMFormat.D1NAMO_DIABETES,
            SupportedCGMFormat.D1NAMO_HEALTHY,
        ):
            return cls._parse_d1namo_corpus(root_path)

        if format_type != SupportedCGMFormat.CGMACROS:
            raise NotImplementedError(
                f"No corpus walker registered for {format_type.value}"
            )

        # Enumerate from the filesystem, never from a numeric range: CGMacros
        # runs 001-049 with gaps at 024, 025, 037 and 040, so a range would
        # both miss subjects and look for ones that do not exist. Sorted, so
        # the mapping's order is deterministic.
        subject_dirs = sorted(
            (d for d in root_path.glob("CGMacros-*") if d.is_dir()),
            key=lambda d: d.name,
        )

        results: Dict[str, UnifiedFormat] = {}
        failures: Dict[str, str] = {}
        for subject_dir in subject_dirs:
            subject_csv = subject_dir / f"{subject_dir.name}.csv"
            if not subject_csv.exists():
                failures[subject_dir.name] = "no subject CSV"
                continue
            try:
                tracks = cls.parse_tracks(subject_csv, track=track)
            except (MalformedDataError, ZeroValidInputError) as e:
                # Collected and reported once with a count rather than one
                # warning per subject, and never silently skipped.
                failures[subject_dir.name] = str(e)[:200]
                continue
            for track_name, frame in tracks.items():
                results[f"{subject_dir.name}/{track_name}"] = frame

        if failures:
            logger.warning(
                "parse_corpus: %d of %d subject(s) yielded no frame: %s",
                len(failures),
                len(subject_dirs),
                "; ".join(f"{k} ({v})" for k, v in sorted(failures.items())),
            )

        if not results:
            raise ZeroValidInputError(
                f"No parseable subjects found under {root_path}"
            )

        return results

    # ===== D1NAMO: a bundle-per-subject corpus =====

    @classmethod
    def _d1namo_timestamp(
        cls,
        frame: pl.DataFrame,
        date_col: str,
        time_col: str,
    ) -> pl.Expr:
        """Combine D1NAMO's split date and time into one datetime expression.

        The healthy subset omits seconds (`11:35`), the diabetes subset does
        not (`19:14:00`), and both appear under the same header. Concatenating
        then probing a format tuple handles both without a per-subset branch.
        """
        combined = pl.concat_str(
            [pl.col(date_col).str.strip_chars(), pl.col(time_col).str.strip_chars()],
            separator=" ",
        )
        for date_fmt in D1NAMO_DATE_FORMATS:
            for time_fmt in D1NAMO_TIME_FORMATS:
                candidate = f"{date_fmt} {time_fmt}"
                probe = frame.select(
                    combined.str.strptime(
                        pl.Datetime("ms"), candidate, strict=False
                    ).alias("probe")
                )
                if probe["probe"].null_count() < len(frame):
                    return combined.str.strptime(
                        pl.Datetime("ms"), candidate, strict=False
                    )
        raise MalformedDataError(
            f"No D1NAMO timestamp format parsed {date_col}+{time_col}; "
            f"tried {D1NAMO_DATE_FORMATS} x {D1NAMO_TIME_FORMATS}"
        )

    @classmethod
    def _process_d1namo_subject(
        cls,
        subject_dir: Union[str, Path],
    ) -> UnifiedFormat:
        """Parse one D1NAMO subject directory — a bundle of modality files.

        Glucose, insulin, meals and annotations are separate files describing
        one person, merged into one frame. Which files exist identifies the
        subset: `insulin.csv` only in diabetes, `annotations.csv` only in
        healthy.

        Fingersticks map to `CALIBRAT`, never `EGV_READ` (D6): only `type ==
        "cgm"` is a continuous sensor reading, and the healthy subset has no
        CGM at all.

        `carbs` stays null throughout — D1NAMO records no carbohydrate
        anywhere, and a zero would assert something the source never said.
        """
        directory = Path(subject_dir)
        glucose_path = directory / "glucose.csv"
        if not glucose_path.exists():
            raise MalformedDataError(
                f"D1NAMO subject directory has no glucose.csv: {directory}"
            )

        frames: List[pl.DataFrame] = []
        schema = UNIFIED_TARGET_SCHEMA[SupportedCGMFormat.D1NAMO_DIABETES]

        # --- glucose.csv: readings, mmol/L, sensor vs fingerstick ---
        raw = pl.read_csv(glucose_path, infer_schema_length=0)
        raw = raw.rename({c: c.strip() for c in raw.columns})
        gcol = D1namoGlucoseColumn.GLUCOSE.value
        raw = raw.with_columns(
            cls._d1namo_timestamp(
                raw,
                D1namoGlucoseColumn.DATE.value,
                D1namoGlucoseColumn.TIME.value,
            ).alias("datetime"),
            # Cast from text, non-strict: a value the schema cannot represent
            # becomes null here and is REPORTED below rather than dropped
            # silently. Leading zeros ("08.2") parse fine; a colon typed for a
            # decimal point ("7:0") does not, and that difference is a finding,
            # not a nuisance.
            pl.col(gcol).str.strip_chars().cast(pl.Float64, strict=False).alias("_gl"),
        )

        unrepresentable = raw.filter(
            pl.col(gcol).str.strip_chars().is_in(D1NAMO_NULL_LITERALS).not_()
            & pl.col(gcol).is_not_null()
            & pl.col("_gl").is_null()
        )
        if len(unrepresentable) > 0:
            # "The source said something we cannot represent" is a different
            # report from "the source did not say" — the empty cells below.
            offenders = (
                unrepresentable.get_column(gcol).value_counts().iter_rows()
            )
            logger.warning(
                "%s: %d glucose reading(s) carried a value the schema cannot "
                "represent and were dropped: %s",
                directory.name,
                len(unrepresentable),
                ", ".join(f"{v!r} x{n}" for v, n in offenders),
            )

        readings = raw.drop_nulls("datetime").filter(pl.col("_gl").is_not_null())
        if len(readings) > 0:
            type_col = pl.col(D1namoGlucoseColumn.TYPE.value).str.strip_chars()
            frames.append(
                readings.with_columns(
                    # Only `cgm` is a sensor trace. Everything else is a
                    # fingerstick and says so.
                    pl.when(type_col.is_in(list(D1NAMO_SENSOR_TYPES)))
                    .then(pl.lit(UnifiedEventType.GLUCOSE.value))
                    .otherwise(pl.lit(UnifiedEventType.CALIBRATION.value))
                    .alias("event_type"),
                    pl.lit(0, dtype=pl.Int64).alias("quality"),
                    cls._glucose_to_canonical(
                        D1NAMO_GLUCOSE_SCHEMA, gcol, pl.col("_gl")
                    ).alias("glucose"),
                    pl.format(
                        '{{"reading_type":"{}"}}',
                        type_col.fill_null(""),
                    ).alias("annotations"),
                ).select("datetime", "event_type", "quality", "glucose", "annotations")
            )

        # --- insulin.csv: diabetes subset only ---
        insulin_path = directory / "insulin.csv"
        if insulin_path.exists():
            ins = pl.read_csv(insulin_path, infer_schema_length=0)
            ins = ins.rename({c: c.strip() for c in ins.columns})
            ins = ins.with_columns(
                cls._d1namo_timestamp(
                    ins,
                    D1namoInsulinColumn.DATE.value,
                    D1namoInsulinColumn.TIME.value,
                ).alias("datetime"),
                pl.col(D1namoInsulinColumn.FAST_INSULIN.value)
                .cast(pl.Float64, strict=False)
                .alias("insulin_fast"),
                pl.col(D1namoInsulinColumn.SLOW_INSULIN.value)
                .cast(pl.Float64, strict=False)
                .alias("insulin_slow"),
            ).drop_nulls("datetime")

            fast = ins.filter(pl.col("insulin_fast").is_not_null()).with_columns(
                pl.lit(UnifiedEventType.INSULIN_FAST.value).alias("event_type"),
                pl.lit(0, dtype=pl.Int64).alias("quality"),
            )
            if len(fast) > 0:
                frames.append(
                    fast.select("datetime", "event_type", "quality", "insulin_fast")
                )
            slow = ins.filter(pl.col("insulin_slow").is_not_null()).with_columns(
                pl.lit(UnifiedEventType.INSULIN_SLOW.value).alias("event_type"),
                pl.lit(0, dtype=pl.Int64).alias("quality"),
            )
            if len(slow) > 0:
                frames.append(
                    slow.select("datetime", "event_type", "quality", "insulin_slow")
                )

        # --- food.csv: two different headers, one per subset ---
        food_path = directory / "food.csv"
        if food_path.exists():
            frames.extend(cls._d1namo_food_frames(food_path, directory))

        if not frames:
            raise ZeroValidInputError(
                f"No usable D1NAMO rows in {directory}"
            )

        unified = pl.concat(frames, how="diagonal")
        unified = unified.with_columns(pl.lit(0).alias("sequence_id"))
        return cls._postprocess_unified(unified, schema=schema)

    @classmethod
    def _d1namo_food_frames(
        cls,
        food_path: Path,
        directory: Path,
    ) -> List[pl.DataFrame]:
        """Meal rows from either subset's `food.csv`.

        The two headers are a different column set, not a rename: the diabetes
        subset carries one EXIF-style `datetime`, the healthy subset a split
        `date` + `time`. Dispatch is on which columns are present.

        Photo references get two distinct reports. A blank cell is "the subject
        recorded no photograph"; a cell naming a file that is not on disk is
        "the source said something we cannot resolve". Collapsing them into one
        message would lose the difference.
        """
        food = pl.read_csv(food_path, infer_schema_length=0)
        food = food.rename({c: c.strip() for c in food.columns})

        if D1namoFoodColumn.DATETIME.value in food.columns:
            stamp = None
            for fmt in D1NAMO_FOOD_DATETIME_FORMATS:
                probe = food.select(
                    pl.col(D1namoFoodColumn.DATETIME.value)
                    .str.strip_chars()
                    .str.strptime(pl.Datetime("ms"), fmt, strict=False)
                    .alias("p")
                )
                if probe["p"].null_count() < len(food):
                    stamp = (
                        pl.col(D1namoFoodColumn.DATETIME.value)
                        .str.strip_chars()
                        .str.strptime(pl.Datetime("ms"), fmt, strict=False)
                    )
                    break
            if stamp is None:
                # Diabetes subject 005 carries the literal "NA" in `datetime`
                # on every one of its 9 meal rows — the only subject in the
                # corpus that does. A meal with no time cannot be placed on a
                # timeline, so the file yields nothing; but that is a reason to
                # drop the *meals*, not the subject's glucose and insulin.
                # Reported prominently, never silently.
                logger.warning(
                    "%s: no meal timestamp could be parsed from %s (all values "
                    "unparseable, e.g. the literal 'NA'). Meals are omitted "
                    "for this subject; glucose and insulin are unaffected.",
                    directory.name,
                    food_path.name,
                )
                return []
        else:
            stamp = cls._d1namo_timestamp(
                food,
                D1namoHealthyFoodColumn.DATE.value,
                D1namoHealthyFoodColumn.TIME.value,
            )

        picture = pl.col(D1namoFoodColumn.PICTURE.value).str.strip_chars()
        balance = pl.col(D1namoFoodColumn.BALANCE.value).str.strip_chars()
        quality_label = pl.col(D1namoFoodColumn.QUALITY.value).str.strip_chars()

        food = food.with_columns(
            stamp.alias("datetime"),
            pl.col(D1namoFoodColumn.CALORIES.value)
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias("calories"),
        ).drop_nulls("datetime")

        # Two separate reports, deliberately.
        photo_dir = directory / "food_pictures"
        on_disk = (
            {p.name for p in photo_dir.iterdir() if p.suffix.lower() == ".jpg"}
            if photo_dir.is_dir()
            else set()
        )
        referenced = [
            v.strip()
            for v in food.get_column(D1namoFoodColumn.PICTURE.value).to_list()
            if v and v.strip()
        ]
        absent = len(food) - len(referenced)
        dangling = sorted({v for v in referenced if v not in on_disk})
        if dangling:
            logger.warning(
                "%s: %d meal row(s) name a photograph that is not on disk: %s. "
                "The cells hold text where a filename belongs, so the "
                "reference cannot be resolved (distinct from a blank cell).",
                directory.name,
                len(dangling),
                ", ".join(repr(d) for d in dangling[:8]),
            )
        if absent:
            logger.info(
                "%s: %d meal row(s) recorded no photograph at all.",
                directory.name,
                absent,
            )

        annotations = pl.format(
            '{{"balance":{},"description":{},"picture":{},"quality":{}}}',
            cls._d1namo_json_str(balance),
            cls._d1namo_json_str(
                pl.col(D1namoFoodColumn.DESCRIPTION.value)
                .str.strip_chars()
                # Free text containing commas is already handled by the CSV
                # reader; quotes would break the JSON, so they are stripped.
                .str.replace_all('"', "")
            ),
            cls._d1namo_json_str(picture),
            cls._d1namo_json_str(quality_label),
        )

        meals = food.with_columns(
            pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
            pl.lit(0, dtype=pl.Int64).alias("quality"),
            annotations.alias("annotations"),
        )
        if len(meals) == 0:
            return []
        # `carbs` is deliberately absent: D1NAMO records no carbohydrate
        # anywhere, and _postprocess_unified will add it as a typed null. A
        # zero would assert something the source never said.
        return [meals.select("datetime", "event_type", "quality", "calories", "annotations")]

    @staticmethod
    def _d1namo_json_str(expr: pl.Expr) -> pl.Expr:
        """Render a text column as a JSON string or literal null.

        `No information` and an empty cell both mean the source did not say, so
        both become JSON `null` rather than the string "No information" — but
        the corrupt `8 Balance""` is a real value we cannot interpret and is
        preserved verbatim for a reader to see.
        """
        return (
            pl.when(expr.is_null() | expr.is_in(list(D1NAMO_NULL_LITERALS)))
            .then(pl.lit("null"))
            .otherwise(pl.format('"{}"', expr.str.replace_all('"', "")))
        )

    @classmethod
    def _parse_d1namo_corpus(cls, root: Path) -> Dict[str, UnifiedFormat]:
        """Walk a D1NAMO subset directory, one bundle per subject.

        Single-track, so keys are bare subject ids with no `/track` suffix.
        Subject ids come from the directory names, which is why the separator
        cannot be `_`: the healthy subset's twelfth subject is literally named
        `012_diabetes`, and a `_` split would mis-key it.
        """
        subject_dirs = sorted(
            (d for d in root.iterdir() if d.is_dir() and (d / "glucose.csv").exists()),
            key=lambda d: d.name,
        )

        results: Dict[str, UnifiedFormat] = {}
        failures: Dict[str, str] = {}
        for subject_dir in subject_dirs:
            try:
                results[subject_dir.name] = cls._process_d1namo_subject(subject_dir)
            except (MalformedDataError, ZeroValidInputError) as e:
                failures[subject_dir.name] = str(e)[:200]

        if failures:
            logger.warning(
                "parse_corpus: %d of %d D1NAMO subject(s) yielded no frame: %s",
                len(failures),
                len(subject_dirs),
                "; ".join(f"{k} ({v})" for k, v in sorted(failures.items())),
            )

        if not results:
            raise ZeroValidInputError(f"No parseable D1NAMO subjects under {root}")

        return results
