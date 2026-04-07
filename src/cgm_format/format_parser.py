"""Format converter for CGM vendor formats working on text data."""

from typing import Dict, List, Union, ClassVar, Optional
from io import StringIO
from typing import Union
import polars as pl
from pathlib import Path
from base64 import b64decode   

from cgm_format.formats.supported import FORMAT_DETECTION_PATTERNS, DETECTION_LINE_COUNT
from cgm_format.interface.cgm_interface import (
    CGMParser,
    SupportedCGMFormat,
    UnifiedFormat,
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    ValidationMethod,
    truncate_error_message,
)

# Import detection patterns from format modules
from cgm_format.formats.unified import (
    UnifiedEventType, 
    Quality, 
    UNIFIED_TIMESTAMP_FORMATS,
    CGM_SCHEMA,
)
from cgm_format.formats.dexcom import (
    DexcomColumn, 
    DEXCOM_METADATA_LINES,
    DEXCOM_TIMESTAMP_FORMATS,
    DEXCOM_HIGH_GLUCOSE_DEFAULT,
    DEXCOM_LOW_GLUCOSE_DEFAULT,
)
from cgm_format.formats.libre import (
    LibreColumn, 
    LibreRecordType, 
    LIBRE_HEADER_LINE,
    LIBRE_TIMESTAMP_FORMATS
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

        if format_type == SupportedCGMFormat.UNIFIED_CGM:
            unified_df = cls._process_unified(text_data)
        elif format_type == SupportedCGMFormat.DEXCOM:
            unified_df = cls._process_dexcom(text_data)
        elif format_type == SupportedCGMFormat.LIBRE:
            unified_df = cls._process_libre(text_data)
        elif format_type == SupportedCGMFormat.MEDTRONIC:
            unified_df = cls._process_medtronic(text_data)
        elif format_type == SupportedCGMFormat.NIGHTSCOUT:
            unified_df = cls._process_nightscout(text_data)
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
    def _postprocess_unified(cls, unified_df: UnifiedFormat) -> UnifiedFormat:
        """Postprocess the unified format dataframe.
        
        Args:
            unified_df: DataFrame in unified format
        """
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
        unified_df = CGM_SCHEMA.validate_dataframe(unified_df, enforce=True)
        
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
        # Get first non-null timestamp value for probing
        sample = df.filter(pl.col(column_name).is_not_null()).select(column_name).head(1)
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
    def _process_unified(cls, text_data: str) -> UnifiedFormat:
        """Process data already in unified format (validation only).
        
        Args:
            text_data: CSV string in unified format
            
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
             
            return cls._postprocess_unified(df)
            
        except pl.exceptions.PolarsError as e:
            error_msg = f"Failed to parse unified format CSV: {e}"
            raise MalformedDataError(cls._truncate_error_message(error_msg))
    
    @classmethod
    def _process_dexcom(
        cls,
        text_data: str, 
        high_glucose_value: int = DEXCOM_HIGH_GLUCOSE_DEFAULT, 
        low_glucose_value: int = DEXCOM_LOW_GLUCOSE_DEFAULT
    ) -> UnifiedFormat:
        """Process Dexcom CSV to unified format.
        
        Args:
            text_data: Dexcom CSV string
            high_glucose_value: Value to replace 'High' readings (default 401 mg/dL)
            low_glucose_value: Value to replace 'Low' readings (default 39 mg/dL)
            
        Returns:
            DataFrame in unified format
            
        Raises:
            MalformedDataError: If parsing fails
        """
        try:
            # Dexcom has: Row 1 = column headers, Rows 2-11 = metadata, Row 12+ = data
            # Use skip_rows_after_header to skip the metadata rows
            # Note: truncate_ragged_lines=True handles variable-length rows (non-EGV events
            # missing Transmitter ID/Time columns). Polars pads missing trailing cells with nulls.
            df = pl.read_csv(
                StringIO(text_data),
                skip_rows_after_header=len(DEXCOM_METADATA_LINES),  # Skip metadata rows after header
                truncate_ragged_lines=True,  # Handle Dexcom's variable-length rows
                infer_schema_length=None,
                ignore_errors=False
            )
            
            # Clean column names
            df = df.rename({col: col.strip().replace('"', '').replace('"', '') for col in df.columns})
            
            # Probe timestamp format once for this file
            timestamp_format = FormatParser._probe_timestamp_format(df, DexcomColumn.TIMESTAMP, DEXCOM_TIMESTAMP_FORMATS)
            
            # Process EGV (glucose) rows
            egv_data = (df
                .filter(pl.col(DexcomColumn.EVENT_TYPE).str.to_lowercase() == "egv")
                .select([
                    pl.col(DexcomColumn.TIMESTAMP).alias("datetime"),
                    pl.col(DexcomColumn.GLUCOSE_VALUE).alias("glucose"),
                    pl.col(DexcomColumn.EVENT_SUBTYPE).alias("subtype"),
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
                    # Replace High/Low with numeric placeholders for processing
                    # High = >400 mg/dL (sensor max), Low = <50 mg/dL (sensor min)
                    pl.col("glucose")
                    .cast(pl.Utf8)
                    .str.replace("High", str(high_glucose_value))
                    .str.replace("Low", str(low_glucose_value))
                    .cast(pl.Float64, strict=False)
                    .alias("glucose"),
                    # Mark out-of-range readings with OUT_OF_RANGE flag (sensor error, not real data)
                    # Values of 50 or 400 mg/dL would indicate severe hypo/hyperglycemia
                    pl.when(pl.col("is_out_of_range"))
                    .then(pl.lit(Quality.OUT_OF_RANGE.value))
                    .otherwise(pl.lit(0))  # 0 = GOOD (no flags)
                    .alias("quality"),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                ])
                .drop(["subtype", "is_out_of_range"])
            )
            
            # Process insulin events
            insulin_data = (df
                .filter(pl.col(DexcomColumn.EVENT_TYPE) == "Insulin")
                .select([
                    pl.col(DexcomColumn.TIMESTAMP).alias("datetime"),
                    pl.col(DexcomColumn.EVENT_SUBTYPE).alias("subtype"),
                    pl.col(DexcomColumn.INSULIN_VALUE).alias("insulin_value"),
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
                .filter(pl.col(DexcomColumn.EVENT_TYPE) == "Carbs")
                .select([
                    pl.col(DexcomColumn.TIMESTAMP).alias("datetime"),
                    pl.col(DexcomColumn.CARB_VALUE).alias("carbs"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Process exercise events
            exercise_data = (df
                .filter(pl.col(DexcomColumn.EVENT_TYPE) == "Exercise")
                .select([
                    pl.col(DexcomColumn.TIMESTAMP).alias("datetime"),
                    pl.col(DexcomColumn.DURATION).alias("duration_str"),
                    pl.col(DexcomColumn.EVENT_SUBTYPE).alias("subtype"),
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
    def _process_libre(cls, text_data: str) -> UnifiedFormat:
        """Process FreeStyle Libre CSV to unified format.
        
        Args:
            text_data: Libre CSV string
            
        Returns:
            DataFrame in unified format
            
        Raises:
            MalformedDataError: If parsing fails
        """
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
            
            # Probe timestamp format once for this file
            timestamp_format = FormatParser._probe_timestamp_format(df, LibreColumn.DEVICE_TIMESTAMP, LIBRE_TIMESTAMP_FORMATS)
            
            # Process historic glucose data (Record Type = 0)
            glucose_data = (df
                .filter(pl.col(LibreColumn.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.HISTORIC_GLUCOSE.value)
                .select([
                    pl.col(LibreColumn.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(LibreColumn.HISTORIC_GLUCOSE).alias("glucose"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.col("glucose").cast(pl.Float64, strict=False),
                    pl.lit(UnifiedEventType.GLUCOSE.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Process insulin events (Record Type = 4)
            insulin_data = (df
                .filter(pl.col(LibreColumn.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.INSULIN.value)
                .select([
                    pl.col(LibreColumn.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(LibreColumn.RAPID_INSULIN).alias("insulin_fast"),
                    pl.col(LibreColumn.LONG_INSULIN).alias("insulin_slow"),
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
                .filter(pl.col(LibreColumn.RECORD_TYPE).cast(pl.Int64) == LibreRecordType.FOOD.value)
                .select([
                    pl.col(LibreColumn.DEVICE_TIMESTAMP).alias("datetime"),
                    pl.col(LibreColumn.CARBOHYDRATES_GRAMS).alias("carbs"),
                ])
                .with_columns([
                    pl.col("datetime").str.strptime(pl.Datetime("ms"), timestamp_format),
                    pl.lit(UnifiedEventType.CARBOHYDRATES.value).alias("event_type"),
                    pl.lit(0).alias("quality"),  # 0 = GOOD (no flags)
                ])
            )
            
            # Combine all data types
            all_data = [glucose_data]
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

        Convenience wrapper around :meth:`parse_nightscout` that reads files
        from disk.  Accepts both JSON and CSV for entries; treatments are
        expected to be JSON (the Nightscout API does not support CSV for
        treatments).

        Args:
            entries_path: Path to entries file (JSON or CSV)
            treatments_path: Optional path to treatments JSON file
            profile_path: Reserved for future use (profile data)

        Returns:
            DataFrame in unified format matching CGM_SCHEMA
        """
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
    
    @staticmethod
    def to_csv_string(dataframe: UnifiedFormat) -> str:
        """Serialize unified format DataFrame to CSV string.
        
        Args:
            dataframe: DataFrame in unified format
            
        Returns:
            CSV string representation
        """
        # Verify input dataframe matches schema
        if FormatParser.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=FormatParser.validation_mode & ValidationMethod.INPUT_FORCED)    
        return dataframe.write_csv(separator=",")
    
    @staticmethod
    def to_csv_file(dataframe: UnifiedFormat, file_path: str) -> None:
        """Save unified format DataFrame to CSV file.
        
        Args:
            dataframe: DataFrame in unified format
            file_path: Path where to save the CSV file
        """
        # Verify input dataframe matches schema
        if FormatParser.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=FormatParser.validation_mode & ValidationMethod.INPUT_FORCED)   
        dataframe.write_csv(file_path)
    

