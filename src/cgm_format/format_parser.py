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
    

