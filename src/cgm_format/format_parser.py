"""Format converter for CGM vendor formats working on text data."""

from typing import Dict, List, Union, ClassVar
from io import StringIO
from typing import Union
import polars as pl
from pathlib import Path
from base64 import b64decode   


from cgm_format.interface.cgm_interface import (
    CGMParser,
    SupportedCGMFormat,
    UnifiedFormat,
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    ValidationMethod,
    EXPECTED_INTERVAL_MINUTES,
    SMALL_GAP_MAX_MINUTES,
)

# Import detection patterns from format modules
from cgm_format.formats.unified import (
    UNIFIED_DETECTION_PATTERNS, 
    UnifiedEventType, 
    Quality, 
    UNIFIED_TIMESTAMP_FORMATS,
    CGM_SCHEMA
)
from cgm_format.formats.dexcom import (
    DEXCOM_DETECTION_PATTERNS, 
    DexcomColumn, 
    DEXCOM_HEADER_LINE,
    DEXCOM_DATA_START_LINE,
    DEXCOM_METADATA_LINES,
    DEXCOM_TIMESTAMP_FORMATS,
    DEXCOM_HIGH_GLUCOSE_DEFAULT,
    DEXCOM_LOW_GLUCOSE_DEFAULT,
)
from cgm_format.formats.libre import (
    LIBRE_DETECTION_PATTERNS,
    LibreColumn, 
    LibreRecordType, 
    LIBRE_HEADER_LINE,
    LIBRE_DATA_START_LINE,
    LIBRE_METADATA_LINES,
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

# CSV format detection patterns (checks first N lines of file)
CGM_DETECTION_PATTERNS: Dict[SupportedCGMFormat, List[str]] = {
    SupportedCGMFormat.UNIFIED_CGM: UNIFIED_DETECTION_PATTERNS,
    SupportedCGMFormat.DEXCOM: DEXCOM_DETECTION_PATTERNS,
    SupportedCGMFormat.LIBRE: LIBRE_DETECTION_PATTERNS,
}

DETECTION_LINE_COUNT = 15  # lines to check

class FormatParser(CGMParser):
    """Main format parser implementing the CGMParser interface.
    
    This class orchestrates the parsing pipeline from raw data to unified format:
    1. Decode raw data (remove BOM, fix encoding)
    2. Detect format (determine vendor)
    3. Parse to unified format (vendor-specific processing)
    """
    
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT

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
        lines = text_data.split('\n',DETECTION_LINE_COUNT+1)[:DETECTION_LINE_COUNT]
        
        # Check each CGM type's patterns
        for cgm_type, patterns in CGM_DETECTION_PATTERNS.items():
            if any(pattern in line for line in lines for pattern in patterns):
                return cgm_type
        
        raise UnknownFormatError(f"Unknown CGM data format. Sample lines: {lines[:3]}")


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
        
        # Mark duplicate timestamps
        #unified_df = cls.mark_time_duplicates(unified_df) # moved to processor

        #Final step: Detect and assign sequences (lossless annotation)
        unified_df = cls.detect_and_assign_sequences(unified_df)

        return unified_df
    
    def _probe_timestamp_format(df: pl.DataFrame, column_name: str, formats: tuple) -> str:
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
        
        raise MalformedDataError(f"Could not parse timestamps with any known format: {formats}")
    
    
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
            
            # Parse datetime column if it's a string
            # Unified format may have datetime as string when loaded from CSV
            if df['datetime'].dtype == pl.Utf8 or df['datetime'].dtype == pl.String:
                # Probe timestamp format for this file
                timestamp_format = FormatParser._probe_timestamp_format(df, 'datetime', UNIFIED_TIMESTAMP_FORMATS)
                
                # Parse datetime column to Datetime type
                df = df.with_columns([
                    pl.col('datetime').str.strptime(pl.Datetime("ms"), timestamp_format)
                ])
            
            # Parse original_datetime column if it's a string
            if df['original_datetime'].dtype == pl.Utf8 or df['original_datetime'].dtype == pl.String:
                # Probe timestamp format for this file (use same logic as datetime)
                timestamp_format = FormatParser._probe_timestamp_format(df, 'original_datetime', UNIFIED_TIMESTAMP_FORMATS)
                
                # Parse original_datetime column to Datetime type
                df = df.with_columns([
                    pl.col('original_datetime').str.strptime(pl.Datetime("ms"), timestamp_format)
                ])
             
            return cls._postprocess_unified(df)
            
        except pl.exceptions.PolarsError as e:
            raise MalformedDataError(f"Failed to parse unified format CSV: {e}")
    
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
            raise MalformedDataError(f"Failed to parse Dexcom CSV: {e}")
    
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
            raise MalformedDataError(f"Failed to parse Libre CSV: {e}")
    
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
    
    # ===== Convenience Methods =====
    
    @classmethod
    def parse_from_file(cls, file_path: str) -> UnifiedFormat:
        """Convenience method to parse a CGM file directly to unified format.
        
        This method chains all stages together:
        1. Read file
        2. Decode raw data
        3. Detect format
        4. Parse to unified format
        
        Args:
            file_path: Path to CGM data file
            
        Returns:
            DataFrame in unified format
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        # Read file as bytes
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Chain all stages
        text_data = cls.decode_raw_data(raw_data)
        format_type = cls.detect_format(text_data)
        return cls.parse_to_unified(text_data, format_type)
    
    @classmethod
    def parse_from_bytes(cls, raw_data: bytes) -> UnifiedFormat:
        """Convenience method to parse raw bytes directly to unified format.
        
        This method chains all stages together:
        1. Decode raw data
        2. Detect format
        3. Parse to unified format
        4. Detect and assign sequences (lossless annotation)
        
        Args:
            raw_data: Raw file contents as bytes
            
        Returns:
            DataFrame in unified format with sequence_id assigned
            
        Raises:
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        text_data = cls.decode_raw_data(raw_data)
        format_type = cls.detect_format(text_data)
        return cls.parse_to_unified(text_data, format_type)
    
    @classmethod
    def parse_from_string(cls, text_data: str) -> UnifiedFormat:
        """Convenience method to parse cleaned string directly to unified format.
        
        This method assumes data is already decoded and chains:
        1. Detect format
        2. Parse to unified format
        3. Detect and assign sequences (lossless annotation)
        
        Args:
            text_data: Cleaned CSV string
            
        Returns:
            DataFrame in unified format with sequence_id assigned
            
        Raises:
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        format_type = cls.detect_format(text_data)
        return cls.parse_to_unified(text_data, format_type)
        
    
    @classmethod
    def parse_file(cls, file_path: Union[str, Path]) -> UnifiedFormat:
        """Parse CGM data from file path.
        
        Convenience method that reads file and parses to unified format.
        Automatically detects format and handles encoding.
        
        Args:
            file_path: Path to CGM data file (CSV format)
            
        Returns:
            DataFrame in unified format
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        return cls.parse_from_bytes(raw_data)
    
    @classmethod
    def parse_base64(cls, base64_data: str) -> UnifiedFormat:
        """Parse CGM data from base64 encoded string.
        
        Useful for web API endpoints that receive base64 encoded CSV data.
        Automatically decodes base64, detects format, and parses to unified format.
        
        Args:
            base64_data: Base64 encoded CSV data string
            
        Returns:
            DataFrame in unified format
            
        Raises:
            ValueError: If base64 decoding fails
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """        
        try:
            raw_data = b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}")
        
        return cls.parse_from_bytes(raw_data)
    
    # ===== STAGE 4: Sequence Detection (Lossless Annotation) =====
    
    @classmethod
    def detect_and_assign_sequences(
        cls, 
        dataframe: UnifiedFormat,
        expected_interval_minutes: int = EXPECTED_INTERVAL_MINUTES,
        large_gap_threshold_minutes: int = SMALL_GAP_MAX_MINUTES
    ) -> UnifiedFormat:
        """Detect large gaps and assign sequence_id column (lossless annotation).
        
        This is the final step in parsing that splits data into continuous sequences
        based on time gaps IN GLUCOSE EVENTS ONLY. Non-glucose events are then
        assigned to the nearest glucose sequence by time.
        
        Large gaps (> large_gap_threshold_minutes) between glucose readings create new sequences.
        sequence_id = 0 means unassigned (no glucose events available for assignment).
        sequence_id >= 1 means assigned to a glucose sequence.
        
        Two-pass approach:
        1. Detect sequences based on glucose event gaps only
        2. Assign non-glucose events to nearest glucose sequence by time
        
        This prevents non-glucose events from "bridging" glucose gaps and incorrectly
        keeping discontinuous glucose data in the same sequence.
        
        Args:
            dataframe: DataFrame in unified format (may or may not have sequence_id)
            expected_interval_minutes: Expected data collection interval (default: 5)
            large_gap_threshold_minutes: Threshold for creating new sequences (default: 15)
            
        Returns:
            DataFrame with sequence_id column assigned
        """
        if len(dataframe) == 0:
            return dataframe
        
        # Verify input dataframe matches schema
        if cls.validation_mode & (ValidationMethod.INPUT | ValidationMethod.INPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(dataframe, enforce=cls.validation_mode & ValidationMethod.INPUT_FORCED)

        large_gap_threshold_seconds: int = large_gap_threshold_minutes * 60
        
        # Get canonical sequence_id dtype from schema
        sequence_id_dtype = next(
            col['dtype'] for col in CGM_SCHEMA.service_columns if col['name'] == 'sequence_id'
        )
        
        # Sort by datetime
        df = dataframe.sort('datetime')
        
        # Check if sequence_id exists and has variation
        has_sequence_id = 'sequence_id' in df.columns
        
        if has_sequence_id:
            unique_seq_ids = df['sequence_id'].n_unique()
            
            # If there are multiple sequences, process each separately to detect internal gaps
            if unique_seq_ids > 1:
                return cls._split_sequences_with_internal_gaps(
                    df, large_gap_threshold_seconds, sequence_id_dtype
                )
        
        # Single sequence or no sequence_id: create sequences based on GLUCOSE GAPS ONLY
        # Pass 1: Filter to glucose events only
        glucose_events = df.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value).sort('datetime')
        
        if len(glucose_events) == 0:
            # No glucose events - all events get sequence_id = 0 (unassigned)
            return df.with_columns([
                pl.lit(0).cast(sequence_id_dtype).alias('sequence_id')
            ])
        
        # Calculate time differences between glucose events only
        glucose_events = glucose_events.with_columns([
            pl.col('datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
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
            # For each non-glucose event, find the closest glucose sequence by time
            # Use join_asof to find nearest glucose event
            sequence_info = glucose_events.select(['datetime', 'sequence_id'])
            
            # Drop sequence_id from non_glucose_events before join to avoid suffix
            non_glucose_no_seq = non_glucose_events.drop('sequence_id')
            
            # Join non-glucose events to nearest glucose event (by time)
            non_glucose_with_seq = non_glucose_no_seq.join_asof(
                sequence_info,
                on='datetime',
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
        if cls.validation_mode & (ValidationMethod.OUTPUT | ValidationMethod.OUTPUT_FORCED):
            CGM_SCHEMA.validate_dataframe(result_df, enforce=cls.validation_mode & ValidationMethod.OUTPUT_FORCED)
        
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
            seq_data = dataframe.filter(pl.col('sequence_id') == seq_id).sort('datetime')
            
            # Filter to glucose events only for gap detection
            glucose_seq = seq_data.filter(pl.col('event_type') == UnifiedEventType.GLUCOSE.value).sort('datetime')
            
            if len(glucose_seq) < 2:
                # Single glucose point or no glucose points, keep as is
                processed_glucose_sequences.append(glucose_seq)
                continue
            
            # Check for internal large gaps in glucose events
            glucose_seq = glucose_seq.with_columns([
                pl.col('datetime').diff().dt.total_seconds().alias('time_diff_seconds'),
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
        
        all_glucose = pl.concat(processed_glucose_sequences).sort('datetime')
        
        # Now reassign non-glucose events to nearest glucose sequence
        non_glucose_events = dataframe.filter(pl.col('event_type') != UnifiedEventType.GLUCOSE.value)
        
        if len(non_glucose_events) == 0:
            # Only glucose events
            result_df = all_glucose
        else:
            # Join non-glucose events to nearest glucose sequence
            sequence_info = all_glucose.select(['datetime', 'sequence_id'])
            
            # Drop sequence_id from non_glucose_events before join to avoid suffix
            non_glucose_no_seq = non_glucose_events.drop('sequence_id')
            
            non_glucose_with_seq = non_glucose_no_seq.join_asof(
                sequence_info,
                on='datetime',
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
    

