"""Abstract Base Class interface for CGM data processing pipeline.

Separated into two concerns:
- CGMParser: Vendor-specific parsing to unified format (Stages 1-3)
- CGMProcessor: Vendor-agnostic unified format processing (Stages 4-5)
"""

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Union, Tuple
from enum import Enum
from pathlib import Path
import polars as pl

# Check pandas availability
try:
    import pyarrow as pa
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# from schema import EventType, Quality

CALIBRATION_GAP_THRESHOLD = 2*60*60+45*60  # 2 hours and 45 minutes
CALIBRATION_PERIOD_HOURS = 24

EXPECTED_INTERVAL_MINUTES = 5
TOLERANCE_INTERVAL_MINUTES = 1.2*EXPECTED_INTERVAL_MINUTES
SMALL_GAP_MAX_MINUTES = EXPECTED_INTERVAL_MINUTES * 3 + 0.8*EXPECTED_INTERVAL_MINUTES #3 intervals +- 40% tolerance on each side

# Expected sequence (- = 1 minue, | = registered value, every 5 minutes):
#|-----|-----|-----|-----|-----|
#Synchronizeable/fillable gap schema (x - missing value, : - synchronized value, | - registered value):
#|---|--:-----x-----x-----:--|---|

MINIMUM_DURATION_MINUTES = 60 # minimum expected duration of a sequence for inference
MAXIMUM_WANTED_DURATION_MINUTES = 480 # maximum duration of a sequence to be included in the inference

# Type alias to highlight that this is the unified format
# (No way to add additional constraints on DF shape in type hints)
UnifiedFormat = pl.DataFrame

class SupportedCGMFormat(Enum):
    """Supported CGM vendor formats."""
    DEXCOM = "dexcom"
    LIBRE = "libre"
    UNIFIED_CGM = "unified"  # Format that this library provides

class ValidationMethod(Flag):
    """Validation method for validating the input and output dataframes."""
    INPUT = auto()
    OUTPUT = auto()
    INPUT_FORCED = auto()
    OUTPUT_FORCED = auto()

NO_VALIDATION = ValidationMethod(0)
class ProcessingWarning(Flag):
    """Warnings that can occur during additional transformations.
    
    These are flags that can be combined using bitwise OR operations.
    Example: warnings = ProcessingWarning.TOO_SHORT | ProcessingWarning.QUALITY
    """
    TOO_SHORT = auto()  # Minimum duration requirement not met
    CALIBRATION = auto()  # Output sequence contains calibration events or 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD
    OUT_OF_RANGE = auto()  # Contains out-of-range values
    IMPUTATION = auto()  # Contains imputed gaps
    TIME_DUPLICATES = auto()  # Sequence contains non-unique time entries
    SYNCHRONIZATION = auto()  # Sequence undergone synchronization corrections
    QUALITY = auto()  # Other quality issues

NO_WARNING = ProcessingWarning(0)

class WarningDescription(Enum):
    """Descriptions of warnings."""
    TOO_SHORT = "Minimum duration requirement not met"
    CALIBRATION = "Sequence contains calibration events or 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD"
    OUT_OF_RANGE = "Contains out-of-range values"
    IMPUTATION = "Contains imputed gaps"
    TIME_DUPLICATES = "Sequence contains non-unique time entries"
    SYNCHRONIZATION = "Sequence undergone synchronization corrections"
    QUALITY = "Other quality issues"

# Simple tuple return types
ValidationResult = Tuple[pl.DataFrame, int, int]  # (dataframe, bad_rows, valid_rows)
InferenceResult = Tuple[UnifiedFormat, ProcessingWarning]  # (dataframe, warnings)



class MalformedDataError(ValueError):
    """Raised when data cannot be parsed or converted properly."""
    pass

class MissingColumnError(MalformedDataError):
    """Raised when a required column is missing from the dataframe."""
    pass

class ExtraColumnError(MalformedDataError):
    """Raised when an extra column is present in the dataframe."""
    pass

class ColumnOrderError(MalformedDataError):
    """Raised when the column order is not correct."""
    pass

class ColumnTypeError(MalformedDataError):
    """Raised when the column type is not correct."""
    pass

class UnknownFormatError(ValueError):
    """Raised when format cannot be determined."""
    pass

class ZeroValidInputError(ValueError):
    """Raised when there are no valid data points in the sequence."""
    pass


class CGMParser(ABC):
    """Abstract base class for vendor-specific CGM data parsing (Stages 1-3).
    
    This interface handles:
    - Stage 1: Preprocessing raw data (BOM removal, encoding fixes)
    - Stage 2: Format detection (identifying vendor)
    - Stage 3: Vendor-specific parsing to unified format
    
    After stage 3, data is in UnifiedFormat and can be serialized or passed to CGMProcessor.

    """
    
    # ===== STAGE 1: Preprocess Raw Data =====
    
    @classmethod
    @abstractmethod
    def decode_raw_data(cls, raw_data: Union[bytes, str]) -> str:
        """Remove BOM marks, encoding artifacts, and other junk from raw input.
        
        Args:
            raw_data: Raw file contents (bytes or string)
            
        Returns:
            Cleaned string data ready for format detection
        """
        pass
    
    # ===== STAGE 2: Format Detection  =====
    
    @classmethod
    @abstractmethod
    def detect_format(cls, text_data: str) -> SupportedCGMFormat:
        """Guess the vendor format based on header patterns in raw CSV string.
        
        This determines which vendor-specific processor to use.
        Works on string data before parsing to avoid vendor-specific CSV quirks.
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            SupportedCGMFormat enum value 
            
        Raises:
            UnknownFormatError: If format cannot be determined
        """
        pass

    # ===== STAGE 3: Device-Specific Parsing to Unified Format =====
    
    @classmethod
    @abstractmethod
    def parse_to_unified(cls, text_data: str, format_type: SupportedCGMFormat) -> UnifiedFormat:
        """Parse vendor-specific CSV to unified format (device-specific parsing).
        
        This stage combines:
        - CSV validation and sanity checks
        - Vendor-specific quirk handling (High/Low values, timezone fixes, etc.)
        - Column mapping to unified schema
        - Populating service fields (event_type, quality)
        - Sequence detection and assignment (sequence_id)
        
        After this stage, processing flow converges to UnifiedFormat with sequence_id assigned.
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            DataFrame in unified format matching CGM_SCHEMA with sequence_id assigned
            
        Raises:
            MalformedDataError: If CSV is unparseable, zero valid rows, or conversion fails
        """
        pass
    
    # ===== STAGE 4: Sequence Detection (Lossless Annotation) =====
    
    @classmethod
    @abstractmethod
    def detect_and_assign_sequences(
        cls,
        dataframe: UnifiedFormat,
        expected_interval_minutes: int = 5,
        large_gap_threshold_minutes: int = 15
    ) -> UnifiedFormat:
        """Detect large gaps and assign sequence_id column (lossless annotation).
        
        This is a final parsing step that splits data into continuous sequences
        based on time gaps. It's a lossless operation that only adds metadata.
        
        **Separation of Concerns:**
        - This method is called automatically at the end of parse_to_unified()
        - Can also be called standalone for re-detecting sequences on existing data
        - Ensures sequence_id is always present in parsed data
        
        Large gaps (> large_gap_threshold_minutes) create new sequences.
        This method is idempotent - if sequence_id already exists, it validates
        and potentially splits sequences with internal large gaps.
        
        Args:
            dataframe: DataFrame in unified format (may or may not have sequence_id)
            expected_interval_minutes: Expected data collection interval (default: 5)
            large_gap_threshold_minutes: Threshold for creating new sequences (default: 15)
            
        Returns:
            DataFrame with sequence_id column assigned
        """
        pass
    
    # ===== Serialization (Roundtrip Support) =====
    
    @staticmethod
    def to_csv_string(dataframe: UnifiedFormat) -> str:
        """Serialize unified format DataFrame to CSV string.
        
        Args:
            dataframe: DataFrame in unified format
            
        Returns:
            CSV string representation of the unified format
        """
        return dataframe.write_csv(separator=",")
    
    @staticmethod
    def to_csv_file(dataframe: UnifiedFormat, file_path: str) -> None:
        """Save unified format DataFrame to CSV file.
        
        Args:
            dataframe: DataFrame in unified format
            file_path: Path where to save the CSV file
        """
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
        from base64 import b64decode
        
        try:
            raw_data = b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}")
        
        return cls.parse_from_bytes(raw_data)
    
    @classmethod
    def mark_time_duplicates(cls, df: "pl.DataFrame") -> "pl.DataFrame":
        """Mark events with duplicate timestamps (keeping first occurrence).
        
        Uses keepfirst logic: the first event at a timestamp is kept clean,
        subsequent events with the same timestamp are marked with TIME_DUPLICATE flag.
        
        Args:
            df: DataFrame in unified format (must have 'datetime' and 'quality' columns)
            
        Returns:
            DataFrame with TIME_DUPLICATE flag added to quality column for duplicate timestamps
        """
        pass


class CGMProcessor(ABC):
    """Abstract base class for unified CGM data processing (Stages 4-5).
    
    This interface handles vendor-agnostic operations on UnifiedFormat data:
    - Stage 4: Postprocessing (synchronization, interpolation)
    - Stage 5: Inference preparation (truncation, validation, warnings)
    
    This class operates only on data already in UnifiedFormat with sequence_id assigned.
    Data should come from CGMParser (which assigns sequences automatically) or have
    sequences assigned via detect_and_assign_sequences() before processing.
    
    **Important**: All methods expect sequence_id to exist in the input dataframe.
    If parsing with FormatParser, sequences are assigned automatically.
    """
    
    # ===== STAGE 4: Postprocessing (Unified Operations) =====
    
    @abstractmethod
    def synchronize_timestamps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Align timestamps to minute boundaries.
        
        Args:
            dataframe: DataFrame in unified format
            
        Returns:
            DataFrame with synchronized timestamps
        """
        pass
    
    @abstractmethod
    def interpolate_gaps(self, dataframe: UnifiedFormat) -> UnifiedFormat:
        """Fill gaps in continuous data with imputed values.
        
        Adds rows with Quality.IMPUTATION flag for missing data points.
        Updates ProcessingWarning.IMPUTATION flag if gaps were filled.
        
        Args:
            dataframe: DataFrame with potential gaps
            
        Returns:
            DataFrame with interpolated values marked with IMPUTATION quality flag
        """
        pass
    
    # ===== STAGE 5: Inference Preprocessing =====
    
    @abstractmethod
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
        - Filter to glucose-only events if requested (drops non-EGV events)
        - Truncate sequences exceeding maximum_wanted_duration
        - Drop duplicate timestamps if requested
        - Raise global output warning flags based on individual row quality
        - Check minimum duration requirements
        
        Returns full UnifiedFormat with all columns (sequence_id, event_type, quality, etc).
        Use to_data_only_df() to strip service columns if needed for ML models.
        
        Args:
            dataframe: Fully processed DataFrame in unified format
            minimum_duration_minutes: Minimum required sequence duration
            maximum_wanted_duration: Maximum desired sequence duration (truncates if exceeded)
            glucose_only: If True, drop non-EGV events before truncation (keeps only GLUCOSE)
            drop_duplicates: If True, drop duplicate timestamps (keeps first occurrence)
            
        Returns:
            Tuple of (unified_format_dataframe, warnings)
            
        Raises:
            ZeroValidInputError: If there are no valid data points
        """
        pass
    
# ============================================================================
# Compatibility Layer: Output Adapters
# ============================================================================

def to_pandas(df: pl.DataFrame) -> "pd.DataFrame":
    """Convert polars DataFrame to pandas.
    
    Raises:
        ImportError: If pandas and pyarrow are not installed
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas and pyarrow are required for this function. "
        )
    return df.to_pandas()

def to_polars(df: "pd.DataFrame") -> pl.DataFrame:
    """Convert pandas DataFrame to polars.
    
    Raises:
        ImportError: If arrow and pandas are not installed
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas and pyarrow are required for this function. "
        )
    return pl.from_pandas(df)

