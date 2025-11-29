"""Abstract Base Class interface for CGM data processing pipeline.

Separated into two concerns:
- CGMParser: Vendor-specific parsing to unified format (Stages 1-3)
- CGMProcessor: Vendor-agnostic unified format processing (Stages 4-5)
"""

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Union, Tuple
from enum import Enum
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

class ProcessingWarning(Flag):
    """Warnings that can occur during additional transformations.
    
    These are flags that can be combined using bitwise OR operations.
    Example: warnings = ProcessingWarning.TOO_SHORT | ProcessingWarning.QUALITY
    """
    TOO_SHORT = auto()  # Minimum duration requirement not met
    CALIBRATION = auto()  # Output sequence contains calibration events
    QUALITY = auto()  # Contains ill or sensor calibration events
    IMPUTATION = auto()  # Contains imputed gaps
    TIME_DUPLICATES = auto()  # Sequence contains non-unique time entries

# Simple tuple return types
ValidationResult = Tuple[pl.DataFrame, int, int]  # (dataframe, bad_rows, valid_rows)
InferenceResult = Tuple[pl.DataFrame, ProcessingWarning]  # (dataframe, warnings)



class MalformedDataError(ValueError):
    """Raised when data cannot be parsed or converted properly."""
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
        - Populating service fields (sequence_id, event_type, quality)
        
        After this stage, processing flow converges to UnifiedFormat.
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            DataFrame in unified format matching CGM_SCHEMA
            
        Raises:
            MalformedDataError: If CSV is unparseable, zero valid rows, or conversion fails
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


class CGMProcessor(ABC):
    """Abstract base class for unified CGM data processing (Stages 4-5).
    
    This interface handles vendor-agnostic operations on UnifiedFormat data:
    - Stage 4: Postprocessing (synchronization, interpolation)
    - Stage 5: Inference preparation (truncation, validation, warnings)
    
    This class operates only on data already in UnifiedFormat, regardless of vendor.
    Can be used with deserialized CSV data or directly after parsing.
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
        
        Adds rows with event_type='impute' for missing data points.
        Updates ProcessingWarning.IMPUTATION flag if gaps were filled.
        
        Args:
            dataframe: DataFrame with potential gaps
            
        Returns:
            DataFrame with interpolated values and impute events
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
        """Prepare data for inference with data-only DF and warning flags.
        
        - Filter to glucose-only events if requested (drops non-EGV events)
        - Truncate sequences exceeding maximum_wanted_duration
        - Drop duplicate timestamps if requested
        - Truncate to data columns only (exclude service columns)
        - Raise global output warning flags based on individual row quality
        - Check minimum duration requirements
        
        Args:
            dataframe: Fully processed DataFrame in unified format
            minimum_duration_minutes: Minimum required sequence duration
            maximum_wanted_duration: Maximum desired sequence duration (truncates if exceeded)
            glucose_only: If True, drop non-EGV events before truncation (keeps only GLUCOSE and IMPUTATION)
            drop_duplicates: If True, drop duplicate timestamps (keeps first occurrence)
            
        Returns:
            Tuple of (data_only_dataframe, warnings)
            
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