"""cgm_format - Unified CGM data format converter for ML training and inference.

This package provides tools to convert vendor-specific CGM data (Dexcom, Libre)
into a standardized unified format for machine learning applications.

Main Components:
    FormatParser: Parse vendor-specific formats to unified format (Stages 1-3)
    FormatProcessor: Process unified data for inference (Stages 4-6)

Quick Start:
    >>> from cgm_format import FormatParser, FormatProcessor
    >>> 
    >>> # Parse any supported CGM file
    >>> unified_df = FormatParser.parse_from_file("data/dexcom_export.csv")
    >>> 
    >>> # Process for inference
    >>> processor = FormatProcessor()
    >>> processed_df = processor.interpolate_gaps(unified_df)
    >>> inference_df, warnings = processor.prepare_for_inference(processed_df)
"""

try:
    from importlib.metadata import version
    __version__ = version("cgm-format")
except Exception:
    # Fallback if package not installed (e.g., during development)
    __version__ = "0.7.0"  # Keep in sync with pyproject.toml

# Core classes
from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor

# Interface classes and exceptions
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    ValidationMethod,
    CGMParser,
    CGMProcessor,
    UnknownFormatError,
    MalformedDataError,
    MissingColumnError,
    ExtraColumnError,
    ColumnOrderError,
    ColumnTypeError,
    ZeroValidInputError,
    ProcessingWarning,
    NO_WARNING,
    WarningDescription,
    InferenceResult,
    ValidationResult,
    UnifiedFormat,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
    CALIBRATION_PERIOD_HOURS,
    to_pandas,
    to_polars,
)

# Schema infrastructure
from cgm_format.interface.schema import (
    EnumLiteral,
    ColumnSchema,
    CGMSchemaDefinition,
)

# Format schemas and enums (commonly used)
from cgm_format.formats.unified import (
    CGM_SCHEMA,
    UnifiedEventType,
    Quality,
    GOOD_QUALITY,
)

from cgm_format.formats.dexcom import (
    DEXCOM_SCHEMA,
    DexcomEventType,
    DexcomEventSubtype,
    DexcomColumn,
)

from cgm_format.formats.libre import (
    LIBRE_SCHEMA,
    LibreRecordType,
    LibreColumn,
)

__all__ = [
    # Main classes
    "FormatParser",
    "FormatProcessor",
    
    # Core interfaces
    "SupportedCGMFormat",
    "ValidationMethod",
    "CGMParser",
    "CGMProcessor",
    "UnifiedFormat",
    
    # Exceptions
    "UnknownFormatError",
    "MalformedDataError",
    "MissingColumnError",
    "ExtraColumnError",
    "ColumnOrderError",
    "ColumnTypeError",
    "ZeroValidInputError",
    
    # Warnings and results
    "ProcessingWarning",
    "NO_WARNING",
    "WarningDescription",
    "InferenceResult",
    "ValidationResult",
    
    # Constants
    "MINIMUM_DURATION_MINUTES",
    "MAXIMUM_WANTED_DURATION_MINUTES",
    "CALIBRATION_GAP_THRESHOLD",
    "CALIBRATION_PERIOD_HOURS",
    
    # Utilities
    "to_pandas",
    "to_polars",
    
    # Schema infrastructure
    "EnumLiteral",
    "ColumnSchema",
    "CGMSchemaDefinition",
    
    # Unified format
    "CGM_SCHEMA",
    "UnifiedEventType",
    "Quality",
    "GOOD_QUALITY",
    
    # Dexcom format
    "DEXCOM_SCHEMA",
    "DexcomEventType",
    "DexcomEventSubtype",
    "DexcomColumn",
    
    # Libre format
    "LIBRE_SCHEMA",
    "LibreRecordType",
    "LibreColumn",
    
    # Version
    "__version__",
]

