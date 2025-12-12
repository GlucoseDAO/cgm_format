"""Interface package for CGM data processing.

This package provides base interfaces and utilities for CGM data processing.
"""

from cgm_format.interface.schema import (
    EnumLiteral,
    ColumnSchema,
    CGMSchemaDefinition,
)
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

__all__ = [
    # Schema definitions
    "EnumLiteral",
    "ColumnSchema",
    "CGMSchemaDefinition",
    # Core interfaces
    "SupportedCGMFormat",
    "ValidationMethod",
    "CGMParser",
    "CGMProcessor",
    # Type aliases
    "UnifiedFormat",
    # Exceptions
    "UnknownFormatError",
    "MalformedDataError",
    "MissingColumnError",
    "ExtraColumnError",
    "ColumnOrderError",
    "ColumnTypeError",
    "ZeroValidInputError",
    # Warnings and result types
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
    # Conversion utilities
    "to_pandas",
    "to_polars",
]

