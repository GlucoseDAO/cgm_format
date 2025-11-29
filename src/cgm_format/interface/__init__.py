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
    CGMParser,
    CGMProcessor,
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    ProcessingWarning,
    InferenceResult,
    ValidationResult,
    MINIMUM_DURATION_MINUTES,
    MAXIMUM_WANTED_DURATION_MINUTES,
    CALIBRATION_GAP_THRESHOLD,
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
    "CGMParser",
    "CGMProcessor",
    # Exceptions
    "UnknownFormatError",
    "MalformedDataError",
    "ZeroValidInputError",
    # Warnings and result types
    "ProcessingWarning",
    "InferenceResult",
    "ValidationResult",
    # Constants
    "MINIMUM_DURATION_MINUTES",
    "MAXIMUM_WANTED_DURATION_MINUTES",
    "CALIBRATION_GAP_THRESHOLD",
    # Conversion utilities
    "to_pandas",
    "to_polars",
]

