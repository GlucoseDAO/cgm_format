"""Interface package for CGM data processing.

This package provides base interfaces and utilities for CGM data processing.
"""

from interface.schema import (
    EnumLiteral,
    ColumnSchema,
    CGMSchemaDefinition,
)
from interface.cgm_interface import (
    SupportedCGMFormat,
    ProcessingWarning,
    CGMParser,
    CGMProcessor,
)

__all__ = [
    "EnumLiteral",
    "SupportedCGMFormat",
    "ProcessingWarning",
    "CGMParser",
    "CGMProcessor",
    "ColumnSchema",
    "CGMSchemaDefinition",
]

