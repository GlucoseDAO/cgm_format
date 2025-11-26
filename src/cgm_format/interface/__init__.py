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

