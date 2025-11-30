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
    __version__ = "0.4.4"  # Keep in sync with pyproject.toml

from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor

__all__ = [
    "FormatParser",
    "FormatProcessor",
    "__version__",
]

