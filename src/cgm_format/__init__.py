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
    >>> unified_df = FormatParser.parse_from_file("data/input/dexcom_export.csv")
    >>> 
    >>> # Process for inference
    >>> processor = FormatProcessor()
    >>> processed_df = processor.interpolate_gaps(unified_df)
    >>> inference_df, warnings = processor.prepare_for_inference(processed_df)
"""

try:
    # Aliased with a leading underscore so the package namespace holds only
    # names that belong in __all__ (tests/test_package_exports.py asserts the
    # two are equal).
    from importlib.metadata import version as _package_version
    __version__ = _package_version("cgm-format")
except Exception:
    # Fallback if package not installed (e.g., during development).
    # Tech debt: RM2 in docs/ROADMAP.md. Two sources of truth drift, and the
    # one you read is the wrong one -- the right fix is an editable install so
    # metadata is always present, then dropping this branch.
    __version__ = "0.10.0"  # Keep in sync with pyproject.toml

# Core classes
from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor, ExtendedFormatProcessor

# Interface classes and exceptions
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    FormatCategory,
    SubjectEntry,
    TrackCoverage,
    MultiTrackSourceError,
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
    ColumnConstraints,
    FrictionlessDialect,
    FrictionlessTableSchema,
    CGMSchemaDefinition,
    derive_schema,
)

# Format schemas and enums (commonly used)
from cgm_format.formats.unified import (
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    EXTENDED_DATA_COLUMNS,
    UnifiedEventType,
    Quality,
    GOOD_QUALITY,
    AnnotationValue,
    AnnotationMapping,
    annotations_to_json,
)

from cgm_format.formats.dexcom import (
    DEXCOM_SCHEMA,
    DexcomEventType,
    DexcomEventSubtype,
    DexcomColumn,
)

from cgm_format.formats.dexcom_eu import (
    DEXCOM_EU_SCHEMA,
    DexcomEUColumn,
    MMOL_TO_MGDL,
)

from cgm_format.formats.libre import (
    LIBRE_SCHEMA,
    LibreRecordType,
    LibreColumn,
)

from cgm_format.formats.libre_eu import (
    LIBRE_EU_SCHEMA,
    LibreEUColumn,
)

from cgm_format.formats.cgmacros import (
    CGMACROS_SCHEMA,
    CGMACROS_TRACKS,
    CGMACROS_MEAN_TRACK,
    CGMacrosColumn,
    CGMacrosMealType,
)

from cgm_format.formats.d1namo import (
    D1NAMO_GLUCOSE_SCHEMA,
    D1NAMO_TRACK,
    D1namoGlucoseColumn,
    D1namoGlucoseType,
    D1namoInsulinColumn,
    D1namoFoodColumn,
    D1namoHealthyFoodColumn,
    D1namoAnnotationColumn,
)

from cgm_format.formats.medtronic import (
    MEDTRONIC_SCHEMA,
    MedtronicColumn,
)

from cgm_format.formats.nightscout import (
    NIGHTSCOUT_ENTRIES_SCHEMA,
    NIGHTSCOUT_TREATMENTS_SCHEMA,
    NightscoutEntryColumn,
    NightscoutTreatmentColumn,
    NightscoutTreatmentEventType,
    NightscoutDirection,
)

from cgm_format.nightscout_downloader import (
    download_nightscout,
)

__all__ = [
    # Main classes
    "FormatParser",
    "FormatProcessor",
    "ExtendedFormatProcessor",

    # Core interfaces
    "SupportedCGMFormat",
    "FormatCategory",
    "SubjectEntry",
    "TrackCoverage",
    "MultiTrackSourceError",
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
    "ColumnConstraints",
    "FrictionlessDialect",
    "FrictionlessTableSchema",
    "CGMSchemaDefinition",
    "derive_schema",

    # Unified format
    "CGM_SCHEMA",
    "UnifiedEventType",
    "Quality",
    "GOOD_QUALITY",

    # Extended unified format (macros, wearables, annotations)
    "CGM_SCHEMA_EXTENDED",
    "EXTENDED_DATA_COLUMNS",
    "AnnotationValue",
    "AnnotationMapping",
    "annotations_to_json",


    # Dexcom format
    "DEXCOM_SCHEMA",
    "DexcomEventType",
    "DexcomEventSubtype",
    "DexcomColumn",

    # Dexcom EU format (mmol/L)
    "DEXCOM_EU_SCHEMA",
    "DexcomEUColumn",
    "MMOL_TO_MGDL",
    
    # Libre format
    "LIBRE_SCHEMA",
    "LibreRecordType",
    "LibreColumn",

    # Libre EU format (mmol/L)
    "LIBRE_EU_SCHEMA",
    "LibreEUColumn",
    "CGMACROS_SCHEMA",
    "CGMACROS_TRACKS",
    "CGMACROS_MEAN_TRACK",
    "CGMacrosColumn",
    "CGMacrosMealType",
    "D1NAMO_GLUCOSE_SCHEMA",
    "D1NAMO_TRACK",
    "D1namoGlucoseColumn",
    "D1namoGlucoseType",
    "D1namoInsulinColumn",
    "D1namoFoodColumn",
    "D1namoHealthyFoodColumn",
    "D1namoAnnotationColumn",
    
    # Medtronic format
    "MEDTRONIC_SCHEMA",
    "MedtronicColumn",
    
    # Nightscout format
    "NIGHTSCOUT_ENTRIES_SCHEMA",
    "NIGHTSCOUT_TREATMENTS_SCHEMA",
    "NightscoutEntryColumn",
    "NightscoutTreatmentColumn",
    "NightscoutTreatmentEventType",
    "NightscoutDirection",
    
    # Nightscout downloader
    "download_nightscout",
    
    # Version
    "__version__",
]

