"""CGM Data Format Schema.

This package provides the base schema infrastructure (interface.schema),
the unified CGM format definition (formats.unified), as well as
vendor-specific format schemas (Dexcom, Libre).
"""

# Unified format
from formats.unified import (
    CGM_SCHEMA,
    UnifiedEventType,
    Quality,
    UNIFIED_DETECTION_PATTERNS,
    UNIFIED_HEADER_LINE,
    UNIFIED_DATA_START_LINE,
    UNIFIED_METADATA_LINES,
    UNIFIED_TIMESTAMP_FORMATS,
)

# Dexcom format
from formats.dexcom import (
    DEXCOM_SCHEMA,
    DexcomEventType,
    DexcomEventSubtype,
    DexcomEventTypeSubtype,
    DexcomColumn,
    DEXCOM_DETECTION_PATTERNS,
    DEXCOM_HEADER_LINE,
    DEXCOM_DATA_START_LINE,
    DEXCOM_METADATA_LINES,
    DEXCOM_TIMESTAMP_FORMATS,
)

# Libre format
from formats.libre import (
    LIBRE_SCHEMA,
    LibreRecordType,
    LibreColumn,
    LIBRE_DETECTION_PATTERNS,
    LIBRE_HEADER_LINE,
    LIBRE_DATA_START_LINE,
    LIBRE_METADATA_LINES,
    LIBRE_TIMESTAMP_FORMATS,
)

# Backward compatibility aliases (deprecated)
UNIFIED_HEADER_LINES = UNIFIED_HEADER_LINE
DEXCOM_HEADER_LINES = DEXCOM_HEADER_LINE
LIBRE_HEADER_LINES = LIBRE_HEADER_LINE

__all__ = [
    # Unified format
    "CGM_SCHEMA",
    "UnifiedEventType",
    "Quality",
    "UNIFIED_DETECTION_PATTERNS",
    "UNIFIED_HEADER_LINE",
    "UNIFIED_DATA_START_LINE",
    "UNIFIED_METADATA_LINES",
    "UNIFIED_TIMESTAMP_FORMATS",
    
    # Dexcom format
    "DEXCOM_SCHEMA",
    "DexcomEventType",
    "DexcomEventSubtype",
    "DexcomEventTypeSubtype",
    "DexcomColumn",
    "DEXCOM_DETECTION_PATTERNS",
    "DEXCOM_HEADER_LINE",
    "DEXCOM_DATA_START_LINE",
    "DEXCOM_METADATA_LINES",
    "DEXCOM_TIMESTAMP_FORMATS",
    
    # Libre format
    "LIBRE_SCHEMA",
    "LibreRecordType",
    "LibreColumn",
    "LIBRE_DETECTION_PATTERNS",
    "LIBRE_HEADER_LINE",
    "LIBRE_DATA_START_LINE",
    "LIBRE_METADATA_LINES",
    "LIBRE_TIMESTAMP_FORMATS",
    
    # Backward compatibility (deprecated)
    "UNIFIED_HEADER_LINES",
    "DEXCOM_HEADER_LINES",
    "LIBRE_HEADER_LINES",
]
