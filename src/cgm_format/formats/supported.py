
from typing import Dict, List, Optional, Tuple
from cgm_format.interface.cgm_interface import SupportedCGMFormat
from cgm_format.interface.schema import CGMSchemaDefinition
from cgm_format.formats.unified import CGM_SCHEMA, UNIFIED_DETECTION_PATTERNS, UNIFIED_DATA_START_LINE
from cgm_format.formats.dexcom import DEXCOM_SCHEMA, DEXCOM_DETECTION_PATTERNS, DEXCOM_DATA_START_LINE
from cgm_format.formats.libre import LIBRE_SCHEMA, LIBRE_DETECTION_PATTERNS, LIBRE_DATA_START_LINE



# Schema map for validation
SCHEMA_MAP: Dict[SupportedCGMFormat, CGMSchemaDefinition] = {
    SupportedCGMFormat.UNIFIED_CGM: CGM_SCHEMA,
    SupportedCGMFormat.DEXCOM: DEXCOM_SCHEMA,
    SupportedCGMFormat.LIBRE: LIBRE_SCHEMA,
}

FORMAT_DETECTION_PATTERNS: Dict[SupportedCGMFormat, List[str]] = {
    SupportedCGMFormat.UNIFIED_CGM: UNIFIED_DETECTION_PATTERNS,
    SupportedCGMFormat.DEXCOM: DEXCOM_DETECTION_PATTERNS,
    SupportedCGMFormat.LIBRE: LIBRE_DETECTION_PATTERNS,
}

FORMAT_DETECTION_LINE_COUNT: Dict[SupportedCGMFormat, int] = {
    SupportedCGMFormat.UNIFIED_CGM: UNIFIED_DATA_START_LINE,
    SupportedCGMFormat.DEXCOM: DEXCOM_DATA_START_LINE,
    SupportedCGMFormat.LIBRE: LIBRE_DATA_START_LINE,
}

DETECTION_LINE_COUNT: int = max(FORMAT_DETECTION_LINE_COUNT.values())*2

# Known issues to suppress per format (can't fix vendor CSV format issues)
KNOWN_ISSUES_TO_SUPPRESS = {
    SupportedCGMFormat.DEXCOM: [
        # Dexcom exports have variable-length rows - non-EGV events don't include
        # trailing Transmitter Time/ID columns (missing cells, not just empty values)
        ('missing-cell', 'Transmitter ID', None),
        ('missing-cell', 'Transmitter Time (Long Integer)', None),
        # Dexcom uses "Low" (<50 mg/dL) and "High" (>400 mg/dL) text markers 
        # instead of numeric values for out-of-range glucose readings
        ('type-error', 'Glucose Value (mg/dL)', 'Low'),
        ('type-error', 'Glucose Value (mg/dL)', 'High'),
        # Some Dexcom exports include UTF-8 BOM marker in header
        ('incorrect-label', 'Index', None),
    ],
    SupportedCGMFormat.UNIFIED_CGM: [], #this is ours, none should be suppressed
    SupportedCGMFormat.LIBRE: [],
}
