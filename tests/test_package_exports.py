"""Test package exports and imports from all __init__.py files.

This test suite verifies that all commonly used classes, functions, and constants
are properly exported at the appropriate package levels for user convenience.
"""

import pytest


class TestMainPackageExports:
    """Test exports from main cgm_format package."""

    def test_main_classes_import(self) -> None:
        """Test that main parser and processor classes can be imported."""
        from cgm_format import FormatParser, FormatProcessor
        
        assert FormatParser is not None
        assert FormatProcessor is not None
        assert hasattr(FormatParser, 'parse_file')
        assert hasattr(FormatProcessor, 'interpolate_gaps')

    def test_version_import(self) -> None:
        """Test that __version__ is available."""
        from cgm_format import __version__
        
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_interface_classes_import(self) -> None:
        """Test that interface classes are available from main package."""
        from cgm_format import (
            SupportedCGMFormat,
            ValidationMethod,
            CGMParser,
            CGMProcessor,
        )
        
        assert SupportedCGMFormat is not None
        assert ValidationMethod is not None
        assert CGMParser is not None
        assert CGMProcessor is not None

    def test_exceptions_import(self) -> None:
        """Test that all exception classes are available from main package."""
        from cgm_format import (
            UnknownFormatError,
            MalformedDataError,
            MissingColumnError,
            ExtraColumnError,
            ColumnOrderError,
            ColumnTypeError,
            ZeroValidInputError,
        )
        
        assert issubclass(UnknownFormatError, ValueError)
        assert issubclass(MalformedDataError, ValueError)
        assert issubclass(MissingColumnError, MalformedDataError)
        assert issubclass(ExtraColumnError, MalformedDataError)
        assert issubclass(ColumnOrderError, MalformedDataError)
        assert issubclass(ColumnTypeError, MalformedDataError)
        assert issubclass(ZeroValidInputError, ValueError)

    def test_warnings_and_results_import(self) -> None:
        """Test that warning flags and result types are available."""
        from cgm_format import (
            ProcessingWarning,
            NO_WARNING,
            WarningDescription,
            InferenceResult,
            ValidationResult,
            UnifiedFormat,
        )
        
        assert ProcessingWarning is not None
        assert NO_WARNING is not None
        assert WarningDescription is not None
        assert InferenceResult is not None
        assert ValidationResult is not None
        assert UnifiedFormat is not None

    def test_constants_import(self) -> None:
        """Test that processing constants are available."""
        from cgm_format import (
            MINIMUM_DURATION_MINUTES,
            MAXIMUM_WANTED_DURATION_MINUTES,
            CALIBRATION_GAP_THRESHOLD,
            CALIBRATION_PERIOD_HOURS,
        )
        
        assert isinstance(MINIMUM_DURATION_MINUTES, int)
        assert isinstance(MAXIMUM_WANTED_DURATION_MINUTES, int)
        assert isinstance(CALIBRATION_GAP_THRESHOLD, int)
        assert isinstance(CALIBRATION_PERIOD_HOURS, int)

    def test_utility_functions_import(self) -> None:
        """Test that utility conversion functions are available."""
        from cgm_format import to_pandas, to_polars
        
        assert callable(to_pandas)
        assert callable(to_polars)

    def test_schema_infrastructure_import(self) -> None:
        """Test that schema infrastructure is available from main package."""
        from cgm_format import (
            EnumLiteral,
            ColumnSchema,
            CGMSchemaDefinition,
        )
        
        assert EnumLiteral is not None
        assert ColumnSchema is not None
        assert CGMSchemaDefinition is not None

    def test_unified_format_imports(self) -> None:
        """Test that unified format schema and enums are available."""
        from cgm_format import (
            CGM_SCHEMA,
            UnifiedEventType,
            Quality,
            GOOD_QUALITY,
            CGMSchemaDefinition,
        )
        
        assert CGM_SCHEMA is not None
        assert isinstance(CGM_SCHEMA, CGMSchemaDefinition)
        assert UnifiedEventType is not None
        assert Quality is not None
        assert GOOD_QUALITY is not None
        assert GOOD_QUALITY.value == 0

    def test_dexcom_format_imports(self) -> None:
        """Test that Dexcom format schema and enums are available."""
        from cgm_format import (
            DEXCOM_SCHEMA,
            DexcomEventType,
            DexcomEventSubtype,
            DexcomColumn,
        )
        
        assert DEXCOM_SCHEMA is not None
        assert DexcomEventType is not None
        assert DexcomEventSubtype is not None
        assert DexcomColumn is not None

    def test_libre_format_imports(self) -> None:
        """Test that Libre format schema and enums are available."""
        from cgm_format import (
            LIBRE_SCHEMA,
            LibreRecordType,
            LibreColumn,
        )
        
        assert LIBRE_SCHEMA is not None
        assert LibreRecordType is not None
        assert LibreColumn is not None

    def test_enum_values_work(self) -> None:
        """Test that imported enums have correct values."""
        from cgm_format import UnifiedEventType, DexcomEventType, LibreRecordType
        
        # Test UnifiedEventType
        assert UnifiedEventType.GLUCOSE == "EGV_READ"
        assert len(UnifiedEventType.GLUCOSE) == 8
        
        # Test DexcomEventType
        assert DexcomEventType.EGV == "EGV"
        
        # Test LibreRecordType
        assert LibreRecordType.HISTORIC_GLUCOSE == 0
        assert LibreRecordType.INSULIN == 4
        assert LibreRecordType.FOOD == 5

    def test_all_exports_listed_in_all(self) -> None:
        """Test that __all__ is properly defined."""
        import cgm_format
        
        assert hasattr(cgm_format, '__all__')
        assert isinstance(cgm_format.__all__, list)
        assert len(cgm_format.__all__) > 0
        
        # Check that key items are in __all__
        assert 'FormatParser' in cgm_format.__all__
        assert 'FormatProcessor' in cgm_format.__all__
        assert 'ValidationMethod' in cgm_format.__all__
        assert 'CGM_SCHEMA' in cgm_format.__all__


class TestInterfacePackageExports:
    """Test exports from cgm_format.interface package."""

    def test_interface_classes(self) -> None:
        """Test that interface classes are properly exported."""
        from cgm_format.interface import (
            SupportedCGMFormat,
            ValidationMethod,
            CGMParser,
            CGMProcessor,
        )
        
        assert SupportedCGMFormat is not None
        assert ValidationMethod is not None
        assert CGMParser is not None
        assert CGMProcessor is not None

    def test_interface_exceptions(self) -> None:
        """Test that all exceptions are properly exported from interface."""
        from cgm_format.interface import (
            UnknownFormatError,
            MalformedDataError,
            MissingColumnError,
            ExtraColumnError,
            ColumnOrderError,
            ColumnTypeError,
            ZeroValidInputError,
        )
        
        assert UnknownFormatError is not None
        assert MalformedDataError is not None
        assert MissingColumnError is not None
        assert ExtraColumnError is not None
        assert ColumnOrderError is not None
        assert ColumnTypeError is not None
        assert ZeroValidInputError is not None

    def test_interface_warnings(self) -> None:
        """Test that warning types are properly exported from interface."""
        from cgm_format.interface import (
            ProcessingWarning,
            NO_WARNING,
            WarningDescription,
        )
        
        assert ProcessingWarning is not None
        assert NO_WARNING is not None
        assert WarningDescription is not None

    def test_interface_schema_infrastructure(self) -> None:
        """Test that schema infrastructure is properly exported."""
        from cgm_format.interface import (
            EnumLiteral,
            ColumnSchema,
            CGMSchemaDefinition,
        )
        
        assert EnumLiteral is not None
        assert ColumnSchema is not None
        assert CGMSchemaDefinition is not None

    def test_interface_constants(self) -> None:
        """Test that constants are properly exported from interface."""
        from cgm_format.interface import (
            MINIMUM_DURATION_MINUTES,
            MAXIMUM_WANTED_DURATION_MINUTES,
            CALIBRATION_GAP_THRESHOLD,
            CALIBRATION_PERIOD_HOURS,
        )
        
        assert MINIMUM_DURATION_MINUTES == 60
        assert MAXIMUM_WANTED_DURATION_MINUTES == 480
        assert CALIBRATION_GAP_THRESHOLD == 9900
        assert CALIBRATION_PERIOD_HOURS == 24

    def test_interface_utilities(self) -> None:
        """Test that utility functions are properly exported from interface."""
        from cgm_format.interface import to_pandas, to_polars
        
        assert callable(to_pandas)
        assert callable(to_polars)

    def test_interface_all_exports(self) -> None:
        """Test that __all__ is properly defined in interface package."""
        import cgm_format.interface
        
        assert hasattr(cgm_format.interface, '__all__')
        assert isinstance(cgm_format.interface.__all__, list)
        assert 'ValidationMethod' in cgm_format.interface.__all__
        assert 'CGMParser' in cgm_format.interface.__all__


class TestFormatsPackageExports:
    """Test exports from cgm_format.formats package."""

    def test_unified_format_exports(self) -> None:
        """Test that unified format is properly exported from formats."""
        from cgm_format.formats import (
            CGM_SCHEMA,
            UnifiedEventType,
            Quality,
            GOOD_QUALITY,
            UNIFIED_DETECTION_PATTERNS,
            UNIFIED_HEADER_LINE,
            UNIFIED_DATA_START_LINE,
            UNIFIED_METADATA_LINES,
            UNIFIED_TIMESTAMP_FORMATS,
        )
        
        assert CGM_SCHEMA is not None
        assert UnifiedEventType is not None
        assert Quality is not None
        assert GOOD_QUALITY.value == 0
        assert UNIFIED_DETECTION_PATTERNS is not None
        assert UNIFIED_HEADER_LINE is not None
        assert UNIFIED_DATA_START_LINE is not None
        assert UNIFIED_METADATA_LINES is not None
        assert UNIFIED_TIMESTAMP_FORMATS is not None

    def test_dexcom_format_exports(self) -> None:
        """Test that Dexcom format is properly exported from formats."""
        from cgm_format.formats import (
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
            DEXCOM_HIGH_GLUCOSE_DEFAULT,
            DEXCOM_LOW_GLUCOSE_DEFAULT,
        )
        
        assert DEXCOM_SCHEMA is not None
        assert DexcomEventType is not None
        assert DexcomEventSubtype is not None
        assert DexcomEventTypeSubtype is not None
        assert DexcomColumn is not None
        assert DEXCOM_DETECTION_PATTERNS is not None
        assert DEXCOM_HEADER_LINE is not None
        assert DEXCOM_DATA_START_LINE is not None
        assert DEXCOM_METADATA_LINES is not None
        assert DEXCOM_TIMESTAMP_FORMATS is not None
        assert isinstance(DEXCOM_HIGH_GLUCOSE_DEFAULT, (int, float))
        assert isinstance(DEXCOM_LOW_GLUCOSE_DEFAULT, (int, float))

    def test_libre_format_exports(self) -> None:
        """Test that Libre format is properly exported from formats."""
        from cgm_format.formats import (
            LIBRE_SCHEMA,
            LibreRecordType,
            LibreColumn,
            LIBRE_DETECTION_PATTERNS,
            LIBRE_HEADER_LINE,
            LIBRE_DATA_START_LINE,
            LIBRE_METADATA_LINES,
            LIBRE_TIMESTAMP_FORMATS,
        )
        
        assert LIBRE_SCHEMA is not None
        assert LibreRecordType is not None
        assert LibreColumn is not None
        assert LIBRE_DETECTION_PATTERNS is not None
        assert LIBRE_HEADER_LINE is not None
        assert LIBRE_DATA_START_LINE is not None
        assert LIBRE_METADATA_LINES is not None
        assert LIBRE_TIMESTAMP_FORMATS is not None

    def test_formats_backward_compatibility(self) -> None:
        """Test that backward compatibility aliases are available."""
        from cgm_format.formats import (
            UNIFIED_HEADER_LINES,
            DEXCOM_HEADER_LINES,
            LIBRE_HEADER_LINES,
        )
        
        assert UNIFIED_HEADER_LINES is not None
        assert DEXCOM_HEADER_LINES is not None
        assert LIBRE_HEADER_LINES is not None

    def test_formats_all_exports(self) -> None:
        """Test that __all__ is properly defined in formats package."""
        import cgm_format.formats
        
        assert hasattr(cgm_format.formats, '__all__')
        assert isinstance(cgm_format.formats.__all__, list)
        assert 'CGM_SCHEMA' in cgm_format.formats.__all__
        assert 'DEXCOM_SCHEMA' in cgm_format.formats.__all__
        assert 'LIBRE_SCHEMA' in cgm_format.formats.__all__


class TestSchemaPackageExports:
    """Test exports from cgm_format.interface.schema module."""

    def test_schema_imports(self) -> None:
        """Test that schema infrastructure can be imported."""
        from cgm_format.interface.schema import (
            EnumLiteral,
            ColumnSchema,
            CGMSchemaDefinition,
        )
        
        assert EnumLiteral is not None
        assert ColumnSchema is not None
        assert CGMSchemaDefinition is not None

    def test_schema_types_are_correct(self) -> None:
        """Test that schema types have expected structure."""
        from cgm_format.interface.schema import CGMSchemaDefinition
        from cgm_format import CGM_SCHEMA
        
        assert isinstance(CGM_SCHEMA, CGMSchemaDefinition)
        assert hasattr(CGM_SCHEMA, 'get_polars_schema')
        assert hasattr(CGM_SCHEMA, 'get_column_names')
        assert hasattr(CGM_SCHEMA, 'validate_dataframe')


class TestCrossPackageImportConsistency:
    """Test that imports from different levels point to same objects."""

    def test_validation_method_same_from_all_sources(self) -> None:
        """Test ValidationMethod is same object from all import sources."""
        from cgm_format import ValidationMethod as vm1
        from cgm_format.interface import ValidationMethod as vm2
        from cgm_format.interface.cgm_interface import ValidationMethod as vm3
        
        assert vm1 is vm2
        assert vm2 is vm3

    def test_schemas_same_from_all_sources(self) -> None:
        """Test schemas are same objects from all import sources."""
        from cgm_format import CGM_SCHEMA as schema1
        from cgm_format.formats import CGM_SCHEMA as schema2
        from cgm_format.formats.unified import CGM_SCHEMA as schema3
        
        assert schema1 is schema2
        assert schema2 is schema3

    def test_exceptions_same_from_all_sources(self) -> None:
        """Test exceptions are same objects from all import sources."""
        from cgm_format import UnknownFormatError as err1
        from cgm_format.interface import UnknownFormatError as err2
        from cgm_format.interface.cgm_interface import UnknownFormatError as err3
        
        assert err1 is err2
        assert err2 is err3

    def test_enums_same_from_all_sources(self) -> None:
        """Test enums are same objects from all import sources."""
        from cgm_format import UnifiedEventType as enum1
        from cgm_format.formats import UnifiedEventType as enum2
        from cgm_format.formats.unified import UnifiedEventType as enum3
        
        assert enum1 is enum2
        assert enum2 is enum3


class TestCommonUsagePatterns:
    """Test that common usage patterns from README work."""

    def test_basic_parsing_imports(self) -> None:
        """Test imports needed for basic parsing work."""
        from cgm_format import FormatParser
        
        assert hasattr(FormatParser, 'parse_file')
        assert hasattr(FormatParser, 'parse_base64')
        assert hasattr(FormatParser, 'parse_from_string')

    def test_complete_pipeline_imports(self) -> None:
        """Test imports needed for complete pipeline work."""
        from cgm_format import FormatParser, FormatProcessor
        
        assert FormatParser is not None
        assert FormatProcessor is not None

    def test_format_detection_imports(self) -> None:
        """Test imports needed for format detection work."""
        from cgm_format import FormatParser, SupportedCGMFormat
        
        assert hasattr(FormatParser, 'detect_format')
        assert hasattr(SupportedCGMFormat, 'DEXCOM')
        assert hasattr(SupportedCGMFormat, 'LIBRE')
        assert hasattr(SupportedCGMFormat, 'UNIFIED_CGM')

    def test_schema_usage_imports(self) -> None:
        """Test imports needed for schema usage work."""
        from cgm_format import CGM_SCHEMA, UnifiedEventType, Quality
        
        assert hasattr(CGM_SCHEMA, 'get_polars_schema')
        assert hasattr(CGM_SCHEMA, 'get_column_names')
        assert hasattr(UnifiedEventType, 'GLUCOSE')
        assert hasattr(Quality, 'OUT_OF_RANGE')
        assert hasattr(Quality, 'IMPUTATION')

    def test_validation_configuration_imports(self) -> None:
        """Test imports needed for validation configuration work."""
        from cgm_format import FormatParser, FormatProcessor, ValidationMethod
        
        assert hasattr(FormatParser, 'validation_mode')
        assert hasattr(ValidationMethod, 'INPUT')
        assert hasattr(ValidationMethod, 'INPUT_FORCED')

    def test_warning_handling_imports(self) -> None:
        """Test imports needed for warning handling work."""
        from cgm_format import FormatProcessor, ProcessingWarning
        
        assert hasattr(ProcessingWarning, 'TOO_SHORT')
        assert hasattr(ProcessingWarning, 'OUT_OF_RANGE')
        assert hasattr(ProcessingWarning, 'IMPUTATION')
        assert hasattr(ProcessingWarning, 'CALIBRATION')
        assert hasattr(ProcessingWarning, 'TIME_DUPLICATES')

    def test_exception_handling_imports(self) -> None:
        """Test imports needed for exception handling work."""
        from cgm_format import (
            UnknownFormatError,
            MalformedDataError,
            ZeroValidInputError,
        )
        
        assert issubclass(UnknownFormatError, ValueError)
        assert issubclass(MalformedDataError, ValueError)
        assert issubclass(ZeroValidInputError, ValueError)
