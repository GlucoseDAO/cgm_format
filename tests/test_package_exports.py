"""Test package exports and imports from all __init__.py files.

This test suite verifies that all commonly used classes, functions, and constants
are properly exported at the appropriate package levels for user convenience.
"""

import types

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
            ColumnConstraints,
            FrictionlessDialect,
            FrictionlessTableSchema,
            CGMSchemaDefinition,
        )
        
        assert EnumLiteral is not None
        assert ColumnSchema is not None
        assert ColumnConstraints is not None
        assert FrictionlessDialect is not None
        assert FrictionlessTableSchema is not None
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

    def test_libre_eu_format_imports(self) -> None:
        """Test that Libre EU format schema and enums are available."""
        from cgm_format import (
            LIBRE_EU_SCHEMA,
            LibreEUColumn,
        )

        assert LIBRE_EU_SCHEMA is not None
        assert LibreEUColumn is not None

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
        """`__all__` matches what the package actually imports, exactly.

        An equality assertion rather than a spot check on four names: this is a
        published package with an API contract (`CLAUDE.md` §5), and the failure
        this guards is a new entry point being added to the import block in
        `__init__.py` and forgotten in `__all__` — which a presence check on
        four unrelated names cannot see.

        The comparison is against the module's own public namespace, so it
        needs no hand-maintained list to drift from. Submodules are excluded —
        `cgm_format.formats` is bound as a side effect of importing from it,
        not because the package exports it — while `__version__` is kept, since
        the package does deliberately export it.
        """
        import cgm_format

        assert isinstance(cgm_format.__all__, list)
        assert len(cgm_format.__all__) == len(set(cgm_format.__all__)), (
            "__all__ contains duplicate entries"
        )

        public_namespace = {
            name
            for name in vars(cgm_format)
            if (not name.startswith('_') or name == '__version__')
            and not isinstance(getattr(cgm_format, name), types.ModuleType)
        }

        assert set(cgm_format.__all__) == public_namespace, (
            "__all__ and the imported public namespace disagree — "
            f"missing from __all__: {sorted(public_namespace - set(cgm_format.__all__))}; "
            f"in __all__ but not imported: {sorted(set(cgm_format.__all__) - public_namespace)}"
        )

    def test_every_name_in_all_is_actually_importable(self) -> None:
        """A name in `__all__` that resolves to nothing breaks `import *`."""
        import cgm_format

        unresolvable = [
            name for name in cgm_format.__all__ if not hasattr(cgm_format, name)
        ]

        assert not unresolvable, f"__all__ names nothing importable: {unresolvable}"


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


