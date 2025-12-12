"""Test format detection and Frictionless schema validation for all CGM formats.

This test ensures that:
1. All CSV files can be detected correctly (Dexcom/Libre/Unified)
2. All detected files validate against their Frictionless schemas
3. Known vendor format issues are properly suppressed

Tests parametrized for all CSV files in the data directory.
"""

import pytest
from pathlib import Path
from typing import Any

from cgm_format import FormatParser
from cgm_format.interface.cgm_interface import SupportedCGMFormat, UnknownFormatError, MalformedDataError
from cgm_format.formats.supported import SCHEMA_MAP, KNOWN_ISSUES_TO_SUPPRESS

# Optional: Use frictionless library if available
try:
    from frictionless import Resource, Schema, Dialect
    HAS_FRICTIONLESS = True
except ImportError:
    HAS_FRICTIONLESS = False


# Test data directory
DATA_DIR = Path(__file__).parent.parent / "data"

def is_medtronic_file(file_path: Path) -> bool:
    """Check if a file is a Medtronic Guardian Connect file."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(500).decode('utf-8', errors='ignore')
            return "Guardian Connect" in header
    except Exception:
        return False


def get_all_csv_files():
    """Get all CSV files from data directory (excluding parsed subdirectory).
    
    Returns:
        List of Path objects for CSV files
    """
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    # Exclude Medtronic files (not yet supported)
    csv_files = [f for f in csv_files if not is_medtronic_file(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {DATA_DIR}")
    
    return sorted(csv_files)


def get_detectable_files():
    """Get CSV files that can be successfully format-detected.
    
    Returns:
        List of tuples (file_path, detected_format)
    """
    csv_files = get_all_csv_files()
    
    detectable = []
    for csv_file in csv_files:
        try:
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            detectable.append((csv_file, detected_format))
        except (UnknownFormatError, MalformedDataError, Exception):
            continue
    
    if not detectable:
        pytest.skip("No detectable CSV files found")
    
    return detectable


def should_suppress_error(error: Any, format_type: SupportedCGMFormat) -> bool:
    """Check if an error should be suppressed based on known format issues.
    
    Args:
        error: Frictionless error object or dict
        format_type: The CGM format type
        
    Returns:
        True if error should be suppressed (known issue), False otherwise
    """
    suppressions = KNOWN_ISSUES_TO_SUPPRESS.get(format_type, [])
    if not suppressions:
        return False
    
    # Extract error type, field name, and cell value from error
    # Try various attribute names that Frictionless uses
    error_type = None
    field_name = None
    cell_value = None
    
    if hasattr(error, 'type'):
        error_type = error.type
    elif hasattr(error, 'code'):
        error_type = error.code
    elif isinstance(error, dict):
        error_type = error.get('type') or error.get('code', '')
    
    if hasattr(error, 'fieldName'):
        field_name = error.fieldName
    elif hasattr(error, 'field_name'):
        field_name = error.field_name
    elif hasattr(error, 'label'):
        field_name = error.label
    elif isinstance(error, dict):
        field_name = error.get('fieldName') or error.get('field_name') or error.get('label', '')
    
    if hasattr(error, 'cell'):
        cell_value = error.cell
    elif isinstance(error, dict):
        cell_value = error.get('cell', '')
    
    if not error_type or not field_name:
        return False
    
    # Check if this error matches any suppression rule
    # Suppression rules are tuples: (error_type, field_name, cell_value)
    # If cell_value in rule is None, we only match on error_type and field_name
    for rule in suppressions:
        rule_error_type, rule_field_name, rule_cell_value = rule
        
        # Check error type and field name
        if error_type == rule_error_type and field_name == rule_field_name:
            # If rule doesn't care about cell value (None), it's a match
            if rule_cell_value is None:
                return True
            # Otherwise, check if cell value matches
            if cell_value == rule_cell_value:
                return True
    
    return False


class TestFormatDetection:
    """Test format detection for all CSV files."""
    
    @pytest.mark.parametrize("csv_file", get_all_csv_files(), 
                            ids=lambda x: x.name)
    def test_detect_format(self, csv_file: Path):
        """Test that format can be detected for each CSV file.
        
        Args:
            csv_file: Path to CSV file to test
        """
        print(f"\n{'='*70}")
        print(f"Testing format detection: {csv_file.name}")
        print(f"{'='*70}")
        
        # Read and decode
        with open(csv_file, 'rb') as f:
            raw_data = f.read()
        
        print(f"\n1. Raw data size: {len(raw_data)} bytes")
        
        # Decode
        text_data = FormatParser.decode_raw_data(raw_data)
        print(f"2. Decoded to {len(text_data)} chars")
        
        # Detect format - this should not raise
        detected_format = FormatParser.detect_format(text_data)
        
        print(f"3. Detected format: {detected_format.value}")
        
        # Verify it's a known format
        assert detected_format in SupportedCGMFormat, \
            f"Detected format {detected_format} is not a known SupportedCGMFormat"
        
        # Verify the format has a schema
        assert detected_format in SCHEMA_MAP, \
            f"No schema defined for format {detected_format.value}"
        
        print(f"\n✅ SUCCESS: Format detected as {detected_format.value}")


class TestSchemaValidation:
    """Test Frictionless schema validation for all detected files."""
    
    @pytest.mark.skipif(not HAS_FRICTIONLESS, 
                       reason="Frictionless library not installed")
    @pytest.mark.parametrize("csv_file,format_type", get_detectable_files(), 
                            ids=lambda x: x.name if isinstance(x, Path) else str(x))
    def test_validate_against_schema(self, csv_file: Path, format_type: SupportedCGMFormat):
        """Test that CSV file validates against its format's Frictionless schema.
        
        Known vendor format issues are automatically suppressed and don't cause test failure.
        
        Args:
            csv_file: Path to CSV file
            format_type: Detected format type
        """
        print(f"\n{'='*70}")
        print(f"Testing schema validation: {csv_file.name}")
        print(f"Format: {format_type.value}")
        print(f"{'='*70}")
        
        # Get the appropriate schema
        schema = SCHEMA_MAP[format_type]
        
        print(f"\n1. Schema info:")
        print(f"   Service columns: {len(schema.service_columns)}")
        print(f"   Data columns: {len(schema.data_columns)}")
        print(f"   Total columns: {len(schema.service_columns) + len(schema.data_columns)}")
        
        # Get the dialect for this schema (handles metadata rows)
        dialect = schema.get_dialect()
        if dialect:
            print(f"\n   Dialect: {dialect}")
        
        # Convert schema to Frictionless format with dialect
        frictionless_schema_dict = schema.to_frictionless_schema()
        
        # Convert to relative path (frictionless requires relative paths for security)
        try:
            relative_path = csv_file.relative_to(Path.cwd())
        except ValueError:
            # If not relative to cwd, just use the file name
            relative_path = csv_file
        
        print(f"\n2. Validating file: {relative_path}")
        
        # Create Schema and Dialect objects for proper validation
        # Extract and remove dialect from schema dict (if present)
        dialect_dict = frictionless_schema_dict.pop('dialect', None)
        
        # If no dialect in schema but schema has one via get_dialect(), use that
        if dialect_dict is None and dialect is not None:
            dialect_dict = dialect
        
        schema_obj = Schema.from_descriptor(frictionless_schema_dict)
        
        # Validate using Resource API
        if dialect_dict:
            dialect_obj = Dialect.from_descriptor(dialect_dict)
            resource = Resource(path=str(relative_path), schema=schema_obj, dialect=dialect_obj)
        else:
            resource = Resource(path=str(relative_path), schema=schema_obj)
        
        report = resource.validate()
        
        # Collect error information
        errors = []
        error_count = 0
        suppressed_count = 0
        
        for task in report.tasks:
            if hasattr(task, 'errors') and task.errors:
                for error in task.errors:
                    # Check if this is a known issue we should suppress
                    if should_suppress_error(error, format_type):
                        suppressed_count += 1
                        continue
                    
                    error_count += 1
                    # Collect error details for reporting
                    if hasattr(error, 'message'):
                        error_msg = error.message
                    elif isinstance(error, dict):
                        error_msg = error.get('message', str(error))
                    else:
                        error_msg = str(error)
                    errors.append(error_msg)
        
        # Get row count
        row_count = report.tasks[0].stats.get('rows', 'unknown') if report.tasks else 'unknown'
        
        print(f"\n3. Validation results:")
        print(f"   Rows: {row_count}")
        print(f"   Errors: {error_count}")
        print(f"   Suppressed (known issues): {suppressed_count}")
        
        if suppressed_count > 0:
            print(f"\n4. Known issues suppressed:")
            suppressions = KNOWN_ISSUES_TO_SUPPRESS[format_type]
            for rule in suppressions:
                error_type, field_name, cell_value = rule
                if cell_value:
                    print(f"   - {error_type} on '{field_name}' with value '{cell_value}'")
                else:
                    print(f"   - {error_type} on '{field_name}'")
        
        if error_count > 0:
            print(f"\n❌ VALIDATION FAILED with {error_count} errors:")
            for i, error_msg in enumerate(errors[:10], 1):  # Show first 10 errors
                print(f"   {i}. {error_msg}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
            
            # Fail the test with clear error message
            pytest.fail(
                f"Schema validation failed for {csv_file.name} ({format_type.value}): "
                f"{error_count} errors found (see output above for details)"
            )
        
        print(f"\n✅ SUCCESS: File validates against {format_type.value} schema")
        print(f"   ({row_count} rows, {suppressed_count} known issues suppressed)")


class TestSchemaDefinitions:
    """Test that all format schemas are properly defined."""
    
    @pytest.mark.parametrize("format_type,schema", SCHEMA_MAP.items(),
                            ids=lambda x: x.value if isinstance(x, SupportedCGMFormat) else "schema")
    def test_schema_has_columns(self, format_type: SupportedCGMFormat, schema):
        """Test that each schema has service and data columns defined.
        
        Args:
            format_type: The CGM format type
            schema: The schema object
        """
        print(f"\n{'='*70}")
        print(f"Testing schema definition: {format_type.value}")
        print(f"{'='*70}")
        
        # Check schema has service columns
        assert hasattr(schema, 'service_columns'), \
            f"Schema for {format_type.value} missing 'service_columns' attribute"
        assert len(schema.service_columns) > 0, \
            f"Schema for {format_type.value} has no service columns"
        
        # Check schema has data columns
        assert hasattr(schema, 'data_columns'), \
            f"Schema for {format_type.value} missing 'data_columns' attribute"
        assert len(schema.data_columns) > 0, \
            f"Schema for {format_type.value} has no data columns"
        
        print(f"\n1. Schema structure:")
        print(f"   Service columns: {len(schema.service_columns)}")
        print(f"   Data columns: {len(schema.data_columns)}")
        print(f"   Total columns: {len(schema.service_columns) + len(schema.data_columns)}")
        
        # Check each column has required fields
        all_columns = schema.service_columns + schema.data_columns
        for col in all_columns:
            assert 'name' in col, f"Column missing 'name' field: {col}"
            assert 'dtype' in col, f"Column '{col.get('name', '?')}' missing 'dtype' field"
        
        print(f"\n2. Column definitions:")
        for col in all_columns:
            col_name = col['name']
            col_dtype = col['dtype']
            unit = f" [{col['unit']}]" if 'unit' in col else ""
            print(f"   - {col_name}: {col_dtype}{unit}")
        
        print(f"\n✅ SUCCESS: Schema for {format_type.value} is properly defined")
    
    @pytest.mark.skipif(not HAS_FRICTIONLESS, 
                       reason="Frictionless library not installed")
    @pytest.mark.parametrize("format_type,schema", SCHEMA_MAP.items(),
                            ids=lambda x: x.value if isinstance(x, SupportedCGMFormat) else "schema")
    def test_schema_converts_to_frictionless(self, format_type: SupportedCGMFormat, schema):
        """Test that each schema can be converted to Frictionless format.
        
        Args:
            format_type: The CGM format type
            schema: The schema object
        """
        print(f"\n{'='*70}")
        print(f"Testing Frictionless conversion: {format_type.value}")
        print(f"{'='*70}")
        
        # This should not raise
        frictionless_schema_dict = schema.to_frictionless_schema()
        
        print(f"\n1. Frictionless schema generated:")
        print(f"   Fields: {len(frictionless_schema_dict.get('fields', []))}")
        
        # Check it has required Frictionless fields
        assert 'fields' in frictionless_schema_dict, \
            f"Frictionless schema for {format_type.value} missing 'fields'"
        assert len(frictionless_schema_dict['fields']) > 0, \
            f"Frictionless schema for {format_type.value} has no fields"
        
        # Try to create a Schema object (will validate the schema)
        schema_obj = Schema.from_descriptor(frictionless_schema_dict)
        assert schema_obj is not None
        
        print(f"   ✅ Valid Frictionless schema created")
        
        # Check for dialect if present
        if 'dialect' in frictionless_schema_dict:
            dialect_dict = frictionless_schema_dict['dialect']
            print(f"\n2. Dialect configuration:")
            for key, value in dialect_dict.items():
                print(f"   {key}: {value}")
            
            # Validate dialect
            dialect_obj = Dialect.from_descriptor(dialect_dict)
            assert dialect_obj is not None
            print(f"   ✅ Valid Frictionless dialect created")
        
        print(f"\n✅ SUCCESS: Schema converts to Frictionless format")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

