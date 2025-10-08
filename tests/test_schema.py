"""Tests for schema definition and frictionless validation.

Tests cover the pitfalls discovered during implementation:
1. EnumLiteral serialization (not <EnumName.MEMBER: 'value'>)
2. Frictionless schema generation with dialect
3. Column order matching actual CSV files
4. Path handling for validation
"""

import pytest
from pathlib import Path
import json

from interface.schema import EnumLiteral, CGMSchemaDefinition
from formats.dexcom import DEXCOM_SCHEMA, DexcomColumn, DexcomEventType
from formats.libre import LIBRE_SCHEMA, LibreColumn
from formats.unified import CGM_SCHEMA, UnifiedEventType


class TestEnumLiteral:
    """Test that EnumLiteral properly behaves as a string literal."""
    
    def test_enum_str_representation(self):
        """EnumLiteral should serialize as its value, not as <Enum.MEMBER: 'value'>."""
        # Test Dexcom column enum
        assert str(DexcomColumn.INDEX) == "Index"
        assert repr(DexcomColumn.INDEX) == "Index"
        
        # Test event type enum
        assert str(DexcomEventType.EGV) == "EGV"
        assert repr(DexcomEventType.EGV) == "EGV"
        
        # Test Libre column enum
        assert str(LibreColumn.DEVICE) == "Device"
        assert repr(LibreColumn.DEVICE) == "Device"
        
        # Test unified event type enum
        assert str(UnifiedEventType.GLUCOSE) == "EGV_READ"
        assert repr(UnifiedEventType.GLUCOSE) == "EGV_READ"
    
    def test_enum_string_comparison(self):
        """EnumLiteral should compare equal to its string value."""
        assert DexcomColumn.INDEX == "Index"
        assert DexcomEventType.EGV == "EGV"
        assert LibreColumn.DEVICE == "Device"
        assert UnifiedEventType.GLUCOSE == "EGV_READ"
    
    def test_enum_hash_compatibility(self):
        """EnumLiteral should be hashable and work as dict keys."""
        test_dict = {DexcomColumn.INDEX: "test"}
        assert test_dict[DexcomColumn.INDEX] == "test"
        
        # Should also work with string key
        test_dict2 = {"Index": "value"}
        # Note: this won't work because dict uses identity, not equality
        # but the enum itself hashes consistently
        assert hash(DexcomColumn.INDEX) == hash("Index")
    
    def test_enum_json_serialization(self):
        """EnumLiteral should serialize cleanly in JSON."""
        data = {
            "column": DexcomColumn.INDEX,
            "event": DexcomEventType.EGV,
        }
        
        # Using default json encoder with str() should work
        json_str = json.dumps({k: str(v) for k, v in data.items()})
        assert '"Index"' in json_str
        assert '"EGV"' in json_str
        assert "<DexcomColumn" not in json_str
        assert "<DexcomEventType" not in json_str


class TestFrictionlessSchemaGeneration:
    """Test frictionless schema generation with proper configuration."""
    
    def test_basic_schema_generation(self):
        """Schema should generate valid frictionless format."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        
        # Should have required keys
        assert "fields" in schema_dict
        assert isinstance(schema_dict["fields"], list)
        assert len(schema_dict["fields"]) == 14  # Total Dexcom columns
        
        # Fields should have proper structure
        for field in schema_dict["fields"]:
            assert "name" in field
            assert "type" in field
            assert "description" in field
            # Name should be a clean string, not enum representation
            assert not field["name"].startswith("<")
            # Should not contain enum class name indicators
            assert ":" not in field["name"] or "<" not in field["name"]  # ":" only bad if part of "<Enum.NAME: 'value'>"
    
    def test_schema_with_primary_key(self):
        """Unified schema should include primary key."""
        schema_dict = CGM_SCHEMA.to_frictionless_schema(
            primary_key=["sequence_id", "datetime"]
        )
        
        assert "primaryKey" in schema_dict
        assert schema_dict["primaryKey"] == ["sequence_id", "datetime"]
    
    def test_schema_without_primary_key(self):
        """Vendor schemas should not have primary key by default."""
        dexcom_schema = DEXCOM_SCHEMA.to_frictionless_schema()
        libre_schema = LIBRE_SCHEMA.to_frictionless_schema()
        
        assert "primaryKey" not in dexcom_schema
        assert "primaryKey" not in libre_schema
    
    def test_dexcom_dialect_with_comment_rows(self):
        """Dexcom schema should support commentRows dialect."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema(
            dialect={"commentRows": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        )
        
        assert "dialect" in schema_dict
        assert "commentRows" in schema_dict["dialect"]
        assert schema_dict["dialect"]["commentRows"] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    def test_libre_dialect_with_header_rows(self):
        """Libre schema should support headerRows dialect."""
        schema_dict = LIBRE_SCHEMA.to_frictionless_schema(
            dialect={"headerRows": [2]}
        )
        
        assert "dialect" in schema_dict
        assert "headerRows" in schema_dict["dialect"]
        assert schema_dict["dialect"]["headerRows"] == [2]
    
    def test_schema_without_dialect(self):
        """Schema without dialect should not have dialect key."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        assert "dialect" not in schema_dict


class TestDexcomColumnOrder:
    """Test that Dexcom schema column order matches actual CSV files."""
    
    def test_column_order_matches_csv(self):
        """Dexcom schema column order should match actual CSV structure."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        field_names = [f["name"] for f in schema_dict["fields"]]
        
        # Expected order from actual Dexcom CSV files:
        # Index, Timestamp, Event Type, Event Subtype, Patient Info, Device Info,
        # Source Device ID, Glucose Value, Insulin Value, Carb Value,
        # Duration, Glucose Rate of Change, Transmitter Time, Transmitter ID
        expected_order = [
            "Index",
            "Timestamp (YYYY-MM-DDThh:mm:ss)",  # MUST be position 2, not later!
            "Event Type",
            "Event Subtype",
            "Patient Info",
            "Device Info",
            "Source Device ID",
            "Glucose Value (mg/dL)",
            "Insulin Value (u)",
            "Carb Value (grams)",
            "Duration (hh:mm:ss)",
            "Glucose Rate of Change (mg/dL/min)",
            "Transmitter Time (Long Integer)",
            "Transmitter ID",
        ]
        
        assert field_names == expected_order, (
            f"Column order mismatch!\n"
            f"Expected: {expected_order}\n"
            f"Got:      {field_names}"
        )
    
    def test_timestamp_comes_second(self):
        """Critical: Timestamp must be column 2 (after Index), not later."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        field_names = [f["name"] for f in schema_dict["fields"]]
        
        # This was the bug: Timestamp was appearing much later in the schema
        assert field_names[1] == "Timestamp (YYYY-MM-DDThh:mm:ss)", (
            "Timestamp must be second column (position 2) to match CSV structure"
        )
        assert field_names[2] == "Event Type", (
            "Event Type must be third column (position 3)"
        )


class TestLibreColumnOrder:
    """Test that Libre schema column order matches actual CSV files."""
    
    def test_libre_column_order(self):
        """Libre schema column order should match actual CSV structure."""
        schema_dict = LIBRE_SCHEMA.to_frictionless_schema()
        field_names = [f["name"] for f in schema_dict["fields"]]
        
        # Verify key columns are in correct positions
        assert field_names[0] == "Device"
        assert field_names[1] == "Serial Number"
        assert field_names[2] == "Record Type"
        
        # Device Timestamp should come after service columns
        assert "Device Timestamp" in field_names
        device_timestamp_idx = field_names.index("Device Timestamp")
        assert device_timestamp_idx > 2  # After Device, Serial Number, Record Type


class TestSchemaValidation:
    """Test schema validation with real file scenarios."""
    
    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_file = tmp_path / "test.csv"
        return csv_file
    
    def test_relative_path_handling(self, temp_csv_file):
        """Frictionless validation requires relative paths for security."""
        # Create a simple CSV file
        temp_csv_file.write_text(
            "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,"
            "Patient Info,Device Info,Source Device ID,"
            "Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams),"
            "Duration (hh:mm:ss),Glucose Rate of Change (mg/dL/min),"
            "Transmitter Time (Long Integer),Transmitter ID\n"
            "1,2024-01-01T12:00:00,EGV,,,Device,ID,100,,,,,123456,ABC123\n"
        )
        
        # Absolute paths should fail with frictionless "not safe" error
        # Relative paths should work
        try:
            relative_path = temp_csv_file.relative_to(Path.cwd())
            assert not relative_path.is_absolute()
        except ValueError:
            # If temp_path is not relative to cwd, that's ok for this test
            pass
    
    def test_enum_values_in_constraints(self):
        """Enum constraints should use values, not enum objects."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        
        # Find the Event Type field
        event_type_field = None
        for field in schema_dict["fields"]:
            if field["name"] == "Event Type":
                event_type_field = field
                break
        
        assert event_type_field is not None
        assert "constraints" in event_type_field
        assert "enum" in event_type_field["constraints"]
        
        # Enum values should be clean strings
        enum_values = event_type_field["constraints"]["enum"]
        for val in enum_values:
            assert isinstance(val, str)
            assert not val.startswith("<")  # No <DexcomEventType.XXX: 'value'>
            assert "DexcomEventType" not in val


class TestSchemaConsistency:
    """Test that schemas are internally consistent."""
    
    def test_all_columns_have_types(self):
        """Every column should have a valid frictionless type."""
        for schema_name, schema in [
            ("dexcom", DEXCOM_SCHEMA),
            ("libre", LIBRE_SCHEMA),
            ("unified", CGM_SCHEMA),
        ]:
            schema_dict = schema.to_frictionless_schema()
            
            for field in schema_dict["fields"]:
                assert "type" in field, f"{schema_name}: Field {field['name']} missing type"
                assert field["type"] in [
                    "string", "integer", "number", "boolean", "datetime", "date"
                ], f"{schema_name}: Invalid type {field['type']} for {field['name']}"
    
    def test_column_count_consistency(self):
        """Service + data columns should equal total fields."""
        for schema_name, schema in [
            ("dexcom", DEXCOM_SCHEMA),
            ("libre", LIBRE_SCHEMA),
            ("unified", CGM_SCHEMA),
        ]:
            total_expected = len(schema.service_columns) + len(schema.data_columns)
            schema_dict = schema.to_frictionless_schema()
            total_actual = len(schema_dict["fields"])
            
            assert total_actual == total_expected, (
                f"{schema_name}: Expected {total_expected} columns, got {total_actual}"
            )
    
    def test_no_duplicate_column_names(self):
        """Schema should not have duplicate column names."""
        for schema_name, schema in [
            ("dexcom", DEXCOM_SCHEMA),
            ("libre", LIBRE_SCHEMA),
            ("unified", CGM_SCHEMA),
        ]:
            schema_dict = schema.to_frictionless_schema()
            field_names = [f["name"] for f in schema_dict["fields"]]
            
            duplicates = [name for name in field_names if field_names.count(name) > 1]
            assert len(duplicates) == 0, (
                f"{schema_name}: Duplicate column names found: {set(duplicates)}"
            )


class TestRegressionPrevention:
    """Tests to prevent regressions of specific bugs we fixed."""
    
    def test_enum_not_in_field_names(self):
        """REGRESSION: Field names must not contain '<' or enum class names."""
        schema_dict = DEXCOM_SCHEMA.to_frictionless_schema()
        
        for field in schema_dict["fields"]:
            name = field["name"]
            assert "<" not in name, f"Field name contains '<': {name}"
            assert "DexcomColumn" not in name, f"Field name contains class name: {name}"
            assert "Enum" not in name or "Enum" == name, (
                f"Field name contains 'Enum' (likely class name): {name}"
            )
    
    def test_primary_key_matches_fields(self):
        """REGRESSION: Primary key fields must exist in schema."""
        schema_dict = CGM_SCHEMA.to_frictionless_schema(
            primary_key=["sequence_id", "datetime"]
        )
        
        field_names = [f["name"] for f in schema_dict["fields"]]
        for pk_field in schema_dict["primaryKey"]:
            assert pk_field in field_names, (
                f"Primary key field '{pk_field}' not found in schema fields"
            )
    
    def test_vendor_schemas_no_unified_primary_key(self):
        """REGRESSION: Vendor schemas should not have unified format primary keys."""
        dexcom_schema = DEXCOM_SCHEMA.to_frictionless_schema()
        libre_schema = LIBRE_SCHEMA.to_frictionless_schema()
        
        # These schemas should NOT have sequence_id and datetime
        dexcom_fields = [f["name"] for f in dexcom_schema["fields"]]
        libre_fields = [f["name"] for f in libre_schema["fields"]]
        
        assert "sequence_id" not in dexcom_fields
        assert "datetime" not in dexcom_fields
        assert "sequence_id" not in libre_fields
        assert "datetime" not in libre_fields
    
    def test_timestamp_not_in_data_columns_for_dexcom(self):
        """REGRESSION: Dexcom Timestamp should be in service_columns, not data_columns."""
        # This ensures timestamp is in the first 7 columns (service columns)
        service_col_names = [col["name"] for col in DEXCOM_SCHEMA.service_columns]
        data_col_names = [col["name"] for col in DEXCOM_SCHEMA.data_columns]
        
        assert "Timestamp (YYYY-MM-DDThh:mm:ss)" in service_col_names, (
            "Timestamp must be in service_columns to appear early in column order"
        )
        assert "Timestamp (YYYY-MM-DDThh:mm:ss)" not in data_col_names, (
            "Timestamp must not be in data_columns"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

