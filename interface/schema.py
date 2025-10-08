"""Base Schema Infrastructure.

This module defines the base types, enums, and schema builder classes
that can be used to define any CGM data format schema.
"""

import polars as pl
from enum import Enum
from typing import Dict, Any, List, Union, Type, TypedDict, NotRequired


class EnumLiteral(str, Enum):
    """
    A general base class for string-based enums that behave like literals.
    Ensures compatibility with str comparisons and retains enum benefits.
    """
    def __new__(cls, value, *args, **kwargs):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self):
        # String representation directly returns the value
        return self.value

    def __eq__(self, other):
        # Allow direct comparison with strings
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self):
        # Use the hash of the value to behave like a string in hashable contexts
        return hash(self.value)
    
    def __repr__(self):
        # For print statements and serialization
        return self.value


class ColumnSchema(TypedDict):
    """Schema definition for a single column."""
    name: str
    dtype: Union[Type[pl.DataType], pl.DataType]
    description: str
    unit: NotRequired[str]
    constraints: NotRequired[Dict[str, Any]]


class CGMSchemaDefinition:
    """Complete schema definition builder for CGM data formats.
    
    This class provides infrastructure for defining and working with
    CGM data schemas, including conversion to various formats (Polars,
    Frictionless Data Table Schema) and generation of cast expressions.
    """
    
    def __init__(
        self,
        service_columns: List[ColumnSchema],
        data_columns: List[ColumnSchema],
        header_line: int = 1,
        data_start_line: int = 2,
        metadata_lines: tuple[int, ...] = (),
        primary_key: List[str] | None = None
    ) -> None:
        """Initialize schema definition.
        
        Args:
            service_columns: Metadata columns (e.g., sequence_id, event_type, quality)
            data_columns: Data columns (e.g., datetime, glucose, carbs, insulin)
            header_line: Line number where the header row is located (1-indexed)
            data_start_line: Line number where data rows start (1-indexed)
            metadata_lines: Tuple of line numbers that are metadata to skip (1-indexed)
            primary_key: Optional list of field names that form the primary key
        """
        self.service_columns = service_columns
        self.data_columns = data_columns
        self.header_line = header_line
        self.data_start_line = data_start_line
        self.metadata_lines = metadata_lines
        self.primary_key = primary_key
        self._dialect = self._generate_dialect(header_line, metadata_lines)
    
    def get_polars_schema(self, data_only: bool = False) -> Dict[str, pl.DataType]:
        """Get Polars dtype schema dictionary.
        
        Args:
            data_only: If True, return only data columns (excludes service columns)
            
        Returns:
            Dictionary mapping column names to Polars data types
        """
        columns = self.data_columns if data_only else self.service_columns + self.data_columns
        return {col["name"]: col["dtype"] for col in columns}
    
    def get_inference_schema(self) -> 'CGMSchemaDefinition':
        """Get a schema with only data columns (for ML inference).
        
        The unified format is a matryoshka: service columns (sequence_id, event_type, quality)
        are stripped for inference, leaving only the core data columns.
        
        Returns:
            New CGMSchemaDefinition with only data columns
        """
        return CGMSchemaDefinition(
            service_columns=[],
            data_columns=self.data_columns,
            header_line=self.header_line,
            data_start_line=self.data_start_line,
            metadata_lines=self.metadata_lines,
            primary_key=self.primary_key  # Keep the same primary key (data columns)
        )
    
    def get_column_names(self, data_only: bool = False) -> List[str]:
        """Get list of all column names.
        
        Args:
            data_only: If True, return only data column names
            
        Returns:
            List of column names in schema order
        """
        columns = self.data_columns if data_only else self.service_columns + self.data_columns
        return [col["name"] for col in columns]
    
    def get_cast_expressions(self, data_only: bool = False) -> List[pl.Expr]:
        """Get Polars expressions for casting columns.
        
        Args:
            data_only: If True, return only data column expressions
            
        Returns:
            List of pl.col().cast() expressions for use with df.with_columns()
        """
        columns = self.data_columns if data_only else self.service_columns + self.data_columns
        return [pl.col(col["name"]).cast(col["dtype"]) for col in columns]
    
    @staticmethod
    def _generate_dialect(header_line: int, metadata_lines: tuple[int, ...]) -> Dict[str, Any] | None:
        """Generate Frictionless dialect configuration from format constants.
        
        Args:
            header_line: Line number where the header row is located (1-indexed)
            metadata_lines: Tuple of line numbers that are metadata to skip (1-indexed)
            
        Returns:
            Dialect dictionary for Frictionless validation, or None if standard format
            
        Examples:
            Dexcom (header=1, metadata=(2,3,4,5,6,7,8,9,10,11)):
                Returns {"commentRows": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
            Libre (header=2, metadata=(1,)):
                Returns {"headerRows": [2]}
            Unified (header=1, metadata=()):
                Returns None (standard CSV format)
        """
        # Standard CSV format: header on line 1, no metadata
        if header_line == 1 and not metadata_lines:
            return None
        
        dialect = {}
        
        # Header is not on line 1 (e.g., Libre with header on row 2)
        if header_line != 1:
            dialect["headerRows"] = [header_line]
        
        # There are metadata lines to skip (e.g., Dexcom with rows 2-11)
        if metadata_lines:
            dialect["commentRows"] = list(metadata_lines)
        
        return dialect if dialect else None
    
    def get_dialect(self) -> Dict[str, Any] | None:
        """Get the Frictionless dialect for this schema.
        
        Returns:
            Dialect dictionary or None if standard format
        """
        return self._dialect
    
    def to_frictionless_schema(
        self, 
        primary_key: List[str] | None = None,
        dialect: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Convert to Frictionless Data Table Schema format.
        
        Returns dictionary in Frictionless Data Table Schema format
        that can be used with the frictionless library for validation.
        
        Args:
            primary_key: Optional list of field names that form the primary key.
                        If None, uses the schema's primary_key (if set).
            dialect: Optional dialect configuration for CSV parsing.
                    If None, uses the auto-generated dialect from format constants.
                    For Dexcom: {"commentRows": [2,3,4,5,6,7,8,9,10,11]} to skip metadata rows
                    For Libre: {"headerRows": [2]} to specify header is on row 2
        
        Returns:
            Dictionary in Frictionless Data Table Schema format
        """
        fields = []
        
        for col in self.service_columns + self.data_columns:
            field = {
                "name": col["name"],
                "type": self._polars_to_frictionless_type(col["dtype"]),
                "description": col["description"],
            }
            if col.get("unit"):
                field["unit"] = col["unit"]
            if col.get("constraints"):
                field["constraints"] = col["constraints"]
            fields.append(field)
        
        schema = {"fields": fields}
        
        # Use provided primary_key, or fall back to schema's primary_key
        effective_primary_key = primary_key if primary_key is not None else self.primary_key
        if effective_primary_key:
            schema["primaryKey"] = effective_primary_key
        
        # Use provided dialect, or fall back to auto-generated dialect
        effective_dialect = dialect if dialect is not None else self._dialect
        if effective_dialect:
            schema["dialect"] = effective_dialect
        
        return schema
    
    @staticmethod
    def _polars_to_frictionless_type(dtype: pl.DataType) -> str:
        """Map Polars dtype to Frictionless Data type.
        
        Args:
            dtype: Polars data type
            
        Returns:
            Frictionless Data type string
        """
        # Use isinstance for parameterized types (e.g., pl.Datetime['ms'])
        if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
            return "datetime"
        elif isinstance(dtype, pl.Date) or dtype == pl.Date:
            return "date"
        elif isinstance(dtype, pl.Enum):
            return "string"
        # Use equality for simple types
        elif dtype == pl.Int64 or dtype == pl.Int32:
            return "integer"
        elif dtype == pl.Float64 or dtype == pl.Float32:
            return "number"
        elif dtype == pl.Utf8 or dtype == pl.String:
            return "string"
        elif dtype == pl.Boolean:
            return "boolean"
        else:
            return "string"
    
    def export_to_json(
        self, 
        output_path: str, 
        primary_key: List[str] | None = None,
        dialect: Dict[str, Any] | None = None
    ) -> None:
        """Export schema to JSON file in Frictionless Data Table Schema format.
        
        Args:
            output_path: Path to output JSON file
            primary_key: Optional list of field names that form the primary key
            dialect: Optional dialect configuration for CSV parsing
        """
        import json
        from pathlib import Path
        
        schema_file = Path(output_path)
        with open(schema_file, "w") as f:
            json.dump(self.to_frictionless_schema(primary_key=primary_key, dialect=dialect), f, indent=2)
            f.write("\n")  # Add trailing newline
        
        print(f"✓ Regenerated {schema_file}")


def regenerate_schema_json(
    schema: CGMSchemaDefinition, 
    calling_module_file: str,
    primary_key: List[str] | None = None,
    dialect: Dict[str, Any] | None = None
) -> None:
    """Regenerate schema JSON file from a schema definition.
    
    Automatically derives the JSON filename from the calling module filename.
    For example, if called from 'formats/unified.py', generates 'formats/unified.json'.
    
    Args:
        schema: The CGMSchemaDefinition instance to export
        calling_module_file: The __file__ variable from the calling module
        primary_key: Optional list of field names that form the primary key.
                    If None, uses schema's primary_key.
        dialect: Optional dialect configuration for CSV parsing.
                If None, uses schema's auto-generated dialect.
        
    Example:
        >>> # In formats/unified.py
        >>> regenerate_schema_json(CGM_SCHEMA, __file__)
        ✓ Regenerated /path/to/formats/unified.json
    """
    from pathlib import Path
    
    # Derive JSON filename from module filename
    module_path = Path(calling_module_file)
    json_path = module_path.with_suffix('.json')
    
    schema.export_to_json(json_path, primary_key=primary_key, dialect=dialect)

