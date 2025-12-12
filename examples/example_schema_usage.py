"""Example usage of the CGM schema definition with format detection and validation."""

from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any

from cgm_format import FormatParser
from cgm_format.interface.cgm_interface import SupportedCGMFormat, UnknownFormatError, MalformedDataError
from cgm_format.formats.supported import SCHEMA_MAP, KNOWN_ISSUES_TO_SUPPRESS

# Optional: Use frictionless library if available
try:
    from frictionless import validate, Schema as FrictionlessSchema
    HAS_FRICTIONLESS = True
except ImportError:
    HAS_FRICTIONLESS = False


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


def detect_all_files(data_dir: Path, parsed_dir: Path | None = None) -> List[Tuple[Path, SupportedCGMFormat, str]]:
    """Detect formats for all CSV files in the data directory and optionally parsed directory.
    
    Args:
        data_dir: Path to directory containing raw CSV files
        parsed_dir: Optional path to directory containing unified format CSV files
        
    Returns:
        List of tuples: (file_path, detected_format, error_msg)
        error_msg is empty string if successful
    """
    results = []
    
    # Collect all CSV files from both directories
    csv_files = sorted(data_dir.glob("*.csv"))
    if parsed_dir and parsed_dir.exists():
        csv_files.extend(sorted(parsed_dir.glob("*.csv")))
    
    for csv_file in csv_files:
        try:
            # Read and decode
            with open(csv_file, 'rb') as f:
                raw_data = f.read()
            text_data = FormatParser.decode_raw_data(raw_data)
            
            # Detect format
            detected_format = FormatParser.detect_format(text_data)
            results.append((csv_file, detected_format, ""))
            
        except (UnknownFormatError, MalformedDataError, Exception) as e:
            results.append((csv_file, None, str(e)))
    
    return results


def validate_with_frictionless(csv_path: Path, format_type: SupportedCGMFormat) -> Tuple[bool, str]:
    """Validate a CSV file against its format's Frictionless schema.
    
    Args:
        csv_path: Path to CSV file
        format_type: Detected format type
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if not HAS_FRICTIONLESS:
        return False, "Frictionless library not available - install with: pip install frictionless"
    
    try:
        # Get the appropriate schema
        schema = SCHEMA_MAP[format_type]
        
        # Schema now contains auto-generated dialect and primary_key
        # based on format constants (HEADER_LINE, DATA_START_LINE, METADATA_LINES)
        frictionless_schema_dict = schema.to_frictionless_schema()
        
        # Convert to relative path (frictionless requires relative paths for security)
        try:
            relative_path = csv_path.relative_to(Path.cwd())
        except ValueError:
            # If not relative to cwd, just use the file name
            relative_path = csv_path
        
        # Create Schema and Dialect objects for proper validation
        from frictionless import Resource, Schema, Dialect
        
        # Extract dialect from schema dict
        dialect_dict = frictionless_schema_dict.pop('dialect', None)
        schema_obj = Schema.from_descriptor(frictionless_schema_dict)
        
        # Validate using Resource API
        # Only pass dialect if it exists (passing None causes issues)
        if dialect_dict:
            dialect_obj = Dialect.from_descriptor(dialect_dict)
            resource = Resource(path=str(relative_path), schema=schema_obj, dialect=dialect_obj)
        else:
            resource = Resource(path=str(relative_path), schema=schema_obj)
        
        report = resource.validate()
        
        if report.valid:
            # Get row count from first task's stats
            row_count = report.tasks[0].stats.get('rows', 'unknown') if report.tasks else 'unknown'
            return True, f"✓ Valid ({row_count} rows)"
        else:
            # Collect error messages, filtering out known issues
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
                        # Only collect first few errors for display
                        if len(errors) < 5:
                            # Handle both dict-like and object-like error formats
                            if hasattr(error, 'message'):
                                error_msg = error.message
                            elif isinstance(error, dict):
                                error_msg = error.get('message', str(error))
                            else:
                                error_msg = str(error)
                            errors.append(f"  - {error_msg}")
            
            # If all errors were suppressed, report as valid
            if error_count == 0 and suppressed_count > 0:
                row_count = report.tasks[0].stats.get('rows', 'unknown') if report.tasks else 'unknown'
                return True, f"✓ Valid ({row_count} rows, {suppressed_count} known issues suppressed)"
            
            # Build result message with remaining errors
            result_msg = f"✗ Invalid ({error_count} errors"
            if suppressed_count > 0:
                result_msg += f", {suppressed_count} known issues suppressed"
            result_msg += ")\n" + "\n".join(errors)
            if error_count > len(errors):
                result_msg += f"\n  ... and {error_count - len(errors)} more errors"
            
            return False, result_msg
            
    except Exception as e:
        return False, f"✗ Validation error: {str(e)}"


def run_format_detection_and_validation(data_dir: Path, parsed_dir: Path | None, output_file: Path) -> None:
    """Run format detection and validation on all CSV files, write results to file.
    
    Args:
        data_dir: Directory containing raw CSV files to validate
        parsed_dir: Optional directory containing unified format CSV files
        output_file: Path to output text file for results
    """
    print(f"\n{'=' * 80}")
    print(f"CGM FORMAT DETECTION AND VALIDATION")
    print(f"{'=' * 80}")
    print(f"Data directory: {data_dir}")
    if parsed_dir:
        print(f"Parsed directory: {parsed_dir}")
    print(f"Output file: {output_file}")
    print(f"Frictionless available: {HAS_FRICTIONLESS}")
    print(f"{'=' * 80}\n")
    
    # Stage 1: Detect all formats
    print("Stage 1: Detecting formats...")
    detection_results = detect_all_files(data_dir, parsed_dir)
    
    # Count formats
    format_counts = Counter()
    failed_detection = []
    successful_detection = []
    
    for csv_file, format_type, error_msg in detection_results:
        if error_msg:
            failed_detection.append((csv_file.name, error_msg))
        else:
            format_counts[format_type] += 1
            successful_detection.append((csv_file, format_type))
    
    print(f"  Total files: {len(detection_results)}")
    print(f"  Successfully detected: {len(successful_detection)}")
    print(f"  Failed detection: {len(failed_detection)}")
    
    # Stage 2: Validate with Frictionless
    print("\nStage 2: Validating with Frictionless schemas...")
    validation_results = []
    
    if HAS_FRICTIONLESS:
        for csv_file, format_type in successful_detection:
            is_valid, msg = validate_with_frictionless(csv_file, format_type)
            validation_results.append((csv_file, format_type, is_valid, msg))
            print(f"  {csv_file.name}: {msg.split(chr(10))[0]}")  # Print first line only
    else:
        print("  Skipping validation - frictionless library not installed")
    
    # Stage 3: Write detailed report to file
    print(f"\nStage 3: Writing report to {output_file}...")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("CGM FORMAT DETECTION AND VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total files scanned: {len(detection_results)}\n")
        f.write(f"Frictionless validation: {'Enabled' if HAS_FRICTIONLESS else 'Disabled (library not installed)'}\n")
        f.write("\n")
        
        # Format detection summary
        f.write("=" * 80 + "\n")
        f.write("FORMAT DETECTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Successfully detected: {len(successful_detection)}/{len(detection_results)}\n")
        f.write(f"Failed detection: {len(failed_detection)}/{len(detection_results)}\n")
        f.write("\n")
        
        f.write("Format breakdown:\n")
        for format_type, count in format_counts.most_common():
            f.write(f"  {format_type.value:15} : {count:3} files\n")
        f.write("\n")
        
        if failed_detection:
            f.write("Failed detections:\n")
            for filename, error in failed_detection:
                f.write(f"  {filename}\n")
                f.write(f"    Error: {error}\n")
            f.write("\n")
        
        # Validation results (if available)
        if HAS_FRICTIONLESS and validation_results:
            f.write("=" * 80 + "\n")
            f.write("FRICTIONLESS SCHEMA VALIDATION RESULTS\n")
            f.write("=" * 80 + "\n")
            
            valid_count = sum(1 for _, _, is_valid, _ in validation_results if is_valid)
            invalid_count = len(validation_results) - valid_count
            
            f.write(f"Valid files: {valid_count}/{len(validation_results)}\n")
            f.write(f"Invalid files: {invalid_count}/{len(validation_results)}\n")
            f.write("\n")
            f.write("NOTE: Known vendor format issues are automatically suppressed:\n")
            f.write("  - Dexcom: Missing Transmitter ID/Time cells in non-EGV rows\n")
            f.write("            (Dexcom exports have variable-length rows)\n")
            f.write("  - Dexcom: 'Low' and 'High' text in Glucose Value field\n")
            f.write("            (Out-of-range markers: <50 and >400 mg/dL)\n")
            f.write("  - Dexcom: UTF-8 BOM marker in CSV header\n")
            f.write("            (Some exports include byte order mark)\n")
            f.write("\n")
            
            # Group by format type
            for format_type in [SupportedCGMFormat]:
                format_results = [(f, fmt, v, m) for f, fmt, v, m in validation_results if fmt == format_type]
                if not format_results:
                    continue
                
                f.write(f"\n{format_type.value} Format ({len(format_results)} files):\n")
                f.write("-" * 80 + "\n")
                
                for csv_file, _, is_valid, msg in format_results:
                    status = "✓ VALID" if is_valid else "✗ INVALID"
                    f.write(f"\n{csv_file.name}\n")
                    f.write(f"  Status: {status}\n")
                    f.write(f"  {msg}\n")
        
        # Schema information
        f.write("\n" + "=" * 80 + "\n")
        f.write("SCHEMA DEFINITIONS\n")
        f.write("=" * 80 + "\n")
        
        for format_type, schema in SCHEMA_MAP.items():
            f.write(f"\n{format_type.value} Schema:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Service columns: {len(schema.service_columns)}\n")
            f.write(f"Data columns: {len(schema.data_columns)}\n")
            f.write(f"Total columns: {len(schema.service_columns) + len(schema.data_columns)}\n")
            
            # List columns
            f.write("\nColumns:\n")
            for col in schema.service_columns + schema.data_columns:
                unit = f" [{col.get('unit')}]" if col.get('unit') else ""
                f.write(f"  - {col['name']}{unit}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Report written to {output_file}")
    print(f"\nSummary:")
    print(f"  Detected: {len(successful_detection)}/{len(detection_results)} files")
    if HAS_FRICTIONLESS:
        valid_count = sum(1 for _, _, is_valid, _ in validation_results if is_valid)
        print(f"  Valid: {valid_count}/{len(validation_results)} files")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    parsed_dir = project_root / "data" / "parsed"
    output_file = project_root / "validation_report.txt"
    
    # Check data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        exit(1)
    
    # Check if parsed directory exists
    if not parsed_dir.exists():
        print(f"Warning: Parsed directory not found: {parsed_dir}")
        parsed_dir = None
    
    # Run format detection and validation
    run_format_detection_and_validation(data_dir, parsed_dir, output_file)
    
    print(f"Done! View the full report at: {output_file}")

