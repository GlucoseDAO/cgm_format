#!/usr/bin/env python3
"""CGM Format CLI Tool - Command-line interface for CGM data processing.

This tool provides access to all parser, processor, and validation features:
- Format detection and parsing
- Data processing (interpolation, synchronization, sequence detection)
- Schema validation
- Full pipeline execution
- Batch processing

Can be used as:
- Installed command: cgm-cli <command>
- Python module: python -m cgm_format.cgm_cli <command>
- Direct script: python scripts/cgm_cli.py <command>
"""

from pathlib import Path
from typing import Optional

import typer
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from cgm_format.format_parser import FormatParser
from cgm_format.format_processor import FormatProcessor
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    UnknownFormatError,
    MalformedDataError,
    ZeroValidInputError,
    ProcessingWarning,
)
from cgm_format.formats.unified import CGM_SCHEMA, UnifiedEventType, Quality
from cgm_format.formats.supported import SCHEMA_MAP, KNOWN_ISSUES_TO_SUPPRESS
# Optional: Frictionless library
try:
    from frictionless import Resource, Schema as FrictionlessSchema, Dialect
    HAS_FRICTIONLESS = True
except ImportError:
    HAS_FRICTIONLESS = False

app = typer.Typer(
    name="cgm-cli",
    help="CGM Format CLI - Parse, process, and validate CGM data",
    add_completion=False,
)
console = Console()


# ===== Format Detection & Parsing Commands =====

@app.command()
def detect(
    input_file: Path = typer.Argument(..., help="Input CSV file to detect format"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
) -> None:
    """Detect the format of a CGM data file."""
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        with open(input_file, 'rb') as f:
            raw_data = f.read()
        
        text_data = FormatParser.decode_raw_data(raw_data)
        detected_format = FormatParser.detect_format(text_data)
        
        console.print(f"\n[green]✓[/green] Detected format: [bold]{detected_format.value}[/bold]")
        
        if verbose:
            console.print(f"\nFile: {input_file}")
            console.print(f"Size: {len(raw_data)} bytes")
            console.print(f"Format: {detected_format.name}")
        
    except UnknownFormatError as e:
        console.print(f"[red]✗ Unknown format: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def parse(
    input_file: Path = typer.Argument(..., help="Input CSV file to parse"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file (unified format)"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show statistics"),
    show_preview: bool = typer.Option(False, "--preview", "-p", help="Show data preview"),
) -> None:
    """Parse a CGM data file to unified format."""
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        with console.status(f"[bold green]Parsing {input_file.name}..."):
            unified_df = FormatParser.parse_file(input_file)
        
        console.print(f"\n[green]✓[/green] Successfully parsed {len(unified_df)} rows")
        
        if show_stats:
            _print_dataframe_stats(unified_df, "Parsed Data")
        
        if show_preview:
            console.print("\n[bold]Data Preview:[/bold]")
            console.print(unified_df.head(10))
        
        if output_file:
            FormatParser.to_csv_file(unified_df, str(output_file))
            console.print(f"\n[green]✓[/green] Saved to: {output_file}")
        
    except (UnknownFormatError, MalformedDataError, ZeroValidInputError) as e:
        console.print(f"[red]✗ Parse error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


# ===== Processing Commands =====

@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input unified format CSV file"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file"),
    interpolate: bool = typer.Option(True, "--interpolate/--no-interpolate", help="Interpolate gaps"),
    synchronize: bool = typer.Option(True, "--sync/--no-sync", help="Synchronize timestamps"),
    interval: int = typer.Option(5, "--interval", "-i", help="Expected interval in minutes"),
    max_gap: int = typer.Option(19, "--max-gap", help="Maximum gap to interpolate (minutes)"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show statistics"),
) -> None:
    """Process unified format data (interpolate gaps, synchronize timestamps)."""
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Parse input file
        with console.status(f"[bold green]Loading {input_file.name}..."):
            df = FormatParser.parse_file(input_file)
        
        original_rows = len(df)
        console.print(f"[green]✓[/green] Loaded {original_rows} rows")
        
        # Detect sequences
        with console.status("[bold green]Detecting sequences..."):
            df = FormatProcessor.detect_and_assign_sequences(
                df,
                expected_interval_minutes=interval,
                large_gap_threshold_minutes=max_gap
            )
        
        sequence_count = df['sequence_id'].n_unique()
        console.print(f"[green]✓[/green] Detected {sequence_count} sequence(s)")
        
        # Interpolate gaps
        if interpolate:
            with console.status("[bold green]Interpolating gaps..."):
                df = FormatProcessor.interpolate_gaps(
                    df,
                    expected_interval_minutes=interval,
                    small_gap_max_minutes=max_gap
                )
            interpolated_rows = len(df)
            added_rows = interpolated_rows - original_rows
            console.print(f"[green]✓[/green] Interpolated {added_rows} rows ({interpolated_rows} total)")
        
        # Synchronize timestamps
        if synchronize:
            with console.status("[bold green]Synchronizing timestamps..."):
                df = FormatProcessor.synchronize_timestamps(
                    df,
                    expected_interval_minutes=interval
                )
            console.print(f"[green]✓[/green] Synchronized timestamps to {interval}-minute grid")
        
        if show_stats:
            _print_dataframe_stats(df, "Processed Data")
        
        if output_file:
            FormatParser.to_csv_file(df, str(output_file))
            console.print(f"\n[green]✓[/green] Saved to: {output_file}")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def pipeline(
    input_file: Path = typer.Argument(..., help="Input CGM data file (any supported format)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file"),
    interval: int = typer.Option(5, "--interval", "-i", help="Expected interval in minutes"),
    max_gap: int = typer.Option(19, "--max-gap", help="Maximum gap to interpolate (minutes)"),
    min_duration: int = typer.Option(15, "--min-duration", help="Minimum sequence duration (minutes)"),
    max_duration: int = typer.Option(1440, "--max-duration", help="Maximum sequence duration (minutes)"),
    glucose_only: bool = typer.Option(False, "--glucose-only", help="Keep only glucose events"),
    drop_duplicates: bool = typer.Option(True, "--drop-duplicates/--keep-duplicates", help="Drop duplicate timestamps"),
    show_warnings: bool = typer.Option(True, "--warnings/--no-warnings", help="Show processing warnings"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show statistics"),
) -> None:
    """Run the complete processing pipeline on a CGM data file.
    
    This executes all stages:
    1. Parse vendor format to unified
    2. Detect and assign sequences
    3. Interpolate gaps
    4. Synchronize timestamps
    5. Prepare for inference (quality checks, warnings)
    6. Convert to data-only format (optional)
    """
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        console.print(f"\n[bold]Running full pipeline on: {input_file.name}[/bold]\n")
        
        # Stage 1: Parse
        with console.status("[bold green]Stage 1/6: Parsing..."):
            unified_df = FormatParser.parse_file(input_file)
        parsed_rows = len(unified_df)
        console.print(f"[green]✓[/green] Stage 1: Parsed {parsed_rows} rows")
        
        # Stage 2: Detect sequences
        with console.status("[bold green]Stage 2/6: Detecting sequences..."):
            unified_df = FormatProcessor.detect_and_assign_sequences(
                unified_df,
                expected_interval_minutes=interval,
                large_gap_threshold_minutes=max_gap
            )
        sequence_count = unified_df['sequence_id'].n_unique()
        console.print(f"[green]✓[/green] Stage 2: Detected {sequence_count} sequence(s)")
        
        # Stage 3: Interpolate
        with console.status("[bold green]Stage 3/6: Interpolating gaps..."):
            unified_df = FormatProcessor.interpolate_gaps(
                unified_df,
                expected_interval_minutes=interval,
                small_gap_max_minutes=max_gap
            )
        interpolated_rows = len(unified_df)
        console.print(f"[green]✓[/green] Stage 3: Interpolated to {interpolated_rows} rows")
        
        # Stage 4: Synchronize
        with console.status("[bold green]Stage 4/6: Synchronizing timestamps..."):
            unified_df = FormatProcessor.synchronize_timestamps(
                unified_df,
                expected_interval_minutes=interval
            )
        console.print(f"[green]✓[/green] Stage 4: Synchronized timestamps")
        
        # Stage 5: Prepare for inference
        with console.status("[bold green]Stage 5/6: Preparing for inference..."):
            inference_df, warnings = FormatProcessor.prepare_for_inference(
                unified_df,
                minimum_duration_minutes=min_duration,
                maximum_wanted_duration=max_duration
            )
        inference_rows = len(inference_df)
        console.print(f"[green]✓[/green] Stage 5: Prepared {inference_rows} rows for inference")
        
        # Show warnings
        if show_warnings and warnings:
            console.print("\n[yellow]⚠ Processing Warnings:[/yellow]")
            for warning in ProcessingWarning:
                if warnings & warning:
                    console.print(f"  • {warning.name}: {_get_warning_description(warning)}")
        elif show_warnings:
            console.print("\n[green]✓ No processing warnings[/green]")
        
        # Stage 6: Convert to data-only format
        with console.status("[bold green]Stage 6/6: Converting to final format..."):
            final_df = FormatProcessor.to_data_only_df(
                inference_df,
                drop_service_columns=True,
                drop_duplicates=drop_duplicates,
                glucose_only=glucose_only
            )
        final_rows = len(final_df)
        console.print(f"[green]✓[/green] Stage 6: Generated {final_rows} final rows")
        
        if show_stats:
            _print_dataframe_stats(final_df, "Final Data")
        
        if output_file:
            # Use regular Polars write_csv since final_df is data-only
            final_df.write_csv(str(output_file))
            console.print(f"\n[green]✓[/green] Saved to: {output_file}")
        
        console.print(f"\n[bold green]Pipeline completed successfully![/bold green]")
        
    except (UnknownFormatError, MalformedDataError, ZeroValidInputError) as e:
        console.print(f"\n[red]✗ Pipeline error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


# ===== Validation Commands =====

@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Input CSV file to validate"),
    format_type: Optional[str] = typer.Option(None, "--format", "-f", help="Format type (unified, dexcom, libre)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation results"),
) -> None:
    """Validate a CSV file against its schema."""
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Detect format if not provided
        if not format_type:
            with console.status("[bold green]Detecting format..."):
                with open(input_file, 'rb') as f:
                    raw_data = f.read()
                text_data = FormatParser.decode_raw_data(raw_data)
                detected_format = FormatParser.detect_format(text_data)
            console.print(f"[green]✓[/green] Detected format: {detected_format.value}")
        else:
            format_map = {
                "unified": SupportedCGMFormat.UNIFIED_CGM,
                "dexcom": SupportedCGMFormat.DEXCOM,
                "libre": SupportedCGMFormat.LIBRE,
            }
            if format_type.lower() not in format_map:
                console.print(f"[red]Error: Unknown format '{format_type}'. Use: unified, dexcom, or libre[/red]")
                raise typer.Exit(1)
            detected_format = format_map[format_type.lower()]
        
        # Parse and validate
        with console.status("[bold green]Validating..."):
            try:
                df = FormatParser.parse_file(input_file)
                # Validate schema
                CGM_SCHEMA.validate_dataframe(df, enforce=False)
                validation_passed = True
            except Exception as e:
                validation_passed = False
                validation_error = str(e)
        
        if validation_passed:
            console.print(f"\n[green]✓ Validation passed![/green]")
            console.print(f"  Rows: {len(df)}")
            console.print(f"  Columns: {len(df.columns)}")
            
            if verbose:
                console.print("\n[bold]Column Information:[/bold]")
                for col in df.columns:
                    dtype = df[col].dtype
                    null_count = df[col].null_count()
                    console.print(f"  • {col}: {dtype} ({null_count} nulls)")
        else:
            console.print(f"\n[red]✗ Validation failed![/red]")
            console.print(f"  Error: {validation_error}")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    input_dir: Path = typer.Argument(..., help="Directory containing CGM data files"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output report file (default: validation_report.txt)"),
    pattern: str = typer.Option("*.csv", "--pattern", "-p", help="File pattern to match"),
    use_frictionless: bool = typer.Option(True, "--frictionless/--no-frictionless", help="Use Frictionless validation"),
    suppress_known: bool = typer.Option(True, "--suppress-known/--show-all", help="Suppress known vendor issues"),
) -> None:
    """Generate comprehensive validation report for all files in directory.
    
    This command:
    - Detects formats for all CSV files
    - Validates using Frictionless schemas (if available)
    - Suppresses known vendor format issues
    - Generates detailed text report
    
    Similar to examples/example_schema_usage.py functionality.
    """
    try:
        if not input_dir.exists():
            console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
            raise typer.Exit(1)
        
        if not input_dir.is_dir():
            console.print(f"[red]Error: Not a directory: {input_dir}[/red]")
            raise typer.Exit(1)
        
        # Default output file
        if not output_file:
            output_file = Path("validation_report.txt")
        
        # Check Frictionless availability
        if use_frictionless and not HAS_FRICTIONLESS:
            console.print("[yellow]⚠ Frictionless library not available - using internal validation only[/yellow]")
            console.print("  Install with: pip install frictionless")
            use_frictionless = False
        
        console.print(f"\n[bold]Generating Validation Report[/bold]")
        console.print(f"Input: {input_dir}")
        console.print(f"Pattern: {pattern}")
        console.print(f"Output: {output_file}")
        console.print(f"Frictionless: {'Enabled' if use_frictionless else 'Disabled'}")
        console.print(f"Suppress known issues: {'Yes' if suppress_known else 'No'}\n")
        
        # Stage 1: Detect formats for all files
        files = sorted(input_dir.glob(pattern))
        if not files:
            console.print(f"[red]Error: No files matching '{pattern}' found in {input_dir}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[bold]Stage 1: Format Detection[/bold] ({len(files)} files)")
        
        detection_results = []
        format_counts = {}
        failed_detection = []
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Detecting formats...", total=len(files))
            
            for file_path in files:
                progress.update(task, description=f"Detecting: {file_path.name}...")
                
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                    text_data = FormatParser.decode_raw_data(raw_data)
                    detected_format = FormatParser.detect_format(text_data)
                    
                    detection_results.append((file_path, detected_format, ""))
                    format_counts[detected_format] = format_counts.get(detected_format, 0) + 1
                    
                except Exception as e:
                    detection_results.append((file_path, None, str(e)))
                    failed_detection.append((file_path.name, str(e)))
                
                progress.advance(task)
        
        successful_detection = [(f, fmt) for f, fmt, err in detection_results if not err]
        
        console.print(f"  [green]✓[/green] Detected: {len(successful_detection)}/{len(files)}")
        console.print(f"  [red]✗[/red] Failed: {len(failed_detection)}")
        
        # Show format breakdown
        if format_counts:
            console.print("\n  Format breakdown:")
            for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
                console.print(f"    • {fmt.value}: {count} files")
        
        # Stage 2: Validate with Frictionless
        validation_results = []
        
        if use_frictionless and successful_detection:
            console.print(f"\n[bold]Stage 2: Frictionless Validation[/bold] ({len(successful_detection)} files)")
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
                task = progress.add_task("Validating...", total=len(successful_detection))
                
                for file_path, format_type in successful_detection:
                    progress.update(task, description=f"Validating: {file_path.name}...")
                    
                    is_valid, msg, error_count, suppressed_count = _validate_with_frictionless(
                        file_path, format_type, suppress_known
                    )
                    validation_results.append((file_path, format_type, is_valid, msg, error_count, suppressed_count))
                    
                    progress.advance(task)
            
            valid_count = sum(1 for _, _, is_valid, _, _, _ in validation_results if is_valid)
            console.print(f"  [green]✓[/green] Valid: {valid_count}/{len(successful_detection)}")
            console.print(f"  [red]✗[/red] Invalid: {len(successful_detection) - valid_count}")
        
        # Stage 3: Write report
        console.print(f"\n[bold]Stage 3: Writing Report[/bold]")
        _write_validation_report(
            output_file, input_dir, detection_results, validation_results,
            use_frictionless, suppress_known
        )
        
        console.print(f"\n[green]✓ Report written to: {output_file}[/green]")
        
        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total files: {len(files)}")
        console.print(f"  Detected: {len(successful_detection)}")
        if use_frictionless:
            valid_count = sum(1 for _, _, is_valid, _, _, _ in validation_results if is_valid)
            console.print(f"  Valid: {valid_count}")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


# ===== Batch Processing Commands =====

@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing CGM data files"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    pattern: str = typer.Option("*.csv", "--pattern", "-p", help="File pattern to match"),
    command: str = typer.Option("parse", "--command", "-c", help="Command to run (parse, process, pipeline)"),
    continue_on_error: bool = typer.Option(True, "--continue/--stop", help="Continue on errors"),
) -> None:
    """Batch process multiple CGM data files."""
    try:
        if not input_dir.exists():
            console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
            raise typer.Exit(1)
        
        if not input_dir.is_dir():
            console.print(f"[red]Error: Not a directory: {input_dir}[/red]")
            raise typer.Exit(1)
        
        files = sorted(input_dir.glob(pattern))
        if not files:
            console.print(f"[red]Error: No files matching '{pattern}' found in {input_dir}[/red]")
            raise typer.Exit(1)
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[bold]Batch processing {len(files)} file(s)[/bold]\n")
        
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=len(files))
            
            for file_path in files:
                progress.update(task, description=f"Processing {file_path.name}...")
                
                try:
                    # Determine output path
                    if output_dir:
                        output_path = output_dir / file_path.name
                    else:
                        output_path = None
                    
                    # Run command
                    if command == "parse":
                        df = FormatParser.parse_file(file_path)
                        if output_path:
                            FormatParser.to_csv_file(df, str(output_path))
                    elif command == "process":
                        df = FormatParser.parse_file(file_path)
                        df = FormatProcessor.detect_and_assign_sequences(df)
                        df = FormatProcessor.interpolate_gaps(df)
                        df = FormatProcessor.synchronize_timestamps(df)
                        if output_path:
                            FormatParser.to_csv_file(df, str(output_path))
                    elif command == "pipeline":
                        df = FormatParser.parse_file(file_path)
                        df = FormatProcessor.detect_and_assign_sequences(df)
                        df = FormatProcessor.interpolate_gaps(df)
                        df = FormatProcessor.synchronize_timestamps(df)
                        df, _ = FormatProcessor.prepare_for_inference(df)
                        df = FormatProcessor.to_data_only_df(df, drop_service_columns=True)
                        if output_path:
                            df.write_csv(str(output_path))
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        raise typer.Exit(1)
                    
                    results["success"] += 1
                    
                except Exception as e:
                    results["failed"] += 1
                    if not continue_on_error:
                        console.print(f"\n[red]✗ Error processing {file_path.name}: {e}[/red]")
                        raise typer.Exit(1)
                
                progress.advance(task)
        
        # Print summary
        console.print(f"\n[bold]Batch processing complete:[/bold]")
        console.print(f"  [green]Success: {results['success']}[/green]")
        if results["failed"] > 0:
            console.print(f"  [red]Failed: {results['failed']}[/red]")
        if results["skipped"] > 0:
            console.print(f"  [yellow]Skipped: {results['skipped']}[/yellow]")
        
        if output_dir:
            console.print(f"\n[green]✓[/green] Output saved to: {output_dir}")
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


# ===== Info & Stats Commands =====

@app.command()
def info(
    input_file: Path = typer.Argument(..., help="Input CSV file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
) -> None:
    """Show information about a CGM data file."""
    try:
        if not input_file.exists():
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Detect format
        with console.status("[bold green]Analyzing file..."):
            with open(input_file, 'rb') as f:
                raw_data = f.read()
            
            text_data = FormatParser.decode_raw_data(raw_data)
            detected_format = FormatParser.detect_format(text_data)
            
            # Parse
            df = FormatParser.parse_file(input_file)
        
        # Display information
        console.print(f"\n[bold]File Information: {input_file.name}[/bold]\n")
        
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Format", detected_format.value)
        table.add_row("Size", f"{len(raw_data):,} bytes")
        table.add_row("Rows", f"{len(df):,}")
        table.add_row("Columns", str(len(df.columns)))
        
        # Time range
        if 'datetime' in df.columns:
            min_time = df['datetime'].min()
            max_time = df['datetime'].max()
            duration = (max_time - min_time).total_seconds() / 3600  # hours
            table.add_row("Time Range", f"{min_time} to {max_time}")
            table.add_row("Duration", f"{duration:.1f} hours")
        
        # Glucose statistics
        if 'glucose' in df.columns:
            glucose_count = df['glucose'].count()
            if glucose_count > 0:
                glucose_mean = df['glucose'].mean()
                glucose_min = df['glucose'].min()
                glucose_max = df['glucose'].max()
                table.add_row("Glucose Readings", f"{glucose_count:,}")
                table.add_row("Glucose Mean", f"{glucose_mean:.1f} mg/dL")
                table.add_row("Glucose Range", f"{glucose_min:.1f} - {glucose_max:.1f} mg/dL")
        
        # Event types
        if 'event_type' in df.columns:
            event_counts = df.group_by('event_type').agg(pl.count().alias('count'))
            table.add_row("Event Types", str(len(event_counts)))
        
        # Sequences
        if 'sequence_id' in df.columns:
            sequence_count = df['sequence_id'].n_unique()
            if sequence_count > 0:
                table.add_row("Sequences", str(sequence_count))
        
        console.print(table)
        
        if detailed:
            console.print(f"\n[bold]Column Details:[/bold]")
            detail_table = Table()
            detail_table.add_column("Column", style="cyan")
            detail_table.add_column("Type", style="yellow")
            detail_table.add_column("Non-Null", style="green")
            detail_table.add_column("Null", style="red")
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = len(df) - df[col].null_count()
                null_count = df[col].null_count()
                detail_table.add_row(col, dtype, str(non_null), str(null_count))
            
            console.print(detail_table)
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)


# ===== Helper Functions =====

def _print_dataframe_stats(df: pl.DataFrame, title: str) -> None:
    """Print statistics about a dataframe."""
    console.print(f"\n[bold]{title} Statistics:[/bold]")
    
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Rows", f"{len(df):,}")
    table.add_row("Columns", str(len(df.columns)))
    
    if 'datetime' in df.columns:
        min_time = df['datetime'].min()
        max_time = df['datetime'].max()
        duration_hours = (max_time - min_time).total_seconds() / 3600
        table.add_row("Time Range", f"{min_time} to {max_time}")
        table.add_row("Duration", f"{duration_hours:.1f} hours")
    
    if 'glucose' in df.columns:
        glucose_count = df['glucose'].count()
        if glucose_count > 0:
            glucose_mean = df['glucose'].mean()
            glucose_std = df['glucose'].std()
            glucose_min = df['glucose'].min()
            glucose_max = df['glucose'].max()
            table.add_row("Glucose Count", f"{glucose_count:,}")
            table.add_row("Glucose Mean ± SD", f"{glucose_mean:.1f} ± {glucose_std:.1f} mg/dL")
            table.add_row("Glucose Range", f"{glucose_min:.1f} - {glucose_max:.1f} mg/dL")
    
    if 'sequence_id' in df.columns:
        sequence_count = df['sequence_id'].n_unique()
        if sequence_count > 0:
            table.add_row("Sequences", str(sequence_count))
    
    if 'event_type' in df.columns:
        event_counts = df.group_by('event_type').agg(pl.count().alias('count'))
        for row in event_counts.iter_rows():
            event_type_value, count = row
            event_name = _get_event_type_name(event_type_value)
            table.add_row(f"  {event_name}", f"{count:,}")
    
    if 'quality' in df.columns:
        # Count quality flags
        imputed = df.filter((pl.col('quality') & Quality.IMPUTATION.value) != 0).height
        out_of_range = df.filter((pl.col('quality') & Quality.OUT_OF_RANGE.value) != 0).height
        time_dup = df.filter((pl.col('quality') & Quality.TIME_DUPLICATE.value) != 0).height
        sync = df.filter((pl.col('quality') & Quality.SYNCHRONIZATION.value) != 0).height
        calibration = df.filter((pl.col('quality') & Quality.SENSOR_CALIBRATION.value) != 0).height
        
        if imputed > 0:
            table.add_row("  Imputed", f"{imputed:,}")
        if out_of_range > 0:
            table.add_row("  Out of Range", f"{out_of_range:,}")
        if time_dup > 0:
            table.add_row("  Time Duplicates", f"{time_dup:,}")
        if sync > 0:
            table.add_row("  Synchronized", f"{sync:,}")
        if calibration > 0:
            table.add_row("  Calibration Period", f"{calibration:,}")
    
    console.print(table)


def _get_event_type_name(event_type_value: int) -> str:
    """Get human-readable event type name."""
    try:
        return UnifiedEventType(event_type_value).name
    except ValueError:
        return f"Unknown ({event_type_value})"


def _get_warning_description(warning: ProcessingWarning) -> str:
    """Get human-readable warning description."""
    descriptions = {
        ProcessingWarning.TOO_SHORT: "Sequence duration below minimum threshold",
        ProcessingWarning.CALIBRATION: "Contains calibration events or post-calibration period",
        ProcessingWarning.OUT_OF_RANGE: "Contains out-of-range glucose readings",
        ProcessingWarning.IMPUTATION: "Contains imputed/interpolated data points",
        ProcessingWarning.TIME_DUPLICATES: "Contains duplicate timestamps",
    }
    return descriptions.get(warning, "Unknown warning")


def _should_suppress_error(error: any, format_type: SupportedCGMFormat, suppress_known: bool) -> bool:
    """Check if an error should be suppressed based on known format issues.
    
    Args:
        error: Frictionless error object or dict
        format_type: The CGM format type
        suppress_known: Whether to suppress known issues
        
    Returns:
        True if error should be suppressed, False otherwise
    """
    if not suppress_known:
        return False
    
    suppressions = KNOWN_ISSUES_TO_SUPPRESS.get(format_type, [])
    if not suppressions:
        return False
    
    # Extract error type, field name, and cell value from error
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
    for rule in suppressions:
        rule_error_type, rule_field_name, rule_cell_value = rule
        
        if error_type == rule_error_type and field_name == rule_field_name:
            if rule_cell_value is None:
                return True
            if cell_value == rule_cell_value:
                return True
    
    return False


def _validate_with_frictionless(
    csv_path: Path,
    format_type: SupportedCGMFormat,
    suppress_known: bool
) -> tuple[bool, str, int, int]:
    """Validate a CSV file against its format's Frictionless schema.
    
    Args:
        csv_path: Path to CSV file
        format_type: Detected format type
        suppress_known: Whether to suppress known vendor issues
        
    Returns:
        Tuple of (is_valid, message, error_count, suppressed_count)
    """
    if not HAS_FRICTIONLESS:
        return False, "Frictionless library not available", 0, 0
    
    try:
        schema = SCHEMA_MAP[format_type]
        frictionless_schema_dict = schema.to_frictionless_schema()
        
        # Convert to relative path
        try:
            relative_path = csv_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = csv_path
        
        # Extract dialect from schema dict
        dialect_dict = frictionless_schema_dict.pop('dialect', None)
        if dialect_dict is None:
            dialect_dict = schema.get_dialect()
        
        schema_obj = FrictionlessSchema.from_descriptor(frictionless_schema_dict)
        
        # Validate using Resource API
        if dialect_dict:
            dialect_obj = Dialect.from_descriptor(dialect_dict)
            resource = Resource(path=str(relative_path), schema=schema_obj, dialect=dialect_obj)
        else:
            resource = Resource(path=str(relative_path), schema=schema_obj)
        
        report = resource.validate()
        
        if report.valid:
            row_count = report.tasks[0].stats.get('rows', 'unknown') if report.tasks else 'unknown'
            return True, f"Valid ({row_count} rows)", 0, 0
        
        # Collect errors
        error_count = 0
        suppressed_count = 0
        errors = []
        
        for task in report.tasks:
            if hasattr(task, 'errors') and task.errors:
                for error in task.errors:
                    if _should_suppress_error(error, format_type, suppress_known):
                        suppressed_count += 1
                        continue
                    
                    error_count += 1
                    if len(errors) < 5:
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
            return True, f"Valid ({row_count} rows, {suppressed_count} known issues suppressed)", 0, suppressed_count
        
        # Build error message
        result_msg = f"Invalid ({error_count} errors"
        if suppressed_count > 0:
            result_msg += f", {suppressed_count} known issues suppressed"
        result_msg += ")\n" + "\n".join(errors)
        if error_count > len(errors):
            result_msg += f"\n  ... and {error_count - len(errors)} more errors"
        
        return False, result_msg, error_count, suppressed_count
        
    except Exception as e:
        return False, f"Validation error: {str(e)}", 0, 0


def _write_validation_report(
    output_file: Path,
    input_dir: Path,
    detection_results: list,
    validation_results: list,
    use_frictionless: bool,
    suppress_known: bool
) -> None:
    """Write comprehensive validation report to file.
    
    Args:
        output_file: Path to output text file
        input_dir: Directory that was scanned
        detection_results: List of (file_path, format_type, error_msg) tuples
        validation_results: List of (file_path, format_type, is_valid, msg, error_count, suppressed_count) tuples
        use_frictionless: Whether Frictionless validation was used
        suppress_known: Whether known issues were suppressed
    """
    from datetime import datetime
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("CGM FORMAT DETECTION AND VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Total files scanned: {len(detection_results)}\n")
        f.write(f"Frictionless validation: {'Enabled' if use_frictionless else 'Disabled'}\n")
        f.write(f"Suppress known issues: {'Yes' if suppress_known else 'No'}\n")
        f.write("\n")
        
        # Format detection summary
        f.write("=" * 80 + "\n")
        f.write("FORMAT DETECTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        
        successful = [(f, fmt) for f, fmt, err in detection_results if not err]
        failed = [(f, err) for f, fmt, err in detection_results if err]
        
        f.write(f"Successfully detected: {len(successful)}/{len(detection_results)}\n")
        f.write(f"Failed detection: {len(failed)}/{len(detection_results)}\n")
        f.write("\n")
        
        # Format breakdown
        if successful:
            format_counts = {}
            for _, fmt in successful:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            
            f.write("Format breakdown:\n")
            for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {fmt.value:15} : {count:3} files\n")
            f.write("\n")
        
        if failed:
            f.write("Failed detections:\n")
            for file_path, error in failed:
                f.write(f"  {file_path.name}\n")
                f.write(f"    Error: {error}\n")
            f.write("\n")
        
        # Validation results
        if use_frictionless and validation_results:
            f.write("=" * 80 + "\n")
            f.write("FRICTIONLESS SCHEMA VALIDATION RESULTS\n")
            f.write("=" * 80 + "\n")
            
            valid_count = sum(1 for _, _, is_valid, _, _, _ in validation_results if is_valid)
            invalid_count = len(validation_results) - valid_count
            
            f.write(f"Valid files: {valid_count}/{len(validation_results)}\n")
            f.write(f"Invalid files: {invalid_count}/{len(validation_results)}\n")
            f.write("\n")
            
            if suppress_known:
                f.write("NOTE: Known vendor format issues are automatically suppressed:\n")
                f.write("  - Dexcom: Missing Transmitter ID/Time cells in non-EGV rows\n")
                f.write("            (Dexcom exports have variable-length rows)\n")
                f.write("  - Dexcom: 'Low' and 'High' text in Glucose Value field\n")
                f.write("            (Out-of-range markers: <50 and >400 mg/dL)\n")
                f.write("  - Dexcom: UTF-8 BOM marker in CSV header\n")
                f.write("            (Some exports include byte order mark)\n")
                f.write("\n")
            
            # Group by format type
            for format_type in [SupportedCGMFormat.UNIFIED_CGM, SupportedCGMFormat.DEXCOM, SupportedCGMFormat.LIBRE]:
                format_results = [
                    (fp, fmt, v, m, ec, sc) for fp, fmt, v, m, ec, sc in validation_results 
                    if fmt == format_type
                ]
                if not format_results:
                    continue
                
                f.write(f"\n{format_type.value} Format ({len(format_results)} files):\n")
                f.write("-" * 80 + "\n")
                
                for file_path, _, is_valid, msg, error_count, suppressed_count in format_results:
                    status = "✓ VALID" if is_valid else "✗ INVALID"
                    f.write(f"\n{file_path.name}\n")
                    f.write(f"  Status: {status}\n")
                    if error_count > 0 or suppressed_count > 0:
                        f.write(f"  Errors: {error_count}, Suppressed: {suppressed_count}\n")
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
            
            f.write("\nColumns:\n")
            for col in schema.service_columns + schema.data_columns:
                unit = f" [{col.get('unit')}]" if col.get('unit') else ""
                f.write(f"  - {col['name']}{unit}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


# ===== Main Entry Point =====

def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()


