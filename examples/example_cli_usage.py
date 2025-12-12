#!/usr/bin/env python3
"""Example CLI Usage Script - Demonstrates all cgm-cli commands.

This script shows how to use the cgm-cli tool from Python by calling it as a subprocess.
It's a practical demonstration of the CLI tool's capabilities.

Usage:
    uv run python examples/example_cli_usage.py
    
    # Or with specific data directory
    uv run python examples/example_cli_usage.py --data-dir /path/to/data
"""

import subprocess
import sys
from pathlib import Path
from typing import List
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer()
console = Console()


def run_cli_command(args: List[str], description: str = "") -> subprocess.CompletedProcess:
    """Run a cgm-cli command and display results.
    
    Args:
        args: Command arguments for cgm-cli
        description: Human-readable description of what this command does
        
    Returns:
        CompletedProcess with stdout/stderr
    """
    if description:
        console.print(f"\n[bold cyan]Example: {description}[/bold cyan]")
    
    # Run as module
    cmd = [sys.executable, "-m", "cgm_format.cgm_cli"] + args
    cmd_str = " ".join(args)
    console.print(f"[dim]$ cgm-cli {cmd_str}[/dim]\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Display output
    if result.stdout:
        console.print(result.stdout)
    
    if result.returncode != 0 and result.stderr:
        console.print(f"[red]{result.stderr}[/red]")
    
    return result


@app.command()
def main(
    data_dir: Path = typer.Option(
        Path(__file__).parent.parent / "data",
        "--data-dir",
        "-d",
        help="Directory containing CGM data files"
    ),
    skip_slow: bool = typer.Option(
        False,
        "--skip-slow",
        help="Skip slow examples (pipeline, batch, report)"
    ),
) -> None:
    """Run through all cgm-cli command examples."""
    
    console.print(Panel.fit(
        "[bold]CGM CLI Tool - Complete Usage Examples[/bold]\n\n"
        "This script demonstrates all cgm-cli commands with real data.\n"
        "Commands are executed via subprocess to show real-world usage.",
        border_style="cyan"
    ))
    
    # Check data directory
    if not data_dir.exists():
        console.print(f"\n[red]Error: Data directory not found: {data_dir}[/red]")
        console.print("Please specify a valid data directory with --data-dir")
        raise typer.Exit(1)
    
    # Find a test file
    csv_files = list(data_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "parsed" not in str(f) and "cli_test" not in str(f)]
    
    if not csv_files:
        console.print(f"\n[red]Error: No CSV files found in {data_dir}[/red]")
        raise typer.Exit(1)
    
    test_file = csv_files[0]
    console.print(f"\n[bold]Using test file:[/bold] {test_file.name}")
    console.print(f"[bold]Data directory:[/bold] {data_dir}")
    
    # Create output directory
    output_dir = data_dir / "cli_examples_output"
    output_dir.mkdir(exist_ok=True)
    console.print(f"[bold]Output directory:[/bold] {output_dir}\n")
    
    # ===== 1. Format Detection =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]1. FORMAT DETECTION COMMANDS[/bold green]")
    console.print("=" * 70)
    
    run_cli_command(
        ["detect", str(test_file)],
        "Detect the format of a CGM data file"
    )
    
    run_cli_command(
        ["detect", str(test_file), "--verbose"],
        "Detect format with detailed information"
    )
    
    # ===== 2. Parsing =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]2. PARSING COMMANDS[/bold green]")
    console.print("=" * 70)
    
    parsed_file = output_dir / f"parsed_{test_file.name}"
    run_cli_command(
        ["parse", str(test_file), "--output", str(parsed_file), "--stats"],
        "Parse CGM file to unified format with statistics"
    )
    
    if parsed_file.exists():
        run_cli_command(
            ["parse", str(test_file), "--preview", "--no-stats"],
            "Parse with data preview (no stats)"
        )
    
    # ===== 3. Validation =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]3. VALIDATION COMMANDS[/bold green]")
    console.print("=" * 70)
    
    run_cli_command(
        ["validate", str(test_file)],
        "Validate file against its detected schema"
    )
    
    if parsed_file.exists():
        run_cli_command(
            ["validate", str(parsed_file), "--format", "unified", "--verbose"],
            "Validate unified format file with details"
        )
    
    # ===== 4. File Information =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]4. FILE INFORMATION COMMANDS[/bold green]")
    console.print("=" * 70)
    
    run_cli_command(
        ["info", str(test_file)],
        "Show basic file information"
    )
    
    run_cli_command(
        ["info", str(test_file), "--detailed"],
        "Show detailed file information with column details"
    )
    
    if skip_slow:
        console.print("\n[yellow]⏭️  Skipping slow examples (--skip-slow)[/yellow]")
        console.print("\nTo see all examples, run without --skip-slow flag")
        return
    
    # ===== 5. Processing =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]5. PROCESSING COMMANDS[/bold green]")
    console.print("=" * 70)
    
    if parsed_file.exists():
        processed_file = output_dir / f"processed_{test_file.name}"
        run_cli_command(
            [
                "process", str(parsed_file),
                "--output", str(processed_file),
                "--interpolate",
                "--sync",
                "--interval", "5",
                "--max-gap", "19",
                "--stats"
            ],
            "Process unified file with interpolation and synchronization"
        )
    
    # ===== 6. Full Pipeline =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]6. FULL PIPELINE COMMAND[/bold green]")
    console.print("=" * 70)
    
    pipeline_file = output_dir / f"pipeline_{test_file.name}"
    run_cli_command(
        [
            "pipeline", str(test_file),
            "--output", str(pipeline_file),
            "--interval", "5",
            "--max-gap", "19",
            "--min-duration", "15",
            "--max-duration", "1440",
            "--drop-duplicates",
            "--warnings",
            "--stats"
        ],
        "Run complete 6-stage processing pipeline"
    )
    
    # ===== 7. Comprehensive Report =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]7. VALIDATION REPORT COMMAND[/bold green]")
    console.print("=" * 70)
    
    report_file = output_dir / "validation_report.txt"
    run_cli_command(
        [
            "report", str(data_dir),
            "--output", str(report_file),
            "--pattern", "*.csv",
            "--frictionless",
            "--suppress-known"
        ],
        "Generate comprehensive validation report for all files"
    )
    
    if report_file.exists():
        console.print(f"\n[bold]Report contents preview:[/bold]")
        with open(report_file) as f:
            preview = f.read(500)
        console.print(f"[dim]{preview}...[/dim]")
    
    # ===== 8. Batch Processing =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]8. BATCH PROCESSING COMMAND[/bold green]")
    console.print("=" * 70)
    
    batch_output = output_dir / "batch_output"
    batch_output.mkdir(exist_ok=True)
    
    run_cli_command(
        [
            "batch", str(data_dir),
            "--output", str(batch_output),
            "--pattern", "*.csv",
            "--command", "parse",
            "--continue"
        ],
        "Batch process all CSV files in directory"
    )
    
    # ===== Summary =====
    console.print("\n" + "=" * 70)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 70)
    
    console.print(f"\n[bold]All examples completed![/bold]")
    console.print(f"\nOutput files saved to: {output_dir}")
    console.print(f"\nGenerated files:")
    for output_file in sorted(output_dir.glob("*")):
        if output_file.is_file():
            size = output_file.stat().st_size
            console.print(f"  • {output_file.name} ({size:,} bytes)")
    
    console.print("\n[bold cyan]Command Categories:[/bold cyan]")
    console.print("  1. [bold]detect[/bold] - Format detection")
    console.print("  2. [bold]parse[/bold] - Parse to unified format")
    console.print("  3. [bold]validate[/bold] - Schema validation")
    console.print("  4. [bold]info[/bold] - File information")
    console.print("  5. [bold]process[/bold] - Interpolation & synchronization")
    console.print("  6. [bold]pipeline[/bold] - Complete processing pipeline")
    console.print("  7. [bold]report[/bold] - Comprehensive validation report")
    console.print("  8. [bold]batch[/bold] - Batch process directories")
    
    console.print("\n[bold cyan]For help on any command:[/bold cyan]")
    console.print("  cgm-cli <command> --help")
    
    console.print("\n[bold green]✓ All examples completed successfully![/bold green]\n")


if __name__ == "__main__":
    app()

