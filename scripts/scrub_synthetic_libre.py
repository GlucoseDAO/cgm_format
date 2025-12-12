#!/usr/bin/env python3
"""
Script to create synthetic FreeStyle Libre CGM data from real data for CI testing.

Transformations:
1. Replace serial number with random GUID of same format
2. Change dates to 12.04.1961 base date (Gagarin's space flight)
3. Replace patient name with "Gagarin"
4. Remove patient notes
5. Apply baseline offset (10-20 random) and random noise (±1) to glucose values
6. Offset all timestamps by random minutes (multiple of 5)
"""

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import typer


app = typer.Typer()


def generate_synthetic_serial() -> str:
    """Generate a random UUID in uppercase format matching the original."""
    return str(uuid.uuid4()).upper()


@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to input FreeStyle Libre CSV file"),
    output_file: Path = typer.Argument(..., help="Path to output synthetic CSV file"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Main entry point."""
    random.seed(seed)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate synthetic identifiers
    synthetic_serial = generate_synthetic_serial()
    
    # Random baseline offset for glucose values (10-20 up or down)
    baseline_direction = random.choice([-1, 1])
    baseline_offset = baseline_direction * random.randint(10, 20)
    
    # Random time offset in minutes (multiple of 5)
    time_offset_minutes = random.choice(range(-60, 65, 5))  # -60 to +60 minutes in 5-min steps
    
    print(f"=== Synthetic Data Generation ===")
    print(f"- New serial number: {synthetic_serial}")
    print(f"- Glucose baseline offset: {baseline_offset:+d} mg/dL")
    print(f"- Time offset: {time_offset_minutes:+d} minutes")
    
    # Read the file - first row is special header
    with open(input_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    
    # Parse and modify header
    header_parts = header_line.split(',')
    if len(header_parts) >= 5:
        header_parts[2] = "12-04-1961 09:40 UTC"  # Generated on date
        header_parts[4] = "Gagarin"  # Patient name
    new_header = ','.join(header_parts)
    
    # Read the data starting from line 2 (0-indexed line 1)
    df = pl.read_csv(
        input_file,
        skip_rows=1,
        encoding='utf-8',
        try_parse_dates=False
    )
    
    # Replace serial number
    df = df.with_columns(
        pl.lit(synthetic_serial).alias("Serial Number")
    )
    
    # Parse dates and apply transformations
    # First, find the base date from the first row
    first_date_str = df["Device Timestamp"][0]
    base_date = datetime.strptime(first_date_str, "%d-%m-%Y %H:%M")
    
    # Apply date transformation: shift to 1961 base + time offset
    def transform_date(date_str: str) -> str:
        if not date_str:
            return date_str
        try:
            parsed = datetime.strptime(date_str, "%d-%m-%Y %H:%M")
            time_diff = parsed - base_date
            new_date = datetime(1961, 4, 12, base_date.hour, base_date.minute) + time_diff
            # Apply time offset
            new_date = new_date + timedelta(minutes=time_offset_minutes)
            return new_date.strftime("%d-%m-%Y %H:%M")
        except (ValueError, TypeError):
            return date_str
    
    df = df.with_columns(
        pl.col("Device Timestamp").map_elements(transform_date, return_dtype=pl.Utf8).alias("Device Timestamp")
    )
    
    # Apply glucose transformations (baseline offset + random noise)
    def transform_glucose(value: int | None) -> int | None:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        try:
            glucose = int(value) if not isinstance(value, int) else value
            noise = random.choice([-1, 0, 1])
            new_value = glucose + baseline_offset + noise
            # Ensure glucose stays in reasonable range (40-400 mg/dL)
            return max(40, min(400, new_value))
        except (ValueError, TypeError):
            return None
    
    # Apply to Historic Glucose
    if "Historic Glucose mg/dL" in df.columns:
        df = df.with_columns(
            pl.col("Historic Glucose mg/dL").map_elements(
                transform_glucose, 
                return_dtype=pl.Int64
            ).alias("Historic Glucose mg/dL")
        )
    
    # Apply to Scan Glucose
    if "Scan Glucose mg/dL" in df.columns:
        df = df.with_columns(
            pl.col("Scan Glucose mg/dL").map_elements(
                transform_glucose,
                return_dtype=pl.Int64
            ).alias("Scan Glucose mg/dL")
        )
    
    # Apply to Strip Glucose
    if "Strip Glucose mg/dL" in df.columns:
        df = df.with_columns(
            pl.col("Strip Glucose mg/dL").map_elements(
                transform_glucose,
                return_dtype=pl.Int64
            ).alias("Strip Glucose mg/dL")
        )
    
    # Remove patient notes (Cyrillic characters indicate patient notes)
    def clean_notes(note: str | None) -> str:
        if note is None or not isinstance(note, str) or note.strip() == "":
            return ""
        # Check for Cyrillic characters
        if any('\u0400' <= char <= '\u04FF' for char in note):
            return ""  # Remove patient notes
        return note  # Keep system notes
    
    if "Notes" in df.columns:
        df = df.with_columns(
            pl.col("Notes").map_elements(clean_notes, return_dtype=pl.Utf8).alias("Notes")
        )
    
    # Write output with custom header
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write(new_header + '\n')
        # Get CSV content from dataframe
        csv_content = df.write_csv()
        f.write(csv_content)
    
    print(f"✓ Synthetic data created: {output_file}")
    print(f"✓ Base date adjusted to: 12-04-1961")
    print(f"✓ Patient name: Gagarin")
    print(f"✓ Patient notes removed")
    print(f"✓ Glucose values: baseline {baseline_offset:+d} + noise (±1)")
    print(f"✓ All timestamps shifted by {time_offset_minutes:+d} minutes")

if __name__ == "__main__":
    app()
