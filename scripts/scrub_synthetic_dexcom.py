#!/usr/bin/env python3
"""
Script to create synthetic Dexcom CGM data from real data for CI testing.

Transformations:
1. Replace transmitter ID with random ID in same format (6 alphanumeric chars)
2. Change dates to 12.04.1961 base date (Gagarin's space flight)
3. Replace patient name with "Gagarin"
4. Apply baseline offset (10-20 random) and random noise (±1) to glucose values
5. Offset all timestamps by random minutes (multiple of 5)
6. Adjust transmitter time accordingly
"""

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import typer


app = typer.Typer()


def generate_synthetic_transmitter_id() -> str:
    """Generate a random transmitter ID in Dexcom format (6 alphanumeric chars)."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(6))


@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to input Dexcom Clarity CSV file"),
    output_file: Path = typer.Argument(..., help="Path to output synthetic CSV file"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Main entry point."""
    random.seed(seed)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Generate synthetic identifiers
    synthetic_transmitter_id = generate_synthetic_transmitter_id()
    
    # Random baseline offset for glucose values (10-20 up or down)
    baseline_direction = random.choice([-1, 1])
    baseline_offset = baseline_direction * random.randint(10, 20)
    
    # Random time offset in minutes (multiple of 5)
    time_offset_minutes = random.choice(range(-60, 65, 5))  # -60 to +60 minutes in 5-min steps
    
    # Time offset for transmitter time (in seconds)
    transmitter_time_offset = time_offset_minutes * 60
    
    print(f"=== Synthetic Dexcom Data Generation ===")
    print(f"- New transmitter ID: {synthetic_transmitter_id}")
    print(f"- Glucose baseline offset: {baseline_offset:+d} mg/dL")
    print(f"- Time offset: {time_offset_minutes:+d} minutes")
    
    # Read the CSV with all columns as strings initially
    df = pl.read_csv(
        input_file,
        encoding='utf-8-lossy',
        try_parse_dates=False,
        infer_schema_length=0  # Read all as strings
    )
    
    # Get column names
    col_names = df.columns
    
    # Replace transmitter ID
    if "Transmitter ID" in col_names:
        df = df.with_columns(
            pl.when(pl.col("Transmitter ID").is_not_null() & (pl.col("Transmitter ID") != ""))
            .then(pl.lit(synthetic_transmitter_id))
            .otherwise(pl.col("Transmitter ID"))
            .alias("Transmitter ID")
        )
    
    # Replace patient name in metadata rows
    if "Patient Info" in col_names:
        df = df.with_columns(
            pl.when(pl.col("Patient Info") == "Patient")
            .then(pl.lit("Gagarin"))
            .otherwise(pl.col("Patient Info"))
            .alias("Patient Info")
        )
    
    # Find base date from first EGV row to maintain relative timing
    timestamp_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    base_date = None
    
    for row in df.iter_rows(named=True):
        if row.get("Event Type") == "EGV" and row.get(timestamp_col):
            try:
                base_date = datetime.strptime(row[timestamp_col], "%Y-%m-%d %H:%M:%S")
                break
            except (ValueError, TypeError):
                continue
    
    if base_date is None:
        raise ValueError("Could not find valid base date in data")
    
    # Apply date transformation
    def transform_timestamp(ts: str | None) -> str | None:
        if not ts or ts.strip() == "":
            return ts
        try:
            parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            time_diff = parsed - base_date
            new_date = datetime(1961, 4, 12, base_date.hour, base_date.minute, base_date.second) + time_diff
            # Apply time offset
            new_date = new_date + timedelta(minutes=time_offset_minutes)
            return new_date.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return ts
    
    if timestamp_col in col_names:
        df = df.with_columns(
            pl.col(timestamp_col).map_elements(transform_timestamp, return_dtype=pl.Utf8).alias(timestamp_col)
        )
    
    # Apply glucose transformations (baseline offset + random noise)
    def transform_glucose(value: str | None) -> str | None:
        if not value or value.strip() == "":
            return value
        try:
            glucose = int(float(value))
            noise = random.choice([-1, 0, 1])
            new_value = glucose + baseline_offset + noise
            # Ensure glucose stays in reasonable range (40-400 mg/dL)
            new_value = max(40, min(400, new_value))
            return str(new_value)
        except (ValueError, TypeError):
            return value
    
    if "Glucose Value (mg/dL)" in col_names:
        df = df.with_columns(
            pl.col("Glucose Value (mg/dL)").map_elements(
                transform_glucose, 
                return_dtype=pl.Utf8
            ).alias("Glucose Value (mg/dL)")
        )
    
    # Adjust transmitter time (if present)
    def transform_transmitter_time(value: str | None) -> str | None:
        if not value or value.strip() == "":
            return value
        try:
            time_val = int(value)
            new_time = time_val + transmitter_time_offset
            return str(new_time)
        except (ValueError, TypeError):
            return value
    
    if "Transmitter Time (Long Integer)" in col_names:
        df = df.with_columns(
            pl.col("Transmitter Time (Long Integer)").map_elements(
                transform_transmitter_time,
                return_dtype=pl.Utf8
            ).alias("Transmitter Time (Long Integer)")
        )
    
    # Write output
    df.write_csv(output_file, include_bom=True)
    
    print(f"✓ Synthetic data created: {output_file}")
    print(f"✓ Base date adjusted to: 12-04-1961")
    print(f"✓ Patient name: Gagarin")
    print(f"✓ Glucose values: baseline {baseline_offset:+d} + noise (±1)")
    print(f"✓ All timestamps shifted by {time_offset_minutes:+d} minutes")
    print(f"✓ Transmitter time adjusted by {transmitter_time_offset:+d} seconds")

if __name__ == "__main__":
    app()

