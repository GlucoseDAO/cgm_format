"""Compare glucose_data_processing vs cgm_format on Livia Dexcom test data.

Uses ``data/livia_test.csv`` from the sibling ``glucose_data_processing`` repo
(Dexcom G6 Clarity export, ~25k raw rows). Does not assert pipelines match —
documents structural and numeric differences to guide alignment work.

Requires:
    ../glucose_data_processing with ``data/livia_test.csv`` and a working ``uv`` env.

Run with report printed to stdout::

    uv run pytest tests/test_livia_gdp_cgm_comparison.py -s -v
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cgm_format import FormatParser, FormatProcessor
from cgm_format.formats.unified import UnifiedEventType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GDP_ROOT = PROJECT_ROOT.parent / "glucose_data_processing"
LIVIA_CSV = GDP_ROOT / "data" / "livia_test.csv"

GDP_MIN_SEQUENCE_LEN = 200
CGM_EXPECTED_INTERVAL_MINUTES = 5
CGM_SMALL_GAP_MAX_MINUTES = 15


@dataclass(frozen=True)
class PipelineSnapshot:
    """Summary statistics for one pipeline output."""

    label: str
    row_count: int
    column_names: tuple[str, ...]
    sequence_count: int | None
    glucose_row_count: int
    timestamp_start: datetime | None
    timestamp_end: datetime | None
    glucose_min: float | None
    glucose_max: float | None
    glucose_mean: float | None

    def format_block(self) -> str:
        lines = [
            f"  [{self.label}]",
            f"    rows: {self.row_count:,}",
            f"    columns ({len(self.column_names)}): {', '.join(self.column_names)}",
        ]
        if self.sequence_count is not None:
            lines.append(f"    sequences: {self.sequence_count:,}")
        lines.append(f"    glucose rows: {self.glucose_row_count:,}")
        if self.timestamp_start and self.timestamp_end:
            lines.append(
                f"    time range: {self.timestamp_start} -> {self.timestamp_end}"
            )
        if self.glucose_min is not None:
            lines.append(
                f"    glucose mg/dL: min={self.glucose_min:.3f} "
                f"max={self.glucose_max:.3f} mean={self.glucose_mean:.3f}"
            )
        return "\n".join(lines)


def _require_livia_csv() -> Path:
    if not LIVIA_CSV.is_file():
        pytest.skip(f"Livia fixture missing: {LIVIA_CSV}")
    if not GDP_ROOT.is_dir():
        pytest.skip(f"glucose_data_processing repo missing: {GDP_ROOT}")
    return LIVIA_CSV


def _run_gdp_ml_ready(livia_csv: Path, work_dir: Path) -> pl.DataFrame:
    """Run glucose-process on a folder containing livia_test.csv."""
    input_dir = work_dir / "livia_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(livia_csv, input_dir / livia_csv.name)

    output_name = "livia_cgm_format_comparison_gdp.csv"
    cmd = [
        "uv",
        "run",
        "glucose-process",
        str(input_dir),
        "-o",
        output_name,
        "--min-length",
        str(GDP_MIN_SEQUENCE_LEN),
        "--no-stats",
    ]
    result = subprocess.run(
        cmd,
        cwd=GDP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "glucose-process failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    output_path = GDP_ROOT / "OUTPUT" / output_name
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected GDP output: {output_path}")
    return pl.read_csv(output_path, infer_schema_length=10_000)


def _run_cgm_parsed(livia_csv: Path) -> pl.DataFrame:
    return FormatParser.parse_file(livia_csv)


def _run_cgm_processed(livia_csv: Path) -> pl.DataFrame:
    df = FormatParser.parse_file(livia_csv)
    df = FormatProcessor.detect_and_assign_sequences(
        df,
        expected_interval_minutes=CGM_EXPECTED_INTERVAL_MINUTES,
        large_gap_threshold_minutes=CGM_SMALL_GAP_MAX_MINUTES,
    )
    df = FormatProcessor.interpolate_gaps(
        df,
        expected_interval_minutes=CGM_EXPECTED_INTERVAL_MINUTES,
        small_gap_max_minutes=CGM_SMALL_GAP_MAX_MINUTES,
    )
    return FormatProcessor.synchronize_timestamps(
        df,
        expected_interval_minutes=CGM_EXPECTED_INTERVAL_MINUTES,
    )


def _run_cgm_inference_pipeline(livia_csv: Path) -> pl.DataFrame:
    """Match ``cgm-cli pipeline`` stages through prepare_for_inference + data-only export."""
    df = _run_cgm_processed(livia_csv)
    df, _warnings = FormatProcessor.prepare_for_inference(
        df,
        minimum_duration_minutes=15,
        maximum_wanted_duration=1440,
    )
    return FormatProcessor.to_data_only_df(df)


def _find_column(columns: list[str], *candidates: str) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _snapshot_dataframe(
    df: pl.DataFrame,
    label: str,
    *,
    timestamp_col: str | None = None,
    glucose_col: str | None = None,
    sequence_col: str | None = None,
    glucose_event_filter: pl.Expr | None = None,
) -> PipelineSnapshot:
    columns = list(df.columns)
    ts_col = timestamp_col or _find_column(
        columns,
        "datetime",
        "timestamp",
        "Timestamp (YYYY-MM-DDThh:mm:ss)",
        "Timestamp",
    )
    g_col = glucose_col or _find_column(
        columns,
        "glucose",
        "Glucose Value (mg/dL)",
        "glucose_value_mgdl",
        "Glucose (mg/dL)",
    )
    seq_col = sequence_col or _find_column(columns, "sequence_id")

    if glucose_event_filter is not None:
        glucose_df = df.filter(glucose_event_filter)
    elif g_col is not None:
        glucose_df = df.filter(pl.col(g_col).is_not_null())
    else:
        glucose_df = df.clear()

    ts_start: datetime | None = None
    ts_end: datetime | None = None
    if ts_col and df.height > 0:
        ts_series = df[ts_col]
        if ts_series.dtype == pl.Utf8:
            ts_series = ts_series.str.strptime(pl.Datetime, strict=False)
        ts_start = ts_series.min()
        ts_end = ts_series.max()

    g_min = g_max = g_mean = None
    if g_col and glucose_df.height > 0:
        numeric = glucose_df[g_col].cast(pl.Float64, strict=False)
        g_min = numeric.min()
        g_max = numeric.max()
        g_mean = numeric.mean()

    seq_count = int(df[seq_col].n_unique()) if seq_col else None

    return PipelineSnapshot(
        label=label,
        row_count=df.height,
        column_names=tuple(columns),
        sequence_count=seq_count,
        glucose_row_count=glucose_df.height,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        glucose_min=g_min,
        glucose_max=g_max,
        glucose_mean=g_mean,
    )


def _glucose_overlap_stats(gdp_df: pl.DataFrame, cgm_df: pl.DataFrame) -> dict[str, Any]:
    """Compare EGV glucose values on overlapping timestamps (5-minute grid strings)."""
    gdp_ts = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    gdp_g = "Glucose Value (mg/dL)"
    cgm_ts = "datetime"
    cgm_g = "glucose"

    gdp_glucose = (
        gdp_df.filter(pl.col(gdp_g).is_not_null())
        .select(
            pl.col(gdp_ts).alias("ts"),
            pl.col(gdp_g).cast(pl.Float64, strict=False).alias("gdp_glucose"),
        )
    )
    cgm_glucose = (
        cgm_df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
        .select(
            pl.col(cgm_ts).dt.strftime("%Y-%m-%dT%H:%M:%S").alias("ts"),
            pl.col(cgm_g).cast(pl.Float64, strict=False).alias("cgm_glucose"),
        )
    )

    joined = gdp_glucose.join(cgm_glucose, on="ts", how="inner")
    if joined.height == 0:
        return {
            "overlap_rows": 0,
            "exact_matches": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "sample_diffs": [],
        }

    diffs = (joined["gdp_glucose"] - joined["cgm_glucose"]).abs()
    sample = (
        joined.with_columns(diffs.alias("abs_diff"))
        .filter(pl.col("abs_diff") > 0.01)
        .select(["ts", "gdp_glucose", "cgm_glucose", "abs_diff"])
        .head(5)
    )
    return {
        "overlap_rows": joined.height,
        "exact_matches": int((diffs <= 0.01).sum()),
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "sample_diffs": sample.to_dicts(),
    }


def build_comparison_report(
    gdp_df: pl.DataFrame,
    cgm_parsed: pl.DataFrame,
    cgm_processed: pl.DataFrame,
    cgm_inference: pl.DataFrame,
) -> str:
    """Build a human-readable comparison report."""
    gdp_snap = _snapshot_dataframe(gdp_df, "GDP ML-ready (fixed-frequency)")
    parsed_snap = _snapshot_dataframe(
        cgm_parsed,
        "cgm_format parsed (unified)",
        glucose_event_filter=pl.col("event_type") == UnifiedEventType.GLUCOSE.value,
    )
    processed_snap = _snapshot_dataframe(
        cgm_processed,
        "cgm_format processed (sequences + interpolate + sync)",
        glucose_event_filter=pl.col("event_type") == UnifiedEventType.GLUCOSE.value,
    )
    inference_snap = _snapshot_dataframe(cgm_inference, "cgm_format inference (CLI pipeline)")

    overlap = _glucose_overlap_stats(gdp_df, cgm_processed)

    parsed_events = (
        cgm_parsed.group_by("event_type")
        .len()
        .sort("len", descending=True)
        .to_dicts()
    )

    lines = [
        "=" * 72,
        "LIVIA TEST: glucose_data_processing vs cgm_format",
        f"Input: {LIVIA_CSV}",
        "=" * 72,
        "",
        "PIPELINE SNAPSHOTS",
        gdp_snap.format_block(),
        "",
        parsed_snap.format_block(),
        "",
        processed_snap.format_block(),
        "",
        inference_snap.format_block(),
        "",
        "PARSED EVENT TYPES (cgm_format)",
        *(f"  {row['event_type']}: {row['len']:,}" for row in parsed_events),
        "",
        "GLUCOSE OVERLAP (GDP ML-ready vs cgm processed EGV rows, inner join on timestamp)",
        f"  overlapping timestamps: {overlap['overlap_rows']:,}",
        f"  exact matches (<=0.01 mg/dL): {overlap['exact_matches']:,}",
        f"  max abs diff: {overlap['max_abs_diff']}",
        f"  mean abs diff: {overlap['mean_abs_diff']}",
    ]
    if overlap["sample_diffs"]:
        lines.append("  sample diffs (first 5):")
        for row in overlap["sample_diffs"]:
            lines.append(
                f"    {row['ts']}: GDP={row['gdp_glucose']:.3f} "
                f"CGM={row['cgm_glucose']:.3f} diff={row['abs_diff']:.3f}"
            )

    lines.extend(
        [
            "",
            "KEY ARCHITECTURAL DIFFERENCES (expected today)",
            "  - GDP resamples to fixed 5-min grid (one row per interval); cgm sync is lossless",
            "  - GDP keeps all sequences >= min_sequence_len; cgm inference keeps latest sequence only",
            "  - GDP removes 24h after calibration gaps; cgm marks calibration quality flags",
            "  - GDP runs data cleaning (drop covariates in large glucose gaps); cgm has no equivalent",
            "  - Output schemas differ (GDP display names vs cgm unified / data-only columns)",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def livia_csv_path() -> Path:
    return _require_livia_csv()


@pytest.fixture(scope="module")
def gdp_ml_ready(livia_csv_path: Path, tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    work_dir = tmp_path_factory.mktemp("livia_gdp")
    return _run_gdp_ml_ready(livia_csv_path, work_dir)


@pytest.fixture(scope="module")
def cgm_parsed_df(livia_csv_path: Path) -> pl.DataFrame:
    return _run_cgm_parsed(livia_csv_path)


@pytest.fixture(scope="module")
def cgm_processed_df(livia_csv_path: Path) -> pl.DataFrame:
    return _run_cgm_processed(livia_csv_path)


@pytest.fixture(scope="module")
def cgm_inference_df(livia_csv_path: Path) -> pl.DataFrame:
    return _run_cgm_inference_pipeline(livia_csv_path)


@pytest.fixture(scope="module")
def comparison_report(
    gdp_ml_ready: pl.DataFrame,
    cgm_parsed_df: pl.DataFrame,
    cgm_processed_df: pl.DataFrame,
    cgm_inference_df: pl.DataFrame,
) -> str:
    return build_comparison_report(
        gdp_ml_ready,
        cgm_parsed_df,
        cgm_processed_df,
        cgm_inference_df,
    )


def test_livia_gdp_pipeline_produces_output(gdp_ml_ready: pl.DataFrame) -> None:
    assert gdp_ml_ready.height > 0
    assert "Glucose Value (mg/dL)" in gdp_ml_ready.columns
    assert gdp_ml_ready["sequence_id"].n_unique() >= 1


def test_livia_cgm_parsed_event_mix(cgm_parsed_df: pl.DataFrame) -> None:
    assert cgm_parsed_df.height > 20_000
    event_types = set(cgm_parsed_df["event_type"].unique().to_list())
    assert UnifiedEventType.GLUCOSE.value in event_types


def test_livia_cgm_processed_retains_event_rows(
    cgm_processed_df: pl.DataFrame,
    gdp_ml_ready: pl.DataFrame,
) -> None:
    """cgm_format keeps insulin/event rows; GDP fixed-frequency grid is glucose-centric."""
    assert cgm_processed_df.height > gdp_ml_ready.height
    non_glucose = cgm_processed_df.filter(
        pl.col("event_type") != UnifiedEventType.GLUCOSE.value
    ).height
    assert non_glucose > 0


def test_livia_comparison_report(
    comparison_report: str,
    gdp_ml_ready: pl.DataFrame,
    cgm_inference_df: pl.DataFrame,
) -> None:
    """Print full comparison report (use ``pytest -s``) and assert pipelines diverge as expected."""
    print("\n" + comparison_report)
    assert "LIVIA TEST" in comparison_report
    assert gdp_ml_ready.height > cgm_inference_df.height
    assert "Glucose Value (mg/dL)" in gdp_ml_ready.columns
    assert list(cgm_inference_df.columns) == [
        "datetime",
        "glucose",
        "carbs",
        "insulin_slow",
        "insulin_fast",
        "exercise",
    ]
