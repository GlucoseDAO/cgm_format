"""Livia reference alignment metrics and reporting.

Compares cgm_format inference pipeline output against a pre-generated reference CSV
produced by glucose_data_processing (fixed-frequency training grid), mapped to
cgm_format data-only column names.

See ``docs/modification_plan.md`` Phase 0 acceptance criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import polars as pl

CGM_DATA_COLUMNS: tuple[str, ...] = (
    "datetime",
    "glucose",
    "carbs",
    "insulin_slow",
    "insulin_fast",
    "exercise",
)

DEFAULT_INFERENCE_MAX_DURATION_MINUTES = 1440
DEFAULT_INFERENCE_MIN_DURATION_MINUTES = 15
GLUCOSE_MATCH_TOLERANCE_MG_DL = 0.01


class IssueSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class AlignmentThresholds:
    """Target 'good enough' alignment after Phase 1+2 (see modification_plan.md)."""

    glucose_mae_max: float = 0.1
    glucose_max_abs_diff_max: float = 5.0
    row_count_ratio_min: float = 0.95
    row_count_ratio_max: float = 1.05
    glucose_exact_match_rate_min: float = 0.95
    duplicate_timestamp_rate_max: float = 0.01


@dataclass
class AlignmentIssue:
    severity: IssueSeverity
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentMetrics:
    reference_rows_window: int
    actual_rows_window: int
    actual_wide_rows_window: int
    row_count_ratio: float
    reference_unique_datetimes: int
    actual_unique_datetimes: int
    actual_duplicate_datetime_rows: int
    overlap_timestamps: int
    glucose_exact_matches: int
    glucose_exact_match_rate: float | None
    glucose_mae: float | None
    glucose_max_abs_diff: float | None
    insulin_fast_ref_non_null: int
    insulin_fast_actual_non_null: int
    insulin_slow_ref_non_null: int
    insulin_slow_actual_non_null: int
    carbs_ref_non_null: int
    carbs_actual_non_null: int
    reference_time_start: datetime | None
    reference_time_end: datetime | None
    actual_time_start: datetime | None
    actual_time_end: datetime | None
    latest_sequence_id: int | None
    issues: list[AlignmentIssue] = field(default_factory=list)

    @property
    def is_good_enough(self) -> bool:
        return not any(
            issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
            for issue in self.issues
        )


def _parse_datetime_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = df.schema[column]
    if dtype == pl.Datetime:
        return df
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col(column).str.strptime(pl.Datetime, strict=False).alias(column)
        )
    return df.with_columns(pl.col(column).cast(pl.Datetime, strict=False).alias(column))


def filter_reference_to_inference_window(
    reference_df: pl.DataFrame,
    *,
    max_duration_minutes: int = DEFAULT_INFERENCE_MAX_DURATION_MINUTES,
) -> tuple[pl.DataFrame, int | None]:
    """Keep latest sequence and last *max_duration_minutes* (matches cgm-cli defaults)."""
    ref = _parse_datetime_column(reference_df, "datetime")
    latest_seq_row = (
        ref.group_by("sequence_id")
        .agg(pl.col("datetime").max().alias("_max_ts"))
        .sort("_max_ts", descending=True)
        .head(1)
    )
    if latest_seq_row.height == 0:
        return ref.clear(), None

    latest_sequence_id = int(latest_seq_row["sequence_id"][0])
    ref_seq = ref.filter(pl.col("sequence_id") == latest_sequence_id)
    max_ts = ref_seq["datetime"].max()
    if max_ts is None:
        return ref_seq.clear(), latest_sequence_id

    cutoff = max_ts - timedelta(minutes=max_duration_minutes)
    return ref_seq.filter(pl.col("datetime") >= cutoff), latest_sequence_id


def collapse_to_wide_grid(actual_df: pl.DataFrame) -> pl.DataFrame:
    """One row per datetime (training reference is wide; cgm inference is event-long)."""
    df = _parse_datetime_column(actual_df, "datetime")
    numeric_cols = [c for c in CGM_DATA_COLUMNS if c != "datetime" and c in df.columns]
    agg_exprs = [pl.col(col).max().alias(col) for col in numeric_cols]
    return df.group_by("datetime").agg(agg_exprs).sort("datetime")


def _timestamp_key(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("datetime").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("_ts_key"))


def compare_reference_to_inference(
    reference_window: pl.DataFrame,
    actual_df: pl.DataFrame,
    thresholds: AlignmentThresholds | None = None,
) -> AlignmentMetrics:
    """Compare reference (GDP grid) vs cgm_format inference output on the same time window."""
    if thresholds is None:
        thresholds = AlignmentThresholds()

    ref = _parse_datetime_column(reference_window, "datetime")
    actual = _parse_datetime_column(actual_df, "datetime")
    actual_wide = collapse_to_wide_grid(actual)

    metrics = AlignmentMetrics(
        reference_rows_window=ref.height,
        actual_rows_window=actual.height,
        actual_wide_rows_window=actual_wide.height,
        row_count_ratio=(
            actual_wide.height / ref.height if ref.height > 0 else 0.0
        ),
        reference_unique_datetimes=ref["datetime"].n_unique(),
        actual_unique_datetimes=actual["datetime"].n_unique(),
        actual_duplicate_datetime_rows=actual.height - actual["datetime"].n_unique(),
        overlap_timestamps=0,
        glucose_exact_matches=0,
        glucose_exact_match_rate=None,
        glucose_mae=None,
        glucose_max_abs_diff=None,
        insulin_fast_ref_non_null=int(ref["insulin_fast"].is_not_null().sum())
        if "insulin_fast" in ref.columns
        else 0,
        insulin_fast_actual_non_null=int(actual_wide["insulin_fast"].is_not_null().sum())
        if "insulin_fast" in actual_wide.columns
        else 0,
        insulin_slow_ref_non_null=int(ref["insulin_slow"].is_not_null().sum())
        if "insulin_slow" in ref.columns
        else 0,
        insulin_slow_actual_non_null=int(actual_wide["insulin_slow"].is_not_null().sum())
        if "insulin_slow" in actual_wide.columns
        else 0,
        carbs_ref_non_null=int(ref["carbs"].is_not_null().sum())
        if "carbs" in ref.columns
        else 0,
        carbs_actual_non_null=int(actual_wide["carbs"].is_not_null().sum())
        if "carbs" in actual_wide.columns
        else 0,
        reference_time_start=ref["datetime"].min(),
        reference_time_end=ref["datetime"].max(),
        actual_time_start=actual["datetime"].min(),
        actual_time_end=actual["datetime"].max(),
        latest_sequence_id=int(ref["sequence_id"][0])
        if "sequence_id" in ref.columns and ref.height > 0
        else None,
    )

    issues: list[AlignmentIssue] = []

    if metrics.row_count_ratio < thresholds.row_count_ratio_min or (
        metrics.row_count_ratio > thresholds.row_count_ratio_max
    ):
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.CRITICAL,
                code="ROW_COUNT_MISMATCH",
                message=(
                    "Wide-grid row count in inference window does not match reference "
                    f"(ratio={metrics.row_count_ratio:.3f}, "
                    f"ref={metrics.reference_rows_window}, "
                    f"actual_wide={metrics.actual_wide_rows_window})"
                ),
                details={
                    "row_count_ratio": metrics.row_count_ratio,
                    "reference_rows": metrics.reference_rows_window,
                    "actual_wide_rows": metrics.actual_wide_rows_window,
                },
            )
        )

    dup_rate = (
        metrics.actual_duplicate_datetime_rows / metrics.actual_rows_window
        if metrics.actual_rows_window > 0
        else 0.0
    )
    if dup_rate > thresholds.duplicate_timestamp_rate_max:
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.HIGH,
                code="DUPLICATE_TIMESTAMPS",
                message=(
                    "cgm_format inference output has multiple rows per datetime "
                    f"(duplicate_rate={dup_rate:.3f}); reference is one row per grid point"
                ),
                details={
                    "duplicate_datetime_rows": metrics.actual_duplicate_datetime_rows,
                    "total_rows": metrics.actual_rows_window,
                },
            )
        )

    ref_keyed = _timestamp_key(ref.select(["datetime", "glucose"]))
    actual_keyed = _timestamp_key(actual_wide.select(["datetime", "glucose"]))
    joined = ref_keyed.join(actual_keyed, on="_ts_key", how="inner", suffix="_actual")

    metrics.overlap_timestamps = joined.height
    if joined.height == 0 and ref.height > 0 and actual_wide.height > 0:
        ref_minutes = ref["datetime"].dt.hour() * 60 + ref["datetime"].dt.minute()
        actual_minutes = actual_wide["datetime"].dt.hour() * 60 + actual_wide["datetime"].dt.minute()
        ref_mod = (ref_minutes % 5).mode().first()
        actual_mod = (actual_minutes % 5).mode().first()
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.CRITICAL,
                code="GRID_PHASE_OFFSET",
                message=(
                    "Reference and actual grids use different phase offsets "
                    f"(ref minute mod 5 = {ref_mod}, actual mod 5 = {actual_mod}); "
                    "timestamps never align exactly"
                ),
                details={
                    "reference_mod5": ref_mod,
                    "actual_mod5": actual_mod,
                },
            )
        )

    if joined.height == 0:
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.CRITICAL,
                code="NO_TIMESTAMP_OVERLAP",
                message="No overlapping timestamps between reference and cgm_format output",
            )
        )
        metrics.issues = issues
        return metrics

    diffs = (joined["glucose"] - joined["glucose_actual"]).abs()
    metrics.glucose_exact_matches = int((diffs <= GLUCOSE_MATCH_TOLERANCE_MG_DL).sum())
    metrics.glucose_exact_match_rate = metrics.glucose_exact_matches / joined.height
    metrics.glucose_mae = float(diffs.mean())
    metrics.glucose_max_abs_diff = float(diffs.max())

    if metrics.glucose_mae > thresholds.glucose_mae_max:
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.HIGH,
                code="GLUCOSE_MAE",
                message=(
                    f"Glucose MAE {metrics.glucose_mae:.4f} mg/dL exceeds "
                    f"threshold {thresholds.glucose_mae_max}"
                ),
                details={"glucose_mae": metrics.glucose_mae},
            )
        )

    if metrics.glucose_max_abs_diff > thresholds.glucose_max_abs_diff_max:
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.HIGH,
                code="GLUCOSE_MAX_DIFF",
                message=(
                    f"Max glucose abs diff {metrics.glucose_max_abs_diff:.3f} mg/dL exceeds "
                    f"threshold {thresholds.glucose_max_abs_diff_max}"
                ),
                details={"glucose_max_abs_diff": metrics.glucose_max_abs_diff},
            )
        )

    if (
        metrics.glucose_exact_match_rate is not None
        and metrics.glucose_exact_match_rate < thresholds.glucose_exact_match_rate_min
    ):
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.MEDIUM,
                code="GLUCOSE_EXACT_MATCH_RATE",
                message=(
                    f"Only {metrics.glucose_exact_match_rate:.1%} of overlapping glucose "
                    f"values match within {GLUCOSE_MATCH_TOLERANCE_MG_DL} mg/dL"
                ),
                details={
                    "exact_match_rate": metrics.glucose_exact_match_rate,
                    "overlap_rows": metrics.overlap_timestamps,
                },
            )
        )

    insulin_fast_gap = abs(
        metrics.insulin_fast_ref_non_null - metrics.insulin_fast_actual_non_null
    )
    if insulin_fast_gap > max(5, int(metrics.insulin_fast_ref_non_null * 0.2)):
        issues.append(
            AlignmentIssue(
                severity=IssueSeverity.MEDIUM,
                code="INSULIN_FAST_COVERAGE",
                message=(
                    "Fast insulin non-null count differs "
                    f"(ref={metrics.insulin_fast_ref_non_null}, "
                    f"actual={metrics.insulin_fast_actual_non_null})"
                ),
            )
        )

    metrics.issues = issues
    return metrics


def format_alignment_report(
    metrics: AlignmentMetrics,
    *,
    input_path: str,
    reference_path: str,
    thresholds: AlignmentThresholds | None = None,
) -> str:
    """Human-readable report for pytest -s and failure messages."""
    if thresholds is None:
        thresholds = AlignmentThresholds()

    lines = [
        "=" * 72,
        "LIVIA REFERENCE ALIGNMENT REPORT",
        f"Input:     {input_path}",
        f"Reference: {reference_path}",
        "=" * 72,
        "",
        "INFERENCE WINDOW (latest sequence, last 1440 min — matches cgm-cli defaults)",
        f"  Latest reference sequence_id: {metrics.latest_sequence_id}",
        f"  Reference rows (5-min grid):  {metrics.reference_rows_window:,}",
        f"  cgm_format rows (raw):        {metrics.actual_rows_window:,}",
        f"  cgm_format rows (wide/grid):  {metrics.actual_wide_rows_window:,}",
        f"  Row count ratio (wide/ref):   {metrics.row_count_ratio:.4f}",
        f"  Reference time range:         {metrics.reference_time_start} -> {metrics.reference_time_end}",
        f"  Actual time range:            {metrics.actual_time_start} -> {metrics.actual_time_end}",
        "",
        "GLUCOSE (inner join on timestamp in window)",
        f"  Overlap timestamps:           {metrics.overlap_timestamps:,}",
        f"  Exact matches (<=0.01):       {metrics.glucose_exact_matches:,}",
        f"  Exact match rate:             {metrics.glucose_exact_match_rate}",
        f"  MAE (mg/dL):                  {metrics.glucose_mae}",
        f"  Max abs diff (mg/dL):         {metrics.glucose_max_abs_diff}",
        "",
        "COVARIATES (non-null counts in window, wide actual)",
        f"  insulin_fast:  ref={metrics.insulin_fast_ref_non_null}, actual={metrics.insulin_fast_actual_non_null}",
        f"  insulin_slow:  ref={metrics.insulin_slow_ref_non_null}, actual={metrics.insulin_slow_actual_non_null}",
        f"  carbs:         ref={metrics.carbs_ref_non_null}, actual={metrics.carbs_actual_non_null}",
        "",
        "ACCEPTANCE THRESHOLDS (Phase 0 — docs/modification_plan.md)",
        f"  glucose MAE <= {thresholds.glucose_mae_max}",
        f"  glucose max abs diff <= {thresholds.glucose_max_abs_diff_max}",
        f"  row count ratio in [{thresholds.row_count_ratio_min}, {thresholds.row_count_ratio_max}]",
        f"  glucose exact match rate >= {thresholds.glucose_exact_match_rate_min:.0%}",
        "",
        f"GOOD ENOUGH: {metrics.is_good_enough}",
        "",
        "ISSUES",
    ]

    if not metrics.issues:
        lines.append("  (none)")
    else:
        for issue in metrics.issues:
            lines.append(f"  [{issue.severity.value}] {issue.code}: {issue.message}")

    lines.extend(
        [
            "",
            "NOTES",
            "  - Reference = glucose_data_processing fixed-frequency grid (training math).",
            "  - Actual = cgm_format cgm-cli pipeline + to_data_only_df (current inference).",
            "  - Perfect match is NOT expected until Phase 1 (resample) + Phase 2 (ML export).",
            "  - Inference may legitimately differ in sequence selection after grid alignment.",
            "=" * 72,
        ]
    )
    return "\n".join(lines)
