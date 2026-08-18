"""Regression gate: cgm_format inference vs GDP reference on Livia Dexcom data.

Fixtures (committed under data/comparison/):
  - livia_test.csv              — raw Dexcom input (copy from glucose_data_processing)
  - glucose_config_livia_reference.yaml — GDP config for reference generation
  - livia_reference.csv         — training-grid reference (regenerate via script)

Regenerate reference::

    uv run python scripts/generate_livia_reference.py --force

Run alignment test (expected to FAIL until pipeline alignment work lands)::

    uv run pytest tests/test_livia_reference_alignment.py -s -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cgm_format import FormatParser, FormatProcessor
from tests.comparison.livia_alignment import (
    DEFAULT_INFERENCE_MAX_DURATION_MINUTES,
    DEFAULT_INFERENCE_MIN_DURATION_MINUTES,
    AlignmentThresholds,
    compare_reference_to_inference,
    filter_reference_to_inference_window,
    format_alignment_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "data" / "comparison"
INPUT_CSV = COMPARISON_DIR / "livia_test.csv"
REFERENCE_CSV = COMPARISON_DIR / "livia_reference.csv"

THRESHOLDS = AlignmentThresholds()


def _require_comparison_fixtures() -> None:
    if not INPUT_CSV.is_file():
        pytest.skip(f"Missing input fixture: {INPUT_CSV}")
    if not REFERENCE_CSV.is_file():
        pytest.skip(
            f"Missing reference fixture: {REFERENCE_CSV}. "
            "Run: uv run python scripts/generate_livia_reference.py --force"
        )


def run_cgm_inference_pipeline(input_csv: Path) -> pl.DataFrame:
    """Mirror ``cgm-cli pipeline`` defaults (interval=5, max_gap=15, max_duration=1440)."""
    unified_df = FormatParser.parse_file(input_csv)
    unified_df = FormatProcessor.detect_and_assign_sequences(
        unified_df,
        expected_interval_minutes=5,
        large_gap_threshold_minutes=15,
    )
    unified_df = FormatProcessor.interpolate_gaps(
        unified_df,
        expected_interval_minutes=5,
        small_gap_max_minutes=15,
    )
    unified_df = FormatProcessor.synchronize_timestamps(
        unified_df,
        expected_interval_minutes=5,
    )
    inference_df, _warnings = FormatProcessor.prepare_for_inference(
        unified_df,
        minimum_duration_minutes=DEFAULT_INFERENCE_MIN_DURATION_MINUTES,
        maximum_wanted_duration=DEFAULT_INFERENCE_MAX_DURATION_MINUTES,
    )
    return FormatProcessor.to_data_only_df(
        inference_df,
        drop_service_columns=True,
        drop_duplicates=True,
        glucose_only=False,
    )


@pytest.fixture(scope="module")
def reference_df() -> pl.DataFrame:
    _require_comparison_fixtures()
    return pl.read_csv(REFERENCE_CSV, infer_schema_length=10_000)


@pytest.fixture(scope="module")
def actual_inference_df() -> pl.DataFrame:
    _require_comparison_fixtures()
    return run_cgm_inference_pipeline(INPUT_CSV)


@pytest.fixture(scope="module")
def reference_window(reference_df: pl.DataFrame) -> pl.DataFrame:
    window_df, _seq_id = filter_reference_to_inference_window(
        reference_df,
        max_duration_minutes=DEFAULT_INFERENCE_MAX_DURATION_MINUTES,
    )
    return window_df


@pytest.fixture(scope="module")
def alignment_metrics(
    reference_window: pl.DataFrame,
    actual_inference_df: pl.DataFrame,
):
    return compare_reference_to_inference(
        reference_window,
        actual_inference_df,
        thresholds=THRESHOLDS,
    )


@pytest.fixture(scope="module")
def alignment_report(alignment_metrics) -> str:
    return format_alignment_report(
        alignment_metrics,
        input_path=str(INPUT_CSV),
        reference_path=str(REFERENCE_CSV),
        thresholds=THRESHOLDS,
    )


def test_livia_reference_fixtures_exist() -> None:
    _require_comparison_fixtures()
    assert INPUT_CSV.stat().st_size > 0
    assert REFERENCE_CSV.stat().st_size > 0


def test_livia_reference_alignment_good_enough(
    alignment_metrics,
    alignment_report: str,
) -> None:
    """Guide for pipeline alignment — should fail until Phase 1+2 are implemented."""
    print("\n" + alignment_report)

    if alignment_metrics.is_good_enough:
        return

    failing = [
        issue
        for issue in alignment_metrics.issues
        if issue.severity.value in ("CRITICAL", "HIGH")
    ]
    summary = "; ".join(f"{issue.code} ({issue.severity.value})" for issue in failing)
    pytest.fail(
        f"Inference output is not aligned with training reference ({summary}).\n\n"
        f"{alignment_report}"
    )
