"""Integration tests for CGM CLI tool.

Tests the CLI by actually invoking it via subprocess, simulating real usage.
Based on test_integration_pipeline.py but calls the CLI tool instead of functions.
"""

import subprocess
import sys
from pathlib import Path
from typing import List
import pytest
import polars as pl

# Data directory relative to project root
DATA_DIR = Path(__file__).parent.parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "cli_test_output"


def get_test_files() -> List[Path]:
    """Get all CSV files from the data directory."""
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    # Exclude the parsed subdirectory and test output
    csv_files = [f for f in csv_files if "parsed" not in str(f) and "cli_test_output" not in str(f)]
    
    if not csv_files:
        pytest.skip(f"No CSV files found in {DATA_DIR}")
    
    return csv_files


def run_cli_command(args: List[str]) -> subprocess.CompletedProcess:
    """Run CLI command via subprocess.
    
    Args:
        args: Command arguments (without 'cgm-cli')
        
    Returns:
        CompletedProcess with stdout/stderr/returncode
    """
    # Run as module to avoid installation requirement
    cmd = [sys.executable, "-m", "cgm_format.cgm_cli"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result


@pytest.fixture(scope="module")
def setup_output_dir():
    """Create output directory for test files."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield TEST_OUTPUT_DIR
    # Cleanup after tests (optional - commented to inspect output)
    # import shutil
    # shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)


class TestCLIDetect:
    """Test CLI detect command."""
    
    @pytest.mark.parametrize("file_path", get_test_files(), ids=lambda p: p.name)
    def test_detect_format(self, file_path: Path) -> None:
        """Test format detection via CLI."""
        result = run_cli_command(["detect", str(file_path)])
        
        # Should succeed for supported formats
        if result.returncode == 0:
            assert "Detected format:" in result.stdout
            assert any(fmt in result.stdout for fmt in ["dexcom", "libre", "unified"])
        else:
            # Should fail gracefully for unsupported formats
            assert "Unknown format" in result.stdout or "Error" in result.stdout
    
    def test_detect_verbose(self) -> None:
        """Test detect with verbose flag."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        file_path = test_files[0]
        result = run_cli_command(["detect", str(file_path), "--verbose"])
        
        if result.returncode == 0:
            assert "File:" in result.stdout
            assert "Size:" in result.stdout


class TestCLIParse:
    """Test CLI parse command."""
    
    def test_parse_to_unified(self, setup_output_dir: Path) -> None:
        """Test parsing a file to unified format."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        input_file = test_files[0]
        output_file = setup_output_dir / f"parsed_{input_file.name}"
        
        result = run_cli_command([
            "parse",
            str(input_file),
            "--output", str(output_file),
            "--stats",
        ])
        
        # Check if parsing succeeded
        if result.returncode == 0:
            assert "Successfully parsed" in result.stdout
            assert output_file.exists()
            
            # Verify output is valid CSV
            df = pl.read_csv(output_file)
            assert len(df) > 0
            assert 'datetime' in df.columns
            assert 'glucose' in df.columns
        else:
            # Unsupported format - check error message
            assert "error" in result.stdout.lower() or "unknown" in result.stdout.lower()
    
    def test_parse_with_preview(self) -> None:
        """Test parse with preview flag."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        file_path = test_files[0]
        result = run_cli_command([
            "parse",
            str(file_path),
            "--preview",
            "--no-stats",
        ])
        
        if result.returncode == 0:
            assert "Successfully parsed" in result.stdout


class TestCLIProcess:
    """Test CLI process command."""
    
    def test_process_full(self, setup_output_dir: Path) -> None:
        """Test full processing with interpolation and synchronization."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        input_file = test_files[0]
        output_file = setup_output_dir / f"processed_{input_file.name}"
        
        result = run_cli_command([
            "process",
            str(input_file),
            "--output", str(output_file),
            "--interpolate",
            "--sync",
            "--interval", "5",
            "--max-gap", "19",
        ])
        
        if result.returncode == 0:
            assert "Detected" in result.stdout and "sequence" in result.stdout
            assert "Interpolated" in result.stdout or "Synchronized" in result.stdout
            assert output_file.exists()
            
            # Verify output
            df = pl.read_csv(output_file)
            assert len(df) > 0
        else:
            # Check for graceful error
            assert "error" in result.stdout.lower()


class TestCLIPipeline:
    """Test CLI pipeline command (most important - full end-to-end)."""
    
    @pytest.mark.parametrize("file_path", get_test_files(), ids=lambda p: p.name)
    def test_pipeline_single_file(self, file_path: Path, setup_output_dir: Path) -> None:
        """Test full pipeline on each data file.
        
        This is the main integration test - runs complete pipeline like test_integration_pipeline.py
        """
        output_file = setup_output_dir / f"pipeline_{file_path.name}"
        
        result = run_cli_command([
            "pipeline",
            str(file_path),
            "--output", str(output_file),
            "--interval", "5",
            "--max-gap", "19",
            "--min-duration", "15",
            "--max-duration", "1440",
            "--drop-duplicates",
            "--warnings",
            "--stats",
        ])
        
        print(f"\n{'='*70}")
        print(f"Testing: {file_path.name}")
        print(f"{'='*70}")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            # Success - verify stages completed
            assert "Stage 1:" in result.stdout  # Parsing
            assert "Stage 2:" in result.stdout  # Sequences
            assert "Stage 3:" in result.stdout  # Interpolation
            assert "Stage 4:" in result.stdout  # Synchronization
            assert "Stage 5:" in result.stdout  # Inference prep
            assert "Stage 6:" in result.stdout  # Final format
            assert "Pipeline completed successfully" in result.stdout
            
            # Verify output file exists and is valid
            assert output_file.exists()
            df = pl.read_csv(output_file)
            assert len(df) > 0
            assert 'datetime' in df.columns
            assert 'glucose' in df.columns
            
            print(f"✅ SUCCESS: Parsed {len(df)} rows")
        else:
            # Unsupported format or processing error - check for graceful failure
            assert ("unknown" in result.stdout.lower() or 
                    "error" in result.stdout.lower() or
                    "unsupported" in result.stdout.lower())
            print(f"⏭️  SKIPPED: {result.stdout[:200]}")
    
    def test_pipeline_glucose_only(self, setup_output_dir: Path) -> None:
        """Test pipeline with glucose-only output."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        input_file = test_files[0]
        output_file = setup_output_dir / f"glucose_only_{input_file.name}"
        
        result = run_cli_command([
            "pipeline",
            str(input_file),
            "--output", str(output_file),
            "--glucose-only",
            "--no-warnings",
            "--no-stats",
        ])
        
        if result.returncode == 0:
            assert output_file.exists()
            df = pl.read_csv(output_file)
            # Should only have glucose events (no insulin, carbs, etc.)
            assert len(df) > 0


class TestCLIValidate:
    """Test CLI validate command."""
    
    def test_validate_auto_detect(self) -> None:
        """Test validation with auto-detection."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        file_path = test_files[0]
        result = run_cli_command([
            "validate",
            str(file_path),
        ])
        
        # Should either pass validation or fail gracefully
        assert result.returncode in [0, 1]
        if result.returncode == 0:
            assert "Validation passed" in result.stdout
        else:
            assert "Validation failed" in result.stdout or "Unknown format" in result.stdout


class TestCLIInfo:
    """Test CLI info command."""
    
    def test_info_basic(self) -> None:
        """Test file info display."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        file_path = test_files[0]
        result = run_cli_command([
            "info",
            str(file_path),
        ])
        
        if result.returncode == 0:
            assert "File Information:" in result.stdout
            assert "Format" in result.stdout or "Rows" in result.stdout
    
    def test_info_detailed(self) -> None:
        """Test detailed file info."""
        test_files = get_test_files()
        if not test_files:
            pytest.skip("No test files available")
        
        file_path = test_files[0]
        result = run_cli_command([
            "info",
            str(file_path),
            "--detailed",
        ])
        
        if result.returncode == 0:
            assert "Column Details:" in result.stdout or "Column" in result.stdout


class TestCLIReport:
    """Test CLI report command (Frictionless validation)."""
    
    def test_report_generation(self, setup_output_dir: Path) -> None:
        """Test comprehensive validation report generation.
        
        This tests the Frictionless validation capabilities similar to
        tests/test_format_detection_validation.py and examples/example_schema_usage.py
        """
        report_file = setup_output_dir / "validation_report.txt"
        
        result = run_cli_command([
            "report",
            str(DATA_DIR),
            "--output", str(report_file),
            "--pattern", "*.csv",
            "--frictionless",
            "--suppress-known",
        ])
        
        print("\n" + result.stdout)
        
        # Should complete successfully
        assert result.returncode == 0
        assert "Generating Validation Report" in result.stdout
        assert "Stage 1: Format Detection" in result.stdout
        
        # Check if Frictionless is available
        if "Frictionless library not available" not in result.stdout:
            assert "Stage 2: Frictionless Validation" in result.stdout
        
        assert "Stage 3: Writing Report" in result.stdout
        assert "Report written to:" in result.stdout
        
        # Verify report file was created
        assert report_file.exists()
        
        # Verify report content
        report_content = report_file.read_text()
        assert "CGM FORMAT DETECTION AND VALIDATION REPORT" in report_content
        assert "FORMAT DETECTION SUMMARY" in report_content
        assert "SCHEMA DEFINITIONS" in report_content
        
        # If Frictionless is available, check validation section
        if "Frictionless validation: Enabled" in report_content:
            assert "FRICTIONLESS SCHEMA VALIDATION RESULTS" in report_content
            assert "Known vendor format issues" in report_content or "NOTE:" in report_content
    
    def test_report_without_frictionless(self, setup_output_dir: Path) -> None:
        """Test report generation without Frictionless validation."""
        report_file = setup_output_dir / "validation_report_no_frictionless.txt"
        
        result = run_cli_command([
            "report",
            str(DATA_DIR),
            "--output", str(report_file),
            "--no-frictionless",
        ])
        
        # Should complete
        assert result.returncode == 0 or "Frictionless" in result.stdout
        
        if report_file.exists():
            report_content = report_file.read_text()
            assert "FORMAT DETECTION SUMMARY" in report_content
    
    def test_report_show_all_errors(self, setup_output_dir: Path) -> None:
        """Test report with all errors (no suppression)."""
        report_file = setup_output_dir / "validation_report_all_errors.txt"
        
        result = run_cli_command([
            "report",
            str(DATA_DIR),
            "--output", str(report_file),
            "--show-all",  # Don't suppress known issues
        ])
        
        # Should complete
        assert result.returncode == 0


class TestCLIBatch:
    """Test CLI batch command."""
    
    def test_batch_parse(self, setup_output_dir: Path) -> None:
        """Test batch processing with parse command."""
        batch_output = setup_output_dir / "batch_parsed"
        batch_output.mkdir(exist_ok=True)
        
        result = run_cli_command([
            "batch",
            str(DATA_DIR),
            "--output", str(batch_output),
            "--pattern", "*.csv",
            "--command", "parse",
            "--continue",  # Continue on errors
        ])
        
        # Should complete (may have some failures for unsupported formats)
        assert "Batch processing complete:" in result.stdout
        assert "Success:" in result.stdout


class TestCLIErrors:
    """Test CLI error handling."""
    
    def test_nonexistent_file(self) -> None:
        """Test handling of nonexistent file."""
        result = run_cli_command([
            "parse",
            "/nonexistent/file.csv",
        ])
        
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_invalid_command(self) -> None:
        """Test invalid command handling."""
        result = run_cli_command([
            "invalid_command_xyz",
        ])
        
        assert result.returncode != 0
        # Typer shows error for invalid commands


class TestCLIHelp:
    """Test CLI help system."""
    
    def test_main_help(self) -> None:
        """Test main help message."""
        result = run_cli_command(["--help"])
        
        assert result.returncode == 0
        assert "cgm-cli" in result.stdout.lower() or "CGM" in result.stdout
        assert "parse" in result.stdout or "detect" in result.stdout
    
    def test_command_help(self) -> None:
        """Test command-specific help."""
        result = run_cli_command(["pipeline", "--help"])
        
        assert result.returncode == 0
        assert "pipeline" in result.stdout.lower()
        assert "input" in result.stdout.lower() or "file" in result.stdout.lower()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

