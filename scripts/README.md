# Scripts

Helper scripts and CLI tools for the cgm_format project.

## cgm_cli.py

**Comprehensive CLI tool for CGM data processing** - provides access to all parser, processor, and validation features.

### Installation

```bash
# Install with CLI dependencies
uv add --optional cli cgm-format

# Or install in development mode
uv sync --extra cli
```

### Usage Modes

The CLI can be used in three ways:

```bash
# 1. As an installed command (after pip/uv install)
cgm-cli <command> [options]

# 2. As a Python module
python -m cgm_format.cgm_cli <command> [options]

# 3. As a direct script
python scripts/cgm_cli.py <command> [options]
```

### Commands

#### Format Detection & Parsing

**detect** - Detect the format of a CGM data file
```bash
cgm-cli detect input.csv [--verbose]
```

**parse** - Parse a CGM data file to unified format
```bash
cgm-cli parse input.csv --output unified.csv [--stats] [--preview]
```

**validate** - Validate a CSV file against its schema
```bash
cgm-cli validate input.csv [--format unified|dexcom|libre] [--verbose]
```

**report** - Generate comprehensive validation report for directory
```bash
cgm-cli report data/input/ --output validation_report.txt \
    [--pattern "*.csv"] \
    [--frictionless/--no-frictionless] \
    [--suppress-known/--show-all]
```

Features:
- Batch format detection for all files
- Optional Frictionless schema validation
- Automatic suppression of known vendor quirks (Dexcom High/Low, variable-length rows)
- Detailed text report with format breakdown, validation results, and schema info
- Similar to `examples/example_schema_usage.py` functionality

#### Data Processing

**process** - Process unified format data (interpolate, synchronize)
```bash
cgm-cli process unified.csv --output processed.csv \
    [--interpolate/--no-interpolate] \
    [--sync/--no-sync] \
    [--interval 5] \
    [--max-gap 15]
```

**pipeline** - Run the complete processing pipeline
```bash
cgm-cli pipeline input.csv --output final.csv \
    [--interval 5] \
    [--max-gap 15] \
    [--min-duration 15] \
    [--max-duration 1440] \
    [--glucose-only] \
    [--drop-duplicates]
```

Pipeline stages:
1. Parse vendor format to unified
2. Detect and assign sequences
3. Interpolate gaps
4. Synchronize timestamps
5. Prepare for inference (quality checks)
6. Convert to data-only format

#### File Information

**info** - Show information about a CGM data file
```bash
cgm-cli info input.csv [--detailed]
```

#### Batch Processing

**batch** - Batch process multiple CGM data files
```bash
cgm-cli batch data/input/ --output processed/ \
    [--pattern "*.csv"] \
    [--command parse|process|pipeline] \
    [--continue/--stop]
```

### Examples

```bash
# Detect format of a file
cgm-cli detect data/input/patient_export.csv

# Parse Dexcom file to unified format
cgm-cli parse data/input/dexcom_export.csv -o data/unified.csv --stats

# Run full pipeline with glucose-only output
cgm-cli pipeline data/input/libre_export.csv -o output.csv --glucose-only

# Get detailed info about a file
cgm-cli info data/unified.csv --detailed

# Validate unified format file
cgm-cli validate data/unified.csv

# Generate comprehensive validation report (like example_schema_usage.py)
cgm-cli report data/input/ -o validation_report.txt --frictionless

# Batch process all CSV files in a directory
cgm-cli batch data/input/ --output data/processed/ --command pipeline

# Process with custom parameters
cgm-cli process data/unified.csv -o processed.csv \
    --interval 5 \
    --max-gap 20 \
    --interpolate \
    --sync
```

### Features

- **Rich output** - Colored terminal output with progress indicators
- **Comprehensive statistics** - Show glucose stats, event counts, quality flags
- **Processing warnings** - Detect and report data quality issues
- **Batch processing** - Process entire directories at once
- **Flexible configuration** - Control all processing parameters
- **Multiple output formats** - Full unified or data-only formats

---

## download_bigideas.py

Fetch the public BIG IDEAs Dexcom + food-log subset from PhysioNet. Empatica
ACC/BVP streams are never downloaded. This is a `dev`-extra setup chore, not a
package feature — people without a local extract run it and point
`CGM_FORMAT_BIGIDEAS_DIR` at the destination.

```bash
uv run python scripts/download_bigideas.py --dest data/input/bigideas
uv run python scripts/download_bigideas.py --dest data/input/bigideas --subjects 1,3
```

Source: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/

## download_d1namo.py

Fetch the two D1NAMO annotation archives from Zenodo (glucose / food / insulin /
meal photos). The Zephyr waveform zips are not downloaded.

```bash
uv run python scripts/download_d1namo.py --dest data/input/d1namo
```

Then point `CGM_FORMAT_D1NAMO_DIR` at the extract.

## regenerate_all_schemas.py

Automatically regenerates all JSON schema files from their Python schema definitions.

### Usage

```bash
# As executable (recommended - uses uv automatically)
./scripts/regenerate_all_schemas.py

# Or using uv explicitly
uv run python scripts/regenerate_all_schemas.py

# Or directly with Python (requires dependencies installed)
python scripts/regenerate_all_schemas.py
```

### What it does

1. Discovers all format modules in `src/cgm_format/formats/` (excluding `__init__.py` and `*_WIP.py`)
2. Dynamically imports each module
3. Calls `regenerate_schema_json()` function if it exists
4. Generates/updates corresponding `.json` schema files

### Output

The script regenerates:
- `dexcom.json` - Dexcom G6/G7 format schema
- `libre.json` - FreeStyle Libre 3 format schema
- `unified.json` - Unified CGM format schema

### When to run

Run this script after:
- Modifying enum values in format definitions
- Adding/removing columns in schemas
- Changing column descriptions or constraints
- Any other schema-related changes

This ensures the JSON schema files stay in sync with the Python schema definitions.

## scrub_synthetic_libre.py

Creates synthetic FreeStyle Libre CGM data from real data for CI testing and demos.

### Usage

```bash
uv run python scripts/scrub_synthetic_libre.py INPUT_FILE OUTPUT_FILE [--seed SEED]
```

### Arguments

- `INPUT_FILE` - Path to input FreeStyle Libre CSV file
- `OUTPUT_FILE` - Path to output synthetic CSV file
- `--seed` - Random seed for reproducibility (default: 42)

### Example

```bash
uv run python scripts/scrub_synthetic_libre.py \
    data/input/FreeStyle_Libre_3__11-12-2024.csv \
    data/input/FreeStyle_Libre_3_synthetic.csv
```

### Transformations Applied

1. **Serial Number** - Replaces with random UUID in same format
2. **Dates** - Changes to 1961-04-12 base date (Gagarin's space flight) while preserving relative timing
3. **Patient Name** - Replaces with "Gagarin"
4. **Patient Notes** - Removes all patient notes (Cyrillic text)
5. **Glucose Values** - Applies random baseline offset (10-20 mg/dL) + noise (±1) to:
   - Historic Glucose mg/dL
   - Scan Glucose mg/dL
   - Strip Glucose mg/dL
6. **Timestamps** - Shifts all timestamps by random minutes (multiple of 5)

### When to use

- Creating synthetic test data for CI/CD pipelines
- Generating demo data without exposing real patient information
- Testing format parsers with realistic but fake data

## scrub_synthetic_dexcom.py

Creates synthetic Dexcom CGM data from real Clarity export files for CI testing and demos.

### Usage

```bash
uv run python scripts/scrub_synthetic_dexcom.py INPUT_FILE OUTPUT_FILE [--seed SEED]
```

### Arguments

- `INPUT_FILE` - Path to input Dexcom Clarity CSV file
- `OUTPUT_FILE` - Path to output synthetic CSV file
- `--seed` - Random seed for reproducibility (default: 42)

### Example

```bash
uv run python scripts/scrub_synthetic_dexcom.py \
    data/input/Clarity_Export__Patient_2025-05-14_154517.csv \
    data/input/Clarity_Export_synthetic.csv
```

### Transformations Applied

1. **Transmitter ID** - Replaces with random 6-character alphanumeric ID
2. **Dates** - Changes to 1961-04-12 base date (Gagarin's space flight) while preserving relative timing
3. **Patient Name** - Replaces with "Gagarin"
4. **Glucose Values** - Applies random baseline offset (10-20 mg/dL) + noise (±1) to EGV readings
5. **Timestamps** - Shifts all timestamps by random minutes (multiple of 5)
6. **Transmitter Time** - Adjusts transmitter time accordingly (in seconds)

### When to use

- Creating synthetic test data for CI/CD pipelines
- Generating demo data without exposing real patient information
- Testing Dexcom format parsers with realistic but fake data

## triage-state.py, triage-archive.py, watch-inbox.sh

The consumer-inbox triage loop for [`docs/FEEDBACK.md`](../docs/FEEDBACK.md) and
[`docs/FEEDBACK_HISTORY.md`](../docs/FEEDBACK_HISTORY.md). The document is both the transcript and
the state: each item's verdict is derived by fingerprinting the reporter's own text, excluding any
reply, so re-running after writing a reply is a no-op. The runbook is
[`docs/CONSUMER_TRIAGE_LOOP.md`](../docs/CONSUMER_TRIAGE_LOOP.md) — read that before triaging, not
this section; the generalized pattern it adapts is
<https://gist.github.com/winternewt/54b94bda01812be937b892146d1bb254>.

These three are the one part of the repo that does **not** run under `uv`: stdlib-only Python 3.11+
and bash, importing nothing from the package, so the reason behind the always-`uv run` rule (the
workspace environment) does not apply. They must stay in the same directory — the archiver resolves
the ledger relative to its own path, and the watcher shells out to it the same way.

```sh
./scripts/triage-state.py                          # every item: new / revised / unmarked-reply / current
./scripts/triage-state.py --pending                # only what needs work
./scripts/triage-state.py --next                   # next unclaimed id, over BOTH documents
./scripts/triage-state.py --backfill               # OLD replies only — see the caveat below
./scripts/triage-state.py docs/FEEDBACK_HISTORY.md # the post-archive lint: everything should read `current`
./scripts/triage-archive.py S1 S2                  # move answered items, verifying the prose moved verbatim
FILE=docs/FEEDBACK.md ./scripts/watch-inbox.sh     # one stdout line when the inbox settles (150s cooldown)
```

`INBOX`, `HISTORY` and `PREFIX` override the paths and the id prefix. Unlike the upstream gist, the
defaults are derived from this script directory rather than from `$PWD`, so the tools work from
anywhere in the tree; that is the only local change to them.

Four things worth knowing before using them:

- **`--backfill` is for replies that predate the ledger, and only those.** Pointed at a reply just
  written, it hashes paragraphs two onward of that reply as though the reporter had written them, and
  stamps the marker inside the first paragraph — where the mistake reads `current` forever instead of
  announcing itself. Write the marker by hand: stamp `sha 000000000000`, run the ledger, and paste back
  the value its `revised` line prints. `docs/CONSUMER_TRIAGE_LOOP.md` §3 Step 3 has the recipe and §6
  has why.
- **The watcher only watches while the tree is on `BRANCH`, which defaults to `main`.** Off it — a
  feature branch, or a detached HEAD — it idles at `BRANCH_PAUSE` (900s) and says so once, because a
  pass permitted to commit should stay off somebody else's half-finished work. This pass does not commit
  (`docs/CONSUMER_TRIAGE_LOOP.md` §5), so the guard is inert here and carried for the day that changes.
  `BRANCH=` switches it off; outside a git work tree it never applies.
- **The `.py` files are Python. Never `bash triage-state.py`** — bash ignores the shebang, executes
  the module docstring as commands, and `import hashlib` reaches ImageMagick's `import`, which
  silently writes 0-byte files named after each import into the working directory.
- **Rehearse an archive on copies, not with `--dry-run`.** The dry run returns before the write, so
  it never reaches the before/after fingerprint comparison that is the thing worth rehearsing:
  `INBOX=/tmp/copy.md HISTORY=/tmp/copy_HISTORY.md ./scripts/triage-archive.py S1`.

The watcher is not armed by anything in the repo; arm it yourself when you want the inbox to page
you, and note that it never fires for a change that predates it, so run the ledger by hand for a
standing backlog.

## Notes

All scrubber scripts:
- Use Polars for efficient data processing
- Support reproducible output via `--seed` parameter
- Preserve CSV structure and data relationships
- Include full type hints
- Use typer for CLI interface

