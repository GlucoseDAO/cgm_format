# cgm_format ↔ glucose_data_processing Alignment Plan

Persistent reference for aligning **inference-time** data processing in `cgm_format` with **training-time** processing in `glucose_data_processing`, so ML model predictions stay consistent.

**Last updated:** 2026-06-17  
**Status:** Investigation complete; implementation not started  
**Related test:** `tests/test_livia_gdp_cgm_comparison.py`  
**Sample report:** `data/comparison/livia_gdp_cgm_comparison_report.txt` (local, gitignored)

---

## 1. Problem statement

A glucose prediction model was **trained on datasets processed by** [`glucose_data_processing`](https://github.com/GlucoseDAO/glucose_data_processing) (GDP). **Server-side inference** uses [`cgm_format`](https://github.com/GlucoseDAO/cgm_format) (this repo).

Both projects share the same origin but **diverged**. If inference preprocessing differs from training preprocessing, model inputs drift and predictions become unreliable—even when raw Dexcom CSV input is identical.

**Goal:** Make inference preprocessing produce **the same numeric inputs at each timestep** as training would for the same raw data, while preserving inference-specific behavior (warnings, last-sequence-only, no aggressive data deletion).

---

## 2. Repositories and roles

| Project | Path (local) | Role |
|---------|--------------|------|
| **cgm_format** | `D:\dev\cgm_format` | Library + `cgm-cli`; server-side parse + process + inference prep |
| **glucose_data_processing** | `D:\dev\glucose_data_processing` | Training pipeline; `glucose-process` CLI; modular `processing/` steps |

### glucose_data_processing processing parts

The live training orchestrator is `glucose_data_processing/glucose_ml_preprocessor.py`, it rely on:

- `processing/steps/gap_detection.py`
- `processing/steps/data_cleaning.py`
- `processing/steps/interpolation.py`
- `processing/steps/filtering.py`
- `processing/steps/fixed_frequency.py`
- `processing/steps/ml_prep.py`
- `formats/dexcom/` converters

**cgm_format live code:** `src/cgm_format/format_parser.py`, `src/cgm_format/format_processor.py`, `src/cgm_format/cgm_cli.py`.

---

## 3. Train vs inference philosophy (explicit product decision)

These pipelines **should not be identical**. They should be **as similar as possible where it affects model math**, and **different where product requirements differ**.

### Training (`glucose_data_processing`)

- Can **remove** corrupted or problematic data (calibration periods, short sequences, covariates in large gaps).
- Keeps **many sequences** that pass `min_sequence_len`.
- Optimizes for **clean, long ML datasets**.

### Inference (`cgm_format`)

- **Cannot** remove data arbitrarily—the server must **predict anyway** using the best available signal.
- Only **one recent sequence** is available in practice (`prepare_for_inference` keeps the latest).
- Problematic data should be **marked** (`quality` flags, `ProcessingWarning`) and surfaced to the user—not silently dropped before prediction.
- Preprocessing must still apply the **same grid math** (interpolation, resampling, covariate placement) that the model saw during training.

### Alignment rule of thumb

| Category | Align with training? |
|----------|---------------------|
| Timestep grid (5-min, one row per interval) | **Yes — critical** |
| Glucose linear interpolation onto grid | **Yes — critical** |
| Insulin/carbs shifted onto grid | **Yes — if model uses covariates** |
| Gap thresholds (5 min expected, 15 min max fill) | **Yes** |
| Glucose-only sequence splits | **Yes** |
| Rounding (`round_precision: 3`) | **Yes — before model input** |
| Remove short sequences | **No** — inference uses last sequence |
| Remove 24h after calibration gap | **No** — mark + warn |
| Remove calibration events | **No** — cgm_format doesn't parse them; warn if relevant |
| Drop covariates in large glucose gaps | **No** — warn; let resample bound by glucose range |
| Multi-sequence export | **No** |
| Output column names | **Adapter layer** (`to_ml_ready_df`) |

---

## 4. Architecture comparison

### 4.1 glucose_data_processing pipeline (training)

Order from `glucose_ml_preprocessor.py` → `_run_processing_pipeline()`:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `formats/` converters | Consolidate CSV; Dexcom High/Low → 401/39; optional calibration removal |
| 2 | `DataCleaner` | Remove covariate rows in large glucose gaps (>15 min) |
| 3 | `GapDetector` | Split on glucose gaps >15 min; **remove 24h** after gaps ≥165 min (2h45m) |
| 4 | `ValueInterpolator` | Fill small gaps (linear); insert missing timestamp rows |
| 5 | `SequenceFilter` | Drop sequences with `< min_sequence_len` (default **200** points) |
| 6 | `FixedFreqGenerator` | **Resample** to fixed 5-min grid; linear glucose; shift events |
| 7 | `SequenceFilter` | Optional `glucose_only` |
| 8 | `MLDataPreparer` | Cast, round, rename to display column names |

Default config: `glucose_data_processing/glucose_config.yaml`

```yaml
expected_interval_minutes: 5
small_gap_max_minutes: 15
min_sequence_len: 200
create_fixed_frequency: true
calibration_period_minutes: 165
remove_after_calibration_hours: 24
round_precision: 3
```

Dexcom-specific (`dexcom` section): `high_glucose_value: 401`, `low_glucose_value: 39`, `remove_calibration: true`.

### 4.2 cgm_format pipeline (inference today)

| Stage | Class / method | Description |
|-------|----------------|-------------|
| 1–3 | `FormatParser.parse_file` | Dexcom → unified schema; High/Low → 401/39 + `OUT_OF_RANGE` flag; no calibration rows |
| 4 | `FormatProcessor.detect_and_assign_sequences` | Split on **glucose-only** gaps >15 min |
| 4 | `FormatProcessor.interpolate_gaps` | Fill small glucose gaps; `snap_to_grid=True` default |
| 5 | `FormatProcessor.synchronize_timestamps` | **Lossless** — rounds timestamps to grid, keeps all event rows |
| 6 | `FormatProcessor.prepare_for_inference` | Latest sequence; truncate duration; mark duplicates/calibration; warnings |
| 7 | `FormatProcessor.to_data_only_df` | Strip service columns → `datetime, glucose, carbs, insulin_slow, insulin_fast, exercise` |

`cgm-cli pipeline` runs stages 1→6 in that order (`src/cgm_format/cgm_cli.py`).

Constants (aligned with GDP): `SMALL_GAP_MAX_MINUTES = 15`, `EXPECTED_INTERVAL_MINUTES = 5`, calibration gap 165 min / 24h marking — see `AGENTS.md`, `interface/cgm_interface.py`.

---

## 5. Dexcom parsing differences

| Behavior | GDP | cgm_format |
|----------|-----|------------|
| High/Low → 401/39 | Yes, at consolidation | Yes, at parse + `Quality.OUT_OF_RANGE` |
| Calibration events | Removed (`remove_calibration: true`) | Not emitted as rows (only EGV, insulin, carbs, exercise parsed) |
| Row structure | Flat stream + `Event Type` column | Unified event codes (`EGV_READ`, `INS_FAST`, …) |
| Insulin | Split to `fast_acting_insulin_u` / `long_acting_insulin_u` | `insulin_fast` / `insulin_slow` |
| Metadata | `original_datetime`, `quality`, `event_type` service columns | Same (idempotent processing design) |
| Timestamps | `%Y-%m-%dT%H:%M:%S`, `%Y-%m-%d %H:%M:%S` | Same |

Insulin subtype logic is equivalent: Fast-Acting → fast, Long-Acting → slow, default → fast.

---

## 6. Critical algorithmic gaps (root causes of mismatch)

### 6.1 Fixed-frequency resampling — **largest gap**

**GDP `FixedFreqGenerator`** (`processing/steps/fixed_frequency.py`):

- Bounds sequence by first/last **glucose** timestamp.
- Builds a **new** 5-minute grid (one row per interval).
- **Linear interpolation** for continuous fields (glucose).
- **Bucket/shift** for occasional fields (insulin, carbs).
- Output row count **changes** (resampling, not rounding).

**cgm_format `synchronize_timestamps`** (`format_processor.py`):

- Documented as **lossless**: keeps **all** source rows; only rounds `datetime` to grid.
- Does **not** produce one-row-per-grid-point ML tensor.
- Event-per-row long format remains (EGV rows + insulin rows + …).

**Impact:** Model trained on GDP sees a regular wide grid; inference today sees irregular multi-row timestamps. This alone explains most prediction drift.

### 6.2 Data cleaning (GDP only)

`DataCleaner.clean_remote_data()` removes insulin/carbs/exercise rows that fall in glucose gaps >15 min **before** gap detection. cgm_format has no equivalent; non-glucose events stay until sequence assignment.

### 6.3 Calibration handling

| | GDP | cgm_format |
|--|-----|------------|
| Gap ≥165 min | **Deletes** following 24 hours of data | **Marks** `SENSOR_CALIBRATION` on quality |
| Calibration events | Removed at parse | Not parsed |

Intentional for inference (predict anyway + warn).

### 6.4 Sequence selection and filtering

| | GDP | cgm_format inference |
|--|-----|----------------------|
| Sequences kept | All with ≥200 points | **Latest** valid sequence only |
| Truncation | None (full sequence) | `maximum_wanted_duration` (CLI default 1440 min) |
| Min length | Hard filter | `minimum_duration_minutes` (CLI default 15 min); may raise `ZeroValidInputError` |

Intentional for inference; still affects **which** window is fed to the model.

### 6.5 Output schema

**GDP ML-ready** (display names from `glucose_config.yaml`):

```
sequence_id, Timestamp (YYYY-MM-DDThh:mm:ss), Event Type,
Glucose Value (mg/dL), Fast-Acting Insulin Value (u),
Long-Acting Insulin Value (u), Carb Value (grams), ...
```

**cgm_format `to_data_only_df`:**

```
datetime, glucose, carbs, insulin_slow, insulin_fast, exercise
```

No `sequence_id`, different names, no rounding step, includes `exercise` (often empty for Dexcom).

### 6.6 Interpolation mechanics

Both use linear glucose interpolation for gaps >5 min and ≤15 min, glucose-only endpoints. Differences:

- GDP: two-phase (fill nulls at existing timestamps + insert rows); works on flat consolidated stream.
- cgm_format: interpolates between consecutive `EGV_READ` rows; grid-aligned placement when `snap_to_grid=True`.
- Until fixed-frequency matches, overlap metrics remain misleading.

---

## 7. What is already aligned

Documented in both READMEs and `AGENTS.md`:

- `expected_interval_minutes = 5`
- `small_gap_max_minutes = 15` (3 grid intervals; grid-multiple threshold)
- Operators: sequence split on `> 15`, interpolation fill on `≤ 15`
- Glucose-only gap detection for sequence splits (current code in both)
- High/Low: 401 / 39 mg/dL
- Calibration gap threshold: 165 minutes; 24-hour window (remove vs mark differs)
- Grid-aligned gap measurement in cgm_format when `snap_to_grid=True` (commutativity with sync)

---

## 8. Empirical comparison (Livia Dexcom test)

### 8.1 Fixture

- **File:** `glucose_data_processing/data/livia_test.csv`
- **Format:** Dexcom G6 Clarity export (~25,751 lines, ~2.4 MB)
- **Content:** EGV + insulin events (25,340 EGV, 291 fast, 86 slow in cgm_format parse)
- **Not copied** into cgm_format; test reads sibling repo path.

### 8.2 Comparison test

**File:** `tests/test_livia_gdp_cgm_comparison.py`

Runs four snapshots:

1. **GDP ML-ready** — `uv run glucose-process <temp_folder>` in GDP repo, `min_sequence_len=200`
2. **cgm parsed** — `FormatParser.parse_file`
3. **cgm processed** — detect sequences → interpolate → synchronize
4. **cgm inference** — processed + `prepare_for_inference` + `to_data_only_df` (matches `cgm-cli pipeline` defaults)

**Run:**

```powershell
cd D:\dev\cgm_format
uv sync --extra dev
uv run pytest tests/test_livia_gdp_cgm_comparison.py -s -v
```

**Save report to file:**

```powershell
uv run pytest tests/test_livia_gdp_cgm_comparison.py::test_livia_comparison_report -s -v `
  2>&1 | Out-File -Encoding utf8 data/comparison/livia_gdp_cgm_comparison_report.txt
```

Skips if `../glucose_data_processing/data/livia_test.csv` is missing.

### 8.3 Measured results (2026-06-17 run)

| Snapshot | Rows | Sequences | Time range | Glucose mean |
|----------|------|-----------|------------|--------------|
| GDP ML-ready | 23,327 | 13 | 2024-03-16 → 2024-06-13 | 142.5 |
| cgm parsed | 25,717 | 1* | 2024-03-16 → 2024-06-13 | 145.3 |
| cgm processed | 25,724 | 14 | 2024-03-16 → 2024-06-13 | 145.3 |
| cgm inference (CLI) | **295** | 1 | **last ~24h only** | 136.2 |

\*Parsed starts with `sequence_id=0` until `detect_and_assign_sequences`.

**Glucose overlap** (GDP ML-ready vs cgm processed EGV, inner join on timestamp string):

- Overlapping timestamps: **22,109**
- Exact matches (≤0.01 mg/dL): **3,038** (~14%)
- Mean absolute diff: **~0.24 mg/dL**
- Max absolute diff: **67.0 mg/dL**
- Typical diffs at start: GDP interpolated grid values (233.953) vs cgm raw/rounded (234.000)

Low exact-match rate is expected **before** fixed-frequency alignment: GDP values are grid-interpolated; cgm processed still uses raw EGV-centric rows.

---

## 9. Implementation plan

### Phase 0 — Baseline and acceptance criteria

**Goal:** Regression gate before/after each change.

1. Extend Livia test to compare GDP vs **new** ML-ready export once implemented.
2. Define v1 acceptance (tune with model team):
   - Same grid point count for overlapping window on last sequence
   - Glucose MAE ≤ ~0.1 mg/dL vs GDP on same window (after resample)
   - Insulin/carbs on-grid counts within tolerance (if IC model)
3. Optional: small golden CSV fixture from GDP for CI without sibling repo.

**Deliverable:** tests fail until Phase 1+2 land; then converge.

### Phase 1 — Fixed-frequency resampling (**highest priority**)

**Add** `FormatProcessor.resample_to_fixed_frequency()` (port logic from GDP `FixedFreqGenerator`):

- Bound by first/last glucose reading (+ half-interval margin for events).
- Build 5-minute grid (same minute-rounding / dominant seconds offset logic as GDP).
- Linear interpolate `glucose`.
- Shift `insulin_fast`, `insulin_slow`, `carbs` onto grid (occasional fields).
- Output: **one row per grid timestamp** (wide rows for ML).

**Pipeline change:**

```
parse → detect_and_assign_sequences → interpolate_gaps
→ resample_to_fixed_frequency   # NEW — replaces sync for ML path
→ prepare_for_inference → to_ml_ready_df
```

Keep `synchronize_timestamps()` for non-ML use cases or implement resample using shared grid helpers (`get_sequence_grid_start`, `calculate_grid_point` already exist).

**Reference implementation:** `glucose_data_processing/processing/steps/fixed_frequency.py`

### Phase 2 — ML-ready export adapter

**Add** `FormatProcessor.to_ml_ready_df()`:

- Map unified columns → GDP display names (config dict, mirror `field_to_display_name_map` in `glucose_config.yaml`).
- Apply `round_precision: 3` (configurable).
- Keep `to_data_only_df()` for non-ML API consumers.

**CLI:** e.g. `cgm-cli pipeline --ml-ready` or `--output-format ml`.

### Phase 3 — Numeric detail audit

1. **Gap detection** — diff cgm `detect_and_assign_sequences` vs GDP `GapDetector` on edge cases.
2. **Interpolation** — verify same small-gap fill after grid step; same alpha formula.
3. **High/Low** — keep `OUT_OF_RANGE` flag at inference (extra vs training; OK).
4. **Data cleaning** — do **not** delete at inference; optional `ProcessingWarning` for covariates in large gaps pre-resample.

### Phase 4 — `prepare_for_inference` refinement

1. Keep **last sequence only** and duration truncation.
2. Consider **soft** handling of short sequences: warn (`TOO_SHORT`) but still predict if any glucose exists (server requirement).
3. Run **after** resample so warnings apply to grid seen by model.
4. Warnings (never silent drop for prediction path): `CALIBRATION`, `OUT_OF_RANGE`, `IMPUTATION`, `TIME_DUPLICATES`, optionally new flags.

### Phase 5 — CLI and server integration

Update `cgm-cli pipeline`:

| Current stage | Target for ML |
|---------------|---------------|
| `synchronize_timestamps` | `resample_to_fixed_frequency` |
| `to_data_only_df` | `to_ml_ready_df` when `--ml-ready` |

Server should call library with same numeric params as training: `expected_interval_minutes=5`, `small_gap_max_minutes=15`, fixed-frequency enabled.

### Phase 6 — Testing and documentation

1. Tighten Livia comparison assertions on grid alignment.
2. Update `docs/PIPELINE.md` / README inference section when behavior lands.
3. Optional checkpoint export (`--save-intermediate`) mirroring GDP steps for debugging.

### Implementation order (minimum viable alignment)

```
Phase 0 (metrics) → Phase 1 (fixed-frequency) → Phase 2 (ML export)
→ Phase 3–6 as follow-ups
```

**Minimum for consistent predictions:** Phase 1 + Phase 2.

### Explicitly do NOT port to inference

- `min_sequence_len: 200` filtering
- Removing 24h after calibration gaps
- Removing calibration events (already absent from parse)
- `DataCleaner` row deletion in large gaps
- Multi-sequence training export

---

## 10. Open decisions (need product/model input)

1. **Exact model input schema** — which columns does the forward pass consume? Is `Event Type` / `sequence_id` included?
2. **Glucose-only vs insulin+carb (IC) model** — if IC, Phase 1 event shifting is mandatory; if glucose-only, resample can simplify.
3. **Inference window length** — confirm `maximum_wanted_duration` (480 vs 1440 min) vs model context length used in training.
4. **Short sequence policy** — predict with warning vs hard error on server?
5. **Livia GluMind configs** — GDP has `glucose_config_livia_glumind_ic.yaml` with custom display names (`Glucose (mg/dL)`, `Bolus Insulin (U)`, …); inference adapter may need per-model config files.

---

## 11. Key file reference

### cgm_format

| Path | Purpose |
|------|---------|
| `src/cgm_format/format_parser.py` | Dexcom parse (`_process_dexcom`) |
| `src/cgm_format/format_processor.py` | interpolate, sync, prepare_for_inference |
| `src/cgm_format/cgm_cli.py` | `pipeline` command orchestration |
| `src/cgm_format/interface/cgm_interface.py` | Constants, warnings, `CGMProcessor` ABC |
| `src/cgm_format/formats/unified.py` | Schema, `Quality`, `UnifiedEventType` |
| `tests/test_livia_gdp_cgm_comparison.py` | Cross-repo comparison test |
| `pyproject.toml` | `[tool.uv] native-tls = true` (corporate PyPI TLS fix) |

### glucose_data_processing

| Path | Purpose |
|------|---------|
| `glucose_ml_preprocessor.py` | Pipeline orchestration |
| `glucose_cli.py` | `glucose-process` CLI |
| `glucose_config.yaml` | Default training config |
| `processing/steps/fixed_frequency.py` | **Port target for Phase 1** |
| `processing/steps/gap_detection.py` | Gap + calibration removal |
| `processing/steps/data_cleaning.py` | Covariate removal in large gaps |
| `processing/steps/interpolation.py` | Small-gap fill |
| `processing/steps/ml_prep.py` | Rounding + display names |
| `formats/dexcom/` | Dexcom consolidation |
| `compare_checkpoints.py` | `glucose-compare` — diff two CSVs |
| `data/livia_test.csv` | Primary comparison fixture |

---

## 12. Environment notes

### uv sync on Windows (corporate proxy)

If `uv sync` fails with `invalid peer certificate: UnknownIssuer`, this repo sets:

```toml
[tool.uv]
native-tls = true
```

Uses OS certificate store. Alternative: `$env:UV_NATIVE_TLS = "1"`.

### Dev dependencies

```powershell
uv sync --extra dev   # polars + cgm-cli + pytest + dotenv
```

### Running GDP from cgm_format machine

GDP expects a **folder** of CSVs, not a single file:

```powershell
cd D:\dev\glucose_data_processing
uv run glucose-process test_data/dexcom_small -o output.csv
# or temp folder with copy of livia_test.csv (see comparison test)
```

---

## 13. Related documentation

| Doc | Content |
|-----|---------|
| `docs/PIPELINE.md` | cgm_format 6-stage pipeline (pre-alignment) |
| `docs/UNIFIED_FORMAT.md` | Unified schema spec |
| `docs/USAGE.md` | Inference workflows |
| `AGENTS.md` | Gap thresholds, commutativity, pitfalls |
| `glucose_data_processing/docs/processing_verification.md` | GDP verification workflow |
| `glucose_data_processing/docs/config.md` | GDP YAML config |

---

## 14. Summary one-liner

**Training and inference preprocessing should differ in what data is kept and what users are warned about, but must match in how the last sequence is converted to a fixed 5-minute grid with interpolated glucose and shifted covariates—the step cgm_format is missing today (`FixedFreqGenerator` equivalent).**
