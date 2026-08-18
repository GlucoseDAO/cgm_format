# cgm_format ↔ glucose_data_processing Alignment Plan

Persistent reference for aligning **inference-time** data processing in `cgm_format` with **training-time** processing in `glucose_data_processing`, so ML model predictions stay consistent.

**Last updated:** 2026-08-18 (rung 3 shipped in 0.12.0)  
**Status:** Rungs 1–3 shipped; rung 4 open; rung 5 blocked (§0.8); rungs 6–7 contested  
**Related test:** `tests/test_livia_gdp_cgm_comparison.py`  
**Sample report:** `data/comparison/livia_gdp_cgm_comparison_report.txt` (local, gitignored)

> **Read §0 first.** Sections 4–9 were written on 2026-06-17 against cgm_format 0.8.2 and a
> *refactored* glucose_data_processing tree. Both premises have since changed. §0 records what
> was re-measured on 2026-08-17 and supersedes the stale parts inline.

---

## 0. Re-validation, 2026-08-17 (post-rebase onto 0.11.0)

Measured by running `tests/test_livia_reference_alignment.py` and the pipeline directly against
the committed fixtures, on branch `logic-synchronization` rebased onto `main` (0.11.0).

### 0.1 The headline blocker in §8.2 was never real against the committed fixture

The 2026-06-17 baseline reported **zero timestamp overlap** — GDP grid on a `:02/:07/:12` phase
vs cgm sync on `:01/:06/:11` (`GRID_PHASE_OFFSET`). Measured against the fixture actually in git,
there is no offset:

| Metric | §8.2 note | measured 2026-08-17 |
|--------|-----------|---------------------|
| Overlapping timestamps (289-point window) | **0** | **289 / 289** |
| Time range agreement | mismatched phase | identical (`2024-06-12 07:16` → `2024-06-13 07:16`) |
| Row-count ratio | 1.0 | 1.0 |

The rebase did not cause this. Running the same gate at the **pre-rebase** tip (322953d) yields
byte-identical metrics — overlap 289, MAE 0.7867, max 3.3, exact-match 8.3%. Two conclusions:

1. The §8.2 "baseline failure" was recorded against a *preliminary* reference CSV that was
   regenerated before `livia_reference.csv` was committed in 322953d. The note was stale the
   moment the fixture landed in the same commit.
2. **The 0.11.0 schema overhaul is numerically neutral for the Dexcom inference path.** Parse
   output is unchanged (25,717 rows / 25,340 EGV — exactly the §8.4 figures), and no commit
   between 7d072cd and `origin/main` touched `get_sequence_grid_start` or the sync grid logic.

**Phase 1 does not need to solve grid phase.** What remains is value-level, not timestamp-level.

### 0.2 The GDP port target named in §4.1 / §11 does not exist

`GlucoseDAO/glucose_data_processing` @ `origin/main` (4a1aa3f) is a **monolith**, not the modular
`processing/steps/` tree this document describes. The real port target is:

| Doc says | Actually is |
|----------|-------------|
| `processing/steps/fixed_frequency.py` → `FixedFreqGenerator` | `glucose_ml_preprocessor.py:675` → `GlucoseMLPreprocessor.create_fixed_frequency_data()` |
| `processing/steps/gap_detection.py` | `glucose_ml_preprocessor.py:268` `detect_gaps_and_sequences()` |
| `processing/steps/interpolation.py` | `glucose_ml_preprocessor.py:389` `interpolate_missing_values()` |
| `processing/steps/ml_prep.py` | `glucose_ml_preprocessor.py:894` `prepare_ml_data()` |

The modular layout exists only on the Windows box (`D:\dev\glucose_data_processing`) and is not
on origin. **Confirm which tree trained the deployed model before porting anything.**

Partially de-risking that question: the committed `livia_reference.csv` reproduces the
**monolith's** two-nearest-point formula exactly (worked example in §0.4 lands on 131.3 to the
digit). So either the reference was generated from the monolith, or the two trees share
interpolation math. Either way, porting §0.4 against the monolith is unlikely to be wasted work.

### 0.3 Covariates are not lost by the pipeline — they are lost by the export flag

§8.2's report line "insulin_fast: ref=5, actual=2" reads like a pipeline gap. It is not. Traced
stage by stage, all 6 insulin events in the window survive to `prepare_for_inference`, **at the
same grid timestamps GDP assigns them**:

```
parsed → sequences → interpolate → synchronize → prepare_for_inference : 5 fast + 1 slow  ✔
to_data_only_df(drop_duplicates=False) : 295 rows, 5 fast + 1 slow, 289 glucose  ✔
to_data_only_df(drop_duplicates=True)  : 289 rows, 2 fast + 0 slow, 287 glucose  ✘
```

`to_data_only_df` implements de-duplication as `unique(subset=['datetime'], keep='first')`
(`format_processor.py:1116`). When an insulin row and a glucose row land on the same grid
timestamp, **one of them is discarded outright** — and which one depends on row order, so it
destroys glucose readings too (289 → 287).

This is live in production: `cgm_cli.py:274` defaults `--drop-duplicates/--keep-duplicates` to
**True**. The library default is `False`, so the CLI is the lossy surface.

The correct operation is **coalesce, not drop**: collapse rows sharing a grid timestamp into one
wide row, merging their non-null data columns. That is simultaneously the bug fix *and* the
one-row-per-grid-point shape Phase 1/2 want.

### 0.4 The residual glucose delta is a re-timing choice, and it is small

MAE **0.78 mg/dL**, max **3.3 mg/dL**, exact-match rate 8.3% across all 289 points. Mechanism,
worked through at the worst point (grid `14:11:00`):

```
raw EGV:  14:06:45 → 150      14:11:45 → 128      14:16:46 → 110
cgm_format : snaps 14:11:45 onto grid 14:11, keeps the measured value   → 128.0
GDP        : linearly interpolates back 45 s to the exact grid instant  → 131.3
             128 + (45/300) × (150 − 128) = 131.3   ✔ matches reference exactly
```

So GDP re-times each value to the exact grid instant; cgm_format keeps the measured value and
relabels its timestamp. GDP's interpolation is arithmetically sound here (the two nearest
readings do bracket the grid point in steady state). Neither is "wrong" — but they differ
**systematically**, always leaning toward the previous reading, and the model saw GDP's version.

Scale check: 0.78 mg/dL is far below Dexcom sensor error (MARD ≈ 9%, i.e. ~10–15 mg/dL at these
levels). It is a systematic bias rather than noise, so it may still matter to a model reading
rate-of-change — but it is not obviously worth bug-for-bug compatibility on its own. **Needs the
model team's call (see §10).**

### 0.5 §10 answered from the model repo (`GlucoseDAO/glucose-forecasting`)

Cloned 2026-08-17. This resolves open decisions §10.1–§10.3 and reframes §10.2 entirely.

**There is no single model.** Two families ship, with different covariates:

| Model | Covariates | Context (`input_steps`) | Horizon |
|-------|-----------|------------------------|---------|
| **GluMind** | glucose + **heart rate + step count** | 80 → **400 min** | 12 → 60 min |
| **SugarOne** | glucose + **basal rate, bolus insulin, carbs** | 128 → **640 min** | 12 → 60 min |

Exact input schemas, from the shipped demo CSVs in `test_data/`:

```
GluMind:   sequence_id, Timestamp (YYYY-MM-DDThh:mm:ss), Event Type, User ID,
           Glucose Value (mg/dL), Heart Rate, Step Count,
           Recommended Split, Study Group, Glucose Observed, Steps Observed, HR Observed
SugarOne:  sequence_id, Timestamp, Event Type, User ID,
           Glucose (mg/dL), Basal Rate (U/h), Bolus Insulin (U), Carbohydrates (g),
           Recommended Split, Study Group
```

So **yes**, `sequence_id` and `Event Type` are part of the frame (§10.1 answered).

**The committed reference is a faithful proxy for what the model actually consumed.** Joining
`test_data/livia_glumind_ready.csv` (139,613 rows, 86 sequences, 2024-03-16 → 2025-09-12) against
`data/comparison/livia_reference.csv` on timestamp: **22,094 of 22,097 overlapping points match
within 0.01 mg/dL** (MAE 1.6e-05, max 0.26). This retires most of §0.2's worry — whichever GDP
tree produced the model-ready CSV, it agrees numerically with our reference. Rung 4 is aimed at
the right target.

**Two mapping problems cgm_format cannot currently solve:**

1. **GluMind's covariates do not exist in the unified schema.** There is no heart-rate or
   step-count column anywhere in cgm_format. Not a gap to close by porting — a schema question.
   Mitigating: Livia's own GluMind file carries HR/steps *empty* with `HR Observed = 0.0`,
   `Steps Observed = 0.0`, and `evaluate_model.py` fills missing covariates with `0.0` (with an
   explicit `--zero-cov` flag for glucose-only runs). So GluMind is runnable glucose-only today.
2. **`insulin_slow` → `Basal Rate (U/h)` is a semantic mismatch, not a rename.** SugarOne's basal
   is a *continuous pump rate in U/h*; cgm_format's `insulin_slow` is a *discrete long-acting
   injection in U* (Dexcom "Long-Acting"). `insulin_fast` → `Bolus Insulin (U)` and `carbs` →
   `Carbohydrates (g)` map cleanly; basal does not. Needs a product decision before rung 2 can
   emit a SugarOne-shaped frame.

**Window sizing (§10.3 answered):** the model floor is 400 min (GluMind) / 640 min (SugarOne)
plus horizon — the CLI's 1440-min `maximum_wanted_duration` is generous, not tight. The number
that actually matters for the server is the *minimum*: below ~80/128 grid points the model cannot
run at all, which sharpens §10.4's short-sequence policy into a hard, per-model floor.

### 0.6 Revised difficulty ladder

Ordered least-difficult / least-contradictory → genuinely contradictory. Rungs 1–2 are pure wins
both repos already agree on; rung 5 is where the projects' philosophies actually collide.

| # | Work | Difficulty | Contested? |
|---|------|-----------|-----------|
| 1 | Coalesce same-timestamp rows in `to_data_only_df` instead of dropping (§0.3) | **Low** | ✅ **Done** — `main` @ 5f8c817 |
| 2 | `to_ml_ready_df()` adapter: SugarOne display names + `round_precision: 3`, `sequence_id`/`Event Type` retained (§0.5) | **Low** | ✅ **Done** — `main` @ 28dd645 |
| 3 | Grid re-timing: interpolate glucose to the exact grid instant, GDP-style (§0.4) | **Medium** | ✅ **Done** — 0.12.0, see §0.9 |
| 4 | Per-model minimum-length floor (80 / 128 grid points) in `prepare_for_inference` (§0.5) | **Medium** | Mildly — hard floor vs "predict anyway" |
| 5 | `insulin_slow` → `Basal Rate (U/h)`: reconcile discrete dose (U) vs pump rate (U/h) (§0.8) | **Blocked** | **Yes — semantic, needs product input** |
| 6 | Heart rate / step count for GluMind — absent from the unified schema entirely (§0.5) | **High** | **Yes — schema expansion, or accept zero-fill** |
| 7 | §3's "No" column: short-sequence filter, 24 h post-calibration deletion, `DataCleaner` row removal | **N/A** | **Yes — intentional, do not port** |

Rungs 1+2 move covariate counts to parity (5 fast / 1 slow, matching the reference) and produce
the wide grid the models expect. Rung 3 is the whole remaining distance to MAE ≤ 0.1. Rungs 5–6
are where cgm_format and the model repo genuinely disagree about what the data *is*, and cannot
be settled by porting code.

### 0.7 Rungs 1–2 as landed (2026-08-17, on `main`)

**Rung 1 — `5f8c817`.** `to_data_only_df(drop_duplicates=True)` now collapses rows sharing a
timestamp (group by `datetime`, first non-null per column) instead of `unique(keep='first')`.
Covariates in the Livia window went 2 fast / 0 slow → **5 fast / 1 slow**, matching the reference
exactly; the 2 glucose readings it was also eating came back (287 → 289). Sorting by `datetime`
additionally fixed unordered output the old `unique()` returned. Regression test added — it fails
on the old code with `assert [130.0, 120.0] == [120.0, 130.0]`, which is that ordering bug.

**Rung 2 — `28dd645`.** `to_ml_ready_df()` plus `cgm-cli pipeline --ml-ready`, targeting
**SugarOne** (chosen 2026-08-17). CLI output is byte-comparable to the shipped
`test_data/livia_sugar_one_ready.csv` header and row format.

Decisions worth remembering:

- **`Event Type` is not a feature channel.** `evaluate_glumind.py:152` states Study Group,
  Recommended Split and Event Type are all optional; `_canonical_feature_cols` derives purely
  from glucose + covariates. It still matters that imputed rows carry `Interpolated`, because
  the NeuralForecast baselines have a `--drop-interpolated` filter keyed on that label.
- **`Basal Rate (U/h)` ships empty by default** — see rung 5. `insulin_slow` is a discrete dose
  in U, basal is a rate in U/h; `evaluate_model.py` zero-fills absent covariates, so an empty
  column costs a covariate rather than injecting a wrong one. `basal_rate_from_insulin_slow=True`
  opts in.
- **Still open:** `Recommended Split` / `Study Group` are emitted empty (training-only), and the
  per-model context floors from §0.5 (80 / 128 grid points) are **not** yet enforced anywhere.

### 0.8 Rung 5 (`Basal Rate (U/h)`) — what is actually missing

Investigated 2026-08-17 against `GlucoseDAO/glucose-forecasting`. This is not a naming decision;
feeding the column wrong is worse than leaving it empty, and the evidence says so quantitatively.

**The two data sources disagree about what the column means.**

| Source | What `Basal Rate (U/h)` holds | Observed values |
|--------|-------------------------------|-----------------|
| **Training** — `loop_ai_ready_joined2.csv` (Loop pump users) | genuine continuous pump rate | not in repo; pump basal is typically ~0.5–2 U/h |
| **Eval demo** — `test_data/livia_sugar_one_ready.csv` | Livia's Dexcom **Long-Acting injections** | n=444, median **26.0**, range **2–30** |

Livia is a Dexcom + pen user with no pump, so her file's basal column can only have come from
long-acting doses — and 26.0 is exactly the `insulin_slow` value cgm_format parses at
`2024-06-12 07:16`. So GDP *did* map long-acting → basal for this export. Copying that mapping
would be following a demo file, not the training distribution.

**Why the mismatch is actively harmful, not merely untidy.** `train_sugar_one.py:249-308` scales
every channel with a per-channel **`MinMaxScaler` fitted on the training set** and applied to the
test set with `fit_scalers=False`. MinMaxScaler maps `[train_min, train_max] → [0, 1]`; it does
not clip. If training basal spans roughly `[0, 3]` U/h, a 26 U injection arrives at the model as
≈ **8.7** — nearly an order of magnitude outside the range the network ever saw on that channel.
An empty column is zero-filled by `evaluate_model.py` and stays in range. **Empty is the safe
default, and cgm_format's current behaviour is correct.** It also means the shipped Livia demo
CSV is itself feeding out-of-distribution values into that channel.

**A second, deployment-level problem surfaced while checking this.** The scalers are **not** in
the checkpoint — `test_model_sugar_one/best_info.json` holds only `epoch` and `val_loss`. At
evaluation, `_load_train_for_scalers` (`evaluate_model.py:522`) **re-fits them from the training
CSV** named in `tuning_meta.json` (a Windows path, `D:\...\loop_ai_ready_joined2.csv`). So serving
SugarOne requires that training CSV on the inference host, or the scalers are wrong. There is a
fallback that fits on the test file itself when the two paths coincide — silently producing a
per-request scaling that has nothing to do with training. This is outside cgm_format, but it
gates whether aligning the basal column matters at all.

**What is missing, concretely:**

1. **The training basal distribution.** Without `loop_ai_ready_joined2.csv` (or just its basal
   min/max) the scale mismatch cannot be quantified, only bounded. One number from the model team
   unblocks this.
2. **A decision on what a pen user's long-acting dose *should* mean to a pump-trained model.**
   The defensible conversion is a rate, not a dose: spread the injection across its duration of
   action, e.g. 26 U over 24 h ≈ **1.08 U/h**, which plausibly lands inside pump range. That is a
   real candidate — but it needs (3).
3. **Duration of action, which the Dexcom export does not record.** Glargine ≈ 24 h, degludec
   ≈ 42 h, detemir ≈ 12–20 h. cgm_format cannot infer the product from the CSV, so this has to
   come from user profile data the library does not currently carry. **This is the actual
   blocker** — not the column mapping.
4. **Confirmation of whether SugarOne is even the right model for pen users.** It was trained on
   pump data with a real basal channel; a Dexcom pen user has no such signal. Running it with
   basal zero-filled may be the honest configuration, and `--zero-cov` exists for exactly that.

Until 1–4 are answered, `to_ml_ready_df` keeps `Basal Rate (U/h)` empty and exposes
`basal_rate_from_insulin_slow=True` as an explicit, documented opt-in for a deployment that has
established its own answer.

### 0.9 Rung 3 (grid re-timing) — **shipped in 0.12.0, 2026-08-18**

**Result: MAE 0.7834 → 0.000176 mg/dL** over the 289-point gate window, max abs diff 0.0005,
exact-match rate 8.7% (25/289) → **100%** (289/289). `tests/test_livia_reference_alignment.py` passes.

Shipped as `synchronize_timestamps(..., retime_glucose=True)`, default on, with a new write-once
`original_glucose` service column as the value anchor and `Quality.GRID_RETIMED` (64) marking
re-timed rows. `interpolate_gaps` and sync now share one interpolant over the same measured anchors.
Full account in `docs/CHANGELOG.md` under 0.12.0.

**Two corrections to what this document said before the work.**

1. **§0.2's claim that GDP uses "the monolith's two-nearest-point formula" is right about the
   mechanism and wrong about what it computes.** `create_fixed_frequency_data` picks its two points
   by absolute time distance (`argsort` on `|t − t_grid|`) with **no requirement that they bracket
   the target**. When they do, the weights sum to 1 and it is ordinary linear interpolation — which
   is why the §0.4 worked example lands on 131.3 to the digit. When both fall on the same side, the
   weights sum to more than 1 and it emits a mirrored pseudo-extrapolation. That happens at the first
   grid point of roughly half of all sequences (grid start rounds down when the first reading's
   seconds are 1–29), and in the interior wherever `interpolate_missing_values` left a 6–9 minute gap
   unfilled (`int(diff/5) - 1 == 0`).

   **We did not reproduce it.** Clamped bracketing interpolation reproduces the whole 25,332-row
   reference to MAE 0.043 with 99.1% of points within 0.01 mg/dL, and the residual outliers are
   exactly the points where GDP produced that artifact — up to 91 mg/dL from anything the sensor
   reported. None fall inside an inference window.

2. **The idempotency hazard this section was written around was real, and the fix is the anchor
   column, not the flag.** `Quality.GRID_RETIMED` marks re-timed rows so a reader can tell derived
   from measured, but it is not what makes the operation re-runnable — skipping already-flagged rows
   would have gone stale the moment a caller re-ran with a different interval. Recomputing every pass
   from `(original_datetime, original_glucose)`, columns no stage writes, is what makes
   `f(f(x)) == f(x)` hold for any parameters.

All five idempotency and commutativity chains pass on every committed fixture, in both re-timing
modes, asserted on values. `tests/test_idempotency.py::TestGridRetiming` adds the two assertions the
chains cannot make: that the anchor still equals a fresh parse after three passes, and that re-timed
values equal the raw readings that bracket them.

<details>
<summary>The original plan for this rung, kept for the reasoning</summary>

### 0.9 (original) Rung 3 (grid re-timing) — approved, but scheduled as its own run

Decision 2026-08-17: matching training's interpolation is the right move, **but it gets a
dedicated run with idempotency as the primary risk**, not a bolt-on to rungs 1–2.

Why idempotency is the hazard here. Every processing stage in cgm_format is designed to be
re-runnable: stages sort and measure against **`original_datetime`**, the service column holding
the reading's true source timestamp, precisely so that a second pass over an already-processed
frame is a no-op. Grid re-timing breaks that contract if implemented naively — it *rewrites the
glucose value itself*. Interpolating from values that were already interpolated on a previous
pass makes each run drift further from the source, and the drift is silent because timestamps
stop changing after pass one.

Requirements for that run:

- Interpolate strictly from readings anchored on `original_datetime`, never from a previously
  re-timed `glucose`. The raw reading must remain recoverable at every pass.
- Mark re-timed values (a `Quality` flag) so a second pass can tell derived from measured, and
  so inference warnings can surface it.
- Idempotency tests are the acceptance gate, not an afterthought: `f(f(x)) == f(x)` on the Livia
  fixture and on the corpora, asserted on **values**, not just row counts and timestamps.
- Only then re-measure the gate; the target is MAE ≤ 0.1 (currently 0.7834).

</details>

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
| Metadata | `original_datetime`, `quality`, `event_type` service columns | Same, plus `original_glucose` — the value anchor grid re-timing reads (idempotent processing design) |
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

> **Partly superseded — see §0.1 and §0.3.** Timestamps now align exactly (289/289); cgm_format
> already places rows on the same grid GDP uses. What is left of this gap is (a) long-vs-wide row
> shape, fixed by coalescing in `to_data_only_df` (§0.3), and (b) grid re-timing of glucose values
> (§0.4) — not grid construction.

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

### 8.2 Reference alignment test (committed fixtures)

**Fixtures** in `data/comparison/` (git-tracked):

| File | Purpose |
|------|---------|
| `livia_test.csv` | Raw Dexcom input (copy of GDP `data/livia_test.csv`) |
| `glucose_config_livia_reference.yaml` | GDP config; cgm-like columns; **`remove_calibration: true`**, **`remove_after_calibration: false`**, **`remove_after_calibration_hours: 0`** |
| `livia_reference.csv` | Training-grid reference (generated; commit after regen) |

**Regenerate reference:**

```powershell
uv run python scripts/generate_livia_reference.py --force
```

**Run alignment gate (expected to FAIL until Phase 1+2):**

```powershell
uv run pytest tests/test_livia_reference_alignment.py -s -v
```

**Logic:** `tests/comparison/livia_alignment.py` compares GDP reference (latest sequence, last 1440 min) vs `cgm-cli pipeline` output on the same input. Thresholds from Phase 0 (`glucose_mae <= 0.1`, row count ratio ~1, etc.).

**Baseline failure (2026-06-17):** row counts match in window (289) but **zero timestamp overlap** — GDP grid at `:02/:07/.../:12` phase vs cgm sync at `:01/:06/.../:16` phase (`GRID_PHASE_OFFSET`).

> **Incorrect — see §0.1.** Against the `livia_reference.csv` committed in this very commit the
> overlap is 289/289 and there is no phase offset; this note describes a preliminary reference
> that was regenerated before commit. The gate does still fail, but only on glucose MAE (0.79)
> and exact-match rate (8.3%).

### 8.3 Exploratory cross-repo test

**File:** `tests/test_livia_gdp_cgm_comparison.py` — runs GDP subprocess live; skips without sibling repo.

### 8.4 Measured results (2026-06-17 exploratory run)

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
| `tests/test_livia_reference_alignment.py` | **Regression gate** vs committed GDP reference |
| `tests/comparison/livia_alignment.py` | Alignment metrics and report builder |
| `tests/test_livia_gdp_cgm_comparison.py` | Exploratory live GDP comparison |
| `scripts/generate_livia_reference.py` | Regenerate `data/comparison/livia_reference.csv` |
| `data/comparison/glucose_config_livia_reference.yaml` | GDP config for reference generation |
| `pyproject.toml` | `[tool.uv] native-tls = true` (corporate PyPI TLS fix) |

### glucose_data_processing

> **Paths below describe the modular tree, which is not on `origin/main` — see §0.2 for the
> monolith line numbers that actually exist in the checked-out repo.**

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
