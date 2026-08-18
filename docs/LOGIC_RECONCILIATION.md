# Logic reconciliation — inference against training

How closely `cgm_format`'s inference preprocessing reproduces the training preprocessing the
deployed models were fitted on, measured rather than asserted, and where the two deliberately
disagree.

**Measured:** 2026-08-18, against `cgm_format` 0.12.0 and the committed fixtures in
`data/comparison/`.
**Gate:** `tests/test_livia_reference_alignment.py`.
**Planning document:** [modification_plan.md](modification_plan.md) — this file reports outcomes,
that one holds the open ladder of remaining work.

---

## Why this exists

A glucose prediction model is fitted on data processed by
[`glucose_data_processing`](https://github.com/GlucoseDAO/glucose_data_processing) ("GDP") and served
on data processed by this library. Both descend from the same code and have diverged. Where they
disagree numerically, the model sees inputs at inference that are not drawn from the distribution it
was fitted on — and it disagrees *silently*, because both pipelines produce a schema-clean frame on
the same 5-minute grid.

The two should **not** be identical. Training may delete data it distrusts; inference has to predict
anyway and mark instead. What must match is the arithmetic: the grid, the interpolation, where
covariates land. [modification_plan.md §3](modification_plan.md) holds the align/don't-align table.

---

## Headline result

The gate compares the last valid sequence, truncated to 1440 minutes — the window a server actually
feeds a model — against `data/comparison/livia_reference.csv`, produced by GDP from the same raw
Dexcom export.

| Metric | 0.11.0 | **0.12.0** | Gate threshold |
|---|---|---|---|
| Glucose MAE | 0.7834 mg/dL | **0.000176 mg/dL** | ≤ 0.1 |
| Max absolute difference | 3.30 mg/dL | **0.000498 mg/dL** | ≤ 5.0 |
| Points matching within 0.01 mg/dL | 25 / 289 (8.7%) | **289 / 289 (100%)** | ≥ 95% |
| Overlapping timestamps | 289 / 289 | 289 / 289 | non-zero |
| Row-count ratio | 1.00 | 1.00 | 0.95–1.05 |

The residual 0.0005 mg/dL is float rounding against the reference's three published decimals. There
is no remaining systematic difference in the inference window.

### What closed the gap

Timestamps already agreed before 0.12.0; the disagreement was entirely in values. `synchronize_timestamps`
moved a reading's timestamp onto the grid and kept the measured number, which is a different claim
from the one the grid instant makes:

```
raw EGV:  14:06:45 → 150      14:11:45 → 128      14:16:46 → 110

grid point 14:11:00
  0.11.0 : snap 14:11:45 onto 14:11, keep the measurement       → 128.0
  GDP    : evaluate the series at the grid instant              → 131.3
  0.12.0 : 128 + (45/300) × (150 − 128)                         → 131.3   ✔
```

The bias was one-directional — always toward the previous reading — which is why 0.78 mg/dL of mean
error came with only 8.7% of points agreeing. (An earlier note in
[modification_plan.md §0.1](modification_plan.md) records 8.3%, measured before the harness settled;
the figure here is what `tests/comparison/livia_alignment.py` reports today, on both modes, and is
the one to reproduce.) Detail of the implementation is in
[PIPELINE.md](PIPELINE.md); the invariant it had to preserve is in [PHILOSOPHY.md](PHILOSOPHY.md).

---

## Reproducing the measurement

```sh
uv sync --extra dev
uv run pytest tests/test_livia_reference_alignment.py -s -v     # prints the metric block above
```

To see that the change is confined to the glucose column:

```sh
uv run cgm-cli pipeline data/comparison/livia_test.csv -o retimed.csv
uv run cgm-cli pipeline data/comparison/livia_test.csv --no-retime-glucose -o measured.csv
```

Both emit 289 rows. `datetime`, `carbs`, `insulin_slow`, `insulin_fast` and `exercise` are
byte-identical; `glucose` differs by MAE 0.7834, max 3.30 — the whole of the old gap, and nothing
else. `--no-retime-glucose` is a supported escape hatch rather than a debugging aid: it is on
`pipeline`, `process` and `batch`, and mirrors `retime_glucose=False` in the library, so the 0.11.0
column is reachable without pinning an old version.

The whole-file breakdown in the next section comes from this, which needs no test fixtures beyond
the two committed CSVs:

```python
import polars as pl, numpy as np
ref = pl.read_csv("data/comparison/livia_reference.csv", try_parse_dates=True)
raw = pl.read_csv("data/comparison/livia_test.csv", infer_schema_length=0, encoding="utf8-lossy")
ts, ev, gl = (next(c for c in raw.columns if k in c) for k in ("Timestamp", "Event Type", "Glucose Value"))

base = raw.select(
    pl.col(ts).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S", strict=False).alias("t"),
    pl.col(ev).alias("ev"), pl.col(gl).alias("s"),
).drop_nulls("t").sort("t")

egv = base.filter(pl.col("ev") == "EGV").select(
    "t",
    pl.when(pl.col("s") == "High").then(pl.lit(401.0))
      .when(pl.col("s") == "Low").then(pl.lit(39.0))
      .otherwise(pl.col("s").cast(pl.Float64, strict=False)).alias("g"),
).drop_nulls("g").unique(subset="t", keep="first").sort("t")

as_epoch = lambda s: s.to_numpy().astype("datetime64[s]").astype(np.int64)
d = np.abs(np.interp(as_epoch(ref["datetime"]), as_epoch(egv["t"]), egv["g"].to_numpy())
           - ref["glucose"].to_numpy())
print(f"agree within 0.01: {(d <= 0.01).sum()} / {len(d)}  ({(d <= 0.01).mean() * 100:.3f}%)")

# attribute each disagreement to the nearest non-EGV or out-of-range marker
for label, rows in (("calibration", base.filter(pl.col("ev") == "Calibration")),
                    ("Low marker", base.filter((pl.col("ev") == "EGV") & (pl.col("s") == "Low")))):
    marks = as_epoch(rows["t"])
    hit = sum(1 for i in np.where(d > 0.01)[0]
              if marks.size and np.min(np.abs(marks - as_epoch(ref["datetime"])[i])) <= 300)
    print(f"  within 5 min of a {label}: {hit}")
```

`np.interp` is deliberately used as an *independent* implementation of the interpolation — it clamps
at the endpoints exactly as the library does, so agreement is evidence about the library rather than
a restatement of it.

---

## Where we still differ, and why that is correct

Across the **whole** reference — all 25,332 rows, not just the inference window — the two pipelines
agree on 99.1% of grid points. Almost every disagreement is attributable, and where it is, the
reference is the one carrying the defect:

| Disagreement | Points | Cause |
|---|---:|---|
| Near a `Low` marker | 201 | The reference carries roughly the last valid reading (56, 59 mg/dL) where the sensor reported `Low`, not the 39 mg/dL floor **its own config declares** (`low_glucose_value: 39`). We emit 39 and flag `Quality.OUT_OF_RANGE`. |
| Near a `Calibration` row | 24 | The reference lets a **fingerstick calibration value** into the CGM glucose channel. At grid `2024-06-08 06:21` the export holds a Calibration row reading 203 fifteen seconds earlier, between EGV readings of 319 and 298; the reference emits 209.0. We parse calibration as `CALIBRAT` and never as glucose. |
| Neither | 4 | Three are the **first grid point of a sequence**, where GDP's same-side selection fires (below); all are ≤ 0.43 mg/dL. The fourth is 1.0 mg/dL and unattributed. |
| **Agree within 0.01 mg/dL** | **25,103** | |

The export contains 166 `Low` markers and 21 `Calibration` rows, so a handful of contaminated grid
points each. Excluding the calibration-adjacent points, whole-file MAE is 0.032 mg/dL.

**None of these fall in an inference window**, which is why the gate is at 100%. They matter anyway:
they are a property of the *training* data, and the calibration one in particular means the models
were fitted with occasional fingerstick values presented as sensor readings. That is worth the model
team knowing; it is not something this library can or should reproduce.

### The interpolation formula we deliberately did not copy

GDP's `create_fixed_frequency_data` (`glucose_ml_preprocessor.py:675`, `origin/main` @ `4a1aa3f`)
selects its two interpolation points by absolute time distance:

```python
sorted_indices = abs((seq_pandas['timestamp'] - fixed_time).dt.total_seconds()).argsort()
idx1, idx2 = sorted_indices.iloc[0], sorted_indices.iloc[1]
...
weight1 = abs((point2['timestamp'] - fixed_time).total_seconds()) / total_time
weight2 = abs((fixed_time - point1['timestamp']).total_seconds()) / total_time
```

Nothing requires the pair to bracket the target. When it does, the weights sum to 1 and this is
ordinary linear interpolation — which is why the worked example above lands on 131.3 to the digit.
When both points fall on the *same* side, the weights sum to more than 1 and the result is a mirrored
pseudo-extrapolation rather than an extrapolation.

The condition is reachable at the first grid point of any sequence whose first reading has seconds in
1–29, because the grid start rounds *down* and so precedes all of that sequence's data; 8 of this
export's 14 sequences meet it. It is also reachable in the interior wherever GDP's count-based fill
left a 6–9 minute gap unfilled (`int(diff / 5) - 1 == 0`), so spacing inside a sequence is not
uniform. Measured, 5 of the 14 sequences disagree at their first grid point, by between 0.02 and 19.9
mg/dL.

We interpolate strictly between the bracketing pair and **clamp** at a sequence's first and last
reading. A sequence exists precisely because the gaps on either side of it are too long to
interpolate across, so a reading beyond one is not an anchor for these points.

Bug-for-bug compatibility was considered and rejected: the artifact produces values tens of mg/dL
from anything the sensor reported, it is unambiguously a defect rather than a modelling choice, and
it does not reach an inference window. Reproducing it would mean shipping known-wrong numbers to
match a mistake.

---

## What is not reconciled

This document covers the numeric path for glucose. Open items live in
[modification_plan.md §0.6](modification_plan.md); the ones that are *not* arithmetic:

- **`insulin_slow` → `Basal Rate (U/h)`** is a semantic mismatch, not a rename — a discrete
  long-acting dose in units against a continuous pump rate in units per hour. Blocked on duration of
  action, which the Dexcom export does not record. We ship the column empty; the model's evaluator
  zero-fills an absent covariate, which stays in range, while a 26 U dose fed to a channel scaled on
  pump rates arrives at the network as roughly 8.7.
- **Heart rate and step count** (GluMind) have no column in the unified schema at all.
- **Per-model context floors** (80 grid points for GluMind, 128 for SugarOne) are not enforced
  anywhere.
- **Training-side deletions we deliberately do not port**: the 200-point minimum sequence length, the
  24-hour purge after a calibration gap, `DataCleaner`'s covariate removal. Inference marks and warns
  instead of deleting, because the server has to answer.

---

## Provenance, and one caveat about it

`data/comparison/livia_reference.csv` is the definition of "correct" used above. It is a committed
artifact that **cannot currently be regenerated** — three places name a
`scripts/generate_livia_reference.py` that has never existed in any commit (F16 in
[dogfooding.md](dogfooding.md)). Independent corroboration that it reflects what the model actually
consumed: joined against the model repo's own shipped `test_data/livia_glumind_ready.csv`, 22,094 of
22,097 overlapping points match within 0.01 mg/dL.

Treat the numbers in this document as measured against that fixture. If the fixture is ever
suspected, the first task is making it reproducible, not re-tuning against it.
