# Consumer feedback — answered

Items moved out of [FEEDBACK.md](FEEDBACK.md) once answered, verbatim: the reporter's prose exactly
as filed, with the reply as its own paragraph at the top of the section. Same split as ROADMAP vs
ROADMAP_HISTORY, and for the same reason — an inbox only grows, and unanswered items become
invisible inside answered ones.

The reporter's text is never edited or re-wrapped on the way here. If it needed clarifying, that
belongs in the reply.

**Ids are never reused**, including the ones archived here and including items answered as
non-issues — the reply is part of the record and a recycled id collides with it. The next id is
computed from this file and `FEEDBACK.md` together: `triage-state.py --next`, the seats kit's ledger
([CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md) §1 has the full command).

## How items arrive

The loop that produces them is [CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md).

Moved by `triage-archive.py S<n>`, never by hand: the one property that matters — the
reporter's prose arriving byte-for-byte — is easy to break in a copy-paste and the tool verifies it,
comparing every fingerprint before and after the write and refusing the write if one moved.

Three things the tool does not do, because each is editorial or a check of a different property:

- **The contents line below is hand-written.** Naming what an item was and how it ended is a
  judgement, and a generated line would be a worse version of the thing the index exists for.
- **A section that travelled without a group heading gets told about, not given one.** The inbox's
  own title is not a group, so an item filed under no heading arrives bare; give it a `# ` heading
  here (who reported it, and when) if it would otherwise read as part of the group above.
- **It verifies the move, not the verdict.** An item the ledger still calls `new` archives without
  complaint. The lint is the ledger pointed at *this* file — `triage-state.py
  docs/FEEDBACK_HISTORY.md` — where a well-formed archived item reads `current`, so anything reading
  `new` or `unmarked-reply` was archived unanswered or lost its marker in transit. Two seconds, and
  it is the only thing between a silent mis-archive and a permanently lost item.

Archive a whole batch in one pass: sections append in the order given, so a group archived in two
batches ends up with its heading twice.

## Contents

*One line per archived item, under 80 characters:
`- **S<n>** <what it was> — <verdict> (<roadmap item or release, if any>)`. It is an index, not a
second copy of the reply — the detail lives in the section's own reply, the one place it cannot
drift from the answer. An item missing from this list is how a tracked item becomes unfindable, and
inbound links are usually file-level, so a reader following one lands on the live inbox and needs a
pointer onward.*

- **S1** no way to check `track="mean"` is defensible — accepted (RM14)
- **S2** treatments JSON typed by inference, ComputeError — fixed (0.12.2)
- **S3** httpx not following redirects — fixed (0.12.2)
- **S4** JSON payloads misrouted to the CSV parser — fixed (0.12.2)
- **S5** temp-dir data, token in logs, dropped readings — fixed (0.12.2) + RM15

---

# sugar-sugar — filed 2026-08-17

## S1

**Status — accepted; filed as [RM14](ROADMAP.md), and the documentation half shipped in this pass.**
Reproduced against 0.12.0 on 2026-08-21, through `parse_tracks` rather than the raw CSV: the two
frames inner-joined on `datetime`, `libre = slope · dexcom + intercept` fitted per subject over all
45 subjects and 629,605 overlapping samples. Median slope 0.70 (0.10–1.16), median intercept 5.4
mg/dL (−16.6 – +110.8), median r 0.82 (0.10–0.97); 31 of 45 subjects under slope 0.8, 15 under r 0.7,
and exactly two — `CGMacros-033` and `CGMacros-035` — inside slope 0.9–1.1 with r > 0.9. Your table
reproduces subject for subject. The behavioural half reproduces too: `parse_tracks(path,
track="mean")` on `CGMacros-013` (slope 0.10, r 0.10) emits **no** warning and returns 15,918 rows
with 14,275 flagged `TRACK_MERGE` — composition reported, validity not.

One correction to the framing, and it is in your favour rather than against it. *"Averaging two
independent readings shrinks noise variance"* is not a claim the shipped surface makes: it appears in
`docs/PLAN_0.10.0.md`, a planning document, and nowhere in a docstring, in `USAGE.md` or in the code.
So there was no false claim to retract — what there was is a **silence** where the caution belonged,
which is the same gap you describe and is why the item still lands as accepted rather than as a doc
correction.

The doc half is fixed now: `docs/USAGE.md` carries the measured spread beside the `track="mean"`
example, says plainly that `TRACK_MERGE` reports composition and not validity, and points at RM14.
That is deliberately not a substitute for the API — it tells the next consumer to run the check, and
it cannot run it for them.

The API half is **filed, not shipped**. RM14 carries your shape as the one to start from —
`TrackAgreement` as a frozen dataclass that computes and does not decide, `FormatParser.track_agreement()`
mirroring `list_subjects`, `is_interchangeable` over two module constants, and a warning on
`track="mean"` that never refuses the call — along with your three rejected repairs, which we agree
with for the reasons you give: a `Quality.TRACK_DISAGREEMENT` bit really would misuse the row-level
mechanism for a per-subject property. Your legality read is right and we reached it independently:
additive throughout, nothing removed, renamed or retyped, frames byte-identical, so **minor**. Two
questions are left open in the item rather than settled here — whether the threshold constants belong
in the library at all or in the caller's policy, and what `mean` should cost when it has to parse
both tracks to fit them. Both are release-shape calls for the maintainer, and nothing in this pass
prejudges them.

Nothing for you to do: you are already playing the Dexcom track alone and documenting why, which is
the right call on this corpus and stays right after RM14 lands. The item exists for the next consumer
who reaches for `mean` because it is the obvious answer, and until it ships `USAGE.md` is what warns
them.
<!-- triaged: 0.12.0 · sha 2462b38b117e -->

Filed on behalf of `sugar-sugar`, during its migration onto 0.10.0. Observed against `cgm-format`
0.10.0, Python 3.14, on the PhysioNet CGMacros corpus (`cgmacros/1.0.0`, 45 subjects) — not on a
consumer fixture.

**What I gave it.** `FormatParser.parse_tracks(path, track="mean")` on each CGMacros subject.

**What I got.** One extended-schema frame per subject, with `Quality.TRACK_MERGE` set on every row
synthesized from two readings and clear on every row that came from one sensor — exactly as D5a
specifies, and correct on its own terms.

**What I expected instead.** Some way, from the API, to learn whether averaging *that subject's* two
series is a defensible operation before opting into it. There is none. The two mechanisms D5a gives
the mean — the `TRACK_MERGE` flag and the estimator-identity docstring — both describe the mean's
**composition**: which rows came from two sensors and which from one. They are per-row, and for what
they cover they work. Neither describes its **validity**: whether the two series are estimates of the
same quantity at all. That is a property of the subject, and it is the question that decides whether
`track="mean"` is honest. The call returns the same clean frame either way.

D5a's own rationale is where this lands: *"Averaging two independent readings shrinks noise
variance."* That holds when both are noisy reads of one truth. Fitting `libre = slope · dexcom +
intercept` over the 629,605 samples where both sensors read, across all 45 subjects, they are not:

| statistic | median | range |
|---|---|---|
| slope | **0.70** | 0.10 – 1.16 |
| intercept | 5.4 mg/dL | −16.6 – +110.8 |
| Pearson r | 0.81 | **0.10** – 0.97 |

A pure baseline offset would give slope ≈ 1 with a large intercept; a pure gain error would give
intercept ≈ 0 with r near 1. Neither shape holds. Libre compresses the excursion to roughly two
thirds rather than sitting a fixed distance below (31 of 45 subjects under slope 0.8); the
compression factor is subject-specific, so no corpus-wide correction recovers it; and 15 of 45
subjects fall under r 0.7, so a third of the corpus differs in shape and not only in level. Six worst
and four best by slope:

| subject | slope | intercept | r | bias | Libre < 70 | Dexcom < 70 |
|---|---:|---:|---:|---:|---:|---:|
| CGMacros-013 | 0.10 | 110.8 | 0.10 | −0.8 | 0.3% | 0.8% |
| CGMacros-008 | 0.21 | 67.7 | 0.33 | −27.4 | 6.7% | 3.0% |
| CGMacros-009 | 0.34 | 61.9 | 0.45 | −23.0 | 1.5% | 2.8% |
| CGMacros-001 | 0.36 | 50.5 | 0.54 | −18.3 | 0.9% | 0.9% |
| CGMacros-006 | 0.38 | 45.5 | 0.41 | −26.3 | 9.6% | 0.8% |
| CGMacros-007 | 0.39 | 10.3 | 0.31 | −58.8 | **82.0%** | 0.1% |
| … 35 subjects … | | | | | | |
| CGMacros-033 | 0.92 | −14.8 | 0.95 | −27.2 | 1.7% | 0.2% |
| CGMacros-035 | 0.95 | −16.0 | 0.91 | −26.3 | 0.0% | 0.0% |
| CGMacros-003 | 0.97 | −8.9 | 0.82 | −12.0 | 1.5% | 0.0% |
| CGMacros-012 | 1.16 | 1.6 | 0.89 | +24.3 | 0.0% | 0.0% |

Against a slope 0.9–1.1 and r > 0.9 bar, two of 45 subjects qualify: 033 and 035. So for 43 of them
the mean is not two readings of one truth averaged — it is two different estimators blended into a
third matching neither, and `TRACK_MERGE` marks those rows merged without suggesting anything is
amiss.

This is not a claim that Libre is wrong, and I am not asking the library to decide which sensor to
trust. The fit shows only that the two series are not interchangeable. (The direction happens to be
unambiguous here — CGMacros-007's Libre series reads below 70 mg/dL for 82% of ten days against
Dexcom's 0.1%, which nobody survives — but that is a domain judgement and it is the consumer's.)

**Why I think it is yours and not mine.** The evidence lives in the relationship between two frames,
so only something holding both can see it. A caller who asks for `mean` gets one frame back and has
no remaining handle on the question; a caller who wants to check first re-derives the fit from
`parse_tracks`, which means every consumer reimplements the same statistic against the same corpus.
The library already reports on things it can see and the caller cannot — a D1NAMO meal referencing a
photograph absent from disk, a CGMacros subject omitting optional columns, subject 005's unparseable
food timestamps. Each is an observation about the source rather than a parse failure, and each is
emitted rather than resolved.

**Smallest shape I can see that closes it.** A measurement beside `TrackCoverage` that computes and
does not decide — `TrackAgreement(reference, comparand, overlap, slope, intercept, correlation, bias,
mean_abs_difference)` as a frozen dataclass, reached by a `FormatParser.track_agreement(file_path,
reference=None, comparand=None)` classmethod mirroring `list_subjects`, raising
`MultiTrackSourceError` for a single-track format and `ZeroValidInputError` when the tracks never
read at the same time. An `is_interchangeable` property over two module constants
(`AGREEMENT_SLOPE_TOLERANCE = 0.10`, `AGREEMENT_MIN_CORRELATION = 0.90`) reads as a documented
convenience rather than an enforced gate; constants rather than registry entries, since the
registries stay pure data and thresholds are policy. Then on `track="mean"`, a warning through the
existing channel when the tracks do not agree, naming slope and r and pointing at
`track_agreement()` — never refusing the call.

Deliberately not suggesting, on charter grounds rather than cost: auto-selecting a "best" track,
which is domain policy and depends on what the consumer does with the frame; gating `mean` behind a
flag or exception, which breaks existing callers when a pipeline tolerating a compressed trace is
entitled to one; and a `Quality.TRACK_DISAGREEMENT` row bit, since disagreement is a per-subject
property and every row would carry the same value, misusing the mechanism `TRACK_MERGE` established.

My read of the legality, offered so there is something to disagree with rather than because I trust
it: additive, so **minor**. Three new names in `__all__`, nothing removed, renamed or retyped;
`CGM_SCHEMA` and `CGM_SCHEMA_EXTENDED` untouched; no existing flag changes meaning; frames
byte-identical to today's. The warning alone would be a patch, but it rides the additive API. One
real cost: requesting `mean` would parse both tracks to fit them, which a `check_agreement=False`
parameter or a per-path cache would cover if unwelcome.

**Reproduction**, no consumer code involved. Every figure above comes from this loop; subjects under
100 overlapping samples are skipped and none were. The corpus is CC BY-NC-SA and travels with
neither repo.

```python
import numpy as np, polars as pl
from pathlib import Path

for d in sorted(Path("CGMacros").glob("CGMacros-*")):
    both = (pl.read_csv(d / f"{d.name}.csv", infer_schema_length=10_000)
              .select(["Libre GL", "Dexcom GL"]).drop_nulls())
    if both.height < 100:
        continue
    x, y = both["Dexcom GL"].to_numpy(), both["Libre GL"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    print(d.name, round(slope, 2), round(float(np.corrcoef(x, y)[0, 1]), 2))
```

Nothing is blocked on this. `sugar-sugar` plays the Dexcom track alone and documents why, falling
back to Libre only for a subject with no Dexcom readings at all; that costs 8.4% of rows, which are
Libre-only samples with 10.1% below 70 mg/dL against Dexcom's 0.4%, so it reads as a contaminant
leaving rather than data lost. I am filing it for the next consumer, who will reach for `mean`
because it is the obvious answer and get a plausible-looking frame back.

# sugar-sugar — filed 2026-08-28

Routed into this inbox on 2026-08-28. The report was written on 2026-08-28 into
`sugar-sugar`'s own tree (`FEEDBACK.md` at its root, commit `6ae4550`) rather than into this
file, so it never reached the ledger and this inbox read empty while four defects were owed.
The reporter's prose below is verbatim; only the issue headings carry this inbox's ids.

Issues found in [`cgm-format`](https://github.com/GlucoseDAO/cgm_format) while working on
Sugar Sugar. Filed here rather than fixed in this repo: `sugar_sugar/corpus.py` /
`sugar_sugar/data.py` are the single boundary to the library, and working around a parser
defect on this side of it is exactly the drift that boundary exists to prevent.

Version under test: **cgm-format 0.12.0** (the latest release on PyPI as of 2026-08-28;
there is nothing newer to bump to). Library paths below are relative to the installed
package. Every claim here was reproduced by running the code, not by reading it.

---

## S2 — Nightscout URL import fails outright on any real instance — `ComputeError` on treatments

**Status — accepted; fixed in 0.12.2.** Reproduced before anything was changed, with a control: your
106-record payload raises `ComputeError` on the current code, and the same payload with the null run
shortened to 50 parses. So it is the inference window, confirmed, and not the int/float mix that also
occurs in this data.

Fixed as you proposed. `NIGHTSCOUT_TREATMENTS_JSON_DTYPES` (`formats/nightscout.py`) declares the six
fields and `_parse_nightscout_treatments_json` passes it; `Float64` is what
`_treatments_json_to_unified` already casts the four numeric ones to, so no downstream semantics
move. The empty returns carry the same schema, so column presence no longer depends on whether any
row survived — the `if "insulin" in columns` guards were doing that work by accident.

`_nightscout_number()` is in, and your argument for it over `infer_schema_length=None` is the reason:
a numeric string infers as `String`, the downstream `cast(Float64, strict=False)` nulls it, and the
event disappears with no warning, which our own §5 forbids. Your `bool` exclusion turned out to
matter more than it reads — the old path *did* coerce `True` into a dose, and the test for it fails
on the pre-fix code.

Four tests in `tests/test_nightscout.py::TestTreatmentsJsonSchema`, built in-test rather than from a
fixture because the defect is a property of record ordering and no committed fixture pins that. All
four were run against the unfixed code and watched to fail first.

<!-- triaged: 0.12.2 · sha 2a82a4449dff -->

**Severity: blocking.** This is the bug a player actually hit; it takes down the whole
"paste your Nightscout URL" feature. Reported against a live Nightscout 15.0.3 instance
(URL withheld — it serves a real patient's data).

### Symptom

`FormatParser.from_nightscout_url(base_url)` raises, before any app code sees a frame:

```
polars.exceptions.ComputeError: could not append value: 47 of type: i64 to the builder;
make sure that all rows have the same schema or consider increasing `infer_schema_length`
```

Traceback bottoms out at `format_parser.py:1480`, in `_parse_nightscout_treatments_json`:

```python
return pl.DataFrame(rows)
```

### Root cause

`_parse_nightscout_treatments_json` (`format_parser.py:1452-1480`) builds `rows` as a list of
dicts and hands it to `pl.DataFrame` with **no schema**, so polars infers dtypes from the
first `infer_schema_length=100` rows only.

Nightscout treatments are heterogeneous by design — a `Temp Basal` carries `rate`/`duration`
and no `carbs`, a `Meal Bolus` carries `carbs`/`insulin` and no `rate`. On the reporting
instance, the 353 treatments had this shape:

| field | None | int | float | first non-null at row |
|---|---|---|---|---|
| `insulin` | 293 | 7 | 53 | 20 |
| `carbs` | **351** | 2 | 0 | **167** |
| `rate` | 77 | 139 | 137 | 0 |
| `duration` | 17 | 336 | 0 | 0 |

`carbs` is `null` for the first **167** rows — past the 100-row inference window — so the
column is inferred as dtype **`Null`**, and appending the `i64` at row 167 fails.

Two things worth stressing, because they rule out the obvious near-miss diagnoses:

- **This is not the int/float mix.** `insulin` mixes int and float (int first appears at row
  122, float at row 20) and appends fine — polars takes the supertype. The failure is
  specifically *any* value into a `Null`-dtype builder. Coercing the JSON's numbers to float
  before parsing does **not** help; it merely changes the message to
  `could not append value: 47.0 of type: f64`.
- **It is not exotic data.** Any Nightscout user who runs a closed loop and logs carbs
  rarely — i.e. a large fraction of them — has a null run longer than 100 rows in at least
  one of these four columns. Bolus-only users hit it via `rate`, pump-only users via `carbs`.

The library already knows to do this correctly on its *other* Nightscout path:
`_parse_nightscout_entries_csv` (`format_parser.py:1613`) passes `infer_schema_length=None`.
The JSON path just lacks the equivalent.

### Minimal repro (no network, no patient data)

```python
import json
from cgm_format import FormatParser

entries = [{"type": "sgv", "dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 120}
           for h in range(5)]
treatments = [{"eventType": "Temp Basal", "created_at": "2026-08-28T00:00:00.000Z",
               "rate": 0.5, "duration": 30, "carbs": None, "insulin": None}
              for _ in range(105)]
treatments.append({"eventType": "Meal Bolus", "created_at": "2026-08-28T05:00:00.000Z",
                   "carbs": 47, "insulin": 2.5})

FormatParser.parse_nightscout(json.dumps(entries), json.dumps(treatments))
# polars.exceptions.ComputeError: could not append value: 2.5 of type: f64 to the builder
```

106 records is enough. Drop the null run below 100 and it passes, which is the whole bug.

### Suggested fix

Declare the schema instead of inferring it. Every one of these columns is cast to `Float64`
downstream anyway (`_treatments_json_to_unified`, `format_parser.py:1517`, `1534`, `1553`),
so an explicit schema changes no semantics — it only stops inference from guessing:

```python
_TREATMENTS_SCHEMA = {
    "created_at": pl.String, "eventType": pl.String,
    "insulin": pl.Float64, "carbs": pl.Float64,
    "rate": pl.Float64, "duration": pl.Float64,
}

def _num(value: object) -> float | None:
    """Nightscout uploaders are inconsistent: ints, floats and numeric strings all occur."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# ... in the row loop, wrap the four numeric fields in _num(), then:
return pl.DataFrame(rows, schema=_TREATMENTS_SCHEMA)
```

`infer_schema_length=None` alone would also clear the reported error, but it stays fragile:
it still guesses, it forces a full scan of every row, and it does not handle the numeric
**strings** some uploaders emit (`"carbs": "47"`), which infer as `String` and are then
silently dropped by the `.cast(pl.Float64, strict=False)` downstream. The explicit schema
plus `_num()` handles both.

### Verified

Applied as a monkeypatch against the reporting instance's real payload: parses cleanly,
1367 unified rows — 1152 `EGV_READ`, 61 `INS_FAST`, 152 `INS_SLOW`, 2 `CARBS_IN` over ~4
days. The numeric-string coercion accounts for 3 events (1 bolus, 2 basals) that are lost
without it.

---

## S3 — Redirects are not followed, so `http://` and apex→www instances fail

**Status — accepted; fixed in 0.12.2.** Confirmed against the pinned httpx 0.28.1:
`follow_redirects` defaults to `False`, and `download_nightscout` was not passing it.
`nightscout_downloader.py` now passes `follow_redirects=True`.

No integration test, because exercising a real redirect needs either a live redirecting host or a
mocked transport, and mocking the thing under test is what our §6 forbids. What is pinned instead is
the argument itself: `TestDownloaderRedirects` captures the client's construction kwargs and asserts
`follow_redirects is True`. It fails on the pre-fix downloader, where the kwargs were `timeout` and
`headers` alone.

<!-- triaged: 0.12.2 · sha 96491a896133 -->

**Severity: high, one-line fix.**

`nightscout_downloader.py:115` constructs the client as:

```python
with httpx.Client(timeout=timeout, headers=headers) as client:
```

httpx differs from `requests` here — verified on the pinned httpx 0.28.1,
`httpx/_client.py:197`, `follow_redirects: bool = False`. `download_nightscout` never passes
the argument.

So any instance that redirects — `http://` → `https://`, apex → `www`, a Heroku/Fly/Netlify
hostname 301ing to its canonical domain, a trailing-slash normalisation — fails at the
`raise_for_status()` on the next line with `httpx.HTTPStatusError: Redirect response '301
Moved Permanently'`, rather than working. Users type bare hostnames constantly.

Fix: `httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)`.

---

## S4 — An empty or non-JSON API response is misrouted into the CSV parser

**Status — accepted; fixed in 0.12.2.** Both cases reproduced exactly as you reported them, including
the mangled column names: `[]` came back as `Missing required column: 'Glucose (mg/dL)'. Got columns:
['[]']`, and the 401 body split on commas into two.

Routing now asks whether the payload parses as JSON rather than whether `"sgv"` appears in the first
2000 characters. `_nightscout_json_array` unwraps it: an error object raises `MalformedDataError`
carrying the server's own message, so an operator debugging a 401 is told `Unauthorized` instead of
being shown a CSV column, and the empty case reaches the `ZeroValidInputError` that was unreachable.

One thing your suggested fix would have broken, so it is worth naming: "parses as JSON" alone lets a
CSV through, because `json.loads("123")` succeeds and a vendor CSV can start with a number. The check
requires a `[` or `{` opener as well, and two tests pin that boundary in both directions — one on the
committed exporter CSV, one on a bare number.

Your latent variant is fixed by the same change and had a test written for it: an entries array whose
first records are `cal` or `mbg` was misrouted identically, and since the downloader writes with
`indent=2`, 2000 characters is only about five records.

<!-- triaged: 0.12.2 · sha b0dcf440e70f -->

**Severity: medium (bad diagnostics, unreachable guard).**

`parse_nightscout` picks its branch with `_is_nightscout_entries_json`
(`format_parser.py:1413-1417`):

```python
stripped = data.strip()
return stripped.startswith("[") and '"sgv"' in stripped[:2000]
```

A Nightscout account with no data returns `200` with body `[]`. That starts with `[` but
contains no `"sgv"`, so the sniff returns `False` and the JSON falls through to
`_parse_nightscout_entries_csv`. Verified:

```
"[]"                                    -> MalformedDataError: Missing required column:
                                           'Glucose (mg/dL)'. Got columns: ['[]']
'{"status":401,"message":"Unauthorized"}' -> MalformedDataError: Missing required column:
                                           'Glucose (mg/dL)'. Got columns:
                                           ['{"status":401', 'message":"Unauthorized"']
```

Two consequences:

- The purpose-built `ZeroValidInputError("Nightscout entries JSON is empty")` guard at
  `format_parser.py:1436` is **unreachable** from a real API response.
- A server's own error text survives only as a mangled polars column name, so an operator
  debugging a 401 is shown a missing-CSV-column error naming a column nobody asked for.

There is a latent variant too: the sniff reads only `stripped[:2000]`, and the downloader
writes the file with `indent=2` (~300-400 chars per entry). An entries array whose first
~5 records are `cal`/`mbg` rather than `sgv` is misrouted the same way.

Fix: decide the branch on whether the payload *parses as JSON*, not on a substring sniff,
and let the empty case reach the existing `ZeroValidInputError`.

---

## S5 — Smaller things noticed in the same code

**Status — accepted in part; five fixed in 0.12.2, one filed as
[RM15](ROADMAP.md).** Each was reproduced before being touched, and each fix has a test that fails on
the pre-fix code.

Fixed: the temp directory (now `TemporaryDirectory`, parsed inside the context — a test asserts the
directory is gone afterwards and another asserts `output_dir` still keeps the files); the token leak
(`_raise_for_status_without_token` redacts the `?token=` parameter and re-raises the *same* exception
type, so your existing `except httpx.HTTPStatusError` keeps working); records omitting `type`; `sgv`
of `0`, which is now kept as the sensor-error value it is rather than falling through `or` to
`glucose`; and the profile endpoint, which now warns and continues instead of failing an import on a
result that is thrown away. `[project.urls]` is added, so `pip show` gives a repository link.

Filed rather than fixed: wrapping network failures in a library exception type. You are right that
matching on `type(exc).__name__` is what an absent contract looks like from outside, and right that
it is ours — the network path is the only surface here that does not raise a typed error. But the
repair is a design decision rather than a defect: it has to settle whether it covers transport
failures only or HTTP status too, and whether it chains the original (which keeps `httpx` in the
traceback, defeating most of the point) or swallows it. Reusing `MalformedDataError` is wrong — a 502
is not malformed data — and the genuinely useful shape, a downloader-agnostic error, is the one
hardest to get right while Nightscout is the only downloader. RM15 has the argument written out.

The token redaction deliberately preserves `httpx.HTTPStatusError` rather than introducing a type, so
RM15 is not pre-empted by it.

<!-- triaged: 0.12.2 · sha 7449a052106f -->

- **`entry.get("sgv") or entry.get("glucose")`** (`format_parser.py:1443`) — an `sgv` of `0`
  is falsy, so the row falls through to `glucose`/`None` and is dropped. `0` is not a
  physiological reading, but it is a real sensor-error value, and silently dropping differs
  from what the code reads as intending. `if "sgv" in entry` is the intended test.
- **`if entry.get("type") != "sgv": continue`** (`format_parser.py:1440`) — uploaders that
  omit `type` (some xDrip+/Loop configurations) have every row skipped, producing
  `ZeroValidInputError("No SGV entries found")` on a payload that carries `sgv` and
  `dateString` on every record.
- **The temp directory is never cleaned up.** `from_nightscout_url` with the default
  `output_dir=None` does `tempfile.mkdtemp(prefix="nightscout_")` (`format_parser.py:1938-1940`)
  and never removes it. Every call leaves raw entries + treatments + profile JSON — a
  patient's full CGM record — in `/tmp` indefinitely. A `TemporaryDirectory` context manager
  when the caller did not ask for persistence would fix it.
- **`profile.json` is fetched, `raise_for_status`-ed, then discarded.** `from_nightscout_url`
  unpacks it as `_` (`format_parser.py:1944`) and `from_nightscout_exports` documents
  `profile_path` as accepted-and-ignored. An instance whose permissions allow `entries` but
  deny `profile` therefore fails the entire import on an endpoint whose result is thrown away.
- **No library exception type wraps network failures.** Connect errors, timeouts, HTTP status
  errors and `json.JSONDecodeError` all escape raw, so a caller must import `httpx` itself to
  catch them. `sugar_sugar/components/startup.py:1107-1129` does exactly that, and has to
  match on `type(exc).__name__` substrings to classify. Relatedly, `HTTPStatusError` messages
  embed the full request URL — including the `?token=` query parameter — so any caller that
  logs the exception leaks the Nightscout read token.
- **`parse_nightscout` and `_process_nightscout` have different error contracts.**
  `_process_nightscout` (`format_parser.py:1787-1791`) wraps its body in
  `except Exception -> MalformedDataError`; `parse_nightscout` (`1813-1845`), which is what
  `from_nightscout_url` actually reaches, has no such wrapper. That asymmetry is why issue 1
  surfaces as a raw polars `ComputeError`.
- **The package ships no `Project-URL` or `Home-page` metadata**, so `pip show cgm-format`
  gives no repository link. The only pointer is a CI badge in the README body.
