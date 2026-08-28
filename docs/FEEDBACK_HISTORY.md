# Consumer feedback — answered

Items moved out of [FEEDBACK.md](FEEDBACK.md) once answered, verbatim: the reporter's prose exactly
as filed, with the reply as its own paragraph at the top of the section. Same split as ROADMAP vs
ROADMAP_HISTORY, and for the same reason — an inbox only grows, and unanswered items become
invisible inside answered ones.

The reporter's text is never edited or re-wrapped on the way here. If it needed clarifying, that
belongs in the reply.

**Ids are never reused**, including the ones archived here and including items answered as
non-issues — the reply is part of the record and a recycled id collides with it. The next id is
computed from this file and `FEEDBACK.md` together: `./scripts/triage-state.py --next`.

## How items arrive

The loop that produces them is [CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md).

Moved by `./scripts/triage-archive.py S<n>`, never by hand: the one property that matters — the
reporter's prose arriving byte-for-byte — is easy to break in a copy-paste and the tool verifies it,
comparing every fingerprint before and after the write and refusing the write if one moved.

Three things the tool does not do, because each is editorial or a check of a different property:

- **The contents line below is hand-written.** Naming what an item was and how it ended is a
  judgement, and a generated line would be a worse version of the thing the index exists for.
- **A section that travelled without a group heading gets told about, not given one.** The inbox's
  own title is not a group, so an item filed under no heading arrives bare; give it a `# ` heading
  here (who reported it, and when) if it would otherwise read as part of the group above.
- **It verifies the move, not the verdict.** An item the ledger still calls `new` archives without
  complaint. The lint is the ledger pointed at *this* file — `./scripts/triage-state.py
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
