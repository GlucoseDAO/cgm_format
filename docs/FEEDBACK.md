# Consumer feedback — open inbox

The intake for people and repos that **consume** `cgm-format` — today `sugar-sugar` and `glucosedao`,
plus anyone using the published package. Layer 3: not our tests, not our own dogfooding, but our
consumers hitting the library in their own work.

**An empty inbox means nothing is owed.** That property only holds if answered items are moved out,
so once an item is answered it goes to [FEEDBACK_HISTORY.md](FEEDBACK_HISTORY.md) verbatim, reply and
all.

The file is run as a **document-as-inbox triage loop**: the document is both the transcript and the
state, and each item's state is derived by fingerprinting the reporter's own text rather than kept
anywhere else. The runbook an agent triaging this file should read is
[CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md) — the algorithm, this repo's legality rules, the
routing table, and what may be done unattended. The generalized pattern it adapts is published at
<https://gist.github.com/winternewt/54b94bda01812be937b892146d1bb254>. Nothing below depends on the
tooling — the ledger reads what is written, so a reply typed by hand in the right shape is a
first-class one.

## The ledger

Three dependency-free scripts in `scripts/`, documented in
[`scripts/README.md`](../scripts/README.md). They are stdlib-only Python and bash, import nothing
from the package, and so are the one place in this repo that does not need `uv run`:

```sh
./scripts/triage-state.py              # every item and its verdict: new / revised / unmarked-reply / current
./scripts/triage-state.py --pending    # just the ones needing work
./scripts/triage-state.py --next       # the next unclaimed id, computed over BOTH files
./scripts/triage-archive.py S1         # move answered items to FEEDBACK_HISTORY.md, verifying the move
```

The two `.py` files are Python: run them, or hand them to `python3` — **never to `bash`**, which
executes the module docstring as commands and reaches ImageMagick's `import`.

## Filing an item

Append it at the bottom under a new `## S<n> — <what happened>` heading. **The next id is S2**, and
that pin is a convenience rather than the authority: `./scripts/triage-state.py --next` computes it
over this file **and** `FEEDBACK_HISTORY.md`. Take it from both, because once answered items move out
the highest id visible here is not the highest ever used, and an empty inbox shows none at all. Ids
are never reused, not even for an item answered as a non-issue.

Say what you observed, on which input, with which version, and what you expected instead. If you are
relaying on behalf of a downstream repo, say so and name the case that motivated it — and restate it
as a fact about **this** library rather than in your own vocabulary: which column cannot express
what, which stage reports a shape it cannot distinguish, what a downstream reader ends up seeing.
Your route names and error codes do not exist here.

## Answering an item

The reply goes first in the section, immediately after the heading, as a paragraph opening
`**Status —` and saying four things: the verdict, where it landed (a roadmap item, a doc, or a
shipped version), what was actually reproduced, and what the reporter should do now. It ends with the
ledger's marker, which carries a fingerprint of the reporter's text and nothing of the reply:

```markdown
## S<n> — <what happened>

**Status — accepted; shipped in <version>.** Reproduced: <what was run and what it did>. <Where the
rest landed, and why any candidate repair was rejected.>
<!-- triaged: <version> · sha <12 hex digits — stamp 000000000000 first, see below> -->
```

Keep the placeholders as placeholders: an id that parses (`## S9`) makes the ledger see a phantom
section and `--next` skip to 10.

Stamp `sha 000000000000` as you write the reply, then run `./scripts/triage-state.py`: with a marker
present the reply is excluded whole, so the `revised` line prints the section's true fingerprint and
that is the value to paste back. Never copy the sha the ledger prints while the reply is in and the
marker is not — the hash then covers your own paragraphs two onward. `--backfill` computes it the same
way, so it is for replies that predate the ledger and only those. Once the reply is in,
archive with `./scripts/triage-archive.py S<n>` and add the item's one-line contents entry to
`FEEDBACK_HISTORY.md` — the archiver deliberately does not write that line, because naming what an
item was and how it ended is editorial.

- **Never edit or re-wrap the reporter's prose**, not when answering and not when archiving. It is
  the record of what was *observed*, not of what was decided. The reply is its own paragraph above
  it, immediately after the heading; the prose itself is never touched. The archiver
  verifies this rather than trusting it: every fingerprint is compared before and after the move and
  the write is refused if one changed. It verifies the *move*, not the *verdict* — so run the ledger
  against `FEEDBACK_HISTORY.md` afterwards, where a well-formed archived item reads `current` and
  anything reading `new` was archived unanswered.
- **Never open a preamble line with `**Status`.** A header line beginning that way reads as a block
  reply covering every id it names, which marks the whole backlog answered. Use a blockquote or
  different wording.
- **"No reply" is not "no work done."** Establish what already shipped before reproducing, and
  certainly before designing — the fix may be in a release the reporter has not upgraded to.
- **Never write a placeholder reply.** An untriaged item stays unanswered; an empty verdict is
  honest, a hedged one is not.
- **A non-issue verdict is not the cheap outcome.** Show what was probed and did not reproduce. A
  bare "works as intended" is not a reply.

Triage order: establish what shipped → reproduce against the **code**, not the docs → decide
**legality before severity** → route → reply → archive. Legality sizes the release (additive → minor;
removal, retype or rename → major; legibility → patch); severity only orders the queue inside it. The
compatibility rules the legality step reads are in `CLAUDE.md` §8 and `docs/PHILOSOPHY.md` — read
them first-hand, never through a summary: a summary drops the qualifier the decision turned on.

## Where an answered item lands

| | verdict | lands in | must contain |
|---|---|---|---|
| **a** | real, repairable, legal | `docs/ROADMAP.md` as an `RMn` item | severity · status · owner · the motivating report |
| **b** | non-issue | the reply only | what was probed and did not reproduce — never a bare "works as intended" |
| **c** | documentation defect | the doc, fixed in the same pass | the reply naming the file changed |
| **d** | real, no acceptable repair | `docs/ROADMAP.md`, status open only | the paragraph saying *why each candidate repair is wrong* |
| **e** | not ours to fix | the third party's issue tracker, restated in their terms, referenced from `docs/dogfooding.md` | the reply naming the upstream issue, plus whatever half *is* ours |

Bucket **(e)** means a third party here — Polars, Frictionless — because nothing of ours sits
upstream of this library. Restate rather than forward: a note written in our vocabulary reads as
somebody else's problem in their tracker. An item is often **(e)** *and* **(a)**: their half filed,
our defensive half shipped, both in the one reply.

An item filed as a documentation gap usually has a code half — look for it. A reporter describes
where *they* got stuck, which is a fact about the docs; what stuck them is often a surface that could
have told them.

---

## S1

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
