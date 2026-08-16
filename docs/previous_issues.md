# Resolved dogfooding findings

Findings from [dogfooding.md](dogfooding.md) that were resolved **in this repo**, each with its
resolution and a code pointer. They keep the `F#` id they had.

**Check here before re-investigating a finding that looks fixed.** That is the whole point of this
file: a resolved finding that is only recorded in a commit message gets rediscovered, re-argued and
sometimes re-fixed.

Ids are never reused, including ids that landed here. Compute the next one from this file and
`dogfooding.md` together.

---

## F4 — the D1NAMO refusal named an entry point that could not do what it said

Found on 2026-08-15 while auditing the 0.10.0 branch for merge readiness, by following the library's
own error message instead of reading it.

`parse_file` on a bare D1NAMO `glucose.csv` refused correctly and advised: *"Pass the subject
directory to `FormatParser.parse_bundle([...])`."* `parse_bundle` took a sequence of **files** and
called `parse_file` on each, so the advice ran into `Path.read_bytes()` on a directory and surfaced
`IsADirectoryError: [Errno 21]`.

The same defect class had already been fixed one layer up in the same release — `cgm-cli info
<directory>` was corrected to name `detect`/`corpus` rather than leak the errno — which is what makes
this worth recording: the CLI fix did not generalize, and nothing pointed at the library path
because no test followed the message's advice.

**Resolution.** `parse_bundle` now resolves a directory member through
`parse_subject_directory` (`interface/cgm_interface.py`, concrete impl in `format_parser.py`), so the
advice is true rather than the message being softened. Guarded by
`tests/test_subject_selection.py::TestDirectoryBundles::test_the_refusal_on_a_bare_modality_names_a_route_that_works`,
which asserts the refusal *and then runs what it names* — a message-string assertion alone would have
passed throughout the bug's life.

## F5 — no public entry point parsed one corpus subject

Found in the same pass, and the reason F4 read as a bug rather than a deferral.

0.10.0 named BUNDLE a first-class source category with D1NAMO as its motivating case — each subject a
directory of modality files — and then shipped no public way to parse one of them. Every route
failed: `parse_bundle([glucose.csv, insulin.csv, food.csv])` hit the F4 refusal, `parse_bundle([dir])`
hit the `IsADirectoryError`, and `parse_corpus`/`detect_path_format` on a subject directory raised
`UnknownFormatError` because their probes match subset roots. The working code existed as
`_process_d1namo_subject` and was private. `parse_corpus(subset_root)` — all 29 subjects — was the
only way in.

Worth separating from F4 because the repair is a different kind: F4 is a wrong sentence, F5 is a
missing capability, and fixing only the sentence would have left the category undeliverable.

**Resolution.** Three additions, all in 0.10.0 before release:

- `detect_subject_format(path)` plus a `SUBJECT_PATH_PROBES` registry — path detection one directory
  level below `detect_path_format`, kept deliberately disjoint from it so a corpus root can never
  parse as a single person (`formats/supported.py`, `format_parser.py`).
- `parse_bundle([subject_dir])` / `parse_subject_directory(dir)` — one subject, one frame.
- `list_subjects(root)` → frozen `SubjectEntry` records with per-track `TrackCoverage`, and
  `parse_corpus(root, subjects=[...])` which prunes **before** parsing. Plus `cgm-cli subjects` and
  `cgm-cli corpus --subject`.

Covered by `tests/test_subject_selection.py`. The two assertions that carry the most weight are that
`parse_bundle([subject_dir])` and `parse_corpus(root, subjects=[id])[id]` return equal frames, and
that `list_subjects`' ids are exactly the subject halves of `parse_corpus`' keys.

**Amended in 0.11.0.** That second assertion holds when every listed subject parses, which is true of
every committed corpus, but it is not an identity and stating it as one was wrong. BIG IDEAs
enumerates subjects from the union of both modalities, so a directory holding only a food log is
*listed* — with no coverage, because we could not look — and is not *keyed*, because parsing it
fails with a typed error naming the missing `Dexcom_*.csv`. That asymmetry is the point of the
union: enumerating by glucose alone would make such a subject invisible rather than reported. Both
walkers warn, so nothing is silent. Pinned by
`tests/test_subject_selection.py::TestListSubjects::test_an_unparseable_subject_is_listed_and_reported`.

---

## F6 — every BIG IDEAs meal row was parsed without an anchor

**Found:** 2026-08-16, reviewing the BIG IDEAs corpus branch against the 16 published subjects.
**Severity:** the corpus's own reason for existing. **Legality:** a bug fix, nothing a consumer
reads changed shape.

`_process_bigideas_subject` built its glucose half by calling `_process_dexcom`, which is a complete
parse: it ends at `_postprocess_unified`, which creates `original_datetime` on the rows it is given.
The meal frames were then concatenated onto that finished frame, and the outer `_postprocess_unified`
skipped anchor creation because of its write-once guard — `if 'original_datetime' not in
unified_df.columns`. The column was present, so the meal rows kept their nulls.

Nothing raised, and the frame passed `CGM_SCHEMA_EXTENDED.validate_dataframe(enforce=False)`. On the
published corpus **all 1,422 meal rows across all 16 subjects** had a null anchor.

`docs/PHILOSOPHY.md` says it plainly: *"`original_datetime` is write-once. Created during parsing,
never overwritten."* These were parsed rows whose anchor was never created at all.

The damage was downstream and total. Everything computes from `original_datetime`, so
`detect_and_assign_sequences` could not place a meal, and `sequence_id = 0` means unassigned:

```
as shipped     meals_in_seq0= 61/61   inference_rows=  97   carb_rows=0
anchor filled  meals_in_seq0=  0/61   inference_rows=  98   carb_rows=1
```

Second symptom, same cause: the emitted frame was not chronological. The schema's total ordering
leads with `original_datetime`, so nulls sorted every meal to the head of the frame instead of
interleaving them with the readings.

**Why the suite did not see it.** `TestPipeline` asserted `interpolated.height >= frame.height`,
which holds whether or not a single meal is ever placed.

**Resolution.** Structural rather than a null-fill, so the shape cannot recur:

- `_process_dexcom` split into `_dexcom_read_rows` (decode, clean, skip the metadata block, probe
  the timestamp) → `_dexcom_event_frames` (per-event sub-frames, stopping **before** the
  unified contract) → `_process_dexcom` (concat, postprocess, unchanged public behaviour — verified
  by identical `hash_rows()` on every committed Dexcom fixture across the refactor).
- `_process_bigideas_subject` now extends `_dexcom_event_frames`' sub-frame list with the meal frames
  and calls `_postprocess_unified` **once**, so the anchor is created for every row in one place.
- The split surfaced a dtype conflict the double-postprocess had been hiding: meal `quality` was
  `Int64` where the Dexcom sub-frames use `pl.lit(0)` (`Int32`). Now both use the plain literal and
  the schema cast happens once, afterwards.

**Guarded by** `tests/test_subject_selection.py::TestParsedCorpusInvariants` — anchors non-null,
frames in time order, and no non-glucose event inside the glucose span left unassigned. Generic over
CGMacros, both D1NAMO subsets and BIG IDEAs, because the defect is per-bundle-parser rather than
per-vendor. All three assertions were demonstrated failing against the pre-fix parser before being
claimed as a guard. The rule is written up in `docs/NEW_SCHEMA.md` under *Gotchas*.

---

## F11 — every Dexcom exercise row lost its duration

**Found:** 2026-08-16, reviewing the BIG IDEAs branch. **Severity:** silent loss of a core data
column on real committed exports. **Legality:** a bug fix — nothing a consumer reads changed shape,
a column that was always null now carries values.

`_dexcom_event_frames` converts Clarity's `Duration (hh:mm:ss)` to seconds:

```python
pl.col("duration_str").str.split(":").list.get(0).cast(pl.Int64) * 3600 +
pl.col("duration_str").str.split(":").list.get(1).cast(pl.Int64) * 60 +
pl.col("duration_str").str.split(":").list.get(2).cast(pl.Int64)
.alias("exercise"),
```

`.alias("exercise")` binds to the seconds term only. A binary op inherits the name of its left
operand, so the sum came out named `duration_str`, was written back onto that column, and was then
removed by the `.drop(["duration_str", "subtype"])` two lines down. No `exercise` column was ever
produced; the diagonal concat and schema enforcement filled it with nulls.

Nothing raised. The frame was schema-clean, the exercise rows were all present with the right
`event_type` and timestamp, and only the value was gone. On `data/input/000-14 oct-28 oct 2019.csv`,
three `Exercise` rows at `00:45:00` all parsed to `exercise = null`; BIG IDEAs subject `014` lost
`00:55:00` and `00:45:00` the same way.

**Why the suite did not see it.** No test read `exercise` off a parsed vendor frame. The unified
round-trip and idempotency suites both pass on a column that is null on every run.

**Resolution.** Parenthesise the sum so the alias covers it, and pass `null_on_oob=True` to
`list.get` so a two-part duration yields null instead of raising.

**Guarded by** `tests/test_format_parser.py::TestDexcomExerciseDuration`, which reads the expected
seconds out of the fixture's own `Duration` cells rather than a literal. Demonstrated failing
against the pre-fix expression before being claimed as a guard.

---

## F12 — a short Clarity metadata block silently ate readings

**Found:** same pass. **Severity:** silent data loss, one reading per missing metadata row.
**Legality:** a bug fix.

`_dexcom_read_rows` passed `skip_rows_after_header=len(DEXCOM_METADATA_LINES)` — 10 for mg/dL
exports, 11 for EU — and then dropped any *surviving* blank-timestamp rows dynamically. That handles
a block **longer** than expected. A **shorter** one is the same mismatch in the other direction and
was fatal: the skip ate that many real readings before anything could look at them, and no
downstream surface could tell they had existed.

Not hypothetical. The block is four info rows plus one row per configured alert, so its length
depends on how many alerts the user set up. A subject with four alerts instead of six loses two
readings. The warning that fires in the other direction reads "metadata length mismatch", so the
message already claimed to cover a case the code did not.

The library even held the evidence and never compared it. On an eight-row probe block,
`list_subjects` reported `values=5` from the raw file while `parse_subject_directory` returned 3
glucose rows.

**Resolution.** Measure the block instead of assuming it. `_dexcom_metadata_row_count` counts the
leading run of rows whose timestamp cell is blank — the parser's own definition of a metadata row —
and that count is what gets skipped. The static value stays as the *expectation* the drift is
reported against, which is the "static floor + dynamic tolerance + a warning" shape the format-drift
rules ask for. The dynamic post-filter stays as a safety net for a blank-timestamp row that is not
part of the leading block.

The drift is now **signed**, and `_dexcom_metadata_drift_message` names the direction, because a
longer block is Clarity emitting a row we do not model and a shorter one is a user with fewer alerts
— different conditions with different follow-ups.

**Guarded by** `tests/test_format_parser.py::TestDexcomMetadataBlockIsMeasured`, parametrised over
block lengths on both sides of the static expectation. Against the pre-fix code the short-block case
failed with `assert 3 == 6`.

---

## F13 — coverage and the parser disagreed on what a glucose row is

**Found:** same pass. **Severity:** `list_subjects` under-reports a G7 export by its fasting
readings. **Legality:** a bug fix.

`_dexcom_event_frames` maps `Event Type` in `("egv", "fasting glucose")` to GLUCOSE events —
`Fasting Glucose` is the G7 spelling. `_bigideas_track_coverage`, written to count the same rows
straight off the source, matched `== "egv"` only. The vocabulary was restated beside the parser
instead of derived from one place, which is exactly the drift `CLAUDE.md` §5 warns about.

`TrackCoverage` exists so a caller can size a corpus **before** parsing it, and its docstring says
the two numbers being independently authored is what makes comparing them a real cross-check. A
disagreement here turns that cross-check into a false alarm: on a probe export with one fasting row,
the parser returned 6 glucose rows and coverage reported 5. Latent on the published corpus, which
records no fasting rows, so it would have surfaced first on someone else's G7 data.

**Resolution.** One `DEXCOM_GLUCOSE_EVENT_TYPES` tuple in `formats/dexcom.py`, read by both.

**Guarded by** `tests/test_bigideas.py::TestCoverageCountsWhatTheParserParses`, which asserts the
two agree on every fixture subject and adds a fasting row to check the case directly.
Demonstrated failing with `assert 5 == 6`.

---

## F14 — meal rows dropped without a timestamp were not reported

**Found:** same pass. **Severity:** a partial loss looked exactly like no loss. **Legality:** a
patch — a warning, nothing a consumer reads changed.

`_bigideas_food_frames` drops meal rows whose timestamp could not be parsed, and warned only when
**every** row failed. Losing 5 meals out of 300 produced a schema-clean, five-rows-shorter frame and
said nothing at all. `CLAUDE.md` §2: never silently drop primary data; §5: aggregate the report by
reason with a count rather than emitting one line per row.

The all-rows-failed branch was already there and correct, which is what made the gap easy to miss —
the case it covers is the one the published corpus happens to exhibit elsewhere (D1NAMO subject
`005`, whose `datetime` reads the literal `NA` on all nine meal rows).

**Resolution.** Report the partial case too: how many of how many, which file, a first-occurrence
deduplicated sample of the food names, and an explicit statement that the surviving meals and all
glucose are unaffected. Names come from `logged_food`, falling back to `searched_food`, and the
message says so rather than printing a placeholder when neither is present.

Zero rows are affected on the published corpus — 1,422 meal rows in, 1,422 out — so this changes no
output, only what a run tells you.

**Same shape remains in D1NAMO's food, insulin and annotation paths**, which drop unparseable rows
the same way. Not repaired here because it is a separate corpus with its own null vocabulary; its
glucose path already has the right shape to copy.

**Guarded by** `tests/test_bigideas.py::TestReportsWhatItDropped`, including the negative case that
a fully parseable food log stays quiet.
