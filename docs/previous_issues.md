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

- `_process_dexcom` split into `_dexcom_read_rows` (decode, clean, drop blank-timestamp metadata
  rows, probe the timestamp) → `_dexcom_event_frames` (per-event sub-frames, stopping **before** the
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
