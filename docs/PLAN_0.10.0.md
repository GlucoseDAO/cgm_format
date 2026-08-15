# 0.10.0 — extended schema + research corpora

## Context

`cgm-format` today assumes **one file, one subject, one device**, and a unified frame with exactly
six data columns. That assumption is nowhere written down because nothing violated it. Research
corpora violate it on three axes at once: cardinality on the input side (many files in), cardinality
on the output side (many frames out), and measurement channels the schema has no home for
(macronutrients, wearable streams, meal photos).

`docs/RESEARCH_CORPORA.md` already contains the design and the verified ground truth. This branch
executes it: **RM8** (extended schema), **RM12** (source categories), **RM9** (faceted output),
**RM10** (CGMacros), **RM13** (D1NAMO), and takes the column half of **RM6** (ketones). Everything is
additive — `CGM_SCHEMA`, every existing entry point and every existing call site keep their present
shape — so the release is **minor: 0.9.0 → 0.10.0**.

**Sequencing.** Waves 1–3 are the enablers and are sequential: the schema seam must exist before a
corpus parser has anywhere to put macronutrients, and the categories must exist before faceted output
has a shape. Waves 4 and 5 are then **independent of each other** and can land in either order or in
parallel — CGMacros is one-file-in/two-frames-out, D1NAMO is many-files-in/one-frame-out-per-subject.
They exercise opposite halves of the design, which is the argument for keeping both in this release:
each catches abstraction errors the other cannot.

Out of scope by decision, not oversight: **RM7** (lazy `scan_csv`) and **RM11** (Loop). RM7 is the
riskiest item in the set — it touches `detect_format`'s signature, `decode_raw_data`'s placement and
the `UnifiedFormat = pl.DataFrame` alias — and Loop is hard-blocked on it, DUA-gated, and needs
~20 GB on disk before one test runs.

---

## Corrections to what the roadmap currently claims

Verified against the code; these change the work, so fix the roadmap text as part of the branch.

1. **RM8's "silently narrowed" claim is half right.** Under the shipped default
   (`validation_mode = INPUT`, non-forced) an extended frame fails **loudly** at the first gated
   check — `MalformedDataError` on column count (`interface/schema.py:388-390`) or `ExtraColumnError`
   (`interface/schema.py:448`). Silent narrowing happens only at the ungated sites, or under
   `*_FORCED` / `NO_VALIDATION`. Two different bugs, two different reproductions — both real.
2. **There are 7 ungated sites, not 2.** Beyond `format_processor.py:410` and `:1302`, five
   structural reads are hard-wired to `CGM_SCHEMA`: `:168`, `:483`, `:680`, `:1081`
   (`to_data_only_df`'s own narrowing), `:1230`. The "24 sites" figure counts the import; it is 23
   usages.
3. **`derive_schema` is not publicly exported** — from neither `cgm_format` nor
   `cgm_format.interface`. RM8's headline idiom is unreachable from the public API today.
4. **`UNIFIED_CGM` is currently first** in `FORMAT_DETECTION_PATTERNS` (`formats/supported.py:31`),
   so `UNIFIED_EXTENDED` goes at the very front of the dict — not merely "before `UNIFIED_CGM`".
5. **`get_stable_sort_keys()` returns every column** (`interface/schema.py:224`), so widening the
   schema silently widens the total ordering. Confirms the `CLAUDE.md` §2.1 test obligation.
6. **`exercise` declares unit `"seconds"`** (`formats/unified.py`) while `UNIT_CONVERSIONS` keys use
   `"s"` — a latent no-op lookup. Fold the fix in.

---

## Decisions taken

| # | Decision | Rationale |
|---|---|---|
| D1 | Processor learns its schema from a `ClassVar`; ship `ExtendedFormatProcessor` with it soldered in | The processor is vendor-agnostic and cannot look anything up — it only ever sees a frame. Matches the existing `validation_mode` / `detection_line_count` ClassVar idiom and leaves the `CGMProcessor` ABC signatures untouched |
| D2 | Parser picks its target schema from a **registry**, not a ClassVar | The parser *does* know the vendor. `UNIFIED_TARGET_SCHEMA[format]` keeps the charter's "adding a vendor means adding registry entries" property |
| D3 | `parse_corpus(root) -> dict[str, UnifiedFormat]`, eager | Fits CGMacros (45) and D1NAMO (29). The iterator form is what Loop needs, and deferring Loop is what makes deferring it cheap. Adding it later is a new sibling, not a retype |
| D4 | Multi-track sources yield **two complete sibling frames**; non-EGV rows (meals, macros, wearables, annotations) are **replicated into both** | Tracks are alternative views, never shards |
| D5 | `parse_file()` on a multi-track file **raises**, naming both tracks and the API that returns them | Preserves `parse_file -> UnifiedFormat`; no silent sensor pick |
| D5a | A synthetic `track="mean"` is available **opt-in only**, and every merged row is flagged `Quality.TRACK_MERGE` | Averaging emits a number no device produced, so it must never be a default and must never be indistinguishable from a reading. See below |
| D6 | D1NAMO healthy subset **included**, fingersticks → `CALIBRAT` | Honest — the same mapping Libre strip readings already use. Never presents a fingerstick as a sensor trace |
| D7 | Extended schema **declares** `heart_rate`, `breathing_rate`, `acceleration`, `steps`; Zephyr waveform→column conversion is **deferred until requested** | 250 Hz waveforms are beyond this format and beyond current glucose-prediction models; averaging ECG is meaningless. HR/BR/accelerometer are averageable and belong in the schema now. Declining loudly in the docs is a decision; omitting quietly is a gap someone re-opens |
| D8 | `ketones` ships on the extended schema (mmol/L, **not** routed through `_glucose_to_canonical`) | Clinical ketones are already mmol/L; borrowing the glucose convention applies one analyte's rule to another |

D7 means only **2 of D1NAMO's 6 zips** are needed (`diabetes_subset_pictures-glucose-food-insulin`,
`healthy_subset_pictures-glucose-food`) — far less than the 11.2 GB full corpus.

---

## Architecture

### Wave 1 — extended schema (RM8, RM6)

`formats/unified.py`: add `CGM_SCHEMA_EXTENDED = derive_schema(CGM_SCHEMA, append_data_columns=(...))`
**beside** `CGM_SCHEMA`, which is not modified. `derive_schema`
(`interface/schema.py:522`) already accepts `append_data_columns` and needs no change — it is used
this way at `formats/libre_eu.py:125`. Export it from `cgm_format` and `cgm_format.interface`.

Appended data columns, in order (this order **is** the widened total ordering):

- food — `calories` (kcal), `protein` (g), `fat` (g), `fiber` (g)
- wearable — `heart_rate` (bpm), `breathing_rate` (breaths/min), `acceleration` (g), `mets`,
  `activity_calories` (kcal), `steps` (count)
- analyte — `ketones` (mmol/L)
- `annotations` (`Utf8`, JSON object), last

`annotations` is a **data** column, not a service column: it therefore joins `primary_key` and the
sort keys, which is what keeps two annotation-only rows at the same timestamp distinguishable (1,553
such rows exist in CGMacros). Cost: it appears in `get_polars_schema(data_only=True)` output —
`to_core_df()` is the narrowing escape hatch. Keeping it a data column also means `derive_schema`
needs no new `append_service_columns` kwarg.

**Deterministic JSON is load-bearing**, not a nicety: `annotations` participates in the sort keys and
in the byte-level round-trip guarantee. Sorted keys, stable float formatting, one serialization
helper used everywhere.

The seam:

- `FormatProcessor.schema: ClassVar[CGMSchemaDefinition] = CGM_SCHEMA`; replace all 23 `CGM_SCHEMA`
  usages in `format_processor.py` with `cls.schema` — **including all 7 ungated sites**, which is the
  half the roadmap under-counted.
- `ExtendedFormatProcessor(FormatProcessor)` with `schema = CGM_SCHEMA_EXTENDED`.
- `format_parser.py:276` (`_postprocess_unified`, ungated by design) takes a `schema` argument,
  defaulted from the new `UNIFIED_TARGET_SCHEMA` registry keyed on the format being parsed.
- `to_csv_string` / `to_csv_file` (`format_parser.py:1496`, `:1511`) become classmethods so they can
  see the target schema. Call sites are unchanged.
- Register `UNIFIED_EXTENDED` **first** in `FORMAT_DETECTION_PATTERNS`, with `annotations` as the
  discriminating pattern; add matching `SCHEMA_MAP`, `FORMAT_DETECTION_LINE_COUNT` and
  `KNOWN_ISSUES_TO_SUPPRESS` entries — `tests/test_format_detection_validation.py:255` indexes the
  last dict directly and `KeyError`s without one.

RM6 note — **the column ships, the item stays open.** 0.10.0 adds `ketones` to the extended schema and
settles the unit (D8) and the merge policy (one series, raw source column recorded in `annotations`).
It does **not** settle RM6's first decision: what `event_type` a ketone row carries. A column with no
event type leaves a ketone row looking like an empty glucose event, which is the failure RM6 names.
That needs a real export with populated ketone cells to implement against — every Libre fixture in the
tree carries the headers with zero populated cells, and `CLAUDE.md` §2 forbids implementing against an
invented row. So RM6 is **narrowed, not closed**: "event type + parser mapping, blocked on a real
fixture." Calling it shipped would overstate what 0.10.0 did.

### Wave 2 — source categories (RM12)

- `FormatCategory` enum (`EXPORT` / `BUNDLE` / `CORPUS`) plus a
  `FORMAT_CATEGORY: dict[SupportedCGMFormat, FormatCategory]` sidecar in `formats/supported.py`.
  Not a field on `SupportedCGMFormat` — that would change a public enum's shape for no gain.
- **Path-shaped detection** as a second mechanism beside the text one: `detect_path_format(path)`,
  driven by a `PATH_DETECTION_PROBES` registry of **glob patterns only**. Data, never callables —
  `docs/NEW_SCHEMA.md` is explicit that schemas and registries stay pure data. Ordered, first match
  wins, mirroring `detect_format`'s contract; raises `UnknownFormatError` on no match.
- `parse_bundle(paths) -> UnifiedFormat` as a concrete classmethod on `CGMParser`
  (`interface/cgm_interface.py:265-357`, where `parse_file`/`parse_from_bytes` already live — they are
  concrete, so no subclass breaks). Generalize `from_nightscout_exports`
  (`format_parser.py:1413`) into it and keep the old name as a thin wrapper; it is public API and
  retaining it is free. Its ignored `profile_path` becomes either used or documented as unused.

### Wave 3 — faceted output (RM9)

- `parse_tracks(path) -> dict[str, UnifiedFormat]` and `parse_corpus(root) -> dict[str, UnifiedFormat]`,
  both concrete on `CGMParser`. `parse_corpus` is built out of `parse_bundle` / `parse_tracks` — the
  composition is most of the value of naming the categories.
- Corpus keys are flat composite strings for multi-track corpora: `"CGMacros-001/libre"`. Keeps the
  return type a plain `dict[str, UnifiedFormat]` rather than a nested mapping. The `/` separator is
  part of the public contract — document it, and note that subject ids may contain `_` but never `/`
  (D1NAMO's `012_diabetes` is exactly why the separator cannot be `_`).
- `MultiTrackSourceError` (new typed exception in `interface/cgm_interface.py`) raised by
  `parse_file` per D5, naming both tracks and `parse_tracks`.
- Identity never enters the frame. The reason is mechanical and belongs in the docstrings:
  `format_parser.py:271` sorts by `datetime`, so many subjects in one frame interleave, and
  `detect_and_assign_sequences` (`format_processor.py:1177`) then splices them into shared sequences
  with nothing raised.
- **Tracks are alternatives, never shards** goes in the docstrings and `docs/UNIFIED_FORMAT.md`, not
  only the design doc: concatenating two track frames double-counts every meal, silently.

**The synthetic `mean` track (D5a).** `parse_file(path, track="mean")` returns a per-timestamp average
of the two sensor series. It is opt-in and never a default, because a mean is a value **no device
produced** — and this library's job is to move numbers a device already produced. Two properties make
it honest rather than a quiet fabrication:

- **A new `Quality.TRACK_MERGE` flag** (next bit, 32) on every row whose value was synthesized from
  two readings. Without it a merged value is indistinguishable from a real one in the frame, which is
  exactly the visibility `IMPUTATION` exists to provide — and a two-sensor mean is not imputation, so
  it needs its own bit. Adding a `Quality` member automatically widens the Frictionless enum
  constraint, which is computed from the Python enum at import time (`formats/unified.py`).
- **The estimator changes identity across the series, and the docstring must say so.** `Libre GL` is
  populated on ~every row, `Dexcom GL` on ~92%, so a mean series is mean-of-two on most rows and
  single-sensor on the rest. Averaging two independent readings shrinks noise variance; a single
  reading does not. Rows with only one contributing sensor are **not** flagged `TRACK_MERGE` — they
  are that sensor's reading — which makes the composition inspectable per row rather than something
  the caller has to infer.

`CGM_SCHEMA` is untouched by this and no existing flag changes meaning, so it stays **minor**. Note
this is the one place the branch synthesizes a glucose value; `docs/PHILOSOPHY.md` should say what it
is and why it is opt-in, in the same change.

### Wave 4 — CGMacros (RM10)

All 45 subjects are already on disk at `sugar-sugar/data/cgmmacros/CGMacros/`, so this is buildable
against real data today. Tests locate it via configuration, **never** a hardcoded `../sugar-sugar/`
path, and skip when absent.

Header drift is the whole job: **9 distinct header variants across 45 subjects**. `METs` vs
`Intensity` (11 subjects) and the trailing-space `Amount Consumed ` are `aliases` + `normalize_headers`
work; the 2 subjects with **no** `Amount Consumed` column need typed nulls, because a
`.select(pl.col(X))` is evaluated even when an upstream filter leaves zero rows — the `ColumnNotFound`
gotcha in `docs/NEW_SCHEMA.md` that used to crash the LibreView insulin sub-frame. Do not guard each
select.

Also: `METs` is stored ×10 (divide, with a comment saying why); annotation-only rows use `OTHEREVT`
rather than an invented code; native cadence is 1-minute and **the parser does not resample** —
callers pass `expected_interval_minutes=1`, and the docstring must name the `small_gap_max_minutes`
caveat (it defaults to 15, which is 15× the interval rather than the intended 3×).

Fold in the pre-existing CLI bug here, since it is what makes any new format visible:
`cgm_cli.py:1127` hardcodes `[UNIFIED_CGM, DEXCOM, LIBRE]`, so `DEXCOM_EU`, `LIBRE_EU`, `MEDTRONIC`
and `NIGHTSCOUT` are **already** omitted from `cgm-cli report` today. Iterate `SCHEMA_MAP` instead.
`cgm_cli.py:344` also hard-validates against `CGM_SCHEMA` regardless of input format — route it
through the target schema.

### Wave 5 — D1NAMO (RM13)

Two registered formats, not one with a flag: the subsets differ in `food.csv`'s header, in
`insulin.csv` / `annotations.csv` presence, and in a disjoint `glucose.csv` `type` vocabulary.
`derive_schema` expresses renames and units, not a different column set.

The bundle category's motivating case: each subject is a directory of files, one per modality.
Traps that must be encoded as tests, not comments — mixed timestamp conventions **inside one subject
directory** (ISO in `glucose.csv`, EXIF-style `2014:10:01 19:27:49` in `food.csv`, day-first in the
Zephyr streams); the literal `012_diabetes` directory name; mmol/L routed through the declared unit;
dangling photo references reported separately from absent ones ("said something we cannot resolve" is
a different report from "did not say"); and the real dirt — `No information`, `08.2`, `8 Balance""`,
commas inside free-text `description`, stray `.DS_Store`.

Downloader is a `scripts/` utility behind the `dev` extra, **never shipped** — a static published
archive is a developer setup chore, not a runtime feature.

### Wave 6 — CLI, docs, version

- New `cgm-cli corpus <root> --out <dir>` rather than overloading `parse` or `batch`; `batch` already
  globs a directory with a different meaning (independent files, one output each). `cgm-cli detect`
  learns path detection.
- **`docs/PHILOSOPHY.md` is amended in the same change, before the code** (`CLAUDE.md` §8: policy
  first, code complies). Two claims move:
  - `:49` "A new sensor format requires zero changes to the processor" — still true per *vendor*, and
    the boundary is intact (the processor still knows no vendors); reword to say the processor is now
    schema-parameterized.
  - `:143` "The processor, schema validation, CLI, and all downstream consumers require zero
    changes" — **already false today** because of `cgm_cli.py:1127`. Correct it rather than leave a
    doc that is known to be wrong. Same coda in `docs/NEW_SCHEMA.md`.
- `docs/UNIFIED_FORMAT.md` gains the extended schema and the tracks warning; `docs/USAGE.md`,
  `docs/PIPELINE.md`, `README.md` gain the new entry points; `docs/RESEARCH_CORPORA.md` loses
  "Nothing here is implemented yet".
- RM8, RM9, RM10, RM12, RM13 **relocate** to `docs/ROADMAP_HISTORY.md` in the RM5 template
  (`Status: shipped <date> in 0.10.0`, legality line, "What shipped", "Left open, not this item").
  Nothing is deleted. RM6 stays open, **rewritten** to its narrowed remainder (event type + parser
  mapping, blocked on a real fixture). RM7 and RM11 stay open with their blocking updated.
- `docs/CHANGELOG.md` gains `## 0.10.0 — <date>` at the top; `pyproject.toml` version bumps **in the
  feature commit**, matching this repo's history. Next finding id is **F3**; next feedback id is
  **S1**.
- **Tagging, pushing and branch creation are yours** — I will not run tree operations.

---

## Critical files

| File | Change |
|---|---|
| `src/cgm_format/formats/unified.py` | `CGM_SCHEMA_EXTENDED`; `Quality.TRACK_MERGE`; `exercise` unit `"seconds"`→`"s"` |
| `src/cgm_format/format_processor.py` | `schema` ClassVar; 23 `CGM_SCHEMA` → `cls.schema` incl. the 7 ungated; `ExtendedFormatProcessor`; `to_core_df()` beside `to_data_only_df` (`:1032`) |
| `src/cgm_format/format_parser.py` | schema-aware `_postprocess_unified` (`:276`); `parse_bundle` generalized from `:1413`; `parse_tracks`; `parse_corpus`; `_process_cgmacros`, `_process_d1namo_*`; dispatch arms at `:212-229` |
| `src/cgm_format/interface/cgm_interface.py` | `SupportedCGMFormat` members; `FormatCategory`; `MultiTrackSourceError`; concrete `parse_bundle`/`parse_tracks`/`parse_corpus` beside `:265-357` |
| `src/cgm_format/interface/schema.py` | export `derive_schema`; no behavior change to `:408-412` |
| `src/cgm_format/formats/supported.py` | `UNIFIED_EXTENDED` **first** in `:31`; `FORMAT_CATEGORY`, `UNIFIED_TARGET_SCHEMA`, `PATH_DETECTION_PROBES`; entries in all four existing dicts per new format |
| `src/cgm_format/formats/{cgmacros,d1namo_diabetes,d1namo_healthy}.py` | new vendor modules per the `docs/NEW_SCHEMA.md` 9-step checklist |
| `src/cgm_format/cgm_cli.py` | `corpus` command; fix `:1127` and `:344` |
| `src/cgm_format/__init__.py` | import block **and** `__all__` for every new symbol |

## Verification

```bash
uv sync --extra dev
uv run pytest -vvv tests/test_schema.py tests/test_package_exports.py   # wave 1
uv run pytest -vvv tests/test_cgmacros.py tests/test_d1namo.py          # waves 4-5
uv run pytest                                                           # full, ~15 min, 600s+ timeout
uv run cgm-cli detect|parse|validate|report <synthetic fixture>
uv run cgm-cli corpus <corpus root> --out data/cli_test_output/
```

Tests that must exist, each demonstrating a property rather than a count:

- **New total ordering** — the widened `get_stable_sort_keys()` and `primary_key` get an explicit
  test (`CLAUDE.md` §2.1), including deterministic `annotations` JSON bytes across two runs.
- **Extended round-trip and idempotency** — extend `tests/test_roundtrip_datetime.py` and
  `tests/test_idempotency.py`; an extended frame must survive CSV → parse → CSV byte-identically, and
  the `UNIFIED_EXTENDED`-before-`UNIFIED_CGM` ordering is what that test actually proves.
- **Both narrowing failure modes** — an extended frame through the core `FormatProcessor` raises
  under the default `INPUT` mode, and is narrowed at the ungated sites under `*_FORCED`. Demonstrate
  each against the pre-fix code before claiming either is fixed.
- **Cross-detection** — every existing fixture still detects as its own format after
  `UNIFIED_EXTENDED` moves to the front of the registry.
- **Exhaustive `__all__`** — `tests/test_package_exports.py:190-202` currently checks only that four
  names are present; make it an equality assertion so a new entry point cannot be added to
  `__init__.py` and forgotten in `__all__`.
- **Tracks are alternatives, not shards** — assert the misuse is real and quantified: concatenating
  the two CGMacros track frames doubles the meal count and the carbohydrate total exactly, with
  nothing raised. Compute both from the fixture at runtime; this is the design's most likely misuse
  and it deserves a test, not only a docstring.
- **`parse_file` raises on multi-track** — `MultiTrackSourceError` names both tracks and
  `parse_tracks`, and the existing single-track fixtures keep working unchanged.
- **The `mean` track is auditable** — on a real CGMacros subject, every row flagged `TRACK_MERGE` has
  both sensors populated and its value lies between them; every unflagged glucose row has exactly one
  sensor populated and equals it. Assert the relationship, not a count. Plus: `TRACK_MERGE` survives
  a round-trip, and `mean` is absent from `parse_tracks`' output (it is a synthetic view, not a
  member of the corpus).
- **Synthetic fixtures reproduce the dirt, not the happy path** — CGMacros: an `Intensity` variant, a
  file with no `Amount Consumed`, annotation-only rows, dirty `Meal Type` spellings. D1NAMO: one
  subject directory per subset, a `012_diabetes`-style name, the EXIF-colon timestamp, an empty
  `food_pictures/`, a dangling `picture` reference. Committed with `git add -f` (finding F2).
- Real corpora stay local behind the skip-if-absent pattern (`tests/test_libre_eu.py:41-55`), located
  by configuration, never a hardcoded sibling-repo path.

## Explicitly out of scope

RM7 (lazy `scan_csv`) · RM11 (Loop) · RM3 (`data/` → `assets/` layout reorg — its own task, never
smuggled) · Zephyr waveform→column conversion (columns ship, conversion waits) · corpus downloaders
inside the package · resampling in any parser · zip extraction in the core (Loop's deflate64 is
user-side regardless).
