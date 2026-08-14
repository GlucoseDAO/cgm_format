# Roadmap

Active-only, forward-only. One `## RMn — name` section per **open** item, each with a
severity/status/owner line. Shipped items move to [ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) with their
rationale — nothing is deleted, only relocated.

Severity orders the queue; it does not size the release. Legality does that (`CLAUDE.md` §8).

---

## RM1 — schema-regen helper prints instead of logging

**Severity:** low · **Status:** open · **Owner:** unassigned

`interface/schema.py:477` does `print(f"✓ Regenerated {schema_file}")`. The house rule is
standard-library `logging` for library diagnostics; `print` belongs only to CLI output the user asked
to see. The helper is called from `scripts/regenerate_all_schemas.py`, so the honest fix is either a
`logger.info` here, or moving the message to the script that owns the user-facing output.

Legality: pure legibility, so a patch.

---

## RM2 — hardcoded version fallback in `__init__.py`

**Severity:** low · **Status:** open · **Owner:** unassigned

`src/cgm_format/__init__.py:27` carries `__version__ = "0.9.0"` as a fallback when
`importlib.metadata.version("cgm-format")` raises. That is a second source of truth for the version,
and it is the one that gets read when the package is not installed — exactly when it is most likely
to be stale.

The fix is an editable install (`uv sync`) so metadata is always present, then dropping the literal.
Worth confirming first that nothing in CI or the docs build imports the package without installing
it.

Legality: patch.

---

## RM3 — `data/` layout does not match the house convention

**Severity:** low · **Status:** open, deliberately deferred · **Owner:** unassigned

The house layout puts committed fixtures in `assets/` and reserves `data/` for git-ignored
`input/` · `interim/` · `output/`. This repo commits `data/input/` and writes generated output to
`data/parsed/`, `data/cli_test_output/` and `data/cli_examples_output/`.

Deferred on purpose, and out of scope for any task that is not this task: every integration test
path, the `.gitignore` ignore-all (F2: `input/` is not allowlisted), the sdist exclusion and `tests/conftest.py`'s
Nightscout download target are built on the current names and all work today. Doing it properly
means one change that moves the fixtures, updates every test path, rewrites the allowlist, and fixes
the docs that name these directories — planned as a unit, not smuggled into an unrelated edit.

Legality: no effect on anything a consumer imports, so a patch — but it touches `data/input/` paths
that examples and docs quote, so the docs move with it.

---

## RM4 — the CLI does not load `.env`

**Severity:** low · **Status:** open · **Owner:** unassigned

`tests/conftest.py` loads `.env` via `python-dotenv`, so `NIGHTSCOUT_URL` / `NIGHTSCOUT_TOKEN` /
`NIGHTSCOUT_API_SECRET` reach the test suite. `cgm_cli.py` does not, so the same credentials that
make tests pass do nothing for `cgm-cli`. The house rule is that every CLI loads `.env` at startup
before reading configuration, with the values documented in a `.env.template`.

Two things to settle before writing code: `python-dotenv` currently sits in `dev`, not `cli`, so
adopting this moves it (or adds it) to the `cli` extra; and the credential-neutralizing rule from
`CLAUDE.md` §6 applies — a test asserting "no credential" must `setenv(VAR, "")`, never `delenv`,
because `load_dotenv(override=False)` skips a key that is merely present.

Legality: additive (a new optional config path), so minor.

---

## RM6 — parse Libre ketone readings into UnifiedFormat

**Severity:** medium · **Status:** open · **Owner:** unassigned

Libre vendor schemas already declare ketone columns: `Ketone mmol/L` on `LIBRE_SCHEMA`, plus
`Historic Ketone mmol/L` and `Scan Ketone mmol/L` appended on `LIBRE_EU_SCHEMA`. `_process_libre`
never selects them. `CGM_SCHEMA` has no ketone field, so `validate_dataframe(enforce=True)` would
drop any extra column the parser tried to smuggle through. The values are gone after Stage 3.

The JonGrove mmol/L export (local, gitignored) carries all three headers and **zero populated
cells**. The committed Libre fixtures are the same. There is no real ketone reading in the tree.
Do not implement against an invented row.

Left out of 0.9.0 on purpose: that release put the columns on the *vendor* schema so Frictionless
accepts the 21-column header. Putting them on the *unified* schema is a shape a consumer reads —
a design, not a parser follow-up.

**Decisions to settle before writing code** (do not infer):

1. **Shape.** A new optional data column (`ketones`, null = the vendor did not say) vs a new
   8-char `UnifiedEventType` vs both. A column without an event type leaves a ketone row looking
   like an empty glucose event; an event type without a column has nowhere to put the number.
   Reusing `glucose` to hold ketones redefines a column a consumer already reads — that would be
   major and silent.
2. **Canonical unit.** Clinical ketones are already mmol/L. Routing them through
   `_glucose_to_canonical` / `CANONICAL_GLUCOSE_UNIT` applies a glucose convention to a different
   analyte. Stay mmol/L, or name a ketone target in `UNIT_CONVERSIONS` — pick one, don't borrow.
3. **Historic vs scan vs the older `Ketone mmol/L` column.** Merge into one series (as scan
   glucose merged into `EGV_READ`) or keep distinct event types. Merging re-opens F1 on a new
   analyte.
4. **Fixture.** Need a real export with nonempty ketone cells. Skip-if-absent until then.

**Legality:** minor if we *add* a column and/or event type. Major if we reuse an existing unified
column.

---

## RM7 — lazy `scan_csv` ingest path

**Severity:** medium · **Status:** open · **Owner:** unassigned

Every public entry point funnels through `parse_file → bytes → decode_raw_data → str →
detect_format`, which materializes the entire file as a Python string before parsing starts. Ingest
is therefore capped at whatever fits in memory, and the cap is silent — there is no size at which the
library says "use the streaming path", because there isn't one.

This is worth doing on its own merits (multi-year Dexcom exports are already the slowest thing in the
test suite), and it is a hard blocker for the Loop corpus, where a single `LOOPDeviceCGM*.txt` is
about 2.3 GB. See [RESEARCH_CORPORA.md](RESEARCH_CORPORA.md).

Shape: a `pl.scan_csv`-based entry point sitting **beside** the eager ones, not replacing them, with
`sink_parquet` for output. The eager path stays the default and stays simplest — most vendor exports
are a few megabytes and lazy evaluation buys them nothing.

**Decisions to settle before writing code:**

1. **Detection on a prefix.** `detect_format` currently takes the whole decoded text. A lazy path has
   only the first N lines, which is all detection actually reads (`DETECTION_LINE_COUNT`) — so this
   is a refactor of the *signature*, not the logic. Confirm that before assuming it's free.
2. **Where `decode_raw_data` fits.** BOM and encoding-artifact normalization only runs on `bytes`
   (`CLAUDE.md` §12). A streaming reader hands Polars a path, not bytes, so either the artifacts are
   handled by a prefix sniff or the lazy path documents that it does not fix them. Do not let it
   silently skip normalization.
3. **What the lazy path returns.** A `LazyFrame` gives the caller the streaming win but breaks the
   `UnifiedFormat = pl.DataFrame` alias every signature uses. Collecting internally is honest but
   forfeits most of the benefit for large inputs.

**Legality:** a new entry point beside the existing ones, nothing removed or retyped, so **minor**.

---

## RM8 — extended unified schema (macros, wearables, annotations)

**Severity:** medium · **Status:** open · **Owner:** unassigned

`CGM_SCHEMA` has six data columns. Research corpora carry channels it has no home for: macronutrient
decomposition (protein, fat, fiber, calories), wearable streams (heart rate, METs, activity calories,
steps), and free-form payloads such as a meal-photo path. Today `validate_dataframe(enforce=True)`
drops anything a parser tries to smuggle through, which is why RM6 exists for ketones and why
CGMacros' entire reason for existing would be discarded at Stage 3.

Proposed shape: `CGM_SCHEMA_EXTENDED = derive_schema(CGM_SCHEMA, append_data_columns=(...))`, with
`annotations` as a `Utf8` column holding a JSON object. `CGM_SCHEMA` itself is **not** modified, and
`ExtraColumnError` is **not** relaxed — that error guards a real invariant, and weakening it to make
extended frames pass the core check trades a correctness check for ergonomics. A `to_core_df()`
narrowing helper sits beside the existing `to_data_only_df()`.

**The substance of this item is not the schema file.** `format_processor.py` refers to `CGM_SCHEMA`
at 24 sites. Most are gated on `validation_mode` and fail loudly. Two are not:

- `format_processor.py:410` — `CGM_SCHEMA.validate_columns(result, enforce=True)`
- `format_processor.py:1302` — `CGM_SCHEMA.validate_dataframe(result_df, enforce=True)`

Both enforce unconditionally, and enforcement drops extra columns (`interface/schema.py:410`,
`dataframe.select(expected_columns.keys())`). An extended frame handed to the processor comes back
**silently narrowed** mid-pipeline — no warning, no error. Threading a `schema` parameter through
`FormatParser._postprocess_unified` and the `FormatProcessor` classmethods is the work; a `ClassVar`
default keeps every existing call site unchanged.

**Also load-bearing:**

- **Detection order.** `UNIFIED_EXTENDED` must register **before** `UNIFIED_CGM` in
  `FORMAT_DETECTION_PATTERNS`, with a pattern unique to it (`annotations`). An extended round-trip
  CSV also contains `sequence_id`, `event_type` and `quality`, so without the ordering it detects as
  `UNIFIED_CGM` and `_process_unified` drops every extended column on re-read. Same rule as
  `DEXCOM_EU` before `DEXCOM` (`formats/supported.py:26-30`).
- **Deterministic JSON.** `annotations` serializes with sorted keys and stable float formatting, or
  two runs produce different bytes for identical data and the byte-level round-trip and idempotency
  guarantees go flaky rather than false.
- **New ordering, new test.** The widened `get_stable_sort_keys()` and `primary_key` are a new total
  ordering, so per `CLAUDE.md` §2.1 they get an explicit test.

Note this partially subsumes RM6: a ketone column on the extended schema is a smaller decision than a
ketone column on `CGM_SCHEMA`. It does not close RM6 — the unit and the historic/scan merge policy
are still open — but it changes what that item is choosing between.

**Legality:** `CGM_SCHEMA` untouched, all new columns on a new schema, nothing removed or retyped, so
**minor**.

---

## RM9 — faceted parse output (multi-track, multi-subject)

**Severity:** medium · **Status:** open · **Owner:** unassigned

The library assumes one file yields one frame. Two real datasets break that: CGMacros records two CGM
sensors simultaneously, and Loop puts ~1000 participants in one shared file set. Both are the same
shape — a source yields N frames differing by a *facet* (device, person), not by content.

Identity belongs in the mapping key, never in a column. The reasoning is mechanical and is written up
in [RESEARCH_CORPORA.md](RESEARCH_CORPORA.md): `_postprocess_unified` sorts by `datetime`, so many
subjects in one frame interleave, and `detect_and_assign_sequences` then splices them into shared
sequences without erroring. Making a `subject_id` column safe means teaching the vendor-agnostic
`FormatProcessor` to group by it everywhere, which contradicts the zero-processor-changes boundary in
[PHILOSOPHY.md](PHILOSOPHY.md).

Precedent: `from_nightscout_exports()` (`format_parser.py:1414`) is already a dataset-level entry
point that takes several files and bypasses `detect_format`.

**Decisions to settle before writing code:**

1. **What `parse_file` does with a multi-track source.** Recommended: return the densest track with a
   prominent warning naming the withheld track *and* the API that returns it — the sanctioned "warn
   prominently, naming the substitute" path in `CLAUDE.md` §2. A hard raise is more explicit but
   breaks `cgm-cli parse|pipeline <file>` out of the box for every user who just wants a look at the
   data. A `track=` override covers the explicit case.
2. **Return type, which differs by scale.** CGMacros' 45 subjects fit an eager
   `dict[str, UnifiedFormat]`. Loop's ~1000 participants across ~20 GB cannot — that wants an
   iterator of `(key, frame)` or a `PtID`-partitioned Parquet writer. Same concept, two ergonomics;
   pick both deliberately rather than discovering the second one under time pressure.
3. **Whether the CLI grows a facet flag.** `cgm-cli parse` writing one output file per track or
   subject is a different output contract than it has today.

**Tracks are alternatives, never shards** — the rows belonging to neither device (meals, wearables,
annotations) are replicated into both frames, so concatenating two tracks double-counts every meal
with nothing raised. This goes in the docstrings and in `docs/UNIFIED_FORMAT.md`, not only in the
design doc.

Relationship to RM12: that item handles cardinality on the **input** side (many files in), this one
the **output** side (many frames out). They are orthogonal, but decision 2 here and decision 3 there
are the same question asked twice — eager mapping versus iterator — and must be answered together.

**Legality:** new entry points, existing ones unchanged, so **minor**.

---

## RM10 — CGMacros format support

**Severity:** medium · **Status:** open, blocked on RM8 + RM9 · **Owner:** unassigned

PhysioNet `cgmacros/1.0.0` — CC BY-NC-SA, open access, no credentialing. 45 subjects × 10 days, two
simultaneous CGM series, macronutrients, Fitbit wearable channels and meal photographs. `sugar-sugar`
already ships `download-cgmacros` and has nowhere to send the result. Full ground truth, read off the
real files, is Appendix A of [RESEARCH_CORPORA.md](RESEARCH_CORPORA.md).

The trap that matters most: **the header is not stable across the 45 subjects — 9 distinct
variants**, including `METs` vs `Intensity` (11 subjects), a `Sugar` column on one, `Steps` +
`RecordIndex` in place of `Calories (Activity)` on another, and two subjects with no
`Amount Consumed` column at all. A parser written and tested against subject 001 will break on
roughly a quarter of the corpus. Most of this is `aliases` work; the absent columns are not.

Other things that will bite:

- **The `.select()` gotcha** (`docs/NEW_SCHEMA.md`, *Gotchas*): a select is evaluated even when an
  upstream filter leaves zero rows, so the two subjects missing `Amount Consumed` raise
  `ColumnNotFound` regardless of whether any row would have used it. Absorb with `aliases` +
  `normalize_headers`, and add genuinely absent columns as typed nulls — do not guard each select.
- **`METs` is stored ×10** per the data dictionary. Divide, with a comment saying why.
- **Annotation-only rows.** 1,553 rows carry a meal-end photo with no meal attached, against 1,644
  with both. Such a row must survive `_postprocess_unified` with no glucose and no carbs; use
  `OTHEREVT` rather than inventing an event code.
- **1-minute cadence.** `expected_interval_minutes` is already a parameter throughout
  (`format_processor.py:103`, `:177`, `:212`), so callers pass `1`. But `small_gap_max_minutes`
  defaults to 15, which is 15× the interval rather than the intended 3×
  (`interface/cgm_interface.py:33-37`). Document the caveat and name the right arguments in the entry
  point's docstring. Do **not** resample in the parser — that is the processor's job and is lossy.
- **Dirty meal vocabulary.** Ten raw spellings for four meals. Normalize case and plural; keep the
  raw string in `annotations` so the normalization stays inspectable.

**Fold in a pre-existing bug here**, since it is what makes any new format visible in the CLI's
report: `cgm_cli.py:1127` hardcodes `[UNIFIED_CGM, DEXCOM, LIBRE]` in the per-format breakdown, so
`DEXCOM_EU`, `LIBRE_EU`, `MEDTRONIC` and `NIGHTSCOUT` are **already** silently omitted from
`cgm-cli report` today. Iterate `SCHEMA_MAP.keys()` instead. This contradicts the "the CLI needs no
changes" claim in both `docs/NEW_SCHEMA.md` and `docs/PHILOSOPHY.md`; correct that sentence in the
same change rather than leaving a doc that is now known to be false.

**Legality:** a new format plus new optional columns, nothing removed, so **minor**.

---

## RM11 — Loop dataset support

**Severity:** low · **Status:** open, blocked on RM7 + RM9 · **Owner:** unassigned

The Loop observational study (NCT03838900, Jaeb Center for Health Research), dataset 560. Roughly a
thousand participants of automated insulin delivery data — the largest real-world AID corpus
available to us.

Lower priority than RM10 for practical reasons, not scientific ones: it is DUA-gated, needs ~20 GB on
disk before a single test runs, and is hard-blocked on RM7. Appendix B of
[RESEARCH_CORPORA.md](RESEARCH_CORPORA.md) has what we know.

**Read `DataGlossary.rtf` from inside the archive before writing any code.** The column lists we
currently hold were reconstructed from third-party parsers' `usecols` arguments, not from the
authoritative file. Do not implement against them.

Constraints already settled:

- **No zip handling in the library.** The archive is deflate64, which the standard library cannot
  open, and the workaround is a third-party package. That must not touch the Polars-only core —
  extraction is a documented user-side step, or a `scripts/` helper at most.
- **Keep `UTCDtTm` verbatim as `datetime`.** `TmZnOffset` is reportedly null for ~63% of patients, so
  reconstructing local time from `PtRoster.PtTimezoneOffset` is a substitution. Opt-in with a
  warning, never silent (`CLAUDE.md` §2).
- **Glucose is mmol/L.** Route it through the column's declared `unit` and `_glucose_to_canonical`,
  never a hardcoded 18.018 — that is what the units mechanism exists for.
- **Dedupe is a parser step, not a caller's problem.** A reported ~35% of CGM rows are exact
  duplicates. First-occurrence with deterministic sort keys, with its own test.
- **Carbs span two mutually exclusive eras** (`LOOPDeviceWizard.txt` to mid-2018,
  `LOOPDeviceFood.txt` after), overlapping in patients but not in time.

**Legality:** a new format, nothing removed, so **minor**.

---

## RM12 — source categories: export, bundle, corpus

**Severity:** medium · **Status:** open · **Owner:** unassigned

Every format the library supports is one shape — **one file, one subject, one device** — and that
assumption is nowhere written down, because until now nothing violated it. It is baked into
`parse_file(path) -> UnifiedFormat` and into `detect_format`, which matches patterns against a text
prefix of a single file.

Research datasets break it along two independent axes: how many *files* come in, and how many
*subjects* are in them. Naming the resulting categories is what lets RM10, RM11 and RM13 share one
mechanism instead of growing three bespoke entry points. Full write-up in
[RESEARCH_CORPORA.md](RESEARCH_CORPORA.md).

| Category | Input | Entry point |
|---|---|---|
| **Export** | one file, one subject | `parse_file(path) -> UnifiedFormat` |
| **Bundle** | several files, one subject, each a different modality | `parse_bundle(...) -> UnifiedFormat` |
| **Corpus** | many subjects | `parse_corpus(root) -> Mapping[str, UnifiedFormat]` |

Two properties carry the design. **The categories compose** — a corpus's member is a bundle or an
export, so `parse_corpus` is built out of `parse_bundle` rather than implemented separately. And
**the bundle category already exists, unrecognized**: `from_nightscout_exports()`
(`format_parser.py:1414`) takes entries + treatments + profile and merges them into one frame with a
diagonal concat, and `from_nightscout_url()` does the same across API endpoints. Both were written as
Nightscout special cases. Generalizing them is most of this item, and it is why the category also
serves **app APIs** — an API pull is several endpoints describing one user, which is a bundle.

**Decisions to settle before writing code:**

1. **Where the category lives.** Recommended: a `FormatCategory` enum plus a
   `FORMAT_CATEGORY: dict[SupportedCGMFormat, FormatCategory]` registry entry, matching the existing
   four-dicts-keyed-by-the-enum pattern in `formats/supported.py`. Adding a category *field* to
   `SupportedCGMFormat` instead would change a public enum's shape, which is a heavier change for no
   gain.
2. **Path-shaped detection.** `detect_format` sniffs decoded text and cannot identify a directory.
   Bundles and corpora are identified by **directory shape** — does `CGMacros-001/CGMacros-001.csv`
   exist, is there a `diabetes_subset/`, are there sibling `LOOPDeviceCGM*.txt`. This is a second
   detection mechanism keyed on paths sitting beside the text one, not another list fed to the
   existing loop. Settle whether it returns the same `SupportedCGMFormat` enum (recommended) and what
   it raises when a directory matches nothing.
3. **Corpus member enumeration.** Subject-per-directory corpora (CGMacros, D1NAMO) can list members
   from the filesystem; subject-as-key-column corpora (Loop's `PtID`) cannot without scanning ~111 M
   rows. This decides whether `parse_corpus` can return an eager mapping or must return an iterator,
   and it is the same question RM9 settles for its return type — the two items must agree.
4. **What the CLI does with a directory argument.** `cgm-cli parse <dir>` writing one output per
   subject is a different output contract than it has today, and `cgm-cli batch` already globs a
   directory with a different meaning (independent files, one output each).

**Corpus downloaders are `scripts/` utilities behind the `dev` extra, never shipped.** Fetching a
static published archive once is a developer setup chore, not a runtime feature of a consuming app,
and the archives are large enough (CGMacros 627 MB, Loop 1.63 GB) that shipping the capability
invites a library user to pull them by accident. `sugar-sugar` already does this correctly —
`download-cgmacros` lives in the consuming app. The existing `nightscout_downloader.py` is not a
counterexample: it fetches the user's **own live data** from their own server, which is a genuine
runtime feature.

Relationship to RM9: this item is about cardinality on the **input** side (many files in), RM9 is
about the **output** side (many frames out). They are orthogonal and both are needed — Loop is
many-in and many-out, D1NAMO is many-in and one-out per subject, CGMacros is one-in and two-out.

**Legality:** a new enum, a new registry dict, new entry points beside the existing ones; nothing
removed or retyped, so **minor**.

---

## RM13 — D1NAMO dataset support

**Severity:** medium · **Status:** open, blocked on RM8 + RM12 · **Owner:** unassigned

Zenodo `5651217` v1.2.0, CC BY-SA 4.0, open access, ~11.2 GB across six zips. 9 type-1 diabetes
subjects and 20 healthy ones, roughly four days each, with glucose, insulin, photographed meals and
continuous Zephyr BioHarness streams. Ground truth — verified by reading the archives, not from
secondary sources — is Appendix C of [RESEARCH_CORPORA.md](RESEARCH_CORPORA.md).

D1NAMO is the **motivating example for the bundle category** (RM12): each subject is a directory of
files, one per modality, and ECG arrives in a *separate archive* under an identical tree that must be
merged by session-directory name. It is also blocked on RM8, because D1NAMO records meal `calories`
and **no carbohydrates at all** — without a `calories` column its food rows carry nothing but a
timestamp and an annotation, so shipping it before the extended schema would mean parsing the corpus
and discarding its most distinctive channel.

**Decisions to settle before writing code:**

1. **Scope of the physiological streams.** ECG is 250 Hz — roughly 86 million samples per subject over
   four days. This is a CGM library whose unified schema is event- and reading-shaped at 5-minute
   cadence; ingesting waveforms is a different product. Recommended: parse the annotation bundle
   (`glucose.csv`, `insulin.csv`, `food.csv`) plus at most the 1 Hz `Summary.csv`, whose `HR` maps
   cleanly to the extended `heart_rate` column, and decline the waveforms explicitly in the docs
   rather than silently. Declining loudly is a decision; omitting quietly is a gap someone re-opens.
2. **Whether the healthy subset belongs here at all.** It contains no CGM — four to six fingersticks a
   day. Those are calibration-style readings and map honestly to `CALIBRAT`, the way Libre strip
   readings already do; mapping them to `EGV_READ` would misrepresent a fingerstick as a sensor trace.
   Settle whether a subset with no continuous glucose is in scope before writing a parser for it.
3. **One format or two.** The subsets differ by more than units — `food.csv` has a *different header*
   in each, `insulin.csv` exists only for diabetes, `annotations.csv` only for healthy, and the
   `glucose.csv` `type` vocabulary is disjoint (`cgm`/`manual` versus meal-relative `BB,AB,BL,AL,BD,AD`).
   `derive_schema` expresses renames and units, not a different column set, so this is probably two
   registered formats rather than one with a flag — but confirm against `derive_schema`'s actual
   capabilities before committing.

**Traps already identified:**

- **Mixed timestamp conventions inside one subject directory.** Annotation files are ISO year-first,
  Zephyr files are **day-first** `%d/%m/%Y %H:%M:%S.%f`, and the diabetes `food.csv` uses EXIF-style
  colons, `2014:10:01 19:27:49`. A parse without an explicit format silently misreads every Zephyr day
  ≤ 12 as a month. Probing tuples per file type, never one shared tuple.
- **Subject ids are not uniform.** The healthy subset's twelfth directory is literally
  `012_diabetes`. A three-digit assumption drops or mis-keys it; this belongs in a test.
- **Glucose is mmol/L** — route through the declared `unit` and `_glucose_to_canonical`, never a
  hardcoded 18.
- **Real dirt in the values**: `No information` as a literal, a `:` typed instead of `.` in a glucose
  reading, leading zeros (`08.2`), a corrupt `8 Balance""` in `balance`, free-text `description`
  fields containing commas, and stray `.DS_Store` files inside data directories.
- **Empty and dangling photo references.** Diabetes subject `005` has an empty `food_pictures/`
  directory, so a `picture` value may name a file that does not exist. Per `CLAUDE.md` §5 this is
  "the source said something we cannot resolve", which is a different report from "the source did not
  say" — do not collapse the two into one warning.
- `Summary.csv` carries sentinels that must not be read as measurements: `-3276.8`, `65535`, `-128`,
  `6553.5`. `BB.csv`/`RR.csv` were observed with an empty `Time` column on every row, checked in one
  session only — verify before relying on either.
- The paper is reported to state 18 Hz for breathing; the files measure 25 Hz. Trust the files, and do
  not re-derive this.

Per RM12, the downloader is a `scripts/` utility behind `dev`, never shipped. CC BY-SA's share-alike
is a further reason the committed fixture is synthetic rather than an excerpt.

**Legality:** a new format (or two), nothing removed, so **minor**.

---

## Idea book

Freeform, unsized, no commitment. An item here has not been triaged.

- A `--strict` / determinism mode for the CLI would be useful for reproducible dataset builds — with
  the caveat stated loudly in the docs that it means *reproducible*, not *correct*. A parser mapping
  the wrong vendor column to `glucose` passes every determinism check we have.
- Aggregate the per-row parse warnings by reason rather than by row before they reach a user running
  `cgm-cli parse` on a full multi-year export. Worth measuring the actual volume on the largest
  fixture first, rather than assuming it is already a problem.
- Nothing currently distinguishes "the vendor did not record a carb entry" from "the vendor recorded
  something the schema cannot hold" in the warning stream. They are different reports and a consumer
  reading the log cannot tell them apart.
