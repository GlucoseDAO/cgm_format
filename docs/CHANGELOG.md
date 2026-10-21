# Changelog

What actually shipped, newest first. Cross-repo integration changes made on our side are recorded
here too, so agents working in `sugar-sugar` or `glucosedao` are not surprised by a schema or CLI
change they did not ask for.

Version numbers are the `version` field of `pyproject.toml` at the commit. Entries before 0.8.4 were
reconstructed from git history on 2026-08-13 and are summaries of what the commits did, not
contemporaneous release notes — treat the commit as the authority where the two disagree.

Legality sizing (see `CLAUDE.md` §8): additive → minor, removal/retype/rename → major, legibility →
patch.

## Unreleased

### Drift in claimed formats (RM22)

- **LibreView, US Libre 3, 12-hour timestamps** (`MM-DD-YYYY hh:mm AM`) parse: a third format in
  `LIBRE_TIMESTAMP_FORMATS`, tried after the two 24-hour ones. Seen failing on a public export.
- **LibreView "Patient report" layout** parses: its header sits on line 3, after a report line and a
  patient line, where the standard export has one metadata line. The parser now finds the header row
  (first cell `Device`, naming `Device Timestamp`) in the first ten lines instead of assuming line 2,
  and says so when there is none. Seen failing on a public Libre Pro export; 1,339 readings now.

### Diagnostics through logging only (RM1)

`CGMSchemaDefinition.export_to_json` logged its "Regenerated" line with `print`; it now goes through
`logging` (info), and its inline imports are module-level. `scripts/regenerate_all_schemas.py` prints
its own line per module, and counts a module with no schema (`formats/supported.py`) as skipped
instead of failed, so a clean run no longer reports "Failed: 1" and exits 0.

## 0.14.0 — 2026-10-21

**Minor.** A caller can state a Nightscout exporter file's date order (RM19); exports that are pieces
of one series join once (`FormatParser.parse_exports`, F21); the pump fold runs from a shell
(`cgm-cli fold`). One repair: a MiniMed 780G's daily automatic-insulin totals, which 0.13.0 read as
boluses at midnight, are no longer insulin rows. Additive for every consumer; `CGM_SCHEMA` unchanged.

### A caller can state a Nightscout exporter file's date order (RM19)

- **`NightscoutDateOrder`** (`MONTH_FIRST`, `DAY_FIRST`) and an optional **`date_order`** on
  `FormatParser.parse_nightscout` and `from_nightscout_exports`. The default `None` keeps 0.12.3's
  inference. Stated, it settles the one case the text cannot: a file on a 12-hour clock with no day
  past the 12th, which 0.12.3 reads month first with a warning. A file that reads in no format of
  the stated order is refused, naming the order. Year-first dates read under either.
- **`parse_nightscout` and `from_nightscout_exports` take the exporter's whole file**, `# CGM
  ENTRIES` and `# TREATMENTS` sections included. Before, they read the section line as the header
  and failed; only `parse_file` / `parse_from_string` could read it. Treatments both inside the file
  and passed beside it raise `MalformedDataError`.
- Not done: a `--date-order` flag on `cgm-cli parse`.

### A 780G's daily automatic-insulin totals are no longer read as boluses

A MiniMed 780G CareLink export carries a section of daily SmartGuard totals whose section line also
says `Pump` (`-------;<device>;Pump;<serial>;Aggregated Auto Insulin Data`). 0.13.0 merged it into the
pump section and read each day's total as a bolus at 00:00:00 (20.494 U on a public 780G sample). The
section is now keyed by its label, not read, and its units and days are given in a warning: the
export does not say when that insulin was delivered. Any other unexpected section is named in a
warning rather than merged. Found by a format survey; shown failing on 0.13.0 first. F20 in
`docs/dogfooding.md` records what remains open for a 780G in auto mode.

### `cgm-cli fold`

The pump fold from a shell: `cgm-cli fold --cgm … --pump … --out-wide … --out-sugarone …` joins the
CGM exports, interpolates small gaps, synchronizes, folds the pump exports and writes the folded frame
and/or the models' input shape, with the API's options and defaults. Its output is byte-identical to
the API's (a test holds it); built so a dataset made with the library can be rebuilt from a shell
script.

### Several exports of one series join once (F21)

`FormatParser.join_exports(frames)` / `parse_exports(paths)` join exports that are pieces of one
series, as an account exported twice over spans that overlap: the export that starts first keeps the
shared span, a later one adds only what comes after, in any order, and a disagreeing overlap is
counted in a warning. Sequences are left for the processor to detect across the whole span. Before
this there was no way to join them: a bundle doubles the overlap, and tracks are other sensors.
Decided here, for review: the earlier export wins a disagreement (the rule overlapping pump exports
already follow). Found joining two real Clarity exports of one G6.

## 0.13.0 — 2026-10-21

**Minor.** Insulin pumps: MiniMed pump CareLink exports parse (RM16), fold onto a CGM export's
5-minute grid (RM17), and render in the trained models' input shape with basal and bolus as separate
channels (RM21). Two silent losses repaired on the way: a second dose on one grid point (RM18) and
fingersticks counted as sensor readings on the Guardian path. The plan is `docs/PLAN_pump.md`.
Additive for a consumer of the core schema: `CGM_SCHEMA` is unchanged; the extended schema gains two
columns before `annotations`; `UnifiedEventType` gains `BG_METER` and `BASAL_RT`.

This release also carries 0.12.3's Nightscout exporter date fix (below), which was never tagged on
its own.

### MiniMed pump exports parse (RM16)

- **`SupportedCGMFormat.MEDTRONIC_PUMP`**, a variant of `MEDTRONIC`, detected before it by the
  pump-only `Estimated Reservoir Volume after Fill (U)` column, parsing into `CGM_SCHEMA_EXTENDED`.
  Before this, every 720G export in the tree failed with `MalformedDataError` on dtype inference.
  The file is read section by section with every cell as text.
- **Boluses** become one `INS_FAST` row each, the programmed and delivered log rows paired by `Bolus
  Number`: the delivered amount, at the programmed time. A square bolus or a dual's square part
  carries its delivery time in the new extended column **`insulin_duration`** (min).
- **Basal** becomes `BASAL_RT` rows (new `UnifiedEventType.BASAL_RATE`) carrying the new extended
  column **`basal_rate`** (U/h): the effective rate in force until the next such row, with `Percent`
  temp basals applied (the export logs only the schedule) and every suspend 0 until
  `NORMAL_PUMPING`, an alarm suspend included. Only changes are emitted.
- **Fingersticks** become `BG_METER` rows (new `UnifiedEventType.GLUCOSE_METER`), value in `glucose`.
- Two assumptions the export cannot confirm, stated in `docs/PLAN_pump.md` §3.4: a percent temp
  delivers schedule × percent, and a cancelled square bolus reports its actual duration (a warning
  names the rows that would show otherwise).
- Priming volumes, event markers and a device clock change are not represented, each said once in an
  aggregated warning with its count.
- New public symbols: `MEDTRONIC_PUMP_SCHEMA`, `MedtronicPumpColumn`, `MedtronicBolusType`,
  `MedtronicTempBasalType`. A hand-authored synthetic export, `data/input/medtronic_pump_synthetic.csv`,
  is written by `scripts/make_synthetic_medtronic_pump.py`.

### A pump folded onto a CGM's grid (RM17)

- **`FormatProcessor.fold_insulin_onto_grid(cgm, pump)`** gives each grid point of every sequence of a
  synchronized CGM frame its bolus insulin (`insulin_fast`, spread doses spread), the pump's basal
  rate averaged over the bin (`basal_rate`, U/h), carbs, and the CGM's own long-acting doses. One row
  per grid point; glucose and grid exactly the CGM's. `pump` may be several frames (overlapping
  monthly exports, their repeated rows counted once). `docs/USAGE.md`, Workflow 5.
- **Basal stays out of `insulin_fast` by default**: every model trained in
  `glucose_forecasting_metabonet` reads basal (U/h) and bolus (U) as separate channels.
  `fold_basal=True` folds the bin's basal dose in.
- **`FoldSource`** (`MERGE`, the default; `PUMP`; `CGM`) picks where boluses and carbs come from:
  under `PUMP`, on the real overlap, 24 of 30 meals logged only in the CGM app never reached a row.
  `MERGE` pairs
  entries at most `merge_pair_slots` grid slots apart (default `MERGE_PAIR_SLOTS = 1`), a 0 meaning
  the source did not say, two non-zeros averaged and warned with both values beyond
  `FOLD_MERGE_INSULIN_TOLERANCE_U` / `FOLD_MERGE_CARBS_TOLERANCE_G`.
- What lands on no row is said once with its total: entries outside every sequence, the source not
  chosen, repeated pump rows, basal in bins the pump covers only in part.
- Checked on real data: on five days where a Dexcom export and a 720G pump record overlap, the bolus
  units on rows plus those reported outside equal what the pump delivered (a test, skipped where
  the exports are absent). Not checked: the pump's clock against the CGM's, which no export
  records.

### What the pre-release audit changed

A fresh read of RM16, RM17, RM18 and RM21 against `docs/PLAN_pump.md` found eleven defects, all fixed
before this release, each with a test shown failing on the code before the fix:

- **Same-second basal ties now keep the later-logged rate.** CareLink writes exports newest first,
  so `Index` falls as the pump logs; the parser had the tie-break backwards on real files (8 and 20
  basal rows in two of the four exports). The synthetic export is now written newest first too.
- **Overlapping pump exports give one answer whatever order they are passed in.** The export that
  starts first keeps the span they share; a later one adds a bolus there only if the earlier has no
  record of it (same amount and duration within 15 minutes). Before, a bolus split across two exports
  (programmed in one, finished in the other) was counted twice, and two exports disagreeing about a
  basal rate gave a list-order-dependent result. A spread bolus whose programmed row an export lacks
  now starts at its finish less its duration.
- **`MERGE`**: a 0 no longer takes a partner from a real entry; entries outside every sequence are
  reported per source, not summed as if two doses.
- **`fold_basal`**: `basal_rate` is null when basal is folded into `insulin_fast` (it was given twice),
  and `source=CGM` with `fold_basal=True` raises, since a CGM log's bins without an entry are unknown.
- **The collapse** (`to_data_only_df(drop_duplicates=True)`, `to_ml_ready_df`) takes `glucose` from
  the sensor series only (`SENSOR_SERIES_EVENTS`): a meter or calibration value no longer replaces a
  reading or stands in where there is none. Two boluses summed on one point keep no
  `insulin_duration`, with a warning. `prepare_for_inference`'s "no valid readings" guard counts
  sensor readings only.
- The real-data test now reads delivered units from the raw CSVs, independent of the parser and the
  fold; it had computed its expectation the way the code does.

### The models' input shape, with pump basal (RM21)

- **`FormatProcessor.to_sugarone_ready_df(folded)`** renders the fold's output in the columns the
  models trained in `glucose_forecasting_metabonet` read (`SUGARONE_COLUMNS`, from that repo's
  `src/common/data/columns.py`): `Basal Rate (U/h)` from `basal_rate`, `Bolus Insulin (U)` from
  `insulin_fast`, `Event Type` `EGV` / `Interpolated` (the label the trainer's `drop_interpolated`
  filters on). Nulls stay null; the training loader fills them itself.
- `to_ml_ready_df` is unchanged, the legacy path for frames without a pump: it cannot carry a basal
  rate.

### Two doses on one grid point are both kept (RM18)

`to_data_only_df(drop_duplicates=True)` and `to_ml_ready_df` collapse the rows sharing a grid point
into one. They kept the first non-null value of every column, so a second bolus or carb entry on the
same point was silently dropped, for every vendor. Amounts (`carbs`, `insulin_slow`, `insulin_fast`,
named in `COALESCE_SUMMED_COLUMNS`) are now summed, and a sum over nothing recorded stays null rather
than becoming 0. A consumer reading those columns sees a larger value only where two entries shared a
grid point, which is the repair. Shown failing on the old code first: 4.0 U and 1.5 U on one point
came back as 1.5 U.

### One source for the version (RM2)

`cgm_format.__version__` comes from the installed metadata only. The fallback literal in
`__init__.py` is gone: it had drifted again (it said 0.12.2 while `pyproject.toml` said 0.12.3). An
import of the package without installing it now raises `PackageNotFoundError` rather than reporting a
stale version; `uv sync` installs it, and CI does.

### Repairs on the Guardian Connect path (`MEDTRONIC`)

- **Fingersticks no longer count as sensor readings.** `BG Reading (mg/dL)` was coalesced into
  `glucose` as `EGV_READ`; it is now a `BG_METER` row. On the committed Guardian fixture, `EGV_READ`
  drops from 22,404 rows to its 22,149 sensor readings and 255 rows become `BG_METER`. A consumer that
  filtered on `EGV_READ` loses the fingersticks from the sensor series, which is the repair; one that
  wants them reads `BG_METER`.
- **`Basal Rate (U/h)` is no longer an `insulin_slow` dose.** A rate is not a dose, and on a pump it
  is rapid-acting insulin. No Guardian file we hold has basal rows, so no shipped output changes;
  one that does gets a warning naming the count, and a pump export keeps the rates as `MEDTRONIC_PUMP`.

### Tests

- New `tests/test_medtronic_pump.py`, expected values written by hand from the fixture's row list.
  Shown failing on the old code first: the old parser read the synthetic export as `MEDTRONIC`, put
  its 11 basal rows into `insulin_slow`, and counted both fingersticks as sensor readings.
- `tests/test_extended_schema.py`: the pinned list of appended extended columns gains `basal_rate`
  and `insulin_duration`, before `annotations` (a changed expectation, by design).
- `tests/test_idempotency.py`: each file now runs through the processor its format targets
  (`ExtendedFormatProcessor` for a pump export) instead of the core processor for every file; the
  pump exports join the suite and hold every idempotency and commutativity check.
- `tests/test_format_detection_validation.py`: pump exports join Guardian ones in being left out of
  single-table Frictionless validation, for the same reason (a second section with its own header).
- New `tests/test_fold.py`: every expected bin value worked out by hand from the synthetic Dexcom and
  pump exports, plus the real-pair accounting identity above.

## 0.12.3 — 2026-10-21 (not tagged; shipped in 0.13.0)

**Patch.** Nightscout exporter CSVs written in a day-first locale now parse, and one written so
that the order cannot be told says so. Nothing added that a consumer reads, nothing removed, nothing
retyped; `CGM_SCHEMA` untouched. **If you read nightscout-exporter CSVs from outside the US,
upgrade.** Before this, such a file either failed or came back with wrong dates and no warning.

Reported by `sugar-sugar` as S6 (`docs/FEEDBACK.md`), relayed from their own `FEEDBACK.md`. Each
case was reproduced on the code before it was touched, on day-first copies of the committed exporter
fixture made at test time; no real non-US export is in the tree.

### Exporter dates were read in the US order whatever the file said

The exporter writes Date and Time with the browser's `toLocaleDateString()` /
`toLocaleTimeString()`, so the shape is the browser's locale. The date probe kept the first format
that did not raise, and `strptime(strict=False)` never raises: it returns null. So US month-first
won every file. A day-first date past the 12th became null and its row was dropped; one up to the
12th was read with day and month swapped (7 April became 4 July), silently; and on the treatments
side a row whose date was null was kept with no time at all. The `except` around the probe was a
guard that could not be reached.

Now one format is chosen per file, over the entries and treatments sections together, as the one
that reads every row. The UK row of the table was itself wrong: en-GB, en-IE and fr-FR write a
24-hour clock (`07/04/2026 19:59:42`), which the old `%d/%m/%Y %I:%M:%S %p` row never matched; it
is added beside the 12-hour day-first row that en-AU, en-NZ and en-IN do write. A file where every
day is 12 or under reads completely both ways on a 12-hour clock: it is still read month first, as
before, and now **one warning says the order was ambiguous and which was assumed**. A US export
that spans only days 1–12 of a month gets that warning too, which is its honest cost. Rows that no
format reads alongside the rest are dropped, as before on the entries side, and now counted in one
warning. Letting a caller state the order outright is filed for the next minor.

### No package change: the triage loop moved to the seats kit

The consumer triage loop now runs on the seats kit's `triage` role: the local
`scripts/triage-state.py`, `scripts/triage-archive.py` and `scripts/watch-inbox.sh` are removed for
the kit's `classes/triage/` copies, whose ledger gives every section of `docs/FEEDBACK.md` and
`docs/FEEDBACK_HISTORY.md` the same verdict and fingerprint as the local one did. The inbox watchers
are the triage seat's, run by its watchdog; `docs/CONSUMER_TRIAGE_LOOP.md`, both inbox headers,
`scripts/README.md` and `CLAUDE.md` §8 and §11 now name the kit's commands, and the gist-sync
procedure is gone. Roadmap ids are claimed with the kit's `item-next.py`. The inbox's pinned next id
is corrected from S2 to S6.

## 0.12.2 — 2026-08-28

**Patch.** Nightscout repairs, all of them defects on paths that already behaved this way. Nothing
added that a consumer reads, nothing removed, nothing retyped; `CGM_SCHEMA` untouched. **If you use
`from_nightscout_url` or `parse_nightscout`, upgrade** — on 0.12.0 the URL import fails outright on
most real instances.

Reported by `sugar-sugar` as S2–S5 (`docs/FEEDBACK_HISTORY.md`), each reproduced against the code
before being touched, each fix carrying a test that was run against the unfixed code and watched to
fail.

### The URL import was broken for most instances

`_parse_nightscout_treatments_json` handed heterogeneous dicts to `pl.DataFrame` with no schema, so
polars inferred dtypes from the first 100 records. Treatments are heterogeneous by design — a Temp
Basal carries `rate` and no `carbs`, a Meal Bolus the reverse — so a column is routinely null for a
long opening run, types as `Null`, and raises `ComputeError` on the first real value. Any closed-loop
user who logs carbs rarely hit it. `NIGHTSCOUT_TREATMENTS_JSON_DTYPES` now declares the six fields,
and `_nightscout_number()` coerces the numeric ones — which also recovers the events some uploaders
send as numeric strings, previously cast to null and dropped without a word.

`download_nightscout` now passes `follow_redirects=True`. httpx defaults it to `False`, unlike
requests, so every redirecting host — `http://` to `https://`, apex to `www`, a PaaS hostname to its
canonical domain — failed rather than working.

### Errors now say what actually happened

Routing between the JSON and CSV branches asked whether `"sgv"` appeared in the first 2000
characters. An empty account returns `[]` and an auth failure returns an object; neither contains it,
so both were parsed as CSV and reported as a missing `'Glucose (mg/dL)'` column. Routing now asks
whether the payload parses as JSON, and a Nightscout error object raises `MalformedDataError`
carrying the server's own message. The `ZeroValidInputError` for an empty account is reachable for
the first time.

### Data that should not have left the machine, and readings that should not have been dropped

`from_nightscout_url` without `output_dir` used `mkdtemp` and never cleaned up, leaving a patient's
full CGM record in `/tmp` after every call. It now uses `TemporaryDirectory`; passing `output_dir`
still keeps the files.

httpx builds `HTTPStatusError` messages from the full request URL, and the Nightscout read token
travels as `?token=`, so any caller logging a failed download wrote a live credential to its logs.
The token is now redacted, and the exception type is preserved so existing `except
httpx.HTTPStatusError` handlers keep working.

Entries omitting `type` — normal for some xDrip+ and Loop uploaders — had every row skipped, and an
`sgv` of `0` was treated as absent rather than as the sensor-error value it is. Both are fixed. A
profile endpoint that refuses no longer fails the whole import, since the profile is discarded
anyway; it warns and names what was skipped.

`[project.urls]` is set, so `pip show cgm-format` gives a repository link.

**Not in this release:** wrapping network failures in a library exception type, filed as RM15. It is
a design decision rather than a defect — see the item for why each candidate shape is wrong.

## 0.12.1 — 2026-08-28

**Patch.** Documentation only: no code, schema or CLI change, and emitted frames are byte-identical
to 0.12.0. This is the first consumer-triage batch (`docs/CONSUMER_TRIAGE_LOOP.md`), answering **S1**
from `sugar-sugar`.

### `track="mean"` reports composition, not validity

`docs/USAGE.md` now says so where it documents `parse_tracks`. `Quality.TRACK_MERGE` marks which rows
were synthesized from two readings, and for that it is correct; nothing in the API says whether the
two sensors were measuring the same quantity, which is a per-subject property and the one that
decides whether averaging them is honest.

Measured through `parse_tracks` on the published CGMacros corpus (45 subjects, 629,605 overlapping
samples), fitting `libre = slope · dexcom + intercept` per subject: median slope 0.70 over a
0.10–1.16 range, median Pearson r 0.82. Thirty-one subjects sit under slope 0.8 and 15 under r 0.7;
exactly two fall inside slope 0.9–1.1 with r > 0.9. For the rest the mean is not two reads of one
truth averaged, and the frame looks no different.

The API half — a `TrackAgreement` measurement and a warning on `mean` when the tracks disagree — is
filed as **RM14** in `docs/ROADMAP.md`, additive and sized minor. It is not in this release; until it
lands the check is the caller's to run, which is what `USAGE.md` now tells them.

S1 is archived verbatim in `docs/FEEDBACK_HISTORY.md`. A separate observation from probing it —
`CGMParser.parse_tracks` / `parse_corpus` omitting selectors their own docstrings name — is recorded
as **F17** in `docs/dogfooding.md`, surfaced rather than repaired because the repair is a decision
about what the ABC is.

## 0.12.0 — 2026-08-18

**Minor.** A new `original_glucose` service column, a new `Quality.GRID_RETIMED` member, a new
`retime_glucose` parameter. Nothing removed, nothing retyped. But **read the migration note below
before upgrading**: the default glucose values coming out of `synchronize_timestamps` change, and a
frame built by hand against the old schema now raises.

### `synchronize_timestamps` re-times glucose onto the grid

A reading taken at `14:11:45` and re-labelled `14:11:00` is not a reading *of* `14:11:00`. Until now
sync moved the timestamp and kept the number, which left every value leaning systematically toward
the previous reading. It now recomputes `glucose` as the measured series evaluated at the grid
instant, and flags the row `Quality.GRID_RETIMED`.

Measured against the `glucose_data_processing` reference that produced the deployed models' training
data (`data/comparison/livia_reference.csv`, 289-point inference window):

| | MAE | max | matches within 0.01 |
|---|---|---|---|
| 0.11.0 | 0.7834 | 3.30 | 8.7% (25/289) |
| **0.12.0** | **0.000176** | 0.0005 | **100% (289/289)** |

`tests/test_livia_reference_alignment.py` was failing on `GLUCOSE_MAE`; it passes now. This closes
rung 3 of `docs/modification_plan.md`.

`interpolate_gaps` and the re-timing half of sync now share one interpolant, evaluated over the same
measured anchors, so the value at a grid point no longer depends on which stage produced its row.
Created points therefore shift slightly too — they anchor on raw reading times rather than on
grid-snapped ones. Idempotency and commutativity are unchanged and still gate the release.

Interpolation clamps at a sequence's first and last reading rather than extrapolating past them.
`glucose_data_processing` picks its two interpolation points by absolute time distance without
requiring them to bracket the target, so where both fall on the same side its weights sum to more
than one; that artifact is not reproduced.

Across the whole reference — not just the inference window — the two pipelines now agree on 99.1% of
its 25,332 grid points. The 229 that differ are the reference's defects, not ours: 201 sit beside a
`Low` marker, where it carries roughly the last valid reading instead of the 39 mg/dL floor its own
config declares, and 24 sit beside a `Calibration` row, where it lets a fingerstick value into the
CGM glucose channel. None fall in an inference window. Full breakdown in
`docs/LOGIC_RECONCILIATION.md`.

### The value anchor, and what it costs you

`original_glucose` is to `glucose` what `original_datetime` is to `datetime`: written once at parse,
read by every later pass, never written again. It is what makes re-timing re-runnable — without it
the second pass would interpolate from the first pass's output and drift silently, because the
timestamps stop moving after pass one.

**Migration, for `sugar-sugar` and `glucosedao`.** Frames that arrive through any `parse_*` entry
point get the column filled automatically, including from a pre-0.12 unified CSV, so the normal path
needs no change. A frame **constructed or cached in memory** against the 0.11 schema and handed
straight to a processor method now raises `MissingColumnError` under the default validation mode.
Either re-parse it or add `original_glucose` (Float64, the measured reading, null where there is no
glucose).

To keep 0.11 numeric behaviour: `synchronize_timestamps(..., retime_glucose=False)`, or
`cgm-cli pipeline --no-retime-glucose`. The measured value is in `original_glucose` either way.

### Also in this release

- **`ProcessingWarning.SYNCHRONIZATION` is finally emitted.** It has existed in the enum since the
  warning system was written and no code path ever raised it. `prepare_for_inference` now does, when
  any row carries `SYNCHRONIZATION` or `GRID_RETIMED` — so a caller can tell that the number under a
  given instant is not the number the device stamped with it. Expect it in warning sets that
  previously omitted it.
- **One-row sequences stop being a special case.** They took a shortcut through `dt.round('1m')`
  that quietly exempted them from the `SYNCHRONIZATION` flag as well. They go through the normal
  path now, which produces the same timestamp and the flags they should always have carried.
- **A dead check in the idempotency suite is alive.** `check_no_null_glucose_egv` filtered
  `event_type == 'EGV'` — the code is `'EGV_READ'` — so it early-returned on all ten of its call
  sites, and the column it would then have read (`glucose_value_mg_dl`) no longer exists. The
  module's docstring had been advertising a constraint nothing enforced. Two more restated-vocabulary
  checks went the same way in this pass: `test_unified_format_schema` now derives its column list
  from the schema, and `cgm-cli`'s warning descriptions now come from `WarningDescription` rather
  than a hand-written dict that covered five of seven members — which is why the CLI greeted the new
  `SYNCHRONIZATION` warning with "Unknown warning". Written up as F15 in `docs/previous_issues.md`.
- **F16, surfaced not fixed** (`docs/dogfooding.md`): the alignment gate's reference fixture names a
  regeneration script in three places, and that script has never existed in any commit. The
  definition of "correct" for our alignment with training is currently not reproducible.
- **Every surface that synchronizes can opt out.** `cgm-cli pipeline`, `cgm-cli process` and
  `cgm-cli batch` all gain `--retime-glucose/--no-retime-glucose` (default on); the library
  equivalent is `synchronize_timestamps(..., retime_glucose=False)`, or
  `FormatProcessor.retime_glucose = False` to change it for a whole run. 0.11.0 behaviour stays
  reachable without pinning an old version.

## 0.11.0 — 2026-08-16

**Minor.** A new `SupportedCGMFormat.BIGIDEAS` member plus registry entries. Nothing a consumer
already reads changed. `parse_file` on a BIG IDEAs `Dexcom_*.csv` is still DEXCOM — that file is a
Clarity export.

### BIG IDEAs corpus (PhysioNet `big-ideas-glycemic-wearable`)

16 subjects, each a directory of `Dexcom_NNN.csv` + `Food_Log_NNN.csv`. Identified by directory
shape (`PATH_DETECTION_PROBES` / `SUBJECT_PATH_PROBES`); a bare food log text-detects as BIGIDEAS
and `parse_to_unified` refuses it, naming `parse_bundle` / `parse_corpus`. Glucose reuses
`_process_dexcom`. Meals target `CGM_SCHEMA_EXTENDED` (carbs, calories, protein, fat, fiber;
food name / amount / unit / sugar in `annotations`). Items are not clustered.

Header drift absorbed, all read off the published 16 subjects:

- `time` → `time_of_day` on 4 subjects (`aliases`)
- headerless 11-column food log on subject `003` (no `time_end` / `sugar` / `total_fat`)
- one blank `time_begin` (subject `012`) falls back to `date` + `time`

Also: `_subject_track_coverage` no longer assumes every non-CGMacros corpus is D1NAMO — it
dispatches explicitly and raises `NotImplementedError` for an unregistered format.

`scripts/download_bigideas.py` fetches Dexcom + food logs from `files.physionet.org`. Empatica
streams are never requested. The script is a `dev`-extra setup chore, never imported by the
package; people without a local extract run it and point `CGM_FORMAT_BIGIDEAS_DIR` at the
destination.

### The Dexcom path is split, so a bundle can reuse it correctly

`_process_dexcom` is now three: `_dexcom_read_rows` (decode, clean, drop blank-timestamp metadata
rows, probe the timestamp), `_dexcom_event_frames` (per-event sub-frames, stopping **before** the
unified contract is enforced), and `_process_dexcom` itself (concat → `_postprocess_unified`).
Public behaviour is unchanged — every committed Dexcom fixture parses to identical `hash_rows()`
across the refactor.

The split is what makes the BIG IDEAs bundle correct. Building the glucose half with
`_process_dexcom` and concatenating meals onto that finished frame left **every meal row with a
null `original_datetime`** — the write-once guards in `_postprocess_unified` no-op on a second pass
— so no meal was ever assigned to a sequence or reached `prepare_for_inference`. All 1,422 meal rows
of the published corpus were affected. Full account in `docs/previous_issues.md` (F6); the rule is
now in `docs/NEW_SCHEMA.md` under *Gotchas*, and
`tests/test_subject_selection.py::TestParsedCorpusInvariants` guards it across every corpus.

### Dexcom fixes the corpus surfaced

Three defects in the shipped Dexcom path, all found by turning the library on the branch's own data
and all pre-dating it. Full accounts in `docs/previous_issues.md`.

- **`exercise` was null on every Dexcom exercise row** (F11). `.alias("exercise")` bound to the last
  term of the HH:MM:SS sum rather than the sum, so the result was named `duration_str` and dropped
  two lines later. Three `Exercise` rows in `data/input/000-14 oct-28 oct 2019.csv` and both of BIG
  IDEAs subject `014` were affected. A core data column now carries its values.
- **A short Clarity metadata block silently ate readings** (F12). The static
  `skip_rows_after_header` handled a block *longer* than expected and lost one reading per row a
  *shorter* one omitted — realistic, since the block length follows how many alerts a user
  configured. The block is now measured (`_dexcom_metadata_row_count`) and the static count is what
  the drift is reported against, signed and with its direction named.
- **`list_subjects` under-counted a G7 export** (F13). The parser accepts `EGV` and `Fasting
  Glucose`; the coverage reader accepted only `EGV`. Both now read one
  `DEXCOM_GLUCOSE_EVENT_TYPES` tuple.

The BIG IDEAs fixtures now carry the real corpus's metadata blocks — 12 rows for `001` and `003`
(with `PatientIdentifier`), 11 for `007` — instead of a tidy 10-row block that exists in no
published subject. That is what puts the drift-aggregation path under test on committed data.

### Reporting and validation

- Meal rows whose timestamp cannot be parsed are reported when *some* of them fail, not only when
  all of them do (F14): how many of how many, from which file, with a sample of the food names.
  Losing 5 meals of 300 used to look exactly like losing none. No row of the published corpus is
  affected — 1,422 in, 1,422 out.
- The Clarity metadata-drift warning is aggregated by a corpus walk: 16 subjects produced 16 copies
  of "the Clarity export format may have changed" for a condition this corpus is *documented* to
  have. One grouped line with signed per-subject drift now.
- `KNOWN_ISSUES_TO_SUPPRESS[DEXCOM]` gains `('type-error', 'Transmitter Time (Long Integer)', None)`
  — these exports write `11101.0` in a field declared a long integer, on every EGV row of all 16
  subjects. Bounded to one field on a column never mapped to the unified frame.
- That takes a real subject's Dexcom file from 500 residual Frictionless errors to 1 (2 on the four
  subjects noted below). What remains is documented rather than suppressed: F7 in
  `docs/dogfooding.md` (a `missing-label` error carries no `fieldName`, so no bounded rule can name
  the absent `Transmitter ID` column), F8 (the headerless food log cannot be validated against a
  header schema at all), and F9 (the blank-timestamp cap stays at **1**, though 4 of the 16 published
  exports carry two such rows — raising it weakens the guard for every Dexcom file and is a
  threshold decision, not a corpus fix).
- BIG IDEAs subject enumeration is now the union of `Dexcom_*.csv` and `Food_Log_*.csv`. A subject
  carrying only a food log used to be invisible to both `list_subjects` and `parse_corpus`; it now
  reaches the parser, fails with a typed error naming the missing file, and appears in the walker's
  failure summary. The consequence is that `list_subjects`' ids and `parse_corpus`' keys agree only
  when every listed subject parses — an amendment to how F5 stated that invariant, written up there
  and pinned by a test rather than left implicit.
- `TrackCoverage.rows` cannot express coverage for a Clarity-shaped source, because filtering to EGV
  before counting makes `values / rows` a constant 1.0. Recorded as F10 in `docs/dogfooding.md`
  rather than repaired: every candidate denominator is a design decision, and one of them would
  redefine a field two other corpora already populate.

## 0.10.0 — 2026-08-15

**Minor.** Everything is additive: no column was removed, renamed or retyped, every existing entry
point keeps its shape, and no existing flag changed meaning. Two new source categories and three new
registered formats.

`CGM_SCHEMA` keeps its exact column names, dtypes and order — which is what a consumer reading a
frame depends on — but two of its *metadata* fields did change, and the blanket phrase "untouched"
would have hidden them: `quality`'s `enum` constraint gained `32` for the new `TRACK_MERGE` bit
(additive), and `exercise` now declares its unit as `"s"` rather than `"seconds"`. Both are visible
in the regenerated `formats/unified.json`.

### The extended schema (RM8)

`CGM_SCHEMA_EXTENDED` sits **beside** `CGM_SCHEMA`, adding food (`calories`, `protein`, `fat`,
`fiber`), wearable (`heart_rate`, `breathing_rate`, `acceleration`, `mets`, `activity_calories`,
`steps`), `ketones`, and `annotations`. The core schema is an exact prefix of it, so an extended
frame narrows to a core one by projection alone (`FormatProcessor.to_core_df`).

`FormatProcessor.schema` is now a `ClassVar` and every one of its 24 `CGM_SCHEMA` usages reads it —
including 7 ungated structural reads that were **silently dropping extra columns mid-pipeline**.
`ExtendedFormatProcessor` carries the extended schema. `to_csv_string` / `to_csv_file` became
classmethods; as staticmethods naming `FormatParser` literally, a subclass override of
`validation_mode` had no effect. Call sites are unchanged.

`Quality.TRACK_MERGE` (bit 32) is new. `annotations_to_json` is the single deterministic serializer
for the `annotations` column — sorted keys, `None` for nothing-to-record — which matters because
that column joins the sort keys and the byte-level round-trip guarantee.

Also: `derive_schema` is now exported, and `exercise` declares unit `"s"` rather than `"seconds"`,
the spelling `UNIT_CONVERSIONS` is actually keyed on.

### Source categories and faceted output (RM12, RM9)

`FormatCategory` (`EXPORT` / `BUNDLE` / `CORPUS`) names an assumption the library had never written
down: one file, one subject, one device.

- `parse_bundle(paths)` — several files, **one** subject, each a different modality.
  `from_nightscout_exports` was already this shape and keeps its name.
- `parse_tracks(path)` — one file, several independent measurements of the same quantity.
- `parse_corpus(root, track=…, subjects=…)` — many subjects, keyed `"subject"` or `"subject/track"`.
- `detect_path_format(path)` identifies a directory by shape, beside the existing text-prefix
  detection. Driven by `PATH_DETECTION_PROBES`: glob patterns only, never callables.

**Reaching one subject.** Naming BUNDLE as a category is not much use if the only way into a
bundle-shaped corpus returns all 29 of its subjects, so three pieces close that:

- **`parse_bundle` accepts a subject directory as a member.** `parse_bundle([root / "001"])` parses
  that whole subject — for a corpus whose members are folders, the directory *is* the bundle.
  Naming the files individually cannot work there and was never meant to: `parse_file` refuses a
  bare D1NAMO `glucose.csv` precisely because one modality is not a record. `parse_subject_directory`
  is the per-subject step, named so a subclass can override it without reimplementing the merge.
- **`list_subjects(root)`** returns frozen `SubjectEntry` records — subject id, format, path,
  modality files present, and a `TrackCoverage` per glucose track (values, rows, first, last). The
  ids are read off the corpus rather than derived, which matters because none of them are
  predictable: D1NAMO's healthy subset holds a directory named `012_diabetes` and CGMacros runs
  `001`–`049` with four numbers missing. Coverage counts cells the *source* filled, deliberately not
  the same number as readings the schema can hold — D1NAMO ships a glucose cell reading `7:0`, which
  is counted here and dropped-with-a-warning by the parser, and collapsing the two would hide that.
  Measured on the published corpora: 1.4s for CGMacros' 45 subjects against 12.8s for the equivalent
  `parse_corpus`.
- **`parse_corpus(root, subjects=[...])`** prunes before parsing, so one id costs one subject's work.
  It composes with `track=`. An id that is not in the corpus **raises**, listing what is available —
  the same lesson `track=` taught earlier in this release: a filter that silently selects nothing
  makes a typo indistinguishable from a subject with no data.

`detect_subject_format(path)` is the third detector, one directory level below `detect_path_format`,
driven by its own `SUBJECT_PATH_PROBES` registry. Two registries rather than one because the shapes
differ by exactly that level, and a probe answering both questions would let a corpus root parse as a
single person. When a subject probe fails on something that is in fact a root, the error says so and
names `parse_corpus`.

`CGMACROS_TRACK_COLUMNS` now maps track name → raw column in one place, read by both the parser and
the coverage reader, so the two cannot drift into disagreeing about which sensor a track name means.

**Two things a consumer must know.** A bundle merges *modalities*, never *subjects* — two people's
files concatenate just as cleanly as two modalities and nothing raises, so the caller owns that
guarantee. And **tracks are alternatives, never shards**: non-sensor rows are replicated into every
track, so concatenating two of them double-counts every meal.

`parse_file` on a multi-track source now raises `MultiTrackSourceError` naming both tracks rather
than silently picking a sensor.

### Research corpora (RM10, RM13)

**CGMacros** — 45 subjects, two concurrent sensors, macronutrients and meal photos. All 9 real
header variants are absorbed declaratively. The opt-in synthetic `mean` track averages the two
sensors and flags every synthesized row `TRACK_MERGE`; rows with only one contributing sensor are
deliberately not flagged, because they are that sensor's real reading.

**D1NAMO** — two registered formats, not one with a flag: the subsets differ in `food.csv`'s column
set, in which modality files exist, and in a disjoint glucose `type` vocabulary. Fingersticks map to
`CALIBRAT`, never `EGV_READ` — the healthy subset has no CGM at all. `carbs` stays null throughout
because D1NAMO records no carbohydrate anywhere.

Neither parser resamples: CGMacros' native cadence is 1 minute, and callers pass
`expected_interval_minutes`.

### CLI

- New `cgm-cli corpus <root> --out <dir>`, with `--subject <id>` (repeatable) to parse a selection.
  `cgm-cli detect` accepts a directory.
- New `cgm-cli subjects <root>` — the ids `--subject` accepts, with each track's value count, row
  count, coverage percentage and span. A subject whose glucose could not be read is shown as
  `unreadable` rather than as a plausible zero.
- **Fixed:** `cgm-cli report` iterated a hardcoded three-format list, so `DEXCOM_EU`, `LIBRE_EU`,
  `MEDTRONIC` and `NIGHTSCOUT` were **already** silently absent from its output before this release.
- **Fixed:** `cgm-cli validate` checked every format against `CGM_SCHEMA` regardless of the schema it
  is actually parsed into.
- Processing commands (`pipeline`, `process`, `batch`) now pick their processor from the frame's
  shape instead of hardcoding the core one.

### Surface consistency

Auditing the API and CLI against the docs turned up four mismatches, all fixed before release:

- **`MultiTrackSourceError` was caught nowhere in the CLI.** Five commands (`parse`, `process`,
  `pipeline`, `validate`, `info`) handed a shell user a message naming `FormatParser.parse_tracks(...)`
  — correct advice for a library caller, useless at a prompt. They now print the refusal plus the
  `cgm-cli corpus` invocation that works. `validate` additionally reported it as a *validation
  failure*, which told the user their data was broken when it was not.
- **`parse_corpus(root, track=...)` silently ignored the track on a single-track corpus**, returning
  every subject while the caller believed they had filtered. Now refuses. Found by running a README
  example verbatim.
- **`cgm-cli info <directory>`** surfaced a raw `Is a directory` errno instead of naming `detect` or
  `corpus`.
- `docs/NEW_SCHEMA.md` still described **four** registries; there are eight, six of them exhaustive
  over `SupportedCGMFormat`. The checklist now lists each one and notes that `detect_format` is
  disjunctive while the two path detectors are conjunctive.
- **Docstrings and docs that outlived what they described.** `parse_bundle`'s public docstring said
  `parse_corpus` and `parse_tracks` were "not implemented yet" — both ship in this release, and that
  text is what `help(FormatParser.parse_bundle)` prints. `docs/PHILOSOPHY.md` carried the same
  parenthetical for `parse_corpus`, and both it and `docs/NEW_SCHEMA.md` described the `cgm-cli
  report` hardcoded-list bug in the present tense after it was fixed. The CLI note is kept but
  rewritten to say what is actually true: the CLI is written against the registries by hand rather
  than derived from them, so it *can* fall behind, and it did.
- The `parse_file` refusal on a bare D1NAMO `glucose.csv` told the caller to pass the subject
  directory to `parse_bundle` — which then raised `IsADirectoryError`, because `parse_bundle` took
  files only. The advice is now true rather than the message being softened.
- Stale line pointers: `docs/ROADMAP.md` RM1/RM2 and `CLAUDE.md` §5 cited `interface/schema.py:477`
  and `__init__.py:27`; the code moved to `:519` and `:33`, and RM2 still quoted the version fallback
  as `0.9.0`.

### Docs

`docs/PHILOSOPHY.md` gained a Source Categories section and corrected two claims: the processor is
now schema-parameterized (the vendor boundary is intact), and "the CLI requires zero changes" was
already false. RM8, RM9, RM10, RM12 and RM13 moved to `docs/ROADMAP_HISTORY.md`. RM6 is narrowed to
its remainder — the `ketones` column ships, but what `event_type` a ketone row carries still needs a
real export with populated cells.

## 0.9.0 — 2026-08-13

- European / mmol/L FreeStyle Libre exports (`LIBRE_EU`, RM5): glucose columns relabeled mmol/L, two
  ketone columns appended. Detected ahead of `LIBRE` the same way `DEXCOM_EU` sits ahead of `DEXCOM`.
- Libre historic and scan readings both parse as `EGV_READ`. Libre files now yield more glucose
  rows than before — a consumer comparing row counts across versions should expect the scan
  readings that used to be dropped. Duplicate timestamps (a scan landing on a historic minute) are
  flagged `TIME_DUPLICATE`, not dropped.
- Libre strip (finger-prick) readings parse as `CALIBRAT`.
- Shared `_glucose_to_canonical` helper: both Dexcom and Libre scale glucose from the unit declared
  in the vendor schema. `derive_schema` can append data columns.

## 0.8.4 — 2026-07-24

- Schema-drift primitives: column aliases, `derive_schema`, declarative units.
- LibreView column rename supported.

## 0.8.3 — 2026-07-18 … 2026-07-23

- Newer Clarity/Dexcom exports with an extra metadata row now parse.
- Frictionless suppression cap fixed — a bounded suppression list stopped silently swallowing more
  than it was meant to.
- Added the agent guide (`CLAUDE.md`, `AGENTS.md` symlink) and `docs/NEW_SCHEMA.md`.

## 0.8.2 — 2026-04-22

- European Dexcom G7 export support.

## 0.8.1 — 2026-04-10

- Version bump and housekeeping.

## 0.7.2 — 2026-04-04 … 2026-04-07

- **Nightscout support**, dual-path: the JSON API (`parse_nightscout`, `from_nightscout_exports`,
  `from_nightscout_url`, combining entries + treatments) and the nightscout-exporter combined CSV
  with `# CGM ENTRIES` / `# TREATMENTS` section headers.
- `api_secret` auth added alongside `token`.
- The built-in Nightscout API CSV endpoints were dropped as unsupported — headerless 5-column
  entries, and treatments returns JSON regardless. `data/input/nightscout_entries.csv` stays as a
  negative control.
- **Medtronic parser**: basal insulin, co-occurring events, exports fixed.
- Gap thresholds harmonized with `glucose_data_processing` (`SMALL_GAP_MAX_MINUTES = 15`, a grid
  multiple; the previous `19` made `interpolate_gaps` and `synchronize_timestamps` disagree on
  borderline gaps).
- Interpolation gap detection always computes from `original_datetime`, never from a mutated
  `datetime` — this is the idempotency anchor.
- Inputs moved under `data/input/`; docs added.

## 0.7.0 — 2025-12-12

- CLI tool (`cgm-cli`), synthetic data fixtures, test coverage.
- Exports, READMEs and tests brushed up.

## 0.6.x — 2025-12-09 … 2025-12-12

- **Schema change**, idempotent processing, rigorous tests (0.6.0).
- Docs update (0.6.1), fixes (0.6.2).

## 0.5.1 — 2025-12-03

- Immutable (frozen-dataclass) schemas plus bugfixes.
- Better tests.

## 0.4.x — 2025-11-30 … 2025-12-01

- `Quality` field reworked in the unified format (0.4.0).
- Format-processor and interface fixes (0.4.1), inference-prep update (0.4.2), further processor
  fixes (0.4.3), docs (0.4.4).

## 0.3.x — 2025-11-27 … 2025-11-30

- Separation of concerns in the inference preprocessor (0.3.7).
- Export fixes (0.3.6), QoL functions (0.3.5), `pyproject` fix (0.3.3), restructuring and cleanup
  (0.3.2).

## 0.2.x — 2025-11-26

- Medtronic format, work in progress (0.2.2).
- Docs and version bump (0.2.1).

## 0.1.1 — 2025-10-08 … 2025-10-30

- MVP: interface, schemas, two supported formats, `FormatParser` implemented, `CGMProcessor` planned.
- `FormatProcessor` implemented.
- Data marked as sensor calibration after a gap longer than 2 h 45 min.
