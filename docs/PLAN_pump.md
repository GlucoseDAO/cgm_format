# Plan: insulin pumps, and folding their insulin onto the glucose grid

Proposal and decisions only, no code. The work is filed in [ROADMAP.md](ROADMAP.md) as **RM16** (the
pump export parses), **RM17** (pump insulin folded onto a CGM's 5-minute grid) and **RM18** (the
cross-vendor collapse sums amounts instead of dropping them).

Drafted 2026-10-21 from four real MiniMed 720G CareLink exports a contributor supplied (monthly,
2026-07 to 2026-10, git-ignored under `data/input/`), then revised against a critic's reading of the
first draft. Every number below is an aggregate read off those files; no row of them enters this
repository.

---

## 1. What the person asked for

> We're gonna support pumps too. And we're gonna need our new multi-track functional right away: an
> ability to join cgm cols with pumps by synced timestamp (one gets glucose). This (pump) has the
> insulin tracks, basal and bolus. This is not to be confused with slow fast, both are actually
> ultra_fast_insulin mostly. What it does: basal is hourly rate. Bolus is +x on top. So each 5 min row
> gets a value of fast equal = basal_hourly/12 + bolus (replace magic with respective config vars).

A CGM export and a pump export are two *modalities* of one subject, which the library calls a
bundle, not a track set (tracks are alternative measurements of the same quantity, CGMacros' two
sensors). But the bundle's merge (a diagonal concat, then sequencing over every row) is the wrong
tool here, and §2.4 shows why: pump rows would move the CGM's grid. So the join is its own operation,
taking the two frames apart, which is also what the person literally asked for.

---

## 2. What the exports actually hold

Measured with the shipped 0.12.2 parser and with section-by-section aggregates.

### 2.1 The shipped parser fails on all four

Detection matches them as `MEDTRONIC` (they carry `Sensor Glucose (mg/dL)` in the header), then
`pl.read_csv` dies on dtype inference: September on `2,0` in `Temp Basal Amount`, the other three on
the second section's repeated header row read as data. The older Guardian Connect fixture still
parses.

The 720G header is the Guardian Connect one widened: 54 columns against 48, seven new (`BG Source`,
`Estimated Reservoir Volume after Fill (U)`, `Alert` where Guardian has `Alarm`, `User Cleared
Alerts`, `BLE Network Device`, `Device Update Event`, `Sensor State`).

Each file has two sections, and **neither holds glucose**: a `Pump` section (about 1,000 to 1,900 rows
a month) and a `Sensor` section whose only filled column is `Event Marker` (`Start of the day` / `End
of the day`). The glucose comes from a separate CGM export. The one in the tree that overlaps is a
Dexcom Clarity export, 2026-07-13 to 2026-07-18, 1,392 readings; the pump starts 2026-07-13.

### 2.2 What the pump section records

September's file; the others agree.

| Record | Rows | Reading |
|---|---|---|
| `Basal Rate (U/h)` | 665 | a rate change, in force until the next; median 13.5 min apart, max 10 h. 211 timestamps carry more than one row, 8 of them with different rates (18 in the 2026-10 file): the pump's own order (`Index`) says which came last |
| `Temp Basal Type` = `Rate` | 33 | the basal row logged at its start is the temp rate (33/33): already effective |
| `Temp Basal Type` = `Percent` | 23 | the basal row logged is the *scheduled* rate (21/23 equal the rate before): the percentage is not applied in the log |
| basal row within 2 min of a temp's end | 56/56 | the return to schedule is logged |
| `Suspend` = `USER_SUSPEND`, `NOTSEATED_SUSPEND` | 48 | followed by a logged 0 U/h row (48/48); `NORMAL_PUMPING` (48) resumes with a logged rate |
| `Suspend` = `ALARM_SUSPEND` | 1 | no 0 U/h row follows (also 1 in August); the alert log has `DELIVERY STOPPED` around it |
| `Rewind` | 6 | reservoir change, between a suspend and its resume |
| `Bolus Type` = `Normal` | 405 | logged twice, linked by `Bolus Number`: programmed (`Bolus Volume Selected`), then finished (`Bolus Volume Delivered`), 1 to 5 min later for 86 of 202, over 5 min for 3 |
| `Bolus Type` = `Dual (normal part)` / `(square part)` | 20 + 20 | the square part's finished row is stamped at its end; `Bolus Duration` runs 00:12 to 02:30 across the four files. A plain `Square` type does not occur here but exists on these pumps |
| bolus rows with neither volume | 1 (Sep), 1 (Aug), 3 (Oct) | nothing to emit |
| `Prime Type` (`Cannula Fill`, `Tubing Fill`) | 13 | priming, not a dose to the body |
| `BG Reading (mg/dL)` | 205 | fingersticks typed into the bolus wizard (`ENTERED_IN_BOLUS_WIZRD`); July also has 3 `ENTERED_IN_BG_ENTRY` |
| `BWZ Carb Input (grams)` | 208 | the meal entered in the wizard |
| `New Device Time` | 0 | empty in all four; a clock change would fill it, and EU summer time ends 2026-10-25 |

### 2.3 Two misreadings in the shipped Medtronic path, not specific to the 720G

- **`Basal Rate (U/h)` becomes `insulin_slow`** as if each row were a dose. It is a rate (U/h), and on
  a pump the same rapid-acting insulin as the bolus, which is the person's "not to be confused with
  slow fast". The Guardian Connect fixture has no basal rows, so no shipped output carried a wrong
  number yet.
- **`BG Reading (mg/dL)` is coalesced into `glucose`.** This already happens on the Guardian path:
  the Guardian fixture's 22,404 `EGV_READ` rows are 22,149 sensor readings plus 255 fingersticks. On a
  pump export it would turn every wizard fingerstick into a CGM reading.

### 2.4 Why bundling before synchronization does not work

Found by the critic, recorded as **F19** in [dogfooding.md](dogfooding.md):

- A sequence's grid is rooted at the earliest `original_datetime` of **all** its rows, insulin
  included. One insulin row 2 minutes before the first Dexcom reading moved 353 of 1,392 grid
  instants and changed 350 re-timed glucose values.
- Non-glucose rows join the nearest glucose sequence with no distance limit: an insulin row 30 days
  after the last reading joined the last sequence.

A pump frame bundled with a CGM frame and then synchronized would re-grid the glucose by whatever the
pump logged first. So the pump frame is never synchronized; it is integrated onto the grid the CGM
frame already has.

### 2.5 What a real pair double-counts

- **Overlapping pump exports.** The 2026-09 and 2026-10 files share 68 identical rows (the second
  starts 2026-09-30 00:00). Two older CareLink exports in the tree share 1,879.
- **The CGM export logs insulin and carbs too.** The Dexcom export has 16 `INS_FAST` and 30 `CARBS_IN`
  rows inside the pump period; 6 of the 16 match a pump bolus (within 15 min and 0.1 U) and 6 of the
  30 match wizard carbs. The rest do not match anything the pump logged.

### 2.6 The existing collapse loses doses

`FormatProcessor._coalesce_duplicate_timestamps` (behind `to_data_only_df(drop_duplicates=True)`)
keeps the first non-null value per column on a grid point, so two boluses on one grid point keep one,
for every vendor. Summing is right for amounts, but Polars sums an all-null group to `0.0`, and a
null `carbs` means "no entry was recorded", never zero.

---

## 3. The design

### 3.1 The pump export parses (RM16)

**A new format, `MEDTRONIC_PUMP`**, a variant of `MEDTRONIC` in the `DEXCOM_EU` / `LIBRE_EU` pattern:
its own detection (a 720G-only header column, registered before `MEDTRONIC`), `derive_schema` for the
seven new columns, an arm on `_process_medtronic`, and `CGM_SCHEMA_EXTENDED` in
`UNIFIED_TARGET_SCHEMA`. A separate format because the output shape differs: `MEDTRONIC` maps to the
core schema and `cgm-cli validate` checks a parse against that entry, so a pump parse under the same
id would fail validation, and a consumer could no longer tell the shape from the format.

Reading: split on the `-------;<device>;<section>` lines, read each section with every column `Utf8`
(`infer_schema=False`), keep `Index` until the rows are ordered, convert numbers with the existing
`_euro_number_to_float`.

What the pump section emits, all rapid-acting, none of it in `insulin_slow`:

- **Boluses**: one `INS_FAST` row per bolus, paired by `Bolus Number`: `insulin_fast` = `Bolus Volume
  Delivered (U)`, at the *programmed* row's time (delivery started then). A `Square` bolus or a
  `Dual (square part)` also carries `insulin_duration` = its `Bolus Duration` in minutes. A bolus with
  no delivered volume emits nothing; an unknown `Bolus Type` emits nothing; each counted in one
  aggregated warning.
- **The effective basal rate**, as `BASAL_RT` rows carrying `basal_rate`: the logged `Basal Rate
  (U/h)` series, ordered by `Index`, the last-logged row winning at a shared timestamp; every
  `Percent` temp applied for its duration, schedule changes inside it included; and **0 U/h from any
  `Suspend` other than `NORMAL_PUMPING` until the next `NORMAL_PUMPING`**, `ALARM_SUSPEND` included.
  A suspended pump delivers nothing, so 0 is the measured value there, not a fill.
- **Carbs**: `BWZ Carb Input (grams)` > 0 as `CARBS_IN`, as today.
- **Fingersticks** (`BG Reading (mg/dL)`, either `BG Source`): a new `UnifiedEventType.BG_METER =
  "BG_METER"` row carrying the value in `glucose`, never `EGV_READ` (the person, Q2). This follows the
  existing pattern: Libre's strip readings already ride `glucose` as `CALIBRAT` rows, and every
  processor stage reads only `EGV_READ` rows as the sensor series. Whether Libre's strip rows should
  become `BG_METER` too is a redefinition of what a consumer reads, so not in this plan.
- **Not represented**, one aggregated warning each with its count: priming volumes and event
  markers. A filled `New Device Time` (the clock changed) is reported with its count; the rows are
  kept as the pump logged them, not shifted and not dropped.

Two **additive** columns on `CGM_SCHEMA_EXTENDED` (inserted before `annotations`, which stays last),
and two new event types (`BASAL_RT`, `BG_METER`):

| Column | dtype | Meaning |
|---|---|---|
| `basal_rate` | `Float64`, U/h | on a `BASAL_RT` row: the effective rate in force from this row's `datetime` until the next `BASAL_RT` row. Null on every other row |
| `insulin_duration` | `Float64`, min | the minutes over which this row's `insulin_fast` was delivered. Null = at once |

`insulin_duration` is not something the person asked for. It is what makes a dual bolus's square part
land where it was delivered: 00:12 to 02:30 of spread, which at 5-minute resolution is up to 30 rows.

`Basal Rate (U/h)` stops feeding `insulin_slow` on the Guardian path too, which has no basal rows in
any file we hold; and fingersticks become `BG_METER` rows on both formats, which takes the Guardian
fixture's 255 fingersticks out of its `EGV_READ` series.

A pump-only frame is a valid parse of insulin and carb events and nothing more. It is not meant to go
through the processor alone: the glucose stages do not refuse it (they run, assign every row
`sequence_id` 0 and snap it to a grid rooted at its own first row), but the result means nothing.
`fold_insulin_onto_grid` is how it is used.

### 3.2 Folding insulin onto the glucose grid (RM17)

```python
cgm   = FormatProcessor.synchronize_timestamps(FormatParser.parse_file(dexcom_csv))  # a core frame
pumps = [FormatParser.parse_file(p) for p in (pump_jul, pump_aug, pump_sep, pump_oct)]
wide  = FormatProcessor.fold_insulin_onto_grid(cgm, pumps)
```

A new classmethod, vendor-agnostic. It takes a synchronized CGM frame, whose grid it reads, and one
pump frame or a sequence of them (monthly exports), which it integrates. Not `parse_bundle`: monthly
exports of one modality are not the disjoint modalities a bundle's members must be. It never synchronizes the pump frame and never
re-sequences the CGM frame, so the glucose grid is exactly what the CGM alone produced (§2.4).

**The rows.** One output row per grid point of each CGM sequence, from the sequence's first grid point
to its last, every point included. A point with a glucose row carries that row's glucose; a point
without one (a gap of 15 min or less that `interpolate_gaps` was not run over) carries null glucose.
The fold invents no glucose.

**The bins.** Grid point `t` owns `[t − I/2, t + I/2)`, `I` = `cls.expected_interval_minutes`.

```
bolus(t)      = Σ insulin_fast of instant doses starting in the bin
              + Σ spread doses × (overlap of [start, start + insulin_duration) with the bin) / insulin_duration
basal_dose(t) = (1 / MINUTES_PER_HOUR) × ∫ basal_rate dt over the bin, in minutes
basal_rate(t) = basal_dose(t) × MINUTES_PER_HOUR / I          (the time-weighted mean U/h)
```

With a constant rate `r` across the bin, `basal_dose` is `r × I / 60`: the person's `basal_hourly /
12` at `I = 5`. `I` is the processor's configured interval; 60 is a named `MINUTES_PER_HOUR`
constant; there is no literal 12. A rate change inside a bin is integrated piecewise, exactly.

**Basal is not folded into `insulin_fast`** (Q1, settled by the training inputs, §5):
`insulin_fast(t) = bolus(t)`, and basal travels separately as `basal_rate(t)` in U/h, which is the
pair every model trained in `glucose_forecasting_metabonet` reads. The folded form, `insulin_fast(t) =
bolus(t) + basal_dose(t)`, is the person's original request and equals MetaboNet's raw `insulin`
total; it stays reachable as an opt-in `fold_basal=True` (default from a class attribute, off), for a
consumer that wants total delivered insulin per row.

**Three-valued.** The pump's coverage is from its first to its last row. A bin wholly inside it gets
a number, 0 included (a suspended hour delivers 0 U, measured). A bin wholly outside it gets null
(the pump did not say). A bin crossing either edge gets null, and the delivery inside it is counted
in the warning below. Basal before the pump frame's first `BASAL_RT` row is unknown, so a bin there
is null even when a bolus falls in it, and that bolus is counted too.

**What is never folded, said once with totals.** Insulin falling outside every CGM sequence's span
(between sequences, before the first, after the last) is not attached to any reading; the warning
gives the total units and the time range. One answer for that case, not three.

**Duplicates.** Monthly exports overlap (§2.5), so before integrating the fold drops pump rows
identical in `datetime`, `event_type` and every data column, and reports how many. Two genuine
boluses identical to the second are not something a pump logs.

**Which source the insulin and carbs come from** is a `source` argument (the person, Q4), an
`EnumLiteral` `FoldSource` with three members, because the CGM frame's own insulin and carb rows
partly duplicate the pump's (§2.5):

- `PUMP` (the default): boluses and carbs from the pump frame; the CGM frame's insulin and carb rows
  are reported in a warning with their totals, not folded.
- `CGM`: boluses and carbs from the CGM frame; the pump contributes basal only.
- `MERGE`: both, **paired within a slot threshold, then placed on one slot**. Per quantity (insulin,
  carbs), each CGM-side entry is paired with at most one pump-side entry whose slot is at most
  `merge_pair_slots` slots from its own (an argument, default from the class attribute
  `merge_pair_slots = 1`, the person's choice), nearest in time first, one-to-one. **Beyond the
  threshold nothing merges and both are kept**, each on its own slot. A pair lands
  on the pump entry's slot (the device that delivered the dose and timed the meal); an entry with no
  partner stays on its own slot. The pair's value follows the person's rule: **a 0 or null on either
  side is "this source did not say"**, so the other side's value is taken as it is (a pump 5 U against
  a CGM 0 is 5, not 2.5); only two non-zero values are averaged, and when they differ by more than a
  tolerance (named constants, `FOLD_MERGE_INSULIN_TOLERANCE_U` and `FOLD_MERGE_CARBS_TOLERANCE_G`) the
  pair is listed in one aggregated warning with both values and both times.

Why a window and not the slot. Snapping each source to the grid is what the bins already do, and it
is not enough: the two devices log one event minutes apart. Measured on the real pair (the five
overlapping days), among entries within 15 minutes of each other and equal in amount:

| | pairs | same slot | other slots |
|---|---|---|---|
| insulin | 6 | 5 | 1, two slots apart (7 min) |
| carbs | 6 | 3 | 3, one, one and three slots apart (up to 10 min) |

Same-slot matching (`merge_pair_slots=0`) would count 1 of 6 doses and 3 of 6 meals twice. The
default of 1 slot catches 5 doses and 5 meals and still keeps two pairs apart: the dose two slots
apart (exactly equal amounts, logged in the CGM app 7 minutes after the pump started it, with no other
dose on either side within 30 minutes, so almost certainly one dose) and the meal three slots apart.
Those count twice at the default and merge at `merge_pair_slots=2` and `3`. The person chose 1 (2026-10-21:
"1 slot is to be default threshold, more = no merge, keep both"). The fold's warning gives the number
of unpaired entries that have a same-kind entry from the other source within three slots, so a run can
see what a wider threshold would merge. A pair also forms between entries whose amounts differ (6 insulin and 2 carb pairs within 15 minutes here): those take
the mean and land in the warning with both values, which is the person's "warn where std deviation
is larger than expected with values". Two different doses within the threshold (a correction right
after a meal bolus) can be paired by mistake; the warning is where that shows, and `PUMP` or `CGM`
avoids pairing altogether.

**The output is a terminal view**: `sequence_id` (the CGM sequence the grid point belongs to),
`datetime`, `glucose`, `carbs`, `insulin_slow` (null), `insulin_fast`, and `basal_rate` as the
time-weighted mean U/h over the bin. It is what `to_sugarone_ready_df` (RM21, §3.5) renders for the
models. That is a different meaning from the parsed `BASAL_RT` row's "in force from here",
which is why the wide view carries no `event_type` and the fold refuses a frame without one: a folded
frame cannot be folded again, by construction rather than by luck. The inputs are not modified;
losslessness holds because the parsed frames keep every row and the fold is a derived view.

**Determinism.** Same inputs, same bytes: the output is ordered by `datetime`, every sum is over a
sorted group, and the duplicate drop keeps the first in a total order. The fold runs after
`synchronize_timestamps` and `interpolate_gaps` in either order, since both leave the grid where it
was; a test holds that.

### 3.3 The collapse sums amounts (RM18)

Separate from the pump work because it changes `to_data_only_df(drop_duplicates=True)` for every
vendor. Amounts (`carbs`, `insulin_slow`, `insulin_fast`) are summed per grid point **when at least
one is non-null, else null**; every other column keeps first-non-null. A Dexcom pen user's two
boluses on one grid point stop losing the second.

### 3.5 The models' input shape (RM21)

`to_ml_ready_df` cannot carry pump basal: it fills `Basal Rate (U/h)` only from `insulin_slow` behind
an opt-in, and it reads a unified frame with `event_type`, which the fold's output does not have. The
person (2026-10-21): "keep this output as legacy, build a to_sugarone_ready_df output node meeting
the format the trained models expect".

So a new classmethod, `to_sugarone_ready_df(folded)`, renders the fold's output in the shape the
models in `glucose_forecasting_metabonet` read, taken from that repo's own definitions at build time
(`src/common/data/columns.py`: `sequence_id`, `User ID`, `Timestamp` in `%Y-%m-%dT%H:%M:%S`,
`Event Type`, `Glucose (mg/dL)`, `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`; the
loader for which are required and in what order). `Basal Rate (U/h)` comes from `basal_rate`,
`Bolus Insulin (U)` from `insulin_fast`; nulls stay null, because the training loader fills them
itself (basal forward-filled, bolus zero-filled) and a 0 would claim none was given. `to_ml_ready_df`
stays as it is, legacy, for frames without a pump.

### 3.4 Decided here, reversible, listed for review

- **`MEDTRONIC_PUMP` is a variant format**, not a mode of `MEDTRONIC`: the target schema is per format
  (§3.1). Rejected: one format with a content-dependent shape.
- **The bin is `[t − I/2, t + I/2)`**, matching how `synchronize_timestamps` snaps a reading to its
  nearest grid point. Rejected: the causal `(t − I, t]`, which keeps a dose up to 2.5 min in the
  future off `t` but disagrees with where every other event already lands.
- **A bolus is placed at its programmed time**, its amount from the delivered row. Rejected: the
  delivered row's time, which lags by up to 5 min and more, and crosses a bin for some.
- **Every suspend is 0 U/h until `NORMAL_PUMPING`.** Rejected: taking `ALARM_SUSPEND` at face value
  (counting basal through a stop the pump reported) and null (the pump did say: it was stopped).
- **Priming is not insulin to the body.** A cannula fill puts a fraction of a unit under the skin, but
  no export records which fraction. Excluded and counted.
- **A `Percent` temp basal delivers scheduled × percent ÷ 100**, which is how the pump's manual
  describes the feature; the export logs only the schedule, so nothing in it can confirm this.
- **A cancelled square bolus reports its actual delivery time in `Bolus Duration`**, so its delivered
  volume is spread over that. The parser warns, naming the count, on any square part whose delivered
  volume is below its selected volume while its duration equals the programmed one: the case where
  this assumption would be false shows up on the data that falsifies it.
- **A clock change is reported, not corrected.** The pump and CGM clocks are independent; aligning
  them needs a reference neither export carries. A pump-to-CGM offset is not detectable by any check
  this plan has (a total-insulin check gives the same total under any shift).

---

## 4. Legality

- RM16: a new format, two extended columns, two new `UnifiedEventType` members. **Minor.**
- RM17: a new classmethod and a new `FoldSource` enum. **Minor.**
- RM18: a repair of a silent loss, its own changelog line. **Patch**, riding the same release.
- RM21: a new classmethod beside `to_ml_ready_df`, which is unchanged. **Minor.**
- `insulin_slow` no longer carrying a Medtronic basal rate, and fingersticks leaving `glucose` (per
  Q2), are repairs of wrong output, each with its own changelog line saying what a consumer
  compensating for the old output sees change.

One release, 0.13.0.

---

## 5. Where this meets the models

`docs/modification_plan.md` §0.8 found that SugarOne, one of the two forecasting models, trains on
Loop pump users with **two** insulin channels, `Basal Rate (U/h)` and `Bolus Insulin (U)`, and that
the channel ships empty because nothing cgm_format parsed held a pump rate (the only candidate was a
Dexcom long-acting dose, which is not one). A pump export is the first source that holds a real rate.
The person's fold puts basal and bolus into one `insulin_fast` number per row, which is what a
single-channel consumer wants and loses the split SugarOne's channels need. Hence Q1, and the
person's answer: "Real training inputs take precedence".

**What the training inputs are** (read 2026-10-21 by an agent in `GlucoseDAO/glucose_forecasting_metabonet`
at `origin/main`, every claim with a file and line in its report):

- Every model family trained there (SugarOne, SugarJepa2-288, CITRAS, KronLSTM, ConvTransformer,
  CF-JEPA, GBM, KANMixer, Chronos, GluoFM) reads four channels: `glucose`, `basal` in **U/h**, `bolus`
  in **U**, `carbs`. Basal and bolus are always separate (`common/data/columns.py`).
- Basal is forward- then back-filled; bolus is zero-filled as an event (`loading.py`). A combined
  channel would be zero-filled as an event, so basal would lose its carry-forward, and nearly every
  row of a channel trained on sparse spikes would carry a small dose.
- MetaboNet's raw parquet holds `basal` as **U per 5-minute bin**, which the repo's adapter multiplies
  by 12 to get U/h, `bolus` in U, and `insulin` = basal + bolus. No model reads `insulin` as a channel;
  it enters only as a fallback into the bolus channel when both are null.
- The shipped SugarOne and SugarJepa2-288 scalers span basal [0, 36] U/h and bolus [0, 25] U.
- The raw data cannot tell a recorded 0 from a missing value; the training code fills nulls itself and
  warns that a zero "would be a claim that none was given". Emitting null for "not recorded" and 0 only
  for a recorded zero is what the models want, and what this library's three-valued rule says anyway.

So the wide row's `insulin_fast` (bolus, U) and `basal_rate` (U/h) are the two training channels
exactly, with no conversion on the consumer's side.

---

## 6. Settled by the person, and what is still open

Answered 2026-10-21:

- **Q1, the basal channel: not folded**, settled by the training inputs. The person: "Before deciding,
  send opus agent into glucose-metabonet repo and see how model actually trains. ... For now default
  (3) don't fold unless clarified otherwise. Real training inputs take precedence." The training repo
  reads basal (U/h) and bolus (U) as separate channels (§5), so the default stays: `insulin_fast` is
  bolus, `basal_rate` is the bin's mean U/h, and the fold is an opt-in (§3.2). Rejected: the fold as
  default (matches only MetaboNet's raw total, which no model reads), and basal as U per 5 minutes
  (MetaboNet's raw unit, 12 times below what the models are scaled on).
- **Q2, fingersticks: a new meter event type**, `BG_METER`, on both Medtronic formats (§3.1).
  Rejected: leaving them out with a warning (the recommendation), and keeping them as `glucose`.
- **Q3, the data: same person, no more exports.** The real-data check covers the five overlapping days
  only.
- **Q4, the source of insulin and carbs: an option**, "pump, cgm, merge-as mean; merge strategy must
  see 0s as none, i.e. 0;5 is not 2.5, it's 5, mean only for two non-zero vals in both sources. Warn
  where std deviation is larger than expected with values" (§3.2). Rejected: pump only (the
  recommendation), a carb-matching heuristic, and summing both. The default is `MERGE`, by the
  person's word before the release ("set merge to default"): on the real overlap `PUMP` left 24 of the
  CGM app's 30 meals off every row. Rejected: `PUMP` as default (the plan's first pick).

---

## 7. Tests and fixtures

- A **synthetic** 720G export committed under `data/input/` with `git add -f` (F2), written by
  `scripts/make_synthetic_medtronic_pump.py`: hand-authored rather than scrubbed from a real export,
  one or two rows per record kind, so that its expected values can be written by hand from the
  script's row list, never by running the parser. The real layout (BOM, metadata, sections) is kept.
- Parser: every row kind in §2.2, including a percent temp across a schedule change, two rate rows on
  one timestamp, a dual bolus, a plain square bolus, a bolus with no volume, an alarm suspend, a
  rewind, a priming, both fingerstick sources as `BG_METER` rows (and none as `EGV_READ`), a filled
  `New Device Time`. The Guardian fixture's `EGV_READ` count drops by exactly its fingerstick rows.
- Fold: constant rate gives `rate × I / 60`; a mid-bin rate change integrates exactly; a square bolus
  spreads; two boluses in one bin sum; a bin outside pump coverage is null and one inside a suspend
  is 0; insulin between sequences is reported, not folded; overlapping pump exports fold to the same
  result as one; the CGM frame's glucose and grid are byte-identical before and after; the output is
  the same whichever order `interpolate_gaps` and `synchronize_timestamps` ran in; folding the output
  is refused. Basal: unfolded by default (`insulin_fast` = bolus, `basal_rate` = the bin mean in U/h),
  and folded with `fold_basal=True` (`insulin_fast` = bolus + `rate × I / 60`). Source: `PUMP`, `CGM`, and
  `MERGE` with a 0 on one side (takes the other), two agreeing non-zeros (the mean), two disagreeing
  beyond tolerance (the mean, and the pair in the warning with both values), a pair one slot apart
  (merged onto the pump's slot), a pair two slots apart (both kept at the default, merged at
  `merge_pair_slots=2`), and two CGM entries near one pump entry (only the nearer pairs).
- RM18: two boluses on one grid point sum; an all-null group stays null.
- The real files: each of the four parses. Over the overlap with the Dexcom export, the folded total
  equals the pump's own total (basal integral + delivered boluses) minus what the warnings report,
  which proves the arithmetic and not the clocks. Skipped where the files are absent.

## 8. Order and size

1. **RM16**, the parser and the synthetic fixture, about 5 hours.
2. **RM17**, the fold, about 4 hours, after RM16.
3. **RM18**, the collapse, about 1 hour, independent.
4. **RM21**, `to_sugarone_ready_df`, about 2 hours, after RM17.

The full suite runs once at the end, in the background.
