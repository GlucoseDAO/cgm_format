# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-11-15, unattended.** 0.14.0, 0.14.1 and 0.14.2 are tagged and on GitHub with their dists;
`dist/` holds all three. **Open: the person's PyPI publish** (`uv publish dist/*`; PyPI is at 0.13.0).
0.14.2 (desk #106: patch) carries RM4 and RM3.

**RM14 built** (`51e4b89`, unreleased): `track_agreement()` and the warning on `track="mean"`. Its release is
0.15.0, a minor: desk #324 (asked 2026-11-15; recommended 0.15.0 once RM15 is built; default no release).
**RM15 built** (`f64c72b`, unreleased; rides #324). **RM7 blocked: desk #325** (asked 2026-11-15; no default): the total-order sort cannot stream (OOM under a 1 GB cap on a 1.95 GB file; unsorted fits), so #107's `LazyFrame` saves only ~25 % unless the reader keeps file order. Measurements in the roadmap item. CI was red from 2026-10-21 to 2026-11-15 on the collection-time
Nightscout pull timing out; fixed in `66e940d` (fall back to fixtures with a warning), green on 3.10-3.14.

**Withdrawn 2026-11-14, default applied** (the secretary's nightly sweep): #63 (the drift fixes built as RM22 and
shipped in 0.14.1; localized headers, Tidepool, AZT1D, Jaeb stay unfiled until a user or dataset asks), #64 (no
Excel), #65 (no split values). #106-#109 got an **Update:** that they still hold. #71 answered 2026-10-27
(option 1): RM3 built as `8825c7b`, pushed, unreleased; it rides the same release as RM4.

The joint Dexcom G6 + 720G dataset is on a private Hugging Face dataset with its repro scripts
(`repro/`, cgm-format pinned to `5ba26a8`, rebuild byte-identical). Whether `build.sh`'s output names
match the person's uploaded names is unconfirmed.

**The plan's order** (`docs/ROADMAP.md`): RM14 and RM15 (built, desk #324), RM7 (desk #325), RM11 (after
RM7), RM20 (next major only), RM6 (waits until 2027-01-21). Desk answers of 2026-11-15: #107 option 1 (RM7
returns a `LazyFrame`, no Parquet helper), #108 option 2 (RM14), #109 option 1 (RM15: `NightscoutError` kinds
that also subclass the httpx errors, a minor).

**In flight:** nothing.

**Acting pm duty "milestones"** (kit v0.1.187, no pm seat here): `docs/seats/LEAD_STATE.md` is the source of
the one post kept in `#cgm-format>state`; after editing it, `watchdog.sh keep lead state '#cgm-format>state'
docs/seats/LEAD_STATE.md` edits that post in place. A landed result also gets one post in
`#cgm-format>milestones`.

**Polars 2:** 2.1.0 is out and the floor `polars>=1.39.3` has no cap, so a fresh install takes it while
the lock holds 1.43.2. The full suite on `a259264` under 2.1.0 (`uv run --with 'polars==2.1.0'`, the
lock untouched), 2026-11-05: 1715 passed, 16 skipped, no warning, the same counts as on 1.x. Nothing
to cap or port. Polars 2 writes its version into every parquet footer, so parquet bytes differ across
polars releases; no test here compares parquet bytes across runs.

- The fold's default source is MERGE, by the person's word before the release.
- Real data: four MiniMed 720G exports and an overlapping Dexcom export (five days, same person, no
  more coming) in the git-ignored `data/input/`. Never name the contributor in a tracked file.
- Committed fixtures are in `assets/fixtures/`, local exports in `data/input/` (RM3); tests take
  paths from `tests/paths.py`.
- A test that loops over every input file with the core processor breaks on an
  extended-schema export: pick the processor by the frame's schema.
- The leak list matches this machine's name, which is also a month: write months as ISO dates.
- The kit's forge gate reads `gh` arguments before the shell does: write file paths out (no shell
  variables) and add no `2>&1` to a `gh` call.
- `/data/sources/glucosedao/glucose_forecasting_metabonet` is a partial clone (checkout failed on
  NTFS names); read it through `git show origin/main:<path>`, never repair it unasked.

**Found, recorded:** F19 (a non-glucose row moves its sequence's grid; joins any sequence however
far). Kit issues filed: #110 (closed as a duplicate of #108), #112 (`automode` prints blank lines the
client refuses).
