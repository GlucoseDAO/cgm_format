# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-11-27, unattended.** 0.14.0 through 0.15.0 are tagged and on GitHub; `dist/` holds all four.
**Open: the person's PyPI publish** (`uv publish dist/*.whl dist/*.tar.gz`; PyPI is at 0.13.0; `dist/` also holds
`notes-0.15.0.md`, the release notes, which is not a dist).
**0.15.0** (2026-11-25, `a268364`, tag `v0.15.0`) carries RM14, RM15 and `--date-order`. Desk #343 answered 1: the
compatibility rule sizes a release and the lead cuts a minor (now in `LEAD.md`'s grant, `3e26d54`). Its GitHub release carries both dists: the sdist went up 2026-11-27, once kit #649 (forge gate on
already-pushed lines) shipped in v0.1.282; #644 before it. Kit pin: v0.1.290 (2026-11-27); held releases arrive at the daily Claude batch, around 12:27 CET, and the watchdog says when to `--take`.
reflect after 2026-12-04: classes/base/BASE.md as of v0.1.287 (a step handed to the person carries its exact action; a wake let go counts itself).
The tag's build worktree goes under `.worktrees/` by the kit's `bin/wt.sh NAME`, never the scratchpad (base, kit
#536, from v0.1.236). The grant covers removing it, but the auto-mode classifier refused `git worktree remove`
(2026-11-25), so it is unlocked and handed over: `unlocked: to delete git -C /data/sources/glucosedao/cgm_format
worktree remove .worktrees/build-0.15.0 && git -C /data/sources/glucosedao/cgm_format branch -d build-0.15.0`. After
each pin move, `bin/automode.py --check`.

**RM7 blocked: desk #362** (re-filed 2026-11-18 with each option's cost, superseding #325; no default): the total-order sort cannot stream (OOM under a 1 GB cap on a 1.95 GB file; unsorted fits), so #107's `LazyFrame` saves only ~25 % unless the reader keeps file order. Measurements in the roadmap item. CI was red from 2026-10-21 to 2026-11-15 on the collection-time
Nightscout pull timing out; fixed in `66e940d` (fall back to fixtures with a warning), green on 3.10-3.14.

**Withdrawn 2026-11-14, default applied** (the secretary's nightly sweep): #63 (the drift fixes built as RM22 and
shipped in 0.14.1; localized headers, Tidepool, AZT1D, Jaeb stay unfiled until a user or dataset asks), #64 (no
Excel), #65 (no split values). #106-#109 got an **Update:** that they still hold. #71 answered 2026-10-27
(option 1): RM3 built as `8825c7b`, pushed, unreleased; it rides the same release as RM4.

The joint Dexcom G6 + 720G dataset is on a private Hugging Face dataset with its repro scripts
(`repro/`, cgm-format pinned to `5ba26a8`, rebuild byte-identical). Whether `build.sh`'s output names
match the person's uploaded names is unconfirmed.

**The plan's order** (`docs/ROADMAP.md`): RM7 (desk #362), RM11 (after RM7), RM20 (next major only), RM6 (waits until 2027-01-21). Desk answers of 2026-11-15: #107 option 1 (RM7
returns a `LazyFrame`, no Parquet helper), #108 option 2 (RM14), #109 option 1 (RM15: `NightscoutError` kinds
that also subclass the httpx errors, a minor).

**In flight:** nothing.

**Acting pm duty "milestones"** (kit v0.1.187, no pm seat here): `docs/seats/LEAD_STATE.md` is the source of
the one post kept in `#cgm-format>state`; after editing it, `watchdog.sh keep lead state '#cgm-format>state'
docs/seats/LEAD_STATE.md` edits that post in place. A landed result also gets one post in
`#cgm-format>milestones`.

**Polars 2:** 2.2.0 is out (2026-11-24) and the floor `polars>=1.39.3` has no cap, so a fresh install takes it
while the lock holds 1.43.2. The full suite on `74ca94b` under 2.2.0 (`uv run --with 'polars==2.2.0'`, the
lock untouched, deprecations as errors), 2026-11-24: 1757 passed, 16 skipped, no warning from the suite.
2.1.0 on `a259264`, 2026-11-05: 1715 passed, 16 skipped. Nothing to cap or port. A 2.2.0 regression another
repository reported (a filter mask from `concat_list` + `list.any` on an all-null list column panics a grouped
`median`) does not reach us: `src/` calls none of the three (grep, 2026-11-24). Polars 2 writes its version into every parquet footer, so parquet bytes differ across
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
