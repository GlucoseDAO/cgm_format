# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-10-21, evening.** First seating done: roadmap state lines (`7018ceb`), `CLAUDE.md` for the
seats (`71d1f85`), the leak gate on, the kit pinned at v0.1.85 (moved by the person's hand; the
auto-mode classifier refuses `move-to-tag` until the allow entry naming it is pasted, which it now
is). Everything committed is pushed.

**The plan's order** (`docs/ROADMAP.md`): RM16 → RM17 (after RM16) → RM18, then RM1 onward. RM19 is
triage's; RM20 waits for the next major.

**In flight: the pump work, planned, not started.** `docs/PLAN_pump.md` is the proposal. The person's
answers are recorded there (§6): fingersticks → `BG_METER`; the fold's `source` option
`PUMP`/`CGM`/`MERGE` (merge reads 0 as "did not say"); basal **not** folded into `insulin_fast`,
because every model in `glucose_forecasting_metabonet` reads basal U/h and bolus U as separate
channels (spot-checked: `src/metabonet/adapter.py` ×12, `src/common/data/columns.py`). The build waits
for the person's go: plans and builds are separate runs.

- Real data: four MiniMed 720G exports and an overlapping Dexcom export (five days, same person, no
  more coming) in the git-ignored `data/input/`. Never name the contributor in a tracked file.
- The four 720G files make `test_validate_against_schema` fail in a local full run until RM16 lands
  (the files are git-ignored, so CI does not see them).
- The leak list matches this machine's name, which is also a month: write months as ISO dates.
- `/data/sources/glucosedao/glucose_forecasting_metabonet` is a partial clone (checkout failed on
  NTFS names); read it through `git show origin/main:<path>`, never repair it unasked.

**Open with the person:** releasing 0.12.3 (triage's S6 exporter fix, version already bumped,
pushed, not tagged).

**Found, recorded:** F19 (a non-glucose row moves its sequence's grid; joins any sequence however
far). Kit issues filed: #110 (closed as a duplicate of #108), #112 (`automode` prints blank lines the
client refuses).
