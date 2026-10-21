# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-10-21, evening.** First seating done: roadmap state lines (`7018ceb`), `CLAUDE.md` for the
seats (`71d1f85`), the leak gate on, the kit pinned at v0.1.85 (moved by the person's hand; the
auto-mode classifier refuses `move-to-tag` until the allow entry naming it is pasted, which it now
is). Everything committed is pushed.

**The plan's order** (`docs/ROADMAP.md`): RM16 → RM17 (after RM16) → RM18, then RM1 onward. RM19 is
triage's; RM20 waits for the next major.

**In flight: the pump work, 0.13.0.** RM16 (parser, `5d916df`) and RM17 (fold, `df57287`) are on main
and pushed; RM21 (`to_sugarone_ready_df`, the models' input shape) and RM18 (the collapse sums
amounts) are next, then a fresh audit of RM16/RM17/RM21 against `docs/PLAN_pump.md`, then the
release. Decisions are in the plan's §6; the merge pairs within `merge_pair_slots` (default 1, the
person's call), basal stays out of `insulin_fast` (the trained models read it apart).

- Real data: four MiniMed 720G exports and an overlapping Dexcom export (five days, same person, no
  more coming) in the git-ignored `data/input/`. Never name the contributor in a tracked file.
- A test that loops over every file in `data/input/` with the core processor breaks on an
  extended-schema export: pick the processor by the frame's schema (done in the idempotency and
  lossless-sync tests).
- The leak list matches this machine's name, which is also a month: write months as ISO dates.
- `/data/sources/glucosedao/glucose_forecasting_metabonet` is a partial clone (checkout failed on
  NTFS names); read it through `git show origin/main:<path>`, never repair it unasked.
- Kit pin at v0.1.87 (moved by the lead); the person still has to /reload-plugins and paste the
  revised allow entry.

**Open with the person:** releasing 0.12.3 (triage's S6 exporter fix, version already bumped,
pushed, not tagged).

**Found, recorded:** F19 (a non-glucose row moves its sequence's grid; joins any sequence however
far). Kit issues filed: #110 (closed as a duplicate of #108), #112 (`automode` prints blank lines the
client refuses).
