# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-10-21, 20:00.** 0.13.0 is released on GitHub (tag `v0.13.0` on `098a75f`, both dists
attached): pump support (RM16, RM17, RM21), RM18, RM2, the pre-release audit's eleven fixes, and
0.12.3's untagged Nightscout fix. Final full suite on the tagged commit: 1,656 passed, 16 skipped.
Published to PyPI by the person the same evening (0.13.0 is PyPI's latest, wheel and sdist). Kit pin at v0.1.87; the person still has to
/reload-plugins in both sessions and paste the revised allow entry.

**The plan's order** (`docs/ROADMAP.md`): RM19 (triage's), RM20 (next major only), then RM1, RM3, RM4,
RM6 (waits until 2027-01-21), RM7, RM11 (after RM7), RM14, RM15.

**In flight:** nothing.

- The fold's default source is MERGE, by the person's word before the release.
- Real data: four MiniMed 720G exports and an overlapping Dexcom export (five days, same person, no
  more coming) in the git-ignored `data/input/`. Never name the contributor in a tracked file.
- A test that loops over every file in `data/input/` with the core processor breaks on an
  extended-schema export: pick the processor by the frame's schema.
- The leak list matches this machine's name, which is also a month: write months as ISO dates.
- The kit's forge gate reads `gh` arguments before the shell does: write file paths out (no shell
  variables) and add no `2>&1` to a `gh` call.
- `/data/sources/glucosedao/glucose_forecasting_metabonet` is a partial clone (checkout failed on
  NTFS names); read it through `git show origin/main:<path>`, never repair it unasked.

**Found, recorded:** F19 (a non-glucose row moves its sequence's grid; joins any sequence however
far). Kit issues filed: #110 (closed as a duplicate of #108), #112 (`automode` prints blank lines the
client refuses).
