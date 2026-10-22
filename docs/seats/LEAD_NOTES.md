# Lead — notes

State, not rules: rewritten to the present before a reset (`README.md` here, "Notes files").

**As of 2026-10-22, unattended.** 0.14.0 and 0.14.1 are tagged and on GitHub with their dists; `dist/`
holds both. **Open: the person's PyPI publish of both** (`uv publish dist/*`; PyPI is at 0.13.0).
0.14.1 (`v0.14.1` on `1792ca2`, full suite 1,712 passed) carries RM22 and RM1.

**On the desk, asked 2026-10-22 while the person was away:** #63 which surveyed formats come next
(default: the drift fixes, now built as RM22); #64 Excel behind the `cli` extra (default: no Excel);
#65 split / study-group values in `to_sugarone_ready_df` (default: no); #71 the `data/` layout (RM3
blocked on it; default: nothing moves).

The joint Dexcom G6 + 720G dataset is on a private Hugging Face dataset with its repro scripts
(`repro/`, cgm-format pinned to `5ba26a8`, rebuild byte-identical). Whether `build.sh`'s output names
match the person's uploaded names is unconfirmed.

**The plan's order** (`docs/ROADMAP.md`): RM20 (next major only), RM3 (blocked: desk #71), RM4 (next
to build: `.env` loaded at CLI startup, `download-nightscout` reading `NIGHTSCOUT_*`, a
`.env.template`, python-dotenv into the `cli` extra; test with a temp `.env` naming a closed local port
and the repo's own credentials stripped from the subprocess env), RM6 (waits until 2027-01-21), RM7, RM11 (after RM7), RM14, RM15. `docs/FORMAT_SURVEY.md` §5 proposes the next items
(drift fixes, localized headers, Tidepool, AZT1D, Jaeb); none filed yet, the person to choose.

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
