**cgm-format: where it stands** (kept by the lead, acting for the pm; edited in place)

✅ 0.14.2 is tagged and on GitHub, and CI is green again on every Python version (a test setup that broke on an unreachable server was fixed): the CLI reads `.env` (RM4), and committed test files live in `assets/fixtures/` (RM3).
⏳ PyPI still serves 0.13.0; publishing 0.14.0, 0.14.1 and 0.14.2 there is the maintainer's step.
✅ The full suite passes on Polars 2.1.0 (1715 passed, 16 skipped), so no upper version cap is needed.

**Next**
1. ⏳ A measure of how well two sensors agree, and a warning when averaging two that disagree, RM14: built on `main`, unreleased (its release, 0.15.0, waits on desk #343, which asks who sizes a minor release).
2. ⏳ Network errors as the library's own exception types, RM15: built on `main`, unreleased (rides 0.15.0 with RM14, desk #343).
3. ❓ Lazy streaming reader for large exports, RM7 (lead; waits on desk #325: measured, the library's fixed row order keeps the whole table in memory, so should the reader keep the file's order instead).
