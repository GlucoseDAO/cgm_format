**cgm-format: where it stands** (kept by the lead, acting for the pm; edited in place)

✅ 0.15.0 is tagged and on GitHub (2026-11-25), sized and cut by the lead on desk #343's answer: a measure of how well two sensors agree with a warning when averaging two that disagree (RM14), Nightscout download errors as the library's own exception types (RM15), and `--date-order` on the CLI (desk #70). All three only add.
⏳ PyPI still serves 0.13.0. The maintainer's step: in the clone's root, in your own terminal with your PyPI token, run `uv publish dist/*.whl dist/*.tar.gz`; done when pypi.org/project/cgm-format shows 0.15.0.
✅ The full suite passes on Polars 2.2.0 (1757 passed, 16 skipped, deprecations as errors), so no upper version cap is needed.

**Next**
1. ❓ Lazy streaming reader for large exports, RM7 (lead; waits on desk #362: measured, the library's fixed row order keeps the whole table in memory, so should the reader keep the file's order instead).
2. ⏳ Loop dataset support, RM11 (after RM7).
