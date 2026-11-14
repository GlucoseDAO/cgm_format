**cgm-format: where it stands** (kept by the lead, acting for the pm; edited in place)

✅ 0.14.1 is tagged and on GitHub. PyPI still serves 0.13.0, and publishing 0.14.0 and 0.14.1 there is the maintainer's step.
⏳ Built on `main` but not released: committed test files moved to `assets/fixtures/` (RM3), and the CLI reads `.env` (RM4).
✅ The full suite passes on Polars 2.1.0 (1715 passed, 16 skipped), so no upper version cap is needed.

**Next**
1. ❓ Choose the version for RM3+RM4, 0.14.2 or 0.15.0 (lead; waits on desk #106).
2. ❓ Lazy streaming reader for large exports, RM7 (lead; waits on desk #107).
3. ❓ A warning when averaging two sensors that disagree, RM14 (lead; waits on desk #108).
4. ❓ Network errors as the library's own exception types, RM15 (lead; waits on desk #109).
