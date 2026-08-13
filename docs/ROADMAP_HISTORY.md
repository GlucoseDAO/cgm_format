# Roadmap history

Shipped roadmap items, moved here from [ROADMAP.md](ROADMAP.md) with the rationale intact. Nothing
is deleted from the roadmap; it is relocated. Newest first.

An item arrives here when the work shipped — keep the `RMn` id it had, and add what actually
changed and where, so a later reader can find the code without a git archaeology session.

---

## RM5 — European Libre mmol/L variant, and Libre scan/strip parsing

**Severity:** medium · **Status:** shipped 2026-08-13 in 0.9.0 · **Owner:** —

This never sat on the active roadmap. It was implemented from a real LibreView export
(`data/input/JonGrove_glucose_13-8-2026.csv`, local, gitignored) in the same change that
found it. Recorded here so the id is reserved and the code is findable.

**Legality:** minor. A new `SupportedCGMFormat.LIBRE_EU` member, two appended ketone columns on a
derived vendor schema, and newly parsed rows of existing unified event types (`EGV_READ`,
`CALIBRAT`). Nothing a consumer already reads was removed or redefined. `EGV_READ` still means
sensor glucose.

**What shipped:**

- `formats/libre_eu.py` / `formats/libre_eu.json` — `LIBRE_EU_SCHEMA` derived from `LIBRE_SCHEMA`
  via `derive_schema` (renames + units + `append_data_columns` for `Historic Ketone mmol/L` and
  `Scan Ketone mmol/L`). Detection pattern `"Historic Glucose mmol/L"`.
- Registry: `LIBRE_EU` before `LIBRE` in `FORMAT_DETECTION_PATTERNS` (`formats/supported.py`), same
  reason `DEXCOM_EU` precedes `DEXCOM`.
- `FormatParser._process_libre(..., european=True)` and a shared
  `FormatParser._glucose_to_canonical` used by both Dexcom and Libre. Canonical unit named once as
  `CANONICAL_GLUCOSE_UNIT` in `formats/unified.py`.
- Record Type 1 (scan glucose) merges into `EGV_READ` with historic readings. Record Type 2 (strip)
  parses as `CALIBRAT`. All Libre files, not only the mmol/L variant.
- Tests: `tests/test_libre_eu.py` (skips when the local fixture is absent); scan coverage and
  `TIME_DUPLICATE` flagging on the committed `FreeStyle_Libre_3_synthetic.csv`.
- Docs: `docs/CHANGELOG.md` 0.9.0, `docs/NEW_SCHEMA.md` (the old "zero parser changes" mmol/L
  claim), `docs/PIPELINE.md`.

**Left open, not this item:** F1 in `docs/dogfooding.md` (whether colliding scan/historic timestamps
should be reconciled rather than flagged); F2 (new `data/input/` fixtures cannot be committed
without `git add -f`); RM6 (Libre ketone columns sit on the vendor schema and are dropped at
parse — unified has nowhere to put them).
