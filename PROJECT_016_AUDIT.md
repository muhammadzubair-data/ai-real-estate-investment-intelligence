# Project 016 — Technical audit and handover

Audit date: 2026-11-10. Scope: uploaded `ai-real-estate-investment-intelligence.zip`.

## Verified
- Archive inspected: Python package, seven Streamlit page modules, ingestion, valuation, forecast, finance modules, test suite, CI workflow, documentation, and illustrative images present.
- Python syntax compilation passed (`python -m compileall -q src app scripts tests`).
- Initial local `pytest -q`: **26 passed, 1 failed, 16 errors**. Errors arise during fixture parquet output because `pyarrow`/`fastparquet` is not installed in this audit environment. The failure is a Streamlit import because Streamlit is also absent. These are environment blockers, not verified application bugs.
- Synthetic fixture training reached model fitting, geographic holdout, and forecast stages before parquet output failed. This is partial pipeline evidence, not a complete build verification.

## Changes made in this package
1. Replaced fragile exact runtime version pins with bounded Python 3.11/3.12 compatible release ranges; this **has not yet been pip-resolved in a clean environment**. In particular the original pins included `numpy==2.5.3` and `shap==0.54.1`, which previously caused install failures in a similar portfolio project.
2. Updated README wording so real-data claims are clearly conditional on downloading and processing the actual public datasets.
3. Removed embedded `.git` directory and generated Python cache from deliverable ZIP; preserved all application code, tests, docs, and visual assets.

## Blockers before claiming a finished live product
- `data/raw/` contains only instructions, not the source UK Land Registry / ONS / OS datasets.
- `data/demo/` contains only `.gitkeep`, not a built `manifest.json`, Parquet data, or serialized model. The Streamlit app requires built data.
- Cannot reproduce real-world metrics, model coverage, 12-city coverage, valuation quality, or deployed dashboard from this ZIP alone.
- `ruff` is unavailable in the audit environment, so lint has not been run locally.
- Run CI for both Python 3.11 and 3.12. Resolve any package constraints and test errors. Do not report a green CI before it actually passes.
- Run the real-data build using `docs/HANDOVER.md`, then `scripts/make_report.py`, validate seven pages, and review data licenses and sensitive address treatment before publishing generated artifacts.

## Recommended publication sequence
1. Upload code and documentation to GitHub (do not call it fully validated).
2. Confirm CI success on both Python versions.
3. Download sources following `data/raw/README.md` and `docs/HANDOVER.md`; run `python scripts/build_data.py`.
4. Run `python scripts/make_report.py`, inspect actual holdout and calibration metrics, and review financial assumptions.
5. Commit only appropriately licensed, size-appropriate derived demo artifacts; do not upload raw national datasets or private addresses.
6. Deploy `app/streamlit_app.py` and smoke-test all seven pages.

**Verdict:** Codebase is structured and suitable for a GitHub verification stage. **Not yet proven deployment-ready** because the public-data build, end-to-end tests, and actual Streamlit runtime are unverified.
