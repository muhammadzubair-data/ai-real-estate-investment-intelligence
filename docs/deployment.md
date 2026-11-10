# Deployment guide

## Streamlit Community Cloud

The app reads the built files in `data/demo/` (about 30 MB), so build them locally first and commit them.

1. Build the data (see [HANDOVER.md](HANDOVER.md)) and commit `data/demo/`.
2. Push the repository to GitHub.
3. On [share.streamlit.io](https://share.streamlit.io) choose **Create app → Deploy a public app from GitHub**.
4. Pick the repository and branch; **Main file path:** `app/streamlit_app.py`; **Python:** 3.12. No secrets needed.
5. Deploy. Dependencies install from `requirements.txt` (its last line, `-e .`, installs the `reintel` package).

Without `data/demo/`, the app opens on a page explaining how to build the data instead of failing.

Maps use the free Carto basemap through Plotly, which needs no API key.

## Local

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py
```

To point the app at another build folder: `REINTEL_DATA=/path/to/folder streamlit run app/streamlit_app.py`.

## Monthly refresh

1. Download the latest `pp-2026.csv` (Land Registry updates it monthly) and the latest ONS rents workbook into
   `data/raw/`.
2. `python scripts/build_data.py` and `python scripts/make_report.py`.
3. Check the latest results against the previous `docs/results.md`, then commit and push. Streamlit Cloud
   redeploys automatically.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request (Python 3.11 and 3.12): `ruff` lint, the full pytest
suite (no downloads needed), and a check that the app starts and answers its health endpoint.

## Moving towards production

* Schedule the monthly refresh (GitHub Actions cron, Airflow or a simple server cron job).
* Add Energy Performance Certificate data for floor area and rooms, after reviewing its licence terms. This is the
  single biggest accuracy improvement available.
* Add live listings from a licensed feed (asking prices, time on market) to replace the recent-sales stand-ins.
* Put user accounts in front of the app if the client's use goes beyond residential price information (see the
  address-data terms in [data_provenance.md](data_provenance.md)).
