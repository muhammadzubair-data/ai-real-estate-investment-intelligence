# Handover: finishing the build

Everything except the real-data run is done and tested. These steps take about 30 minutes, most of it downloading.
Commands are for Windows PowerShell; on Mac/Linux use `source .venv/bin/activate` instead.

## 1. Set up (once)

```powershell
cd ai-real-estate-investment-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q          # should end with "passed"; no data needed
```

## 2. Download the five public files into `data\raw\`

Keep the original file names.

| File | Link |
|---|---|
| `pp-2024.csv` (157 MB) | https://price-paid-data.publicdata.landregistry.gov.uk/pp-2024.csv |
| `pp-2025.csv` (129 MB) | https://price-paid-data.publicdata.landregistry.gov.uk/pp-2025.csv |
| `pp-2026.csv` (latest) | https://price-paid-data.publicdata.landregistry.gov.uk/pp-2026.csv |
| ONS rents workbook (`...privaterents...xlsx`) | https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics (latest edition) |
| `codepo_gb.zip` | https://osdatahub.os.uk/downloads/open/CodePointOpen (choose CSV) |

## 3. Build the data and models

```powershell
python scripts\build_data.py
```

It prints progress for each stage and finishes with a summary. A benchmark at the same scale took about a minute
on a 2-vCPU machine; reading the national files adds a few minutes.

**If it stops at the rents step** with "Could not find local-authority rents", ONS has changed the workbook layout.
Run this and send me the output:

```powershell
python scripts\inspect_rents.py data\raw\<the rents file>.xlsx
```

## 4. Write the real results into the docs

```powershell
python scripts\make_report.py
```

This fills the Results sections of `README.md`, `docs/model_card.md` and `docs/case_study.md`, and writes
`docs/results.md` and `docs/client_message.txt` (the Upwork handover message, with the real figures). Read the
Results section once: if a baseline beat LightGBM on your data, the text says so honestly. Keep it that way.

## 5. Look at the app

```powershell
streamlit run app\streamlit_app.py
```

Click through all seven pages. Things to check with real data:

* maps on the Valuation and Geospatial pages show the Carto basemap (needs internet);
* the city list on page 1 has all 12 cities (a city missing from the ONS rents will show blank yields; the build
  prints a warning naming it);
* the Forecasting page says whether each forecast beats "no change".

## 6. Screenshots and thumbnail

```powershell
pip install playwright
python -m playwright install chromium
python scripts\screenshots.py
```

This writes `assets\screenshots\*.png` and `assets\thumbnail.png` using the real numbers.

## 7. Commit, push, deploy

```powershell
git add -A
git commit -m "Build with real Land Registry, ONS and OS data"
git remote add origin https://github.com/muhammadzubair-data/ai-real-estate-investment-intelligence.git
git push -u origin main
```

`data/raw/` is ignored (large, re-downloadable); `data/demo/` (about 30 MB) is committed so the app runs on
Streamlit Cloud. Then follow [deployment.md](deployment.md) (main file `app/streamlit_app.py`).

## 8. Send the client message

Paste `docs/client_message.txt` into the Upwork inbox (plain text, no tables) along with the repo and app links.
