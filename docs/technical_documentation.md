# Technical documentation

## 1. Architecture

A batch build (`python scripts/build_data.py`, i.e. `python -m reintel.build`) reads the raw public files, writes
validated tables, models and evaluation results into `data/demo/`, and a Streamlit app reads them.

| Stage | Module | Output in `data/demo/` |
|---|---|---|
| Validate Price Paid Data | `data/ingest.py` | `quality_report.json` |
| Geocode postcodes | `data/ingest.py` (pyproj, EPSG:27700 → 4326) | coordinates in `sales.parquet` |
| Parse ONS rents | `data/rents.py` | `rents.parquet` |
| Features + comparables | `valuation/features.py` | `sales.parquet`, `comparables.parquet` |
| Models + evaluation | `valuation/models.py`, `build.py` | `valuer_*.txt`, `valuer_meta.json`, `metrics.json`, `shap_global.csv` |
| Market indices + forecasts | `market/forecast.py` | `market_monthly.parquet`, `forecasts.parquet`, `market_risk.csv` |
| Schema exports | `build.py` | `properties.parquet`, `transactions.parquet`, `market_monthly.parquet` |
| Provenance | `build.py` | `manifest.json` |

The schema exports follow the column lists in the client's `schemas/*.csv`, with the extra fields of
`market_monthly` (mix-adjusted price, gross yield) appended after the required columns.

## 2. Ingestion and validation

See [data_provenance.md](data_provenance.md) for the rules. Points worth knowing:

* Files are read in 500,000-row chunks and filtered to the covered local authorities straight away, so memory use
  stays low even though each yearly file covers all of England and Wales.
* `property_id` = `"P"` + the first 11 hex characters of SHA-1(postcode | PAON | SAON | street). It links repeat sales
  of the same home without keeping the address.
* Code-Point Open eastings and northings are converted to WGS84 latitude and longitude with pyproj.

## 3. Features (leakage-safe)

`month_features(cur, s, m, idx)` computes features for rows valued in month *m* from sales completed in months
*m − 13* to *m − 2*. Historical prices are first moved to the reference month with a city × type index (the 3-month
rolling log-median), so a sale from 11 months ago counts at today's level.

| Feature | Definition |
|---|---|
| `log_sector_type_median`, `sector_type_count` | median (log) and count of same-type sales in the postcode sector |
| `log_sector_median`, `sector_count` | same, all types |
| `log_district_type_median` | same-type median in the postcode district |
| `log_city_type_median` | same-type median in the city |
| `log_comp_median`, `comp_mean_km`, `comp_max_km` | 10 nearest same-type sales of *other* properties (haversine BallTree) |
| `log_prev_price_indexed`, `months_since_prev`, `has_prev_sale` | the property's own last public sale, moved to today |
| `city_momentum_3m` | city log-median of the last 3 months minus the last 12 |
| `latitude`, `longitude`, `month_of_year` | location and season |
| `city`, `property_type`, `tenure`, `new_build` | categorical and flags |

The same function values a property "today" in the app (`valuation/service.py`), so the live valuation and the
evaluated model use identical code.

## 4. Models and evaluation

* Warm-up: the first 5 months are not modelled (they have little evidence behind them).
* Test: the latest 6 complete months; validation: the 3 months before; training: everything earlier.
* LightGBM: 63 leaves, learning rate 0.04, early stopping (100 rounds) on validation; deterministic training.
* Interval: quantile models at 0.10 and 0.90, widened by *q* = the 80th percentile of the validation non-conformity
  score max(lo − y, y − hi) (conformalised quantile regression).
* Deployed model: refitted on training + validation with the early-stopped tree count; *q* from the validation step.
* Geographic holdout: districts selected by an MD5 hash of the district code (deterministic, about 20%).
* Leave-one-city-out: point model only, trained on the other cities.

Metrics are in £ after converting back from logs: mean and median absolute error, RMSE, mean and median absolute
percentage error, share within 10% and 20%, median bias, interval coverage and median interval width.

## 5. Underwriting engine (`finance/underwriting.py`)

Definitions are written at the top of the module and repeated here:

| Metric | Formula |
|---|---|
| Gross rent | monthly rent × 12 |
| Effective rent | gross rent × (1 − vacancy) |
| Operating expenses | management % × effective rent + maintenance % × gross rent + insurance + service charge/ground rent + other |
| NOI | effective rent − operating expenses − reserves |
| Debt service | 12 × monthly payment (annuity, or interest-only; 0% rate repays evenly) |
| Cash flow | NOI − debt service (pre-tax) |
| Gross yield | gross rent ÷ price |
| Cap rate | NOI ÷ total acquisition cost (price + SDLT + legal + refurbishment) |
| Cash-on-cash | cash flow ÷ cash invested (acquisition cost − loan + mortgage fee) |
| DSCR | NOI ÷ debt service (∞ when all cash) |
| Break-even occupancy | (fixed costs + reserves + debt service) ÷ (gross rent × (1 − management %)) |
| IRR | bisection on [−cash invested, cash flows…, final cash flow + sale × (1 − selling costs) − loan balance] |

SDLT uses the England & Northern Ireland residential bands from 1 April 2025 (0% to £125k, 2% to £250k, 5% to
£925k, 10% to £1.5m, 12% above), plus 5 points for an additional property (from £40,000) and 2 points for
non-residents. `screen()` is a vectorised year-one version used by the Opportunity Finder; a test checks it against
`analyse()` row by row.

## 6. Market indices, forecasts and risk (`market/forecast.py`)

* **Mix-adjusted index:** for each city, the weighted average of per-type log medians with weights fixed at the
  city's overall type mix (months where under 60% of the mix is observed are left blank).
* **Forecast:** damped-trend exponential smoothing on the log series; α, β, φ chosen from a small grid by rolling-origin
  backtest of 1–3 month errors. Prices: 6 months ahead. Rents: 12 months ahead.
* **Ranges:** empirical 10th and 90th percentiles of backtest errors at each horizon (growing with √h where the
  backtest is too short), always containing the point forecast.
* **Benchmark:** every forecast is compared with "no change"; the app says plainly when the model does not beat it.
* **Risk:** year-on-year change, monthly and annualised volatility, maximum drawdown and worst 3-month change.

## 7. Performance

Benchmark on a fixture of the same format with 8 markets and 200,000 sales: full build in about 1 minute with a
0.9 GB memory peak (2 vCPUs). Every app page renders in under 3 seconds. Committed artifacts are about 30 MB at that
scale.

## 8. Tests

`pytest -q` runs without downloading anything. The pipeline tests use `tests/fixtures.py`, which writes small invented
files in the exact Land Registry, Code-Point Open and rents formats.

* finance: hand-worked SDLT, mortgage, all-cash, financed, 0% rate, break-even, IRR, scenarios, sensitivity, and
  vectorised screening against the full model
* ingestion: removal of category B, other types, bad prices, out-of-market rows and duplicates; addresses never published
* leakage: features unchanged when the valuation month's prices are corrupted and later sales deleted
* models: time-ordered splits, every baseline and holdout reported, metric formulas, model save/load, comparables
  never include the property itself
* rents: ONS workbook parsing (notes rows, "Bristol, City of", type vs bedroom columns) and a clear error otherwise
* forecasts: straight-line extrapolation, intervals, mix adjustment ignoring mix changes, risk metrics
* app: every page renders; the scenario page reacts to inputs; a clear message when data has not been built
