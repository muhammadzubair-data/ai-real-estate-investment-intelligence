# Model card: property valuation model

## Purpose

Estimate the current market value of a home in one of the covered English cities, with an 80% range, comparable
sales and the main reasons behind the figure. It supports research, screening and conversations. It is **not a
formal valuation** (a RICS Red Book valuation needs an inspection), it is not lending advice, and it should never be
the only basis for an offer.

## What the model knows, and what it does not

| Knows | Does not know |
|---|---|
| Location (coordinates, postcode sector and district, city) | Floor area, bedrooms, bathrooms |
| Property type, tenure, new build | Condition, renovations, garden, parking, outlook |
| Recent sales nearby (12-month medians at sector, district and city level) | Asking prices, time on market |
| The 10 nearest comparable sales of the same type, adjusted to today | Lease length remaining on leaseholds |
| The property's own previous sale, adjusted to today | Anything about the buyer or seller |
| City price momentum | |

Because size and condition are missing, the error on any single property is much larger than for models that
have them. The app says this on every valuation.

## Model

* **LightGBM** gradient boosting on log price; trees chosen by early stopping on validation months, then refitted on
  training and validation months.
* **80% range:** two quantile LightGBM models (10th and 90th percentiles), widened by conformalised quantile regression
  on the validation months so that the range holds about 80% of real prices.
* **Explanations:** SHAP TreeExplainer, shown as the percentage effect of each input on the estimate.
* **Baselines reported next to it:** postcode-sector median (same type, last 12 months, falling back to district
  then city), the median of the 10 nearest comparable sales, and ridge regression on the same features.

## Evaluation design

* Leakage-safe features: a sale in month *m* sees only sales completed up to *m − 2*, over 12 months. Tested in
  `tests/test_pipeline.py::test_features_use_only_past_evidence`.
* Time-based split: warm-up months (not modelled) → training → three validation months → the latest six months as
  the test set.
* Extra holdouts: properties never seen in training, about 20% of postcode districts removed from training, and each
  city left out in turn.
* Slices: city, property type, price tier, new build, comparables quality, geocoding precision.

## Performance

<!-- RESULTS:START -->
_Filled in by `python scripts/make_report.py` after the data build._
<!-- RESULTS:END -->

## Known limitations and risks

* **Missing size and condition data** is the largest source of error. A large house and a small one on the same
  street get similar estimates.
* **Regression to the mean:** the cheapest homes in an area tend to be over-valued and the dearest under-valued.
* **New builds** sell at a premium that the market does not repeat on resale; they are flagged.
* **Thin evidence:** sectors with few sales of the same type, or comparables more than 2 km away, give wider and
  less reliable estimates; the app warns in both cases.
* **Registration delay:** the latest one or two months are under-recorded; the build drops a clearly incomplete
  final month.
* **Category A only:** repossessions and company or buy-to-let purchases are excluded, so the model reflects ordinary
  owner-occupier sales.
* **Fairness:** the model uses only property and location facts. It contains no information about people. Location
  can still carry historic patterns, so outputs should not be used to make decisions about individuals.

## Use and monitoring

* Rebuild monthly with the new Land Registry file; compare the latest month's error with the test results.
* Retrain when the median error on new sales drifts more than 2 percentage points above the reported test figure.
* Any change to features, data or the interval method is evaluated with the same holdouts before release.
