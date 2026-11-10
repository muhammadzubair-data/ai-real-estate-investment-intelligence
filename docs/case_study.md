# Case study: valuing homes and screening rental deals from public records

## The problem

Property investors and small agencies usually work from portal listings, a spreadsheet of rents and a feel for each
area. Valuations come from a handful of comparables picked by hand, cash-flow models live in separate files, and
nobody checks how far off the estimates usually are.

## What I built

One platform that runs on real UK open data, with every number traceable to a public source.

1. **Clean, licensed data.** Every home sale registered with HM Land Registry in 12 English cities since 2024, ONS
   private rents for each city and property type, and OS postcode locations. Repossessions, company transfers,
   duplicates and errors are removed and logged. House numbers are dropped so no individual address is published.
2. **A valuation model that shows its working.** Each estimate comes with an 80% range, the 10 nearest comparable
   sales on a map, the inputs that moved the price, and warnings when local evidence is thin.
3. **Testing that would satisfy a sceptic.** The model is scored only on later sales it never saw, on homes it never
   saw, on whole postcode districts it never saw, and on each city with that city left out. A test proves no
   future information leaks into any valuation.
4. **An investment engine.** Stamp duty (including the second-home surcharge), mortgage, running costs, NOI, cap
   rate, cash-on-cash, DSCR, break-even occupancy and IRR. Every formula is checked against hand-worked examples.
5. **Seven dashboard pages:** market overview, valuation, opportunity finder, yield and cash flow, map,
   forecasting and risk, and a scenario simulator with editable upside and downside cases.

## Results

<!-- RESULTS:START -->
_Filled in by `python scripts/make_report.py` after the data build._
<!-- RESULTS:END -->

## What I'd flag honestly

* Land Registry doesn't record floor area, bedrooms or condition, so the model can't tell a large house from a
  small one on the same street. That limit is stated everywhere it matters, not hidden.
* Rents are area averages by property type, not rents for a specific home.
* There are no public listings, so the opportunity finder treats recent sales as if they were on offer at their
  sale price. Because the model valued each one before the sale was known, the price-vs-estimate signal is still
  fair.
* Two years of monthly prices is a short history for forecasting, so every forecast is shown next to a simple
  "no change" benchmark, with ranges taken from the method's own past errors.

## Stack

Python, pandas, scikit-learn, LightGBM, SHAP, pyproj, Streamlit, Plotly, pytest, GitHub Actions.
