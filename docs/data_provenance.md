# Data provenance & licence report

Every dataset used is public, free and licensed for commercial and non-commercial reuse under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) (OGL),
subject to the attribution and address-data terms below. No data is scraped from listing portals, and no data is
synthetic: the test suite uses small invented files in the same formats, which are never shipped as results.

## Sources

| ID | Dataset | Publisher | What we use | Licence | Download |
|---|---|---|---|---|---|
| `HMLR-PPD` | Price Paid Data, yearly files 2024, 2025, 2026 (to date) | HM Land Registry | Every residential sale: price, date, postcode, type, new build, tenure | OGL v3.0 + address-data terms | [gov.uk](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) |
| `OS-CPO` | Code-Point Open | Ordnance Survey | Postcode coordinates (British National Grid → latitude/longitude) | OGL v3.0 | [OS Data Hub](https://osdatahub.os.uk/downloads/open/CodePointOpen) |
| `ONS-PIPR` | Price Index of Private Rents, UK: monthly price statistics | Office for National Statistics | Average private rent by local authority and property type | OGL v3.0 | [ons.gov.uk](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics) |
| — | Stamp Duty Land Tax rates | HMRC (gov.uk) | Residential bands and the additional-property surcharge, coded in `underwriting.py` | OGL v3.0 | [gov.uk](https://www.gov.uk/stamp-duty-land-tax/residential-property-rates) |

The `source_id` column in every exported table refers to these IDs.

## Required attribution

Shown in the footer of every app page and in the README:

> Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the Open
> Government Licence v3.0. Contains OS data © Crown copyright and database right 2026. Contains Royal Mail data ©
> Royal Mail copyright and database right 2026. Contains National Statistics data © Crown copyright and database right
> 2026. Source: Office for National Statistics licensed under the Open Government Licence v3.0.

## Address data: what we keep and why

Price Paid Data contains address data derived from Ordnance Survey AddressBase and Royal Mail's PAF. HM Land Registry
states that Royal Mail and Ordnance Survey permit its use for personal or non-commercial purposes, and for display
for the purpose of providing residential property price information services. Other uses need Royal Mail's
agreement (address.management@royalmail.com).

To stay well inside those terms:

* house numbers and names (PAON, SAON) and street names are **dropped at ingestion** and never stored or shown;
* each property gets an anonymous ID (a one-way hash of its address) so repeat sales can be linked;
* only the postcode, postcode sector and coordinates are kept, which is enough for comparables and maps, and they
  are displayed only as residential price information.

**If the client wants to use this commercially beyond price information** (for example, a lead-generation tool for
specific addresses), confirm the terms with Royal Mail first.

## Cleaning rules (logged in `data/demo/quality_report.json`, shown on page 1)

| Rule | Why |
|---|---|
| Keep only the covered local authorities | the 12 markets |
| Latest version of each record; drop record status `D` | Land Registry corrections and deletions |
| Drop PPD category `B` | repossessions, buy-to-let and company purchases, transfers under power of sale: not ordinary market sales |
| Drop property type `O` ("other") | commercial, land and mixed-use |
| Drop prices under £10,000 or over £10m, and invalid dates | data errors and non-residential deals |
| Drop the same property sold on the same day for the same price twice | duplicate registrations |
| Drop the latest month if it has under half the usual sales | registrations arrive with a delay of weeks to months |
| Postcodes not in Code-Point Open placed at the postcode-sector centre, flagged `geo_precision` | retired postcodes |

## Markets

Birmingham, Bradford, Bristol, Coventry, Leeds, Leicester, Liverpool, Manchester, Newcastle upon Tyne, Nottingham,
Sheffield and Southampton: English local authorities, so that sales, rents and Stamp Duty refer to the same
place. Welsh areas are excluded because Wales uses Land Transaction Tax rather than SDLT.

## What the data does not contain

| Field in the client schema | Status | Consequence |
|---|---|---|
| `area_sqft`, `bedrooms`, `bathrooms`, `year_built` | not in Price Paid Data | left empty in the exports; the model cannot use size or condition, the main source of error for single properties |
| `listing_date`, `asking_price` | no public listings data | the Opportunity Finder uses recent sales at their sale price as stand-ins, and says so |
| `expected_monthly_rent` | ONS area average for the property type | not a property-specific rent; labelled everywhere it appears |
| `mortgage_rate_annual` | not included | mortgage rate is a user assumption in every calculator |

Energy Performance Certificate data (floor area, rooms) could fill the size gap, but it is released under its own
licence terms with registration and restrictions on address data, so it is left as a documented extension.

## Reproducing the build

1. Download the five files above into `data/raw/` (see `data/raw/README.md` for exact names).
2. `python scripts/build_data.py` writes everything the app reads into `data/demo/`.
3. `python scripts/make_report.py` writes the results into the README, model card and `docs/results.md`.
