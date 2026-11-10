"""Central configuration: markets, file locations, attribution."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"          # downloaded source files (not committed)
DEMO_DIR = PROJECT_ROOT / "data" / "demo"        # processed, committed data the app reads

CURRENCY = "GBP"
CURRENCY_SYMBOL = "£"

# Markets = English local authorities (Land Registry "District"), so every sale,
# rent statistic and SDLT rule refers to the same area. Welsh authorities are
# excluded because Wales uses Land Transaction Tax, not SDLT.
MARKETS = {
    "BIRMINGHAM": "Birmingham",
    "MANCHESTER": "Manchester",
    "LEEDS": "Leeds",
    "LIVERPOOL": "Liverpool",
    "SHEFFIELD": "Sheffield",
    "BRISTOL": "Bristol",
    "CITY OF BRISTOL": "Bristol",
    "NEWCASTLE UPON TYNE": "Newcastle upon Tyne",
    "NOTTINGHAM": "Nottingham",
    "LEICESTER": "Leicester",
    "COVENTRY": "Coventry",
    "BRADFORD": "Bradford",
    "SOUTHAMPTON": "Southampton",
}

# Land Registry Price Paid Data column layout (files have no header row)
PPD_COLUMNS = [
    "transaction_id", "price", "date", "postcode", "property_type", "old_new", "duration",
    "paon", "saon", "street", "locality", "town", "district", "county", "ppd_category", "record_status",
]
PROPERTY_TYPES = {"D": "Detached", "S": "Semi-detached", "T": "Terraced", "F": "Flat/maisonette", "O": "Other"}
TENURE = {"F": "Freehold", "L": "Leasehold"}

ATTRIBUTION = (
    "Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the "
    "Open Government Licence v3.0. Contains OS data © Crown copyright and database right 2026. Contains Royal Mail "
    "data © Royal Mail copyright and database right 2026. Contains National Statistics data © Crown copyright and "
    "database right 2026. Source: Office for National Statistics licensed under the Open Government Licence v3.0."
)
