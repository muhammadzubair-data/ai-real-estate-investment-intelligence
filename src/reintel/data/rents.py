"""Area rents from the ONS Price Index of Private Rents (PIPR).

PIPR publishes average monthly private rents by local authority: all
properties, and split by property type (and by bedrooms). We use the local
authority figure for each market and property type.

Important: Land Registry records do not include bedrooms, so a property's rent
here is the **area average for its property type**, not a property-specific
rent estimate. The app says so wherever the figure appears.

The parser does not rely on fixed sheet names or column positions (ONS
reorganises its workbooks from time to time). It scans every sheet for a table
with an area-name column, a date column and one or more "rental price"
columns, and recognises property-type columns by keyword. If it cannot find
what it needs it raises an error explaining what it saw; ``python
scripts/inspect_rents.py`` prints the workbook layout to help.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from reintel.config import MARKETS

TYPE_KEYWORDS = [  # order matters: "semi" before "detached"
    ("semi", "Semi-detached"), ("detached", "Detached"), ("terrace", "Terraced"),
    ("flat", "Flat/maisonette"), ("maisonette", "Flat/maisonette"),
]
BEDROOM_WORDS = ("bed", "room")


def norm_area(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r",?\s*city of$", "", s)
    s = re.sub(r"^city of\s+", "", s)
    return re.sub(r"[^a-z ]", "", s).strip()


CITY_BY_NORM = {norm_area(v): v for v in MARKETS.values()}


def load_rents(path: Path) -> pd.DataFrame:
    """Return tidy rents: city, property_type ('All' or a type), month, rent (GBP / month)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["month"])
        return df[["city", "property_type", "month", "rent"]]
    return parse_pipr(path)


def _find_header(raw: pd.DataFrame) -> int | None:
    for i in range(min(len(raw), 40)):
        cells = [str(c).lower() for c in raw.iloc[i].tolist()]
        if any("area name" in c or c.strip() in ("area", "geography") for c in cells) and \
                any("rent" in c and ("price" in c or "£" in c or "average" in c) for c in cells):
            return i
    return None


def _classify(col: str) -> str | None:
    """'All', a property type, or None (index, change, bedroom split, other)."""
    c = col.lower()
    if "rent" not in c or "index" in c or "change" in c or "%" in c:
        return None
    if any(w in c for w in BEDROOM_WORDS):
        return None
    for kw, label in TYPE_KEYWORDS:
        if kw in c:
            return label
    return "All"


def parse_pipr(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    found = []
    seen = {}
    for name, raw in sheets.items():
        h = _find_header(raw)
        seen[name] = None if h is None else [str(c) for c in raw.iloc[h].tolist()]
        if h is None:
            continue
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[h].tolist()]
        area_col = next((c for c in df.columns if "area name" in c.lower()),
                        next((c for c in df.columns if c.lower() in ("area", "geography")), None))
        date_col = next((c for c in df.columns if re.search(r"time period|date|month|period", c, re.I)), None)
        rent_cols = {c: _classify(c) for c in df.columns}
        rent_cols = {c: t for c, t in rent_cols.items() if t}
        if area_col is None or date_col is None or not rent_cols:
            continue
        df["city"] = df[area_col].map(lambda a: CITY_BY_NORM.get(norm_area(a)))
        df = df[df.city.notna()]
        if df.empty:
            continue
        df["month"] = pd.to_datetime(df[date_col], errors="coerce", format="mixed").dt.to_period("M").dt.to_timestamp()
        df = df[df.month.notna()]
        for c, t in rent_cols.items():
            part = df[["city", "month"]].assign(property_type=t, rent=pd.to_numeric(df[c], errors="coerce"))
            found.append(part.dropna(subset=["rent"]))
    if not found:
        layout = "\n".join(f"  sheet '{k}': {'no rent table found' if v is None else v[:12]}" for k, v in seen.items())
        raise ValueError(
            "Could not find local-authority rents for the covered cities in the ONS workbook.\n"
            f"What was found:\n{layout}\n"
            "Run `python scripts/inspect_rents.py <file>` and share the output, or create "
            "data/raw/rents_normalised.csv with columns city, property_type, month, rent.")
    out = pd.concat(found, ignore_index=True)
    # one value per city/type/month (if a city appears in several sheets, keep the first)
    out = out.drop_duplicates(["city", "property_type", "month"]).sort_values(["city", "property_type", "month"])
    missing = set(MARKETS.values()) - set(out.city)
    if missing:
        print(f"Warning: no ONS rents found for {sorted(missing)}; their yields will be blank.")
    return out[["city", "property_type", "month", "rent"]].reset_index(drop=True)
