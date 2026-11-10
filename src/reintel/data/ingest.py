"""Ingest, validate and de-duplicate the public source files.

Sources (all Open Government Licence v3.0, see docs/data_provenance.md):

* HM Land Registry Price Paid Data, yearly CSVs ``pp-YYYY.csv``
* OS Code-Point Open (postcode coordinates), ``codepo_gb.zip``
* ONS Price Index of Private Rents (local-authority rents), ``priceindexofprivaterents*.xlsx``

Every step records what it removed in a quality log, which is saved as
``data/demo/quality_report.json`` and shown in the app.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from reintel.config import MARKETS, PPD_COLUMNS, PROPERTY_TYPES, TENURE

MIN_PRICE, MAX_PRICE = 10_000, 10_000_000


def norm_postcode(s: pd.Series) -> pd.Series:
    return s.fillna("").str.upper().str.replace(r"\s+", "", regex=True)


def postcode_parts(pc: pd.Series) -> pd.DataFrame:
    """Split a space-free postcode into district (outward code) and sector."""
    pc = norm_postcode(pc)
    outward = pc.str[:-3]
    inward = pc.str[-3:]
    return pd.DataFrame({"postcode_district": outward, "postcode_sector": outward + " " + inward.str[0]})


def property_key(df: pd.DataFrame) -> pd.Series:
    """Stable anonymous property ID from the address; the address itself is then dropped."""
    raw = (norm_postcode(df.postcode) + "|" + df.paon.fillna("").str.upper().str.strip() + "|"
           + df.saon.fillna("").str.upper().str.strip() + "|" + df.street.fillna("").str.upper().str.strip())
    return raw.map(lambda s: "P" + hashlib.sha1(s.encode()).hexdigest()[:11])


def load_price_paid(paths: list[Path], log: dict) -> pd.DataFrame:
    frames = []
    total = 0
    for p in sorted(paths):
        chunks = pd.read_csv(p, header=None, names=PPD_COLUMNS, dtype=str, chunksize=500_000,
                             keep_default_na=False, na_values=[""])
        for ch in chunks:
            total += len(ch)
            frames.append(ch[ch.district.isin(MARKETS)])
    df = pd.concat(frames, ignore_index=True)
    log["rows_read_national"] = int(total)
    log["rows_in_markets"] = int(len(df))
    log["source_files"] = [p.name for p in sorted(paths)]

    df["transaction_id"] = df.transaction_id.str.strip("{}")
    # record_status: A = added, C = changed, D = deleted. Keep the latest version of each record.
    before = len(df)
    df = df.drop_duplicates("transaction_id", keep="last")
    deleted = df.record_status == "D"
    df = df[~deleted]
    log["removed_superseded_or_deleted"] = int(before - len(df))

    # Category A = standard price paid; B = repossessions, buy-to-let/company purchases, transfers
    # under a power of sale etc., which are not arm's-length market prices for a home.
    b = df.ppd_category != "A"
    log["removed_category_b"] = int(b.sum())
    df = df[~b]
    other = ~df.property_type.isin(["D", "S", "T", "F"])
    log["removed_property_type_other"] = int(other.sum())
    df = df[~other]

    df["price"] = pd.to_numeric(df.price, errors="coerce")
    df["sale_date"] = pd.to_datetime(df.date.str[:10], errors="coerce")
    bad = df.price.isna() | df.sale_date.isna() | (df.price < MIN_PRICE) | (df.price > MAX_PRICE)
    log["removed_price_or_date_out_of_range"] = int(bad.sum())
    df = df[~bad]
    nopc = norm_postcode(df.postcode).str.len() < 5
    log["removed_missing_postcode"] = int(nopc.sum())
    df = df[~nopc].copy()

    df["property_id"] = property_key(df)
    # Same property, same day, same price under two IDs = duplicate registration
    before = len(df)
    df = df.sort_values(["sale_date", "transaction_id"]).drop_duplicates(
        ["property_id", "sale_date", "price"], keep="first")
    log["removed_duplicate_sales"] = int(before - len(df))

    df["postcode"] = norm_postcode(df.postcode)
    df = pd.concat([df.reset_index(drop=True), postcode_parts(df.postcode).reset_index(drop=True)], axis=1)
    df["city"] = df.district.map(MARKETS)
    df["property_type"] = df.property_type.map(PROPERTY_TYPES)
    df["tenure"] = df.duration.map(TENURE).fillna("Unknown")
    df["new_build"] = (df.old_new == "Y").astype(int)
    keep = ["transaction_id", "property_id", "sale_date", "price", "postcode", "postcode_district",
            "postcode_sector", "city", "property_type", "tenure", "new_build"]
    # PAON / SAON / street (house numbers and names) are deliberately dropped.
    return df[keep].sort_values(["sale_date", "transaction_id"]).reset_index(drop=True)


def load_codepoint(zip_path: Path, postcodes: set[str], log: dict) -> pd.DataFrame:
    """Postcode → latitude/longitude from OS Code-Point Open (British National Grid)."""
    from pyproj import Transformer

    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if re.search(r"Data/CSV/[a-z]+\.csv$", n, re.I)]
        areas = {re.match(r"[A-Z]+", p).group(0).lower() for p in postcodes if re.match(r"[A-Z]+", p)}
        for n in names:
            if Path(n).stem.lower() not in areas:
                continue
            d = pd.read_csv(io.BytesIO(z.read(n)), header=None, usecols=[0, 1, 2, 3], dtype={0: str},
                            names=["postcode", "quality", "easting", "northing"])
            d["postcode"] = norm_postcode(d.postcode)
            rows.append(d[d.postcode.isin(postcodes)])
    cp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["postcode", "quality", "easting", "northing"])
    cp = cp[(cp.easting > 0) & (cp.northing > 0)]
    tr = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(cp.easting.to_numpy(float), cp.northing.to_numpy(float))
    cp = cp.assign(latitude=np.round(lat, 6), longitude=np.round(lon, 6))
    log["postcodes_needed"] = len(postcodes)
    log["postcodes_geocoded"] = int(cp.postcode.nunique())
    return cp[["postcode", "latitude", "longitude", "quality"]].drop_duplicates("postcode")


def attach_coordinates(sales: pd.DataFrame, cp: pd.DataFrame, log: dict) -> pd.DataFrame:
    out = sales.merge(cp[["postcode", "latitude", "longitude"]], on="postcode", how="left")
    # Terminated postcodes are missing from the current Code-Point file: fall back to the
    # centroid of the same postcode sector, and flag it.
    miss = out.latitude.isna()
    sector_xy = out[~miss].groupby("postcode_sector")[["latitude", "longitude"]].mean()
    fill = out.loc[miss, "postcode_sector"].map(sector_xy.latitude)
    out.loc[miss, "latitude"] = fill
    out.loc[miss, "longitude"] = out.loc[miss, "postcode_sector"].map(sector_xy.longitude)
    out["geo_precision"] = np.where(miss, "sector centroid", "postcode")
    still = out.latitude.isna()
    log["sales_geocoded_to_postcode"] = int((~miss).sum())
    log["sales_geocoded_to_sector_centroid"] = int((miss & ~still).sum())
    log["removed_not_geocoded"] = int(still.sum())
    return out[~still].reset_index(drop=True)
