"""Tiny fake source files in the real formats, so the whole pipeline can be tested
in CI without downloading the national datasets.

These rows are invented test data, used only by the test suite.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = {"LEEDS": ("LS", 430_000, 433_500, 230_000), "BRISTOL": ("BS", 359_000, 172_500, 330_000)}
TYPES = {"D": 1.6, "S": 1.15, "T": 0.95, "F": 0.75}


def make_sources(folder: Path, n_per_market: int = 2200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows, cp_rows = [], {}
    tid = 0
    for district, (area, e0, n0, base) in MARKETS.items():
        sectors = [f"{area}{d} {s}" for d in range(1, 9) for s in range(1, 5)]
        sector_xy = {sc: (e0 + rng.normal(0, 3000), n0 + rng.normal(0, 3000), rng.lognormal(0, 0.25))
                     for sc in sectors}
        homes = []
        for i in range(int(n_per_market * 0.8)):
            sc = sectors[rng.integers(len(sectors))]
            pc = f"{sc}{chr(65 + rng.integers(26))}{chr(65 + rng.integers(26))}"
            homes.append((pc, sc, rng.choice(list(TYPES), p=[0.2, 0.3, 0.3, 0.2]), f"{i + 1}", f"STREET {i % 40}"))
            x, y, _ = sector_xy[sc]
            cp_rows.setdefault(area.lower(), []).append(
                (pc, 10, round(x + rng.normal(0, 150)), round(y + rng.normal(0, 150))))
        for _ in range(n_per_market):
            pc, sc, t, paon, street = homes[rng.integers(len(homes))]
            day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 820)))
            months = (day.year - 2024) * 12 + day.month - 1
            price = base * TYPES[t] * sector_xy[sc][2] * (1 + 0.003 * months) * rng.lognormal(0, 0.12)
            tid += 1
            rows.append([f"{{T{tid:08d}}}", int(price), f"{day:%Y-%m-%d} 00:00", pc, t,
                         "Y" if rng.random() < 0.05 else "N", "L" if t == "F" else "F", paon, "", street,
                         "", district, district, "COUNTY", "A", "A"])
    # rows that must be removed by validation
    rows.append(["{TBAD00001}", 150000, "2024-05-01 00:00", "LS1 1AA", "S", "N", "F", "9", "", "X", "",
                 "LEEDS", "LEEDS", "C", "B", "A"])                                     # category B
    rows.append(["{TBAD00002}", 150000, "2024-05-01 00:00", "LS1 1AA", "O", "N", "F", "9", "", "X", "",
                 "LEEDS", "LEEDS", "C", "A", "A"])                                     # other type
    rows.append(["{TBAD00003}", 500, "2024-05-01 00:00", "LS1 1AA", "S", "N", "F", "9", "", "X", "",
                 "LEEDS", "LEEDS", "C", "A", "A"])                                     # price too low
    rows.append(["{TBAD00004}", 150000, "2024-05-01 00:00", "M1 1AA", "S", "N", "F", "9", "", "X", "",
                 "MANCHESTERX", "MANCHESTERX", "C", "A", "A"])                         # out of market
    dup = list(rows[10])
    dup[0] = "{TDUP00001}"
    rows.append(dup)                                                                   # duplicate sale
    df = pd.DataFrame(rows)
    years = pd.to_datetime(df[2].str[:10]).dt.year
    paths = []
    for y in sorted(years.unique()):
        p = folder / f"pp-{y}.csv"
        df[years == y].to_csv(p, header=False, index=False)
        paths.append(p)

    zp = folder / "codepo_gb.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for area, r in cp_rows.items():
            d = pd.DataFrame(r).drop_duplicates(0)
            for c in range(4, 10):
                d[c] = "X"
            z.writestr(f"Data/CSV/{area}.csv", d.to_csv(header=False, index=False))

    months = pd.date_range("2022-01-01", "2026-03-01", freq="MS")
    rent = []
    for city, b in (("Leeds", 1000), ("Bristol", 1500)):
        for i, m in enumerate(months):
            for t, mult in (("All", 1), ("Detached", 1.5), ("Semi-detached", 1.15), ("Terraced", 1.0),
                            ("Flat/maisonette", 0.85)):
                rent.append((city, t, m, round(b * mult * (1 + 0.005 * i))))
    rp = folder / "rents_normalised.csv"
    pd.DataFrame(rent, columns=["city", "property_type", "month", "rent"]).to_csv(rp, index=False)
    return {"price_paid": paths, "codepoint": zp, "rents": rp}
