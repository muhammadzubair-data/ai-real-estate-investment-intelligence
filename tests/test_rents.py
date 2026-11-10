"""The ONS rents parser copes with the workbook layout: notes rows above the table,
'Bristol, City of' style names, property-type and bedroom columns, index columns."""
import pandas as pd
import pytest

from reintel.data.rents import load_rents, norm_area, parse_pipr


def _workbook(path):
    notes = pd.DataFrame([["Price Index of Private Rents"], ["Source: ONS"], [None]])
    header = ["Time period", "Area code", "Area name", "Region or country name", "Index", "Annual change",
              "Rental price", "Rental price one bed", "Rental price two bed", "Rental price detached",
              "Rental price semidetached", "Rental price terraced", "Rental price flat maisonette"]
    rows = []
    for m in ("Jan-2025", "Feb-2025"):
        rows.append([m, "E06000023", "Bristol, City of", "South West", 140, 5.1, 1800, 1300, 1600, 2600, 2100,
                     1900, 1500])
        rows.append([m, "E08000035", "Leeds", "Yorkshire", 135, 6.0, 1100, 800, 1000, 1500, 1200, 1050, 950])
        rows.append([m, "E09000001", "Somewhere Else", "London", 150, 4, 3000, 2000, 2500, 5000, 4000, 3500, 2800])
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([["Contents"], ["Table 1"]]).to_excel(w, sheet_name="Contents", header=False, index=False)
        pd.concat([notes, pd.DataFrame([header] + rows)]).to_excel(w, sheet_name="Table 1", header=False, index=False)


def test_parse_pipr_layout(tmp_path):
    p = tmp_path / "pipr.xlsx"
    _workbook(p)
    r = parse_pipr(p)
    assert set(r.city) == {"Bristol", "Leeds"}
    assert set(r.property_type) == {"All", "Detached", "Semi-detached", "Terraced", "Flat/maisonette"}
    bristol = r[(r.city == "Bristol") & (r.month == pd.Timestamp("2025-01-01"))].set_index("property_type").rent
    assert bristol["All"] == 1800 and bristol["Semi-detached"] == 2100 and bristol["Detached"] == 2600
    assert len(r) == 2 * 2 * 5                                       # 2 cities × 2 months × 5 series


def test_parse_pipr_explains_failure(tmp_path):
    p = tmp_path / "bad.xlsx"
    pd.DataFrame([["nothing useful"]]).to_excel(p, header=False, index=False)
    with pytest.raises(ValueError, match="inspect_rents"):
        parse_pipr(p)


def test_area_name_normalisation():
    assert norm_area("Bristol, City of") == norm_area("Bristol") == "bristol"
    assert norm_area("Newcastle upon Tyne") == "newcastle upon tyne"


def test_normalised_csv_route(tmp_path):
    p = tmp_path / "rents_normalised.csv"
    pd.DataFrame({"city": ["Leeds"], "property_type": ["All"], "month": ["2025-01-01"], "rent": [1000]}).to_csv(p, index=False)
    assert load_rents(p).rent.iloc[0] == 1000
