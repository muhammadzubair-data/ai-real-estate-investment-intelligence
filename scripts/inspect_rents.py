"""Print the layout of the ONS rents workbook (sheet names and header rows).

    python scripts/inspect_rents.py data/raw/priceindexofprivaterentsukmonthlypricestatistics.xlsx
"""
import sys

import pandas as pd

path = sys.argv[1]
for name, raw in pd.read_excel(path, sheet_name=None, header=None, nrows=15).items():
    print(f"=== sheet: {name}")
    for i, row in raw.iterrows():
        cells = [str(c) for c in row.tolist() if str(c) != "nan"]
        if cells:
            print(f"  row {i}: {cells[:14]}")
