"""Fill the results into the docs from data/demo/metrics.json (run after build_data.py).

    python scripts/make_report.py

Updates, between <!-- RESULTS:START --> and <!-- RESULTS:END --> markers:
    README.md, docs/model_card.md, docs/case_study.md
Writes:
    docs/results.md            full results tables
    docs/client_message.txt    plain-text handover message for the Upwork inbox
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "demo"


def gbp(x):
    return f"£{x:,.0f}"


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def replace_block(path: Path, body: str, tag: str = "RESULTS") -> None:
    s = path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- {tag}:START -->)(.*?)(<!-- {tag}:END -->)", re.S)
    if not pat.search(s):
        raise SystemExit(f"No {tag} markers in {path}")
    path.write_text(pat.sub(lambda m: m.group(1) + "\n" + body.strip() + "\n" + m.group(3), s), encoding="utf-8")


def main() -> None:
    m = json.loads((DATA / "metrics.json").read_text())
    man = json.loads((DATA / "manifest.json").read_text())
    q = json.loads((DATA / "quality_report.json").read_text())
    th = m["temporal_holdout"]
    dep = th["LightGBM (deployed)"]
    nb = th["Neighbourhood median"]
    geo = m["geographic_holdout"]
    loco = m["leave_one_city_out"]
    unseen = m["unseen_properties"]
    p = m["periods"]

    main_rows = [[name, gbp(r["mae"]), pct(r["median_ape"]), pct(r["mape"]), pct(r["within_10pct"], 0),
                  pct(r["within_20pct"], 0)] for name, r in th.items()]
    main_tbl = table(main_rows, ["Model", "Mean abs. error", "Median % error", "Mean % error", "Within 10%", "Within 20%"])
    improve = 1 - dep["median_ape"] / nb["median_ape"]
    if improve > 0.005:
        vs_base = f"{pct(improve, 0)} lower than the neighbourhood-median baseline ({pct(nb['median_ape'])})"
    elif improve < -0.005:
        vs_base = f"{pct(-improve, 0)} higher than the neighbourhood-median baseline ({pct(nb['median_ape'])})"
    else:
        vs_base = f"about the same as the neighbourhood-median baseline ({pct(nb['median_ape'])})"
    best = min(th, key=lambda k: th[k]["median_ape"])
    loco_mape = sorted(loco, key=lambda r: r["median_ape"])
    summary = (
        f"Tested on **{m['rows']['test']:,} sales from {p['test'][0]} to {p['test'][1]}** that the models never saw "
        f"(trained on {p['train'][0]} to {p['train'][1]}, tuned on {p['valid'][0]} to {p['valid'][1]}).\n\n"
        f"{main_tbl}\n\n"
        f"* The deployed LightGBM model's median error is **{pct(dep['median_ape'])}**, {vs_base}."
        + (f" The lowest error on this test came from **{best}**." if best != "LightGBM (deployed)" else "")
        + "\n"
        f"* Its 80% price range contained the actual price **{pct(dep['interval_coverage'], 0)}** of the time "
        f"(median width {pct(dep['interval_median_width_pct'], 0)}).\n"
        f"* Properties never seen in training ({m['rows']['test_unseen_properties']:,} sales): median error "
        f"{pct(unseen['median_ape'])}.\n"
        f"* Whole postcode districts held out of training ({geo['held_out_districts']} of {geo['districts_total']}): "
        f"median error {pct(geo['held_out']['median_ape'])} vs {pct(geo['same_sales_full_model']['median_ape'])} when "
        f"those districts were in training.\n"
        f"* Each city valued by a model trained only on the other cities: median error from "
        f"{pct(loco_mape[0]['median_ape'])} ({loco_mape[0]['city']}) to {pct(loco_mape[-1]['median_ape'])} "
        f"({loco_mape[-1]['city']}).\n"
        f"* Land Registry has no floor area, bedrooms or condition, which explain much of any single price. "
        f"Errors of this size are expected without them."
    )
    for f in ("README.md", "docs/model_card.md"):
        replace_block(ROOT / f, summary)

    # full results file
    sl = m["slices"]
    parts = ["# Results\n", f"Generated from `data/demo/metrics.json`. Currency: GBP. Built {man['built_on']}.\n",
             "## Temporal holdout\n", summary, "\n## By city\n",
             table([[r["city"], r["n"], gbp(r["mae"]), pct(r["median_ape"]), pct(r["within_10pct"], 0),
                     pct(r.get("interval_coverage"), 0)] for r in sorted(sl["city"], key=lambda r: r["median_ape"])],
                   ["City", "Sales", "Mean abs. error", "Median % error", "Within 10%", "80% range coverage"]),
             "\n## By property type\n",
             table([[r["property_type"], r["n"], pct(r["median_ape"]), pct(r.get("interval_coverage"), 0)]
                    for r in sl["property_type"]], ["Type", "Sales", "Median % error", "80% range coverage"]),
             "\n## By price tier (actual sale price)\n",
             table([[r["price_tier"], r["n"], pct(r["median_ape"]), pct(r["bias_median_pct"])] for r in sl["price_tier"]],
                   ["Tier", "Sales", "Median % error", "Median bias (estimate vs price)"]),
             "\nThe cheapest homes tend to be over-valued and the dearest under-valued: part of this is regression "
             "to the mean (slicing by the actual price), part is missing size and condition data.\n",
             "\n## By evidence quality\n",
             table([[r["comps_quality"], r["n"], pct(r["median_ape"])] for r in sl["comps_quality"]],
                   ["Comparables", "Sales", "Median % error"]),
             "\n## New build vs existing\n",
             table([["New build" if r["new_build"] == 1 else "Existing", r["n"], pct(r["median_ape"]),
                     pct(r["bias_median_pct"])] for r in sl["new_build"]],
                   ["", "Sales", "Median % error", "Median bias"]),
             "\n## Leave one city out\n",
             table([[r["city"], r["n"], pct(r["median_ape"]), gbp(r["mae"]), gbp(r["in_sample_city_mae"])]
                    for r in loco_mape], ["City", "Sales", "Median % error", "Mean abs. error",
                                          "Mean abs. error with city in training"]),
             "\n## Data quality\n",
             table([[k, f"{v:,}" if isinstance(v, int) else v] for k, v in q.items() if k != "source_files"],
                   ["Step", "Value"])]
    (ROOT / "docs" / "results.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    cs = (f"* Built on **{man['counts']['sales']:,} real sales** of {man['counts']['properties']:,} homes in "
          f"{man['counts']['markets']} English cities ({man['date_range'][0]} to {man['date_range'][1]}).\n"
          f"* On {m['rows']['test']:,} later sales it had never seen, the model's typical error was "
          f"**{pct(dep['median_ape'])}**, {vs_base.replace('the neighbourhood-median baseline', 'the usual local-median approach')}.\n"
          f"* Its 80% price range held the real price {pct(dep['interval_coverage'], 0)} of the time.")
    replace_block(ROOT / "docs" / "case_study.md", cs)

    msg = f"""Hi,

The real estate intelligence platform is ready. Here is what's included:

- Real data: {man['counts']['sales']:,} HM Land Registry sales across {man['counts']['markets']} English cities ({man['date_range'][0]} to {man['date_range'][1]}), ONS private rents and OS postcode locations, all under the Open Government Licence, with a data provenance and licence report
- Data checks that remove repossessions, company transfers, duplicates and bad records, with a log of every row removed
- Valuation model with comparable sales, an 80% price range, the reasons behind each estimate, and warnings when local evidence is thin
- Honest testing: later months the model never saw, whole postcode districts held out, each city held out in turn, and a test that proves no future sales leak into any valuation
- Full investment calculator: stamp duty, mortgage, NOI, cap rate, cash-on-cash, DSCR, break-even occupancy and IRR, with worked-example tests
- Seven-page Streamlit dashboard: market overview, valuation engine, opportunity finder, yield and cash flow, map view, forecasting and risk, and a scenario simulator
- Automated tests, GitHub Actions CI, a model card, technical documentation and a deployment guide

Results on {m['rows']['test']:,} recent sales the model had never seen:
- Typical (median) valuation error: {pct(dep['median_ape'])}, {vs_base.replace('the neighbourhood-median baseline', 'a simple local median price')}
- The 80% price range held the real price {pct(dep['interval_coverage'], 0)} of the time
- {pct(dep['within_10pct'], 0)} of valuations were within 10% of the sale price

One thing to be clear about: Land Registry doesn't record floor area, bedrooms or condition, so the model values from location, property type, tenure and local sales. The app says this wherever it matters.

I can deploy it to Streamlit Cloud for you, or walk you through it on a short call. Let me know which you prefer.

Best regards,
Muhammad Zubair
"""
    (ROOT / "docs" / "client_message.txt").write_text(msg, encoding="utf-8")
    print("Updated README.md, docs/model_card.md, docs/case_study.md; wrote docs/results.md and docs/client_message.txt")


if __name__ == "__main__":
    main()
