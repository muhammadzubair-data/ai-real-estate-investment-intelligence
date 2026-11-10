"""Capture README screenshots of all seven pages and render the portfolio thumbnail.

One-time setup:   pip install playwright && python -m playwright install chromium
Then:             python scripts/screenshots.py

Starts the app on port 8599, captures assets/screenshots/*.png and writes
assets/thumbnail.png (1280×720) using the real results in data/demo/metrics.json.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "assets" / "screenshots"
PORT = 8599
PAGES = {
    "": "01_market_overview", "p2_valuation": "02_valuation_engine", "p3_opportunities": "03_opportunity_finder",
    "p4_cashflow": "04_yield_cash_flow", "p5_geo": "05_geospatial", "p6_forecast": "06_forecasting_risk",
    "p7_scenarios": "07_scenario_simulator",
}


def wait_for_app() -> None:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("The app did not start. Run `streamlit run app/streamlit_app.py` to see the error.")


def capture(page) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for slug, name in PAGES.items():
        page.goto(f"http://localhost:{PORT}/{slug}", wait_until="networkidle", timeout=120_000)
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60_000)
        for _ in range(60):                                # wait until Streamlit stops running
            if not page.locator('[data-testid="stStatusWidget"]').count():
                break
            time.sleep(1)
        time.sleep(4)                                      # let charts and map tiles settle
        page.screenshot(path=str(SHOTS / f"{name}.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 1000})
        print("captured", name)


def thumbnail(page) -> None:
    m = json.loads((ROOT / "data" / "demo" / "metrics.json").read_text())
    man = json.loads((ROOT / "data" / "demo" / "manifest.json").read_text())
    dep = m["temporal_holdout"]["LightGBM (deployed)"]
    img = base64.b64encode((SHOTS / "02_valuation_engine.png").read_bytes()).decode()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:1280px;height:720px;background:#0d366b;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#fff;overflow:hidden}}
.wrap{{position:absolute;inset:0;display:grid;grid-template-columns:620px 1fr}}
.left{{padding:62px 50px 56px 64px;display:flex;flex-direction:column}}
.tag{{font-size:15px;letter-spacing:.14em;text-transform:uppercase;color:#9ec5f4;font-weight:600}}
h1{{font-size:50px;line-height:1.05;font-weight:750;letter-spacing:-.02em;margin-top:18px}}
.sub{{font-size:19px;color:#cde2fb;margin-top:18px;line-height:1.45}}
.stats{{margin-top:auto;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.s{{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:16px}}
.s b{{display:block;font-size:32px;font-weight:700}}
.s span{{display:block;font-size:13.5px;color:#cde2fb;margin-top:6px;line-height:1.35}}
.right{{padding:48px 48px 48px 0;display:flex;align-items:center}}
.card{{background:#fcfcfb;border-radius:16px;width:100%;height:100%;overflow:hidden;box-shadow:0 20px 50px rgba(0,0,0,.35)}}
.card img{{width:100%;height:100%;object-fit:cover;object-position:100% 0}}
.foot{{position:absolute;left:64px;bottom:22px;font-size:12.5px;color:#86b6ef}}
</style></head><body><div class="wrap"><div class="left">
<div class="tag">PropTech · Valuation AI · Underwriting</div>
<h1>Real Estate Investment Intelligence</h1>
<div class="sub">Values UK homes from {man['counts']['sales']:,} real sales, finds rental opportunities and underwrites every deal. Seven-page Streamlit app on open government data.</div>
<div class="stats">
<div class="s"><b>{100 * dep['median_ape']:.1f}%</b><span>median valuation error on unseen recent sales</span></div>
<div class="s"><b>{100 * dep['interval_coverage']:.0f}%</b><span>of prices inside the 80% range</span></div>
<div class="s"><b>{man['counts']['markets']}</b><span>English cities, Land Registry + ONS + OS data</span></div>
</div></div>
<div class="right"><div class="card"><img src="data:image/png;base64,{img}"></div></div></div>
<div class="foot">Python · LightGBM · SHAP · Streamlit · Plotly &nbsp;·&nbsp; tested on later months, unseen districts and unseen cities</div>
</body></html>"""
    page.set_viewport_size({"width": 1280, "height": 720})
    page.set_content(html)
    page.wait_for_timeout(500)
    page.screenshot(path=str(ROOT / "assets" / "thumbnail.png"))
    print("rendered thumbnail")


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("Install first: pip install playwright && python -m playwright install chromium")
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                            "--server.port", str(PORT), "--server.headless", "true"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_app()
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000})
            capture(page)
            thumbnail(page)
            b.close()
    finally:
        app.terminate()


if __name__ == "__main__":
    main()
