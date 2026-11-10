"""Shared loading, theming and UI helpers for the Streamlit app."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reintel.config import ATTRIBUTION, DEMO_DIR  # noqa: E402


def data_dir() -> Path:
    """Folder with the built artifacts (REINTEL_DATA overrides the default data/demo)."""
    return Path(os.environ.get("REINTEL_DATA", DEMO_DIR))

# Validated palette: fixed categorical order, status colours reserved for status.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#c4352d", "#e98a7e", "#f0efec", "#7fb0e8", "#1f5fae"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TYPE_ORDER = ["Detached", "Semi-detached", "Terraced", "Flat/maisonette"]
TYPE_COLORS = dict(zip(TYPE_ORDER, SERIES))

pio.templates["re"] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK2, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE, colorway=SERIES, hovermode="closest",
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
))
pio.templates.default = "re"

CSS = """
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1 { font-size: 1.75rem !important; letter-spacing: -0.01em; }
.kpi { background:#fff; border:1px solid rgba(11,11,11,.10); border-radius:10px; padding:14px 16px; height:100%; }
.kpi .lbl { color:var(--ink2); font-size:.80rem; margin-bottom:4px; }
.kpi .val { color:var(--ink); font-size:1.6rem; font-weight:650; line-height:1.15; }
.kpi .sub { color:var(--muted); font-size:.78rem; margin-top:4px; }
.badge { background:#f0efec; border-radius:8px; padding:6px 12px; font-size:.78rem; color:var(--ink2);
         display:inline-block; margin-bottom:6px; }
.warn { background:#fff8e6; border-left:3px solid #fab219; padding:8px 12px; margin:6px 0; border-radius:4px;
        font-size:.88rem; color:var(--ink); }
.pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:.75rem; font-weight:600;
        border:1px solid rgba(11,11,11,.12); background:#fff; color:var(--ink); margin-right:4px; }
.footer { color:var(--muted); font-size:.72rem; margin-top:2.5rem; border-top:1px solid #e1e0d9; padding-top:.6rem; }
</style>
"""


NOT_BUILT = """
### The data has not been built yet

This app reads processed files from `data/demo/`. Build them once from the public sources:

1. Download the files listed in `data/raw/README.md` into `data/raw/`
   (HM Land Registry Price Paid Data, ONS private rents, OS Code-Point Open).
2. Run `python scripts/build_data.py` (a few minutes).
3. Restart the app.

Full instructions: `docs/HANDOVER.md`.
"""


def require_data() -> None:
    if not (data_dir() / "manifest.json").exists():
        st.markdown(CSS, unsafe_allow_html=True)
        st.info(NOT_BUILT)
        st.stop()


def setup(title: str, subtitle: str = "") -> None:
    require_data()
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def badge(text: str | None = None) -> None:
    m = manifest()
    label = m.get("data_label", "Real HM Land Registry sales")
    default = f"{label} · {m['date_range'][0]} to {m['date_range'][1]} · all figures in GBP (£) · informational only, not investment advice"
    st.markdown(f'<span class="badge">{text or default}</span>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">{ATTRIBUTION}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def gbp(x: float, short: bool = True) -> str:
    if x is None or pd.isna(x):
        return "–"
    x = float(x)
    sign = "−" if x < 0 else ""
    x = abs(x)
    if short and x >= 1e6:
        return f"{sign}£{x / 1e6:,.2f}m"
    if short and x >= 1e4:
        return f"{sign}£{x / 1e3:,.0f}k"
    return f"{sign}£{x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * float(x):.{d}f}%"


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def load(name: str) -> pd.DataFrame:
    return _load(str(data_dir() / name))


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_json(name: str) -> dict:
    return _load_json(str(data_dir() / name))


def manifest() -> dict:
    return load_json("manifest.json")


def sales() -> pd.DataFrame:
    return load("sales.parquet")


@st.cache_resource(show_spinner="Loading the valuation model…")
def _service(path: str):
    from reintel.valuation.models import Valuer
    from reintel.valuation.service import ValuationService
    return ValuationService(_load(str(Path(path) / "sales.parquet")), Valuer.load(Path(path)))


def valuation_service():
    return _service(str(data_dir()))


def city_list() -> list[str]:
    return manifest()["markets"]
