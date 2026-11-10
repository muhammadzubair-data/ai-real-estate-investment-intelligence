"""Every dashboard page renders without errors (against the committed demo data, or a
fixture build when the demo data is absent)."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = sorted((ROOT / "app" / "views").glob("p*.py"))


@pytest.fixture(scope="module")
def data_dir(request):
    demo = ROOT / "data" / "demo"
    if (demo / "manifest.json").exists():
        return demo
    _, out, _ = request.getfixturevalue("fixture_build")
    return out


@pytest.mark.parametrize("page", VIEWS, ids=[p.stem for p in VIEWS])
def test_page_renders(page, data_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REINTEL_DATA", str(data_dir))
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(page), default_timeout=240)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_scenario_page_reacts_to_inputs(data_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REINTEL_DATA", str(data_dir))
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(ROOT / "app" / "views" / "p7_scenarios.py"), default_timeout=240)
    at.run()
    before = at.markdown[3].value if len(at.markdown) > 3 else ""
    at.sidebar.slider[0].set_value(0.0)          # loan-to-value -> all cash
    at.run()
    assert not at.exception
    assert any("all cash" in m.value for m in at.markdown)
    assert before is not None and os.environ.get("REINTEL_DATA")


def test_app_explains_missing_data(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("REINTEL_DATA", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(ROOT / "app" / "views" / "p1_market.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert "has not been built yet" in at.info[0].value
