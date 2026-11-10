import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def fixture_build(tmp_path_factory):
    """Run the whole pipeline once on small fake files in the real source formats."""
    from fixtures import make_sources
    from reintel.build import build
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("out")
    src = make_sources(raw)
    manifest = build(sources=src, out=out, data_label="TEST FIXTURE (invented rows)")
    return src, out, manifest
