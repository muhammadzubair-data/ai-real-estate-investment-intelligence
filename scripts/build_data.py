"""Rebuild data/demo/ from the raw public files in data/raw/.

    python scripts/build_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reintel.build import main  # noqa: E402

if __name__ == "__main__":
    main()
