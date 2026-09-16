"""pytest configuration: make ``src/`` importable and expose repository paths."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sample_table(root: Path) -> Path:
    return root / "data" / "sample" / "kiba_sample.tab"
