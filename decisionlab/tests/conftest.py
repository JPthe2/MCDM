"""Shared fixtures: a small decision problem with benefit and cost criteria."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def phone_matrix() -> pd.DataFrame:
    """Classic 5 x 4 TOPSIS example (price is a cost criterion)."""
    return pd.DataFrame(
        [[250, 16, 12, 5],
         [200, 16, 8, 3],
         [300, 32, 16, 4],
         [275, 32, 8, 4],
         [225, 16, 16, 2]],
        index=["A", "B", "C", "D", "E"],
        columns=["Price", "Storage", "Camera", "Looks"], dtype=float)


@pytest.fixture
def phone_types() -> list[str]:
    return ["cost", "benefit", "benefit", "benefit"]


@pytest.fixture
def small_matrix() -> pd.DataFrame:
    """4 alternatives x 4 criteria; 'Cost' and 'Time' are cost criteria."""
    return pd.DataFrame(
        [[100, 8, 5, 0.90],
         [120, 9, 3, 0.95],
         [90, 6, 7, 0.80],
         [110, 7, 4, 0.85]],
        index=["S1", "S2", "S3", "S4"],
        columns=["Cost", "Quality", "Time", "Reliability"], dtype=float)


@pytest.fixture
def small_types() -> list[str]:
    return ["cost", "benefit", "cost", "benefit"]
