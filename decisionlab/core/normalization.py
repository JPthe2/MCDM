"""Normalisation schemes used by the weighting and ranking methods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.validation import BENEFIT

# Value given to every cell of a constant column in min-max normalisation.
# A constant criterion cannot discriminate between alternatives; giving everyone
# the same (best) score keeps it neutral for ranking and makes its entropy maximal.
CONSTANT_COLUMN_VALUE = 1.0


def constant_mask(frame: pd.DataFrame, rtol: float = 1e-12) -> np.ndarray:
    """Boolean array: True for columns whose values are all (numerically) equal."""
    values = frame.to_numpy(dtype=float)
    spread = values.max(axis=0) - values.min(axis=0)
    scale = np.maximum(np.abs(values).max(axis=0), 1.0)
    return spread <= rtol * scale


def constant_columns(frame: pd.DataFrame, rtol: float = 1e-12) -> list[str]:
    """Names (as strings) of columns whose values are all equal, for messages."""
    return [str(c) for c, const in zip(frame.columns, constant_mask(frame, rtol)) if const]


def min_max_normalize(frame: pd.DataFrame, types: Sequence[str]) -> pd.DataFrame:
    """Linear min-max normalisation into [0, 1], respecting criterion direction.

    Benefit: ``r = (x - min) / (max - min)``; Cost: ``r = (max - x) / (max - min)``.
    Works for negative values. Constant columns become ``CONSTANT_COLUMN_VALUE``.
    """
    values = frame.to_numpy(dtype=float)
    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    spread = col_max - col_min
    constant = constant_mask(frame)
    result = np.empty_like(values)
    for j, kind in enumerate(types):
        if constant[j]:
            result[:, j] = CONSTANT_COLUMN_VALUE
        elif kind == BENEFIT:
            result[:, j] = (values[:, j] - col_min[j]) / spread[j]
        else:
            result[:, j] = (col_max[j] - values[:, j]) / spread[j]
    return pd.DataFrame(result, index=frame.index, columns=frame.columns)


def vector_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Vector (Euclidean) normalisation: ``r_ij = x_ij / sqrt(sum_i x_ij^2)``.

    Direction is *not* applied here (TOPSIS handles it when choosing the ideal
    solutions). An all-zero column carries no information and is returned as zeros.
    """
    values = frame.to_numpy(dtype=float)
    norms = np.sqrt((values ** 2).sum(axis=0))
    safe = np.where(norms > 0, norms, 1.0)
    result = np.where(norms > 0, values / safe, 0.0)
    return pd.DataFrame(result, index=frame.index, columns=frame.columns)
