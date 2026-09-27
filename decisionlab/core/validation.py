"""Input validation for decision matrices, criterion types and weights.

Every function raises :class:`~core.models.MCDMError` with a user-facing message
when the input cannot be used, so that no method ever silently returns a
misleading result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from core.models import MCDMError

BENEFIT = "benefit"
COST = "cost"
WEIGHT_TOLERANCE = 1e-6


def validate_names(names: Sequence[Any], kind: str) -> list[str]:
    """Check that names are non-empty and unique; return them stripped.

    Args:
        names: Alternative or criterion names.
        kind: ``"alternative"`` or ``"criterion"``, used in error messages.
    """
    cleaned = ["" if n is None or (isinstance(n, float) and np.isnan(n)) else str(n).strip()
               for n in names]
    if not cleaned:
        raise MCDMError(f"At least one {kind} is required.")
    empty = [i + 1 for i, n in enumerate(cleaned) if not n]
    if empty:
        raise MCDMError(f"Every {kind} needs a name. Missing name at position(s): "
                        f"{', '.join(map(str, empty))}.")
    seen: set[str] = set()
    duplicates = sorted({n for n in cleaned if n in seen or seen.add(n)})
    if duplicates:
        raise MCDMError(f"{kind.capitalize()} names must be unique. Duplicated: "
                        f"{', '.join(duplicates)}.")
    return cleaned


def validate_criteria_types(types: Sequence[str], n_criteria: int) -> list[str]:
    """Normalise criterion types to ``"benefit"``/``"cost"`` and check the count."""
    if len(types) != n_criteria:
        raise MCDMError(f"Expected {n_criteria} criterion types, got {len(types)}.")
    normalised: list[str] = []
    for position, raw in enumerate(types, start=1):
        value = str(raw).strip().lower()
        if value not in (BENEFIT, COST):
            raise MCDMError(f"Criterion {position} has type '{raw}'. "
                            "Each criterion must be either 'Benefit' or 'Cost'.")
        normalised.append(value)
    return normalised


def find_invalid_cells(frame: pd.DataFrame) -> list[tuple[str, str, Any]]:
    """Return ``(alternative, criterion, raw_value)`` for every missing/non-numeric cell."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() | ~np.isfinite(numeric.astype(float))
    return [(str(frame.index[i]), str(frame.columns[j]), frame.iat[i, j])
            for i, j in zip(*np.nonzero(bad.to_numpy()))]


def validate_decision_matrix(matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
                             ) -> pd.DataFrame:
    """Validate a decision matrix and return it as a float DataFrame.

    Rows are alternatives and columns are criteria. Plain arrays receive default
    labels ``A1..Am`` and ``C1..Cn``.

    Raises:
        MCDMError: if the matrix is empty, not 2-D, has duplicate labels, or
            contains missing, non-numeric or infinite values.
    """
    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy()
    else:
        array = np.asarray(matrix, dtype=object)
        if array.ndim != 2:
            raise MCDMError("The decision matrix must be two-dimensional "
                            "(alternatives x criteria).")
        frame = pd.DataFrame(array,
                             index=[f"A{i + 1}" for i in range(array.shape[0])],
                             columns=[f"C{j + 1}" for j in range(array.shape[1])])
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise MCDMError("The decision matrix is empty. Add at least one alternative "
                        "and one criterion.")
    validate_names(list(frame.index), "alternative")
    validate_names(list(frame.columns), "criterion")

    invalid = find_invalid_cells(frame)
    if invalid:
        preview = "; ".join(
            f"{alt} / {crit} = {'(empty)' if _is_blank(val) else repr(val)}"
            for alt, crit, val in invalid[:8])
        more = f" (and {len(invalid) - 8} more)" if len(invalid) > 8 else ""
        raise MCDMError(f"The decision matrix has {len(invalid)} missing or non-numeric "
                        f"value(s): {preview}{more}. Enter a number in every cell.")
    return frame.apply(pd.to_numeric).astype(float)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def validate_weights(weights: Sequence[float] | np.ndarray | pd.Series, n_criteria: int,
                     tolerance: float = WEIGHT_TOLERANCE) -> np.ndarray:
    """Check that weights are finite, non-negative and sum to 1 within ``tolerance``.

    Weights are *not* silently rescaled: a sum different from 1 is an error.
    """
    array = np.asarray(weights, dtype=float).ravel()
    if array.size != n_criteria:
        raise MCDMError(f"Expected {n_criteria} weights (one per criterion), got {array.size}.")
    if not np.all(np.isfinite(array)):
        raise MCDMError("Weights contain missing or infinite values.")
    if np.any(array < -tolerance):
        raise MCDMError("Weights must be non-negative.")
    total = float(array.sum())
    if abs(total - 1.0) > tolerance:
        raise MCDMError(f"Weights must sum to 1 (current sum: {total:.6f}).")
    return np.clip(array, 0.0, None)


def normalise_weight_vector(raw: np.ndarray) -> np.ndarray:
    """Scale a non-negative vector to sum to 1; error if it is all zeros."""
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        raise MCDMError("Cannot normalise weights: all raw weight values are zero.")
    return raw / total


def check_ranking_inputs(matrix: pd.DataFrame | np.ndarray, types: Sequence[str],
                         weights: Sequence[float] | np.ndarray | pd.Series,
                         ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Validate everything a ranking method needs, in one call."""
    frame = validate_decision_matrix(matrix)
    kinds = validate_criteria_types(types, frame.shape[1])
    if isinstance(weights, pd.Series) and set(weights.index) == set(frame.columns):
        weights = weights.reindex(frame.columns)
    w = validate_weights(weights, frame.shape[1])
    return frame, kinds, w
