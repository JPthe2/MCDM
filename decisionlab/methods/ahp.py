"""AHP — Analytic Hierarchy Process (Saaty, 1980).

Weights are the normalised principal right eigenvector of a reciprocal pairwise
comparison matrix built on Saaty's 1–9 scale. Consistency is measured with
``CI = (λ_max − n) / (n − 1)`` and ``CR = CI / RI``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.models import MCDMError, WeightingResult
from core.validation import validate_names

# Saaty's Random Index values (average CI of random reciprocal matrices), n = 1..15.
RANDOM_INDEX: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}
MAX_CRITERIA = max(RANDOM_INDEX)
CR_THRESHOLD = 0.10
SAATY_MIN, SAATY_MAX = 1.0 / 9.0, 9.0

SAATY_SCALE: dict[int, str] = {
    1: "Equal importance",
    2: "Equal to moderate",
    3: "Moderate importance",
    4: "Moderate to strong",
    5: "Strong importance",
    6: "Strong to very strong",
    7: "Very strong importance",
    8: "Very strong to extreme",
    9: "Extreme importance",
}


def build_pairwise_matrix(n: int, judgments: Mapping[tuple[int, int], float]) -> np.ndarray:
    """Build a full reciprocal matrix from upper-triangle judgments.

    Args:
        n: Number of criteria.
        judgments: ``{(i, j): a_ij}`` for ``i < j``; ``a_ij > 1`` means criterion ``i``
            is more important than ``j``. Missing pairs default to 1 (equal).

    Returns:
        ``n x n`` matrix with unit diagonal and ``a_ji = 1 / a_ij``.
    """
    matrix = np.ones((n, n), dtype=float)
    for (i, j), value in judgments.items():
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise MCDMError(f"Invalid comparison pair ({i}, {j}) for {n} criteria.")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise MCDMError("Pairwise comparison values must be positive numbers.")
        if i > j:
            i, j, value = j, i, 1.0 / value
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value
    return matrix


def validate_pairwise_matrix(matrix: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Check squareness, positivity, unit diagonal, reciprocity and Saaty-scale bounds."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise MCDMError("The AHP comparison matrix must be square and non-empty.")
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise MCDMError("All AHP comparison values must be positive, finite numbers.")
    if not np.allclose(np.diag(a), 1.0, atol=tolerance):
        raise MCDMError("The diagonal of the AHP matrix must be 1 "
                        "(each criterion is equally important to itself).")
    if not np.allclose(a * a.T, 1.0, atol=tolerance * 10):
        raise MCDMError("The AHP matrix is not reciprocal: a_ji must equal 1 / a_ij.")
    if np.any(a < SAATY_MIN - tolerance) or np.any(a > SAATY_MAX + tolerance):
        raise MCDMError("AHP judgments must lie on Saaty's scale between 1/9 and 9.")
    return a


def random_index(n: int) -> float:
    """Saaty's Random Index for an ``n x n`` matrix."""
    if n < 1:
        raise MCDMError("Random Index requires at least one criterion.")
    if n > MAX_CRITERIA:
        raise MCDMError(f"AHP supports at most {MAX_CRITERIA} criteria "
                        "(no Random Index is tabulated beyond that). Use BWM, Entropy "
                        "or CRITIC for larger problems.")
    return RANDOM_INDEX[n]


def ahp_weights(matrix: np.ndarray, criteria: Sequence[str] | None = None) -> WeightingResult:
    """Compute AHP weights with the principal eigenvector method.

    Args:
        matrix: Reciprocal pairwise comparison matrix.
        criteria: Criterion names (defaults to ``C1..Cn``).

    Returns:
        WeightingResult whose ``details`` contain ``pairwise_matrix``, ``lambda_max``,
        ``CI``, ``RI``, ``CR``, ``consistent`` and ``normalised_matrix``.
    """
    a = validate_pairwise_matrix(matrix)
    n = a.shape[0]
    names = validate_names(criteria, "criterion") if criteria is not None else \
        [f"C{i + 1}" for i in range(n)]
    if len(names) != n:
        raise MCDMError(f"Got {len(names)} criterion names for a {n}x{n} matrix.")
    ri = random_index(n)
    warnings: list[str] = []

    if n == 1:
        weights = np.array([1.0])
        lambda_max = 1.0
        warnings.append("Only one criterion: it receives 100% of the weight and no "
                        "consistency check is needed.")
    else:
        eigenvalues, eigenvectors = np.linalg.eig(a)
        k = int(np.argmax(eigenvalues.real))
        lambda_max = float(eigenvalues[k].real)
        # Perron vector of a positive matrix has a single sign; abs() fixes sign flips.
        vector = np.abs(eigenvectors[:, k].real)
        weights = vector / vector.sum()

    ci = max((lambda_max - n) / (n - 1), 0.0) if n > 1 else 0.0
    cr = ci / ri if ri > 0 else 0.0
    if n == 2:
        warnings.append("With two criteria every AHP matrix is perfectly consistent (CR = 0).")
    if cr > CR_THRESHOLD:
        warnings.append(f"Consistency ratio CR = {cr:.3f} exceeds {CR_THRESHOLD:.2f}. "
                        "Your pairwise judgments are inconsistent; consider revising them "
                        "(e.g. if A > B and B > C, then A should be > C).")

    labels = pd.Index(names)
    details = {
        "pairwise_matrix": pd.DataFrame(a, index=labels, columns=labels),
        "normalised_matrix": pd.DataFrame(a / a.sum(axis=0), index=labels, columns=labels),
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": cr <= CR_THRESHOLD,
    }
    return WeightingResult("AHP", pd.Series(weights, index=labels, name="Weight"),
                           details, warnings)
