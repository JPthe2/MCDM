"""Entropy Weight Method (Shannon entropy applied to criteria).

A criterion whose normalised values are spread evenly across alternatives has high
entropy and carries little information, so it gets a small weight.

Convention: the matrix is min-max normalised with criterion direction applied
(benefit ↑, cost ↓), so zeros and negative raw values are handled. Constant
criteria get a uniform distribution, i.e. entropy 1 and weight 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.models import MCDMError, WeightingResult
from core.normalization import constant_columns, min_max_normalize
from core.validation import validate_criteria_types, validate_decision_matrix


def entropy_weights(matrix: pd.DataFrame | np.ndarray,
                    types: Sequence[str]) -> WeightingResult:
    """Compute objective weights with the entropy method.

    Steps: ``r`` = min-max normalised matrix; ``p_ij = r_ij / Σ_i r_ij``;
    ``e_j = −k Σ_i p_ij ln p_ij`` with ``k = 1/ln m`` and ``0 ln 0 = 0``;
    ``d_j = 1 − e_j``; ``w_j = d_j / Σ_j d_j``.
    """
    frame = validate_decision_matrix(matrix)
    kinds = validate_criteria_types(types, frame.shape[1])
    m, n = frame.shape
    if m < 2:
        raise MCDMError("The Entropy method needs at least two alternatives "
                        "(entropy is undefined for a single alternative).")

    normalised = min_max_normalize(frame, kinds)
    r = normalised.to_numpy()
    column_sums = r.sum(axis=0)
    # A column summing to 0 cannot occur after min-max, but guard anyway (uniform p).
    p = np.where(column_sums > 0, r / np.where(column_sums > 0, column_sums, 1.0), 1.0 / m)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    k = 1.0 / np.log(m)
    entropy = np.clip(-k * plogp.sum(axis=0), 0.0, 1.0)
    divergence = 1.0 - entropy

    warnings: list[str] = []
    constant = constant_columns(frame)
    if constant:
        warnings.append("Constant criteria carry no information and receive zero weight: "
                        + ", ".join(constant) + ".")
    if divergence.sum() <= 1e-12:
        raise MCDMError("All criteria have zero divergence (every criterion is constant "
                        "across alternatives), so the Entropy method cannot tell them apart. "
                        "Use Equal weights, AHP or BWM instead.")
    if n == 1:
        warnings.append("Only one criterion: it receives 100% of the weight.")
    weights = divergence / divergence.sum()

    labels = frame.columns
    details = {
        "normalised_matrix": normalised,
        "proportions": pd.DataFrame(p, index=frame.index, columns=labels),
        "summary": pd.DataFrame({"Entropy e_j": entropy, "Divergence d_j": divergence,
                                 "Weight w_j": weights}, index=labels),
        "k": k,
    }
    return WeightingResult("Entropy", pd.Series(weights, index=labels, name="Weight"),
                           details, warnings)
