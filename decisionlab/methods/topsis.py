"""TOPSIS — Technique for Order Preference by Similarity to Ideal Solution (Hwang & Yoon, 1981)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.models import RankingResult, build_ranking_table
from core.normalization import vector_normalize
from core.validation import BENEFIT, check_ranking_inputs

TIE_SCORE = 0.5


def topsis(matrix: pd.DataFrame | np.ndarray, types: Sequence[str],
           weights: Sequence[float] | np.ndarray | pd.Series) -> RankingResult:
    """Rank alternatives with classical TOPSIS.

    1. Vector normalisation ``r_ij = x_ij / sqrt(Σ_i x_ij²)``.
    2. Weighting ``v_ij = w_j r_ij``.
    3. Ideal best ``A+`` (max for benefit, min for cost) and ideal worst ``A−``.
    4. Euclidean distances ``D_i+`` and ``D_i−``.
    5. Closeness ``CC_i = D_i− / (D_i+ + D_i−)``; rank by descending ``CC_i``.

    If every alternative coincides with both ideals (all alternatives identical on
    all weighted criteria), closeness is undefined; every alternative then gets
    0.5 and a warning explains the tie.
    """
    frame, kinds, w = check_ranking_inputs(matrix, types, weights)
    warnings: list[str] = []
    if frame.shape[0] == 1:
        warnings.append("Only one alternative: it is ranked first by default.")
    if np.any(frame.to_numpy() < 0):
        warnings.append("The matrix contains negative values. Vector normalisation keeps "
                        "their order, but closeness values are harder to interpret.")
    zero_columns = [str(c) for c in frame.columns if np.allclose(frame[c], 0.0)]
    if zero_columns:
        warnings.append("All-zero criteria cannot be vector-normalised and are ignored: "
                        + ", ".join(zero_columns) + ".")

    normalised = vector_normalize(frame)
    weighted = normalised * w
    is_benefit = np.array([k == BENEFIT for k in kinds])
    v = weighted.to_numpy()
    ideal_best = np.where(is_benefit, v.max(axis=0), v.min(axis=0))
    ideal_worst = np.where(is_benefit, v.min(axis=0), v.max(axis=0))

    d_plus = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))
    denominator = d_plus + d_minus
    if np.all(denominator <= 1e-15):
        closeness = np.full(frame.shape[0], TIE_SCORE)
        if frame.shape[0] > 1:
            warnings.append("All alternatives are identical on the weighted criteria, so "
                            "TOPSIS cannot separate them; all are tied at 0.5.")
    else:
        closeness = np.where(denominator > 0, d_minus / np.where(denominator > 0,
                                                                  denominator, 1.0), TIE_SCORE)

    scores = pd.Series(closeness, index=frame.index, name="Closeness CC")
    ideal = pd.DataFrame({"Ideal best (A+)": ideal_best, "Ideal worst (A−)": ideal_worst},
                         index=frame.columns).T
    distances = pd.DataFrame({"D+ (to ideal best)": d_plus, "D− (to ideal worst)": d_minus,
                              "Closeness CC": closeness}, index=frame.index)
    details = {
        "normalised_matrix": normalised,
        "weighted_matrix": weighted,
        "ideal_solutions": ideal,
        "distances": distances,
        "weights": pd.Series(w, index=frame.columns, name="Weight"),
    }
    return RankingResult("TOPSIS", scores, build_ranking_table(scores, "Score"),
                         details, warnings)
