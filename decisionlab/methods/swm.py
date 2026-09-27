"""SWM — Simple Weighted Method (weighted sum model / SAW).

Each criterion is min-max normalised into [0, 1] with its direction applied, so the
best raw value scores 1 and the worst scores 0; the final score is
``S_i = Σ_j w_j r_ij`` and therefore also lies in [0, 1].
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.models import RankingResult, build_ranking_table
from core.normalization import CONSTANT_COLUMN_VALUE, constant_columns, min_max_normalize
from core.validation import check_ranking_inputs


def swm(matrix: pd.DataFrame | np.ndarray, types: Sequence[str],
        weights: Sequence[float] | np.ndarray | pd.Series) -> RankingResult:
    """Rank alternatives by the weighted sum of min-max normalised scores.

    Benefit: ``r = (x − min) / (max − min)``; Cost: ``r = (max − x) / (max − min)``.
    Negative values are handled naturally by min-max. Constant criteria are set to
    ``CONSTANT_COLUMN_VALUE`` for every alternative, so they do not affect the order.
    """
    frame, kinds, w = check_ranking_inputs(matrix, types, weights)
    warnings: list[str] = []
    if frame.shape[0] == 1:
        warnings.append("Only one alternative: it is ranked first by default.")
    constant = constant_columns(frame)
    if constant and frame.shape[0] > 1:
        warnings.append(f"Constant criteria give every alternative the same normalised "
                        f"value ({CONSTANT_COLUMN_VALUE:g}) and do not affect the ranking: "
                        + ", ".join(constant) + ".")

    normalised = min_max_normalize(frame, kinds)
    contributions = normalised * w
    scores = contributions.sum(axis=1).rename("Weighted score")
    if frame.shape[0] > 1 and np.ptp(scores.to_numpy()) <= 1e-12:
        warnings.append("All alternatives obtained the same score; they are tied.")

    details = {
        "normalised_matrix": normalised,
        "weighted_matrix": contributions,
        "weights": pd.Series(w, index=frame.columns, name="Weight"),
    }
    return RankingResult("SWM", scores, build_ranking_table(scores, "Score"),
                         details, warnings)
