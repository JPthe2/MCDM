"""CRITIC — Criteria Importance Through Intercriteria Correlation (Diakoulaki et al., 1995).

A criterion is important when it (a) varies a lot across alternatives (high σ_j)
and (b) is weakly correlated with other criteria (brings unique information).

Conventions: min-max normalisation with direction applied; sample standard
deviation (ddof = 1 — the choice cancels out after normalising the weights);
Pearson correlation; a constant criterion has σ = 0 and correlation 0 with others.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from core.models import MCDMError, WeightingResult
from core.normalization import constant_columns, constant_mask, min_max_normalize
from core.validation import validate_criteria_types, validate_decision_matrix


def critic_weights(matrix: pd.DataFrame | np.ndarray,
                   types: Sequence[str]) -> WeightingResult:
    """Compute objective weights with CRITIC: ``C_j = σ_j Σ_k (1 − r_jk)``, ``w_j = C_j / Σ C``."""
    frame = validate_decision_matrix(matrix)
    kinds = validate_criteria_types(types, frame.shape[1])
    m, n = frame.shape
    if m < 2:
        raise MCDMError("The CRITIC method needs at least two alternatives to measure "
                        "variability and correlation.")
    labels = frame.columns
    warnings: list[str] = []

    normalised = min_max_normalize(frame, kinds)
    x = normalised.to_numpy()
    sigma = x.std(axis=0, ddof=1)
    constant = constant_columns(frame)
    sigma[constant_mask(frame)] = 0.0

    corr = np.eye(n)
    varying = sigma > 0
    if varying.sum() >= 2:
        idx = np.flatnonzero(varying)
        corr[np.ix_(idx, idx)] = np.corrcoef(x[:, idx], rowvar=False)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    conflict = (1.0 - corr).sum(axis=1)
    information = sigma * conflict

    if n == 1:
        weights = np.array([1.0])
        warnings.append("Only one criterion: it receives 100% of the weight.")
    else:
        if constant:
            warnings.append("Constant criteria have zero standard deviation and receive "
                            "zero weight: " + ", ".join(constant) + ".")
        if information.sum() <= 1e-12:
            raise MCDMError("All criteria have zero information content (they are constant "
                            "or perfectly correlated), so CRITIC cannot distinguish them. "
                            "Use Equal weights, AHP or BWM instead.")
        weights = information / information.sum()

    details = {
        "normalised_matrix": normalised,
        "correlation_matrix": pd.DataFrame(corr, index=labels, columns=labels),
        "summary": pd.DataFrame({"Std. deviation σ_j": sigma,
                                 "Conflict Σ(1 − r_jk)": conflict,
                                 "Information C_j": information,
                                 "Weight w_j": weights}, index=labels),
    }
    return WeightingResult("CRITIC", pd.Series(weights, index=labels, name="Weight"),
                           details, warnings)
