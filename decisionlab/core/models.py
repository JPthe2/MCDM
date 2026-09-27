"""Result containers shared by all weighting and ranking methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class MCDMError(ValueError):
    """Raised when inputs are invalid or a method cannot produce a meaningful result.

    The message is written for end users: it states what is wrong and what to fix.
    """


@dataclass
class WeightingResult:
    """Output of a criteria weighting method.

    Attributes:
        method: Human-readable method name (e.g. ``"AHP"``).
        weights: Criterion weights indexed by criterion name; non-negative, sum to 1.
        details: Intermediate results (DataFrames, Series, scalars) for transparency.
        warnings: Non-fatal issues the user should know about.
    """

    method: str
    weights: pd.Series
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RankingResult:
    """Output of an alternative ranking method.

    Attributes:
        method: Human-readable method name (e.g. ``"TOPSIS"``).
        scores: Final score per alternative (higher is better).
        ranking: Table with columns ``Rank``, ``Alternative``, ``Score`` sorted by rank.
        details: Intermediate matrices and vectors for transparency.
        warnings: Non-fatal issues the user should know about.
    """

    method: str
    scores: pd.Series
    ranking: pd.DataFrame
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_ranking_table(scores: pd.Series, score_label: str = "Score") -> pd.DataFrame:
    """Sort alternatives by descending score and assign competition ranks.

    Ties share the same (lowest) rank, e.g. 1, 2, 2, 4.
    """
    ranks = scores.rank(method="min", ascending=False).astype(int)
    table = pd.DataFrame(
        {"Rank": ranks.values, "Alternative": scores.index, score_label: scores.values}
    )
    # Stable sort keeps the user's original order among tied alternatives.
    return table.sort_values(["Rank"], kind="stable").reset_index(drop=True)
