"""Plain-language explanations of why each alternative received its score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.models import RankingResult


def achievement_matrix(result: RankingResult) -> pd.DataFrame:
    """Share of the achievable range reached on each criterion (0 = worst, 1 = best).

    SWM: the min-max normalised value. TOPSIS: ``1 − |v − A+| / |A+ − A−|`` on the
    weighted normalised matrix. Criteria where all alternatives are equal get 1.
    """
    if result.method == "TOPSIS":
        weighted: pd.DataFrame = result.details["weighted_matrix"]
        ideal = result.details["ideal_solutions"].iloc[0]
        worst = result.details["ideal_solutions"].iloc[1]
        span = (ideal - worst).abs().replace(0, np.nan)
        return (1 - (weighted - ideal).abs() / span).fillna(1.0)
    return result.details["normalised_matrix"].copy()


def explain_ranking(result: RankingResult, top_k: int = 2) -> pd.DataFrame:
    """Build one explanation row per alternative, in rank order.

    A criterion is a *strength* when the alternative reaches at least half of the
    achievable range on it, and a *weakness* otherwise. Both lists are ordered by
    impact on the score: ``w_j · a_ij`` for strengths and ``w_j · (1 − a_ij)`` for
    weaknesses.
    """
    weights: pd.Series = result.details["weights"]
    active = weights[weights > 0].index
    achieved = achievement_matrix(result)[active]
    rows = []
    for _, row in result.ranking.iterrows():
        alt = row["Alternative"]
        a = achieved.loc[alt]
        w = weights[active]
        gain = (w * a)[a >= 0.5].sort_values(ascending=False)
        loss = (w * (1 - a))[a < 0.5].sort_values(ascending=False)
        strengths = [f"{c} ({a[c]:.0%})" for c in gain.index[:top_k]]
        weaknesses = [f"{c} ({a[c]:.0%})" for c in loss.index[:top_k]]
        if result.method == "TOPSIS":
            dist = result.details["distances"].loc[alt]
            summary = (f"Distance to ideal best {dist.iloc[0]:.4f} and to ideal worst "
                       f"{dist.iloc[1]:.4f}, so closeness = {dist.iloc[1]:.4f} / "
                       f"({dist.iloc[0]:.4f} + {dist.iloc[1]:.4f}) = {row['Score']:.4f}.")
        else:
            summary = (f"Weighted sum of normalised scores = {row['Score']:.4f} out of a "
                       f"possible 1 ({row['Score'] * 100:.1f}%).")
        rows.append({
            "Rank": row["Rank"],
            "Alternative": alt,
            "Score": row["Score"],
            "Main strengths": ", ".join(strengths) or "—",
            "Main weaknesses": ", ".join(weaknesses) or "—",
            "Explanation": summary,
        })
    return pd.DataFrame(rows)
