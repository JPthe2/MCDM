"""Static explanatory text and formulas displayed in the interface."""

from __future__ import annotations

WEIGHTING_INFO: dict[str, dict[str, str]] = {
    "AHP": {
        "icon": "🔺",
        "kind": "Subjective — based on your judgments",
        "summary": "You compare criteria two at a time on Saaty's 1–9 scale (e.g. *Price is "
                   "strongly more important than Service*). AHP turns those comparisons "
                   "into weights and checks that your judgments are consistent.",
        "when": "Use when you (or experts) can judge the relative importance of criteria "
                "and there are no more than about 7–9 criteria.",
    },
    "BWM": {
        "icon": "⚖️",
        "kind": "Subjective — based on your judgments",
        "summary": "You pick the most and least important criteria, then compare the best "
                   "criterion against all others and all others against the worst. An "
                   "optimisation finds weights that agree best with your judgments.",
        "when": "Use when you want expert-based weights with far fewer comparisons than "
                "AHP (2n − 3 instead of n(n − 1)/2).",
    },
    "Entropy": {
        "icon": "📡",
        "kind": "Objective — derived from the data",
        "summary": "Criteria whose values differ a lot between alternatives carry more "
                   "information and get more weight; criteria on which all alternatives "
                   "are similar get little weight.",
        "when": "Use when you have no preferences and want the data to speak, or as a "
                "cross-check of subjective weights.",
    },
    "CRITIC": {
        "icon": "🧭",
        "kind": "Objective — derived from the data",
        "summary": "A criterion is important when it varies a lot (high standard deviation) "
                   "and brings unique information (low correlation with other criteria). "
                   "Two criteria that always move together share their importance.",
        "when": "Use when criteria may overlap (e.g. *horsepower* and *top speed*) and you "
                "do not want to double-count them.",
    },
    "Equal weights": {
        "icon": "🟰",
        "kind": "Neutral baseline",
        "summary": "Every criterion gets the same weight, 1/n.",
        "when": "Use as a neutral reference point, or when all criteria truly matter "
                "equally.",
    },
}

RANKING_INFO: dict[str, dict[str, str]] = {
    "TOPSIS": {
        "icon": "🎯",
        "summary": "Builds an imaginary *ideal* alternative (best value on every criterion) "
                   "and an *anti-ideal* (worst on every criterion). Each real alternative "
                   "is scored by how close it is to the ideal and how far from the "
                   "anti-ideal. Score = closeness coefficient between 0 and 1.",
        "when": "Good general-purpose choice; it rewards balanced alternatives that are "
                "never far from the best.",
    },
    "SWM": {
        "icon": "➕",
        "summary": "Rescales every criterion to 0–1 (1 = best alternative, 0 = worst), "
                   "multiplies by the weight and adds everything up. Score = weighted "
                   "average performance between 0 and 1.",
        "when": "Simplest and most transparent; each criterion's contribution to the "
                "score is directly visible.",
    },
}

FORMULAS: dict[str, list[tuple[str, str]]] = {
    "AHP": [
        ("Reciprocal comparison matrix", r"A = [a_{ij}],\quad a_{ii}=1,\quad a_{ji} = 1/a_{ij}"),
        ("Principal eigenvector (weights)", r"A\,\mathbf{w} = \lambda_{\max}\,\mathbf{w},"
                                            r"\qquad \textstyle\sum_j w_j = 1"),
        ("Consistency index", r"CI = \dfrac{\lambda_{\max} - n}{n - 1}"),
        ("Consistency ratio", r"CR = \dfrac{CI}{RI(n)} \quad (\text{acceptable if } CR \le 0.10)"),
    ],
    "BWM": [
        ("Optimisation model",
         r"\min \xi \ \text{ s.t. } \left|\dfrac{w_B}{w_j} - a_{Bj}\right| \le \xi,\ "
         r"\left|\dfrac{w_j}{w_W} - a_{jW}\right| \le \xi,\ \textstyle\sum_j w_j = 1,\ w_j \ge 0"),
        ("Consistency ratio", r"CR = \dfrac{\xi^*}{CI(a_{BW})}"),
    ],
    "Entropy": [
        ("Min-max normalisation (direction-aware)",
         r"r_{ij} = \dfrac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}\ \text{(benefit)},"
         r"\quad r_{ij} = \dfrac{\max_i x_{ij} - x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}\ \text{(cost)}"),
        ("Proportions", r"p_{ij} = \dfrac{r_{ij}}{\sum_i r_{ij}}"),
        ("Entropy", r"e_j = -\dfrac{1}{\ln m}\sum_i p_{ij}\ln p_{ij}, \quad 0\ln 0 = 0"),
        ("Divergence and weights", r"d_j = 1 - e_j, \qquad w_j = \dfrac{d_j}{\sum_k d_k}"),
    ],
    "CRITIC": [
        ("Min-max normalisation (direction-aware)",
         r"r_{ij} \in [0,1],\ 1 = \text{best alternative on criterion } j"),
        ("Contrast intensity", r"\sigma_j = \text{standard deviation of } r_{\cdot j}"),
        ("Conflict (correlation)", r"\rho_{jk} = \text{Pearson correlation of criteria } j, k"),
        ("Information content and weights",
         r"C_j = \sigma_j \sum_{k}(1 - \rho_{jk}), \qquad w_j = \dfrac{C_j}{\sum_k C_k}"),
    ],
    "Equal weights": [("Weights", r"w_j = \dfrac{1}{n}")],
    "TOPSIS": [
        ("Vector normalisation", r"r_{ij} = \dfrac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}"),
        ("Weighted matrix", r"v_{ij} = w_j\, r_{ij}"),
        ("Ideal solutions",
         r"A^+_j = \max_i v_{ij}\ (\text{benefit}),\ \min_i v_{ij}\ (\text{cost});\quad "
         r"A^-_j = \text{the opposite}"),
        ("Distances", r"D_i^{\pm} = \sqrt{\textstyle\sum_j (v_{ij} - A^{\pm}_j)^2}"),
        ("Closeness coefficient", r"CC_i = \dfrac{D_i^-}{D_i^+ + D_i^-} \in [0, 1]"),
    ],
    "SWM": [
        ("Min-max normalisation (direction-aware)",
         r"r_{ij} = \dfrac{x_{ij} - \min_i x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}\ \text{(benefit)},"
         r"\quad r_{ij} = \dfrac{\max_i x_{ij} - x_{ij}}{\max_i x_{ij} - \min_i x_{ij}}\ \text{(cost)}"),
        ("Weighted sum", r"S_i = \sum_j w_j\, r_{ij} \in [0, 1]"),
    ],
}

CONVENTIONS = """
- **Criterion direction.** *Benefit*: higher is better. *Cost*: lower is better.
- **Constant criteria** (same value for every alternative) cannot discriminate. In
  min-max normalisation every alternative receives 1 on such a criterion; Entropy and
  CRITIC give it zero weight.
- **SWM normalisation** is linear min-max, so the best alternative on a criterion scores 1
  and the worst scores 0; negative raw values are handled.
- **TOPSIS** uses vector normalisation; an all-zero criterion is ignored. If all
  alternatives are identical, closeness is undefined and every alternative is tied at 0.5.
- **Entropy and CRITIC** use direction-aware min-max normalisation. CRITIC uses the
  sample standard deviation (the choice cancels out in the final weights).
- **AHP** uses the principal eigenvector and Saaty's Random Index (n ≤ 15). With 1 or 2
  criteria CR is 0 by definition.
- **BWM** solves Rezaei's non-linear model with SciPy SLSQP (weights ≥ 10⁻⁶ so ratios are
  defined), started from the linear-BWM solution and two other deterministic points.
- **Ties** share the same rank (1, 2, 2, 4).
"""

SAATY_HELP = """
| Value | Meaning |
|---|---|
| 1 | Equal importance |
| 3 | Moderate importance of one over the other |
| 5 | Strong importance |
| 7 | Very strong importance |
| 9 | Extreme importance |
| 2, 4, 6, 8 | Intermediate values |
"""

INTENSITY_WORDS = {1: "equally important as", 2: "slightly more important than",
                   3: "moderately more important than", 4: "moderately-to-strongly more "
                   "important than", 5: "strongly more important than",
                   6: "strongly-to-very-strongly more important than",
                   7: "very strongly more important than",
                   8: "very-to-extremely more important than",
                   9: "extremely more important than"}
