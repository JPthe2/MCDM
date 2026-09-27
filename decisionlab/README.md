# ⚖️ DecisionLab — Multi-Criteria Decision Engine

DecisionLab is an interactive Streamlit application for **Multi-Criteria Decision Making (MCDM)**.
Define a decision problem, enter alternatives and criteria, choose how to weight the criteria and
how to rank the alternatives, and get a transparent, explainable, exportable ranking.

| Criteria weighting | Alternative ranking |
|---|---|
| **AHP** — Analytic Hierarchy Process (eigenvector method, consistency ratio) | **TOPSIS** — closeness to the ideal solution |
| **BWM** — Best-Worst Method (non-linear model, SciPy SLSQP) | **SWM** — Simple Weighted Method (weighted sum of min-max normalised scores) |
| **Entropy** weight method | |
| **CRITIC** — Criteria Importance Through Intercriteria Correlation | |
| **Equal weights** (also available as an automatic baseline) | |

Any weighting method can be combined with any ranking method (e.g. AHP + TOPSIS, BWM + SWM,
Entropy + TOPSIS, CRITIC + SWM). All algorithms are implemented from scratch with NumPy, pandas
and SciPy — no black-box MCDM library is used.

---

## Quick start

Requires **Python 3.11+** (tested on Python 3.13).

**Windows (PowerShell)**

```powershell
cd decisionlab
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
cd decisionlab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Click **🚀 Explore the example** to load a ready-made
supplier-selection problem, or **📝 Start from scratch**.

**Run the tests**

```bash
python -m pytest
```

---

## Guided workflow

| Step | What you do |
|---|---|
| **Home** | Overview of MCDM and of the methods; load the example or start blank. |
| **1 · Decision problem** | Title, number and names of alternatives and criteria (editable tables). |
| **2 · Criteria** | For each criterion: name, **Benefit** (higher is better) or **Cost** (lower is better), optional unit and description. |
| **3 · Decision matrix** | Type values directly into the grid; add/remove alternatives and criteria; upload a CSV; download a template or the current matrix; clear values; live validation of missing values. |
| **4 · Criteria weights** | Choose AHP, BWM, Entropy, CRITIC or Equal weights. The input form adapts to the method (pairwise sliders for AHP, best/worst vectors for BWM), with a live weights preview and diagnostics. |
| **5 · Ranking method** | Choose TOPSIS or SWM (with plain-language explanations); optionally add an equal-weights baseline. |
| **6 · Results** | **Calculate Ranking** validates everything and shows the dashboard. |

The sidebar shows a check mark for each completed step and an overall progress bar. Inputs are
kept when you move between steps. If you change an input after calculating, the results page tells
you that the results are out of date.

### Results dashboard

- **🏆 Ranking**: recommended alternative, rank, position, score bar, winner indicator, score chart,
  and the comparison with equal weights (rank changes and a rank dot plot).
- **⚖️ Weights**: weight table (value and %), horizontal importance chart, method diagnostics
  (λmax / CI / RI / CR for AHP; ξ*, consistency ratio and solver status for BWM; entropy,
  divergence, σ, correlation matrix and information content for Entropy and CRITIC).
- **📊 Charts**: heatmap of the decision matrix (coloured by direction-aware relative performance,
  labelled with raw values), score contributions per criterion (SWM) or distances to the ideal
  solutions (TOPSIS), and a radar chart to compare up to 8 alternatives.
- **🧮 Calculations**: LaTeX formulas and every intermediate matrix (normalised, weighted, ideal
  vectors, distances, closeness, contributions).
- **💡 Explanation**: why each alternative received its score, with its main strengths and
  weaknesses (weighted by importance).
- **⬇️ Export**: rankings CSV, weights CSV, a complete multi-sheet **Excel** report, and a **PDF**
  summary.

---

## CSV format

The first column holds alternative names and the header row holds criterion names. An optional
row whose first cell is `type` sets each criterion's type. Comma, semicolon and tab separators are
accepted; decimal commas are accepted with the semicolon separator.

```csv
Alternative,Price,Quality,Delivery time,Reliability,After-sales service
type,Cost,Benefit,Cost,Benefit,Benefit
Alpha Components,42,7,12,92,6
Beta Supply Co.,38,6,15,85,8
```

Non-numeric cells are left empty and listed, so you can fix them in the editor. See
[sample_data/supplier_selection.csv](sample_data/supplier_selection.csv).

---

## Methods and formulas

### AHP
Reciprocal pairwise matrix on Saaty's 1–9 scale (`a_ii = 1`, `a_ji = 1/a_ij`). The weights are the
normalised principal right eigenvector. `CI = (λmax − n)/(n − 1)` and `CR = CI / RI(n)`, using
Saaty's Random Index for n = 1…15. A warning appears when CR > 0.10.

### BWM
Rezaei's (2015) non-linear model:
`min ξ  s.t. |w_B/w_j − a_Bj| ≤ ξ, |w_j/w_W − a_jW| ≤ ξ, Σw_j = 1, w_j ≥ 0`.
The app reports the weights, optimal ξ*, consistency ratio `ξ*/CI(a_BW)` and solver status.

### Entropy
`p_ij = r_ij / Σ_i r_ij`, `e_j = −(1/ln m) Σ_i p_ij ln p_ij` (with `0 ln 0 = 0`), `d_j = 1 − e_j`,
`w_j = d_j / Σ d_j`.

### CRITIC
`C_j = σ_j Σ_k (1 − ρ_jk)`, `w_j = C_j / Σ C_j`, computed on the normalised matrix.

### TOPSIS
Vector normalisation → weighting → ideal best/worst by criterion type → Euclidean distances
`D+`, `D−` → closeness `CC = D− / (D+ + D−)` → rank by descending CC.

### SWM
Direction-aware min-max normalisation → `S_i = Σ_j w_j r_ij` → rank by descending S.

---

## Conventions (where the literature leaves a choice)

| Topic | Convention used |
|---|---|
| Normalisation for SWM, Entropy, CRITIC and the charts | Linear min-max. Benefit: `(x − min)/(max − min)`; Cost: `(max − x)/(max − min)`. Works with zeros and negative values. |
| Constant criterion (same value for all alternatives) | Normalised value 1 for every alternative, so it does not affect the order. Entropy and CRITIC give it weight 0, and a warning is shown. |
| All criteria constant (Entropy/CRITIC) | Error, because no objective weights exist; the message suggests AHP, BWM or Equal weights. Nothing falls back silently. |
| TOPSIS all-zero column | Cannot be vector-normalised, so it is ignored (zeros) and a warning is shown. |
| TOPSIS identical alternatives | Closeness is undefined; every alternative gets 0.5 (a tie) and a warning is shown. |
| TOPSIS negative values | Allowed (vector normalisation preserves order); a warning notes that interpretation is harder. |
| CRITIC standard deviation | Sample (ddof = 1). The constant factor cancels in the final weights. Correlation involving a constant criterion is 0. |
| AHP with 1 or 2 criteria | CR = 0 by definition. More than 15 criteria is rejected (no tabulated RI). |
| BWM solver | SLSQP with `w_j ≥ 10⁻⁶` so the ratios are defined. It starts from the linear-BWM solution (`linprog`/HiGHS), the geometric-mean vector and equal weights, and keeps the best feasible solution. It is deterministic. |
| BWM a_BW | The Best-to-Worst judgment is entered once and shared by both vectors. |
| Ties | Competition ranking: 1, 2, 2, 4. |
| Weights | Must be finite, non-negative and sum to 1 (tolerance 10⁻⁶). They are never silently rescaled. |
| Entropy/CRITIC with one alternative | Error: at least two alternatives are required. |

---

## Project structure

```
decisionlab/
├── app.py                  # Streamlit UI (pages, dashboard); no maths here
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml  # theme
├── core/
│   ├── models.py           # MCDMError, WeightingResult, RankingResult, rank table
│   ├── validation.py       # matrix / names / types / weights validation
│   ├── normalization.py    # min-max and vector normalisation, constant detection
│   └── explain.py          # per-alternative strengths/weaknesses
├── methods/
│   ├── ahp.py  bwm.py  entropy.py  critic.py  equal.py   # weighting
│   └── topsis.py  swm.py                                 # ranking
├── utils/
│   ├── visualizations.py   # Plotly figures (colour-blind-safe palette)
│   ├── export.py           # CSV, Excel (openpyxl), PDF (fpdf2)
│   └── csv_io.py           # CSV import/template
├── ui/
│   ├── state.py            # session state, stable ids, example problem
│   └── content.py          # method descriptions, formulas, conventions
├── sample_data/supplier_selection.csv
└── tests/                  # pytest suite for every method + validation/IO/export
```

Every method returns a `WeightingResult` or `RankingResult` holding the result, the intermediate
tables (`details`) and non-fatal `warnings`. Invalid input raises `MCDMError` with a message that
tells the user what to fix.

### Using the methods from Python

```python
import pandas as pd
from methods import ahp_weights, build_pairwise_matrix, topsis

matrix = pd.DataFrame([[250, 16, 12], [200, 16, 8], [300, 32, 16]],
                      index=["A", "B", "C"], columns=["Price", "Storage", "Camera"])
types = ["cost", "benefit", "benefit"]
weights = ahp_weights(build_pairwise_matrix(3, {(0, 1): 3, (0, 2): 5, (1, 2): 2}),
                      list(matrix.columns))
print(weights.weights, weights.details["CR"])
print(topsis(matrix, types, weights.weights).ranking)
```

---

## Testing

The suite in `tests/` (84 tests) covers:

- known results: Saaty's 3×3 AHP example (λmax ≈ 3.0385) and the classic 5×4 TOPSIS example;
- weights that are normalised and non-negative, and correct benefit/cost handling;
- AHP consistency (CI, RI, CR) and BWM constraint satisfaction at the optimum;
- zero values, negative values, constant criteria, ties, single alternative or criterion;
- invalid inputs (missing or non-numeric cells, duplicate names, bad types, invalid weights, invalid
  AHP matrices and BWM vectors);
- reproducibility, CSV round-trips, and the Excel and PDF exports.

```bash
python -m pytest -v
```
