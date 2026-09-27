"""BWM — Best-Worst Method (Rezaei, 2015), non-linear model.

    minimise ξ
    s.t.  |w_B / w_j − a_Bj| ≤ ξ   for all j
          |w_j / w_W − a_jW| ≤ ξ   for all j
          Σ_j w_j = 1,  w_j ≥ 0

The model is solved with SciPy's SLSQP from several deterministic starting points,
one of which is the solution of the linear BWM (solved exactly with ``linprog``).
"""

from __future__ import annotations

import warnings as _warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from core.models import MCDMError, WeightingResult
from core.validation import validate_names

# Consistency Index of the non-linear BWM by a_BW (Rezaei, 2015, Table 3).
CONSISTENCY_INDEX: dict[int, float] = {
    1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23,
}
# Lower bound on each weight so the ratio constraints stay defined.
MIN_WEIGHT = 1e-6
FEASIBILITY_TOL = 1e-6


def validate_bwm_inputs(best: int, worst: int, best_to_others: Sequence[float],
                        others_to_worst: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Check BWM comparison vectors; return them as float arrays."""
    a_b = np.asarray(best_to_others, dtype=float)
    a_w = np.asarray(others_to_worst, dtype=float)
    n = a_b.size
    if n == 0 or a_w.size != n:
        raise MCDMError("Best-to-Others and Others-to-Worst vectors must have one value "
                        "per criterion.")
    if not (0 <= best < n and 0 <= worst < n):
        raise MCDMError("Best and worst criteria must be valid criterion positions.")
    if n > 1 and best == worst:
        raise MCDMError("The best and worst criteria must be different.")
    for label, vec in (("Best-to-Others", a_b), ("Others-to-Worst", a_w)):
        if not np.all(np.isfinite(vec)):
            raise MCDMError(f"{label} comparisons contain missing values.")
        if np.any(vec < 1) or np.any(vec > 9):
            raise MCDMError(f"{label} comparisons must be on the 1–9 scale.")
    if a_b[best] != 1:
        raise MCDMError("The best criterion compared with itself must equal 1.")
    if a_w[worst] != 1:
        raise MCDMError("The worst criterion compared with itself must equal 1.")
    if n > 1 and a_b[worst] != a_w[best]:
        raise MCDMError(
            f"The Best-to-Worst preference must be the same in both vectors "
            f"(Best-to-Others gives {a_b[worst]:g}, Others-to-Worst gives {a_w[best]:g}).")
    return a_b, a_w


def _ordinal_warnings(a_b: np.ndarray, a_w: np.ndarray, best: int, worst: int,
                      names: list[str]) -> list[str]:
    """Flag judgments that contradict the choice of best/worst criteria."""
    warnings: list[str] = []
    a_bw = a_b[worst]
    for j, name in enumerate(names):
        if a_b[j] > a_bw:
            warnings.append(f"Best is preferred over '{name}' ({a_b[j]:g}) more than over "
                            f"the worst criterion ({a_bw:g}). This is inconsistent.")
        if a_w[j] > a_bw:
            warnings.append(f"'{name}' is preferred over Worst ({a_w[j]:g}) more than the "
                            f"best criterion is ({a_bw:g}). This is inconsistent.")
    return warnings


def _linear_bwm(a_b: np.ndarray, a_w: np.ndarray, best: int, worst: int) -> np.ndarray | None:
    """Linear BWM (Rezaei, 2016): |w_B − a_Bj w_j| ≤ ξ, |w_j − a_jW w_W| ≤ ξ."""
    n = a_b.size
    rows: list[np.ndarray] = []
    for j in range(n):
        for coeffs in (_row(n, {best: 1.0, j: -a_b[j]}), _row(n, {j: 1.0, worst: -a_w[j]})):
            rows.append(np.append(coeffs, -1.0))
            rows.append(np.append(-coeffs, -1.0))
    a_ub = np.vstack(rows)
    b_ub = np.zeros(len(rows))
    a_eq = np.append(np.ones(n), 0.0)[None, :]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0],
                  bounds=[(MIN_WEIGHT, 1)] * n + [(0, None)], method="highs")
    return res.x[:n] if res.success else None


def _row(n: int, entries: dict[int, float]) -> np.ndarray:
    row = np.zeros(n)
    for idx, value in entries.items():
        row[idx] += value
    return row


def _max_deviation(w: np.ndarray, a_b: np.ndarray, a_w: np.ndarray, best: int,
                   worst: int) -> float:
    """The smallest ξ consistent with weights ``w`` (objective of the model)."""
    dev_b = np.abs(w[best] / w - a_b)
    dev_w = np.abs(w / w[worst] - a_w)
    return float(max(dev_b.max(), dev_w.max()))


def bwm_weights(best: int, worst: int, best_to_others: Sequence[float],
                others_to_worst: Sequence[float],
                criteria: Sequence[str] | None = None) -> WeightingResult:
    """Solve the non-linear BWM model.

    Args:
        best: Position of the most important criterion.
        worst: Position of the least important criterion.
        best_to_others: ``a_Bj`` — preference of Best over each criterion (a_BB = 1).
        others_to_worst: ``a_jW`` — preference of each criterion over Worst (a_WW = 1).
        criteria: Criterion names (defaults to ``C1..Cn``).

    Returns:
        WeightingResult with ``details`` holding ``xi`` (optimal ξ), ``consistency_ratio``,
        ``solver_status``, ``solver_success`` and the comparison vectors.
    """
    a_b, a_w = validate_bwm_inputs(best, worst, best_to_others, others_to_worst)
    n = a_b.size
    names = validate_names(criteria, "criterion") if criteria is not None else \
        [f"C{i + 1}" for i in range(n)]
    if len(names) != n:
        raise MCDMError(f"Got {len(names)} criterion names for {n} comparisons.")
    labels = pd.Index(names)

    if n == 1:
        return WeightingResult(
            "BWM", pd.Series([1.0], index=labels, name="Weight"),
            {"xi": 0.0, "consistency_ratio": 0.0, "solver_status": "Trivial (one criterion)",
             "solver_success": True, "best": names[0], "worst": names[0],
             "comparisons": _comparison_table(labels, a_b, a_w)},
            ["Only one criterion: it receives 100% of the weight."])

    warnings = _ordinal_warnings(a_b, a_w, best, worst, names)

    def objective(x: np.ndarray) -> float:
        return x[-1]

    def inequality(x: np.ndarray) -> np.ndarray:
        w, xi = x[:-1], x[-1]
        ratio_b = w[best] / w - a_b
        ratio_w = w / w[worst] - a_w
        return np.concatenate([xi - ratio_b, xi + ratio_b, xi - ratio_w, xi + ratio_w])

    constraints = [{"type": "ineq", "fun": inequality},
                   {"type": "eq", "fun": lambda x: np.sum(x[:-1]) - 1.0}]
    bounds = [(MIN_WEIGHT, 1.0)] * n + [(0.0, None)]

    starts: list[np.ndarray] = []
    linear = _linear_bwm(a_b, a_w, best, worst)
    if linear is not None:
        starts.append(linear)
    geometric = np.sqrt(a_w / a_b)  # blends both vectors; exact for consistent input
    starts.append(geometric / geometric.sum())
    starts.append(np.full(n, 1.0 / n))

    best_solution: np.ndarray | None = None
    best_xi = np.inf
    status_message = "Solver did not run."
    solver_success = False
    for start in starts:
        start = np.clip(start, MIN_WEIGHT, None)
        start = start / start.sum()
        x0 = np.append(start, _max_deviation(start, a_b, a_w, best, worst))
        with _warnings.catch_warnings():
            # SLSQP may step marginally outside the bounds and clip; this is harmless.
            _warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
        w = np.clip(res.x[:-1], MIN_WEIGHT, None)
        w = w / w.sum()
        xi = _max_deviation(w, a_b, a_w, best, worst)
        feasible = np.all(inequality(np.append(w, xi)) >= -FEASIBILITY_TOL)
        if feasible and xi < best_xi - 1e-12:
            best_solution, best_xi = w, xi
            status_message, solver_success = res.message, bool(res.success)

    if best_solution is None:
        raise MCDMError("The BWM optimisation failed to find a feasible solution. "
                        "Check that your comparisons are on the 1–9 scale and consistent.")
    if not solver_success:
        warnings.append(f"SLSQP reported: '{status_message}'. The best feasible solution "
                        "found is shown; treat it with caution.")

    a_bw = int(round(a_b[worst]))
    ci = CONSISTENCY_INDEX.get(a_bw, 0.0)
    cr = best_xi / ci if ci > 0 else 0.0
    if cr > 0.1:
        warnings.append(f"BWM consistency ratio is {cr:.3f} (ξ* = {best_xi:.4f}). Values "
                        "close to 0 are preferred; consider revising your comparisons.")

    details = {
        "xi": best_xi,
        "consistency_index": ci,
        "consistency_ratio": cr,
        "solver_status": str(status_message),
        "solver_success": solver_success,
        "best": names[best],
        "worst": names[worst],
        "comparisons": _comparison_table(labels, a_b, a_w),
    }
    return WeightingResult("BWM", pd.Series(best_solution, index=labels, name="Weight"),
                           details, warnings)


def _comparison_table(labels: pd.Index, a_b: np.ndarray, a_w: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Best-to-Others (a_Bj)": a_b, "Others-to-Worst (a_jW)": a_w},
                        index=labels)
