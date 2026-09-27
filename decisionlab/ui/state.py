"""Session-state model for the Streamlit app.

Alternatives and criteria carry stable integer ids (the DataFrame index), so that
renaming, adding or removing items never scrambles the matrix or the AHP/BWM
judgments, which are keyed by id.

Widgets whose key starts with ``p_`` are *persistent*: their values are re-assigned
at the top of every run so they survive navigation between steps.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils.csv_io import ParsedMatrix, parse_matrix_csv

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "supplier_selection.csv"
SAMPLE_TITLE = "Selecting a component supplier"
SAMPLE_CRITERIA_INFO = {
    "Price": ("€/unit", "Unit price quoted for the component."),
    "Quality": ("score 1–10", "Audit score of product quality."),
    "Delivery time": ("days", "Average lead time from order to delivery."),
    "Reliability": ("% on time", "Share of past deliveries that arrived on time."),
    "After-sales service": ("score 1–10", "Responsiveness and support quality."),
}
SAMPLE_ALT_INFO = {
    "Alpha Components": "Established regional supplier.",
    "Beta Supply Co.": "Low-cost importer.",
    "Gamma Industrial": "Premium manufacturer.",
    "Delta Parts": "Mid-range specialist.",
    "Epsilon Trading": "Broker with flexible terms.",
}
# Sample AHP judgments by criterion *name*: (i, j) -> a_ij (i more important if > 1).
SAMPLE_AHP = {
    ("Price", "Quality"): 2, ("Price", "Delivery time"): 3, ("Price", "Reliability"): 2,
    ("Price", "After-sales service"): 5, ("Quality", "Delivery time"): 2,
    ("Quality", "Reliability"): 1, ("Quality", "After-sales service"): 3,
    ("Delivery time", "Reliability"): 0.5, ("Delivery time", "After-sales service"): 2,
    ("Reliability", "After-sales service"): 3,
}
SAMPLE_BWM = {"best": "Price", "worst": "After-sales service",
              "bo": {"Price": 1, "Quality": 2, "Delivery time": 4, "Reliability": 3,
                     "After-sales service": 7},
              "ow": {"Price": 7, "Quality": 4, "Delivery time": 2, "Reliability": 3,
                     "After-sales service": 1}}

MAX_ALTERNATIVES = 50
MAX_CRITERIA = 20
ALT_COLUMNS = ["Alternative", "Description"]
CRIT_COLUMNS = ["Criterion", "Type", "Unit", "Description"]

ss = st.session_state


# ----------------------------------------------------------------------------- setup

def keep_persistent_widgets() -> None:
    """Re-assign ``p_*`` widget values so Streamlit does not drop them between steps."""
    for key in list(ss.keys()):
        if isinstance(key, str) and key.startswith("p_"):
            ss[key] = ss[key]


def init_state() -> None:
    if "initialised" in ss:
        return
    ss.initialised = True
    ss.next_id = 0
    ss.editor_version = {}
    ss.results = None
    ss.results_signature = None
    ss.flash = None
    load_example()


def new_id() -> int:
    ss.next_id += 1
    return ss.next_id


def bump(*editors: str) -> None:
    """Force the named data editors to re-mount from canonical state."""
    for name in editors:
        ss.editor_version[name] = ss.editor_version.get(name, 0) + 1


def editor_key(name: str) -> str:
    return f"ed_{name}_{ss.editor_version.get(name, 0)}"


def clear_method_inputs() -> None:
    for key in [k for k in ss.keys() if isinstance(k, str) and
                (k.startswith("p_ahp_") or k.startswith("p_bwm_"))]:
        del ss[key]


def set_problem(title: str, matrix: pd.DataFrame, types: list[str],
                units: dict[str, str] | None = None,
                descriptions: dict[str, str] | None = None,
                alt_descriptions: dict[str, str] | None = None) -> None:
    """Replace the whole problem (names, types, values) with fresh ids."""
    units, descriptions, alt_descriptions = units or {}, descriptions or {}, alt_descriptions or {}
    alt_ids = [new_id() for _ in matrix.index]
    crit_ids = [new_id() for _ in matrix.columns]
    ss.alts = pd.DataFrame({"Alternative": [str(a) for a in matrix.index],
                            "Description": [alt_descriptions.get(str(a), "")
                                            for a in matrix.index]}, index=alt_ids)
    ss.crits = pd.DataFrame({"Criterion": [str(c) for c in matrix.columns],
                             "Type": types,
                             "Unit": [units.get(str(c), "") for c in matrix.columns],
                             "Description": [descriptions.get(str(c), "")
                                             for c in matrix.columns]}, index=crit_ids)
    ss.decision_values = pd.DataFrame(matrix.to_numpy(dtype=float), index=alt_ids, columns=crit_ids)
    ss.p_title = title
    ss.p_n_alt = len(alt_ids)
    ss.p_n_crit = len(crit_ids)
    clear_method_inputs()
    ss.results = None
    bump("alts", "crits_names", "crits", "matrix")


def load_example() -> None:
    parsed = parse_matrix_csv(SAMPLE_PATH.read_bytes())
    set_problem(SAMPLE_TITLE, parsed.matrix, parsed.types or ["Benefit"] * parsed.matrix.shape[1],
                units={k: v[0] for k, v in SAMPLE_CRITERIA_INFO.items()},
                descriptions={k: v[1] for k, v in SAMPLE_CRITERIA_INFO.items()},
                alt_descriptions=SAMPLE_ALT_INFO)
    name_to_id = dict(zip(ss.crits["Criterion"], ss.crits.index))
    for (a, b), value in SAMPLE_AHP.items():
        i, j = name_to_id[a], name_to_id[b]
        ss[ahp_key(i, j)] = to_slider(value)
    ss.p_bwm_best = name_to_id[SAMPLE_BWM["best"]]
    ss.p_bwm_worst = name_to_id[SAMPLE_BWM["worst"]]
    for name, value in SAMPLE_BWM["bo"].items():
        ss[f"p_bwm_bo_{name_to_id[name]}"] = value
    for name, value in SAMPLE_BWM["ow"].items():
        ss[f"p_bwm_ow_{name_to_id[name]}"] = value
    ss.p_weight_method = "AHP"
    ss.p_rank_method = "TOPSIS"
    ss.p_baseline = True


def start_blank(n_alt: int = 3, n_crit: int = 3) -> None:
    matrix = pd.DataFrame(np.nan, index=[f"Alternative {i + 1}" for i in range(n_alt)],
                          columns=[f"Criterion {j + 1}" for j in range(n_crit)])
    set_problem("My decision", matrix, ["Benefit"] * n_crit)
    ss.p_weight_method = "Equal weights"
    ss.p_rank_method = "TOPSIS"
    ss.p_baseline = False


# ------------------------------------------------------------------- AHP slider codes
# Slider value v: v >= 2 → row criterion is v times more important; v <= -2 → the column
# criterion is |v| times more important; 1 → equal.

AHP_OPTIONS = [9, 8, 7, 6, 5, 4, 3, 2, 1, -2, -3, -4, -5, -6, -7, -8, -9]


def ahp_key(i: int, j: int) -> str:
    return f"p_ahp_{i}_{j}"


def to_slider(a_ij: float) -> int:
    if a_ij >= 1:
        return int(round(a_ij))
    return -int(round(1 / a_ij))


def from_slider(v: int) -> float:
    return float(v) if v >= 1 else 1.0 / abs(v)


# --------------------------------------------------------------------- structure edits

def _unique_name(prefix: str, existing: list[str]) -> str:
    k = len(existing) + 1
    while f"{prefix} {k}" in existing:
        k += 1
    return f"{prefix} {k}"


def add_alternative() -> None:
    if len(ss.alts) >= MAX_ALTERNATIVES:
        ss.flash = ("warning", f"At most {MAX_ALTERNATIVES} alternatives are supported.")
        return
    i = new_id()
    ss.alts.loc[i] = [_unique_name("Alternative", ss.alts["Alternative"].tolist()), ""]
    ss.decision_values.loc[i] = np.nan
    ss.p_n_alt = len(ss.alts)
    bump("alts", "matrix")


def add_criterion() -> None:
    if len(ss.crits) >= MAX_CRITERIA:
        ss.flash = ("warning", f"At most {MAX_CRITERIA} criteria are supported.")
        return
    j = new_id()
    ss.crits.loc[j] = [_unique_name("Criterion", ss.crits["Criterion"].tolist()),
                       "Benefit", "", ""]
    ss.decision_values[j] = np.nan
    ss.p_n_crit = len(ss.crits)
    bump("crits_names", "crits", "matrix")


def remove_alternative(alt_id: int) -> None:
    if len(ss.alts) <= 1:
        ss.flash = ("warning", "At least one alternative is required.")
        return
    ss.alts = ss.alts.drop(index=alt_id)
    ss.decision_values = ss.decision_values.drop(index=alt_id)
    ss.p_n_alt = len(ss.alts)
    bump("alts", "matrix")


def remove_criterion(crit_id: int) -> None:
    if len(ss.crits) <= 1:
        ss.flash = ("warning", "At least one criterion is required.")
        return
    ss.crits = ss.crits.drop(index=crit_id)
    ss.decision_values = ss.decision_values.drop(columns=crit_id)
    for key in [k for k in ss.keys() if isinstance(k, str) and k.startswith("p_") and
                (k.endswith(f"_{crit_id}") or f"_{crit_id}_" in k)]:
        del ss[key]
    for key in ("p_bwm_best", "p_bwm_worst"):
        if ss.get(key) == crit_id:
            del ss[key]
    ss.p_n_crit = len(ss.crits)
    bump("crits_names", "crits", "matrix")


def resize_alternatives() -> None:
    """Callback for the 'number of alternatives' input: add/remove at the end."""
    target = int(ss.p_n_alt)
    while len(ss.alts) < target:
        add_alternative()
    while len(ss.alts) > target:
        remove_alternative(ss.alts.index[-1])


def resize_criteria() -> None:
    target = int(ss.p_n_crit)
    while len(ss.crits) < target:
        add_criterion()
    while len(ss.crits) > target:
        remove_criterion(ss.crits.index[-1])


def clear_values() -> None:
    ss.decision_values.loc[:, :] = np.nan
    bump("matrix")
    ss.flash = ("info", "All matrix values were cleared.")


def import_csv(data: bytes) -> ParsedMatrix:
    """Load an uploaded matrix, keeping units/descriptions of matching criteria."""
    parsed = parse_matrix_csv(data)
    old = ss.crits.set_index("Criterion")
    types = parsed.types or [old["Type"].get(c, "Benefit") for c in parsed.matrix.columns]
    set_problem(ss.get("p_title", "My decision"), parsed.matrix, types,
                units=old["Unit"].to_dict(), descriptions=old["Description"].to_dict(),
                alt_descriptions=ss.alts.set_index("Alternative")["Description"].to_dict())
    return parsed


# ------------------------------------------------------------------------ accessors

def alt_names() -> list[str]:
    return [str(a).strip() if pd.notna(a) else "" for a in ss.alts["Alternative"]]


def crit_names() -> list[str]:
    return [str(c).strip() if pd.notna(c) else "" for c in ss.crits["Criterion"]]


def crit_types() -> list[str]:
    return [str(t) if pd.notna(t) else "" for t in ss.crits["Type"]]


def named_matrix() -> pd.DataFrame:
    """The decision matrix labelled with current names (may contain NaN)."""
    frame = ss.decision_values.loc[ss.alts.index, ss.crits.index].copy()
    frame.index = alt_names()
    frame.columns = crit_names()
    return frame


def crit_label(crit_id: int) -> str:
    return str(ss.crits.at[crit_id, "Criterion"])


def signature() -> str:
    """Hash of every input that affects the results."""
    parts: list[Any] = [ss.alts.to_json(), ss.crits.to_json(), ss.decision_values.to_json(),
                        ss.get("p_weight_method"), ss.get("p_rank_method"),
                        ss.get("p_baseline")]
    parts += sorted((k, str(ss[k])) for k in ss.keys()
                    if isinstance(k, str) and (k.startswith("p_ahp_") or k.startswith("p_bwm_")))
    return hashlib.sha1(repr(parts).encode()).hexdigest()
