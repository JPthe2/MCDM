"""DecisionLab — Multi-Criteria Decision Engine.

Streamlit entry point. All mathematics lives in ``core/`` and ``methods/``; this file
only collects inputs, calls those functions and presents the results.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.explain import explain_ranking  # noqa: E402
from core.models import MCDMError, RankingResult, WeightingResult  # noqa: E402
from core.normalization import constant_columns  # noqa: E402
from core.validation import (find_invalid_cells, validate_criteria_types,  # noqa: E402
                             validate_decision_matrix, validate_names)
from methods import (RANKING_METHODS, WEIGHTING_METHODS, ahp_weights,  # noqa: E402
                     build_pairwise_matrix, bwm_weights, critic_weights, entropy_weights,
                     equal_weights, swm, topsis)
from methods.ahp import CR_THRESHOLD, MAX_CRITERIA as AHP_MAX_CRITERIA  # noqa: E402
from ui import state as S  # noqa: E402
from ui.content import (CONVENTIONS, FORMULAS, INTENSITY_WORDS, RANKING_INFO,  # noqa: E402
                        SAATY_HELP, WEIGHTING_INFO)
from utils import export as X  # noqa: E402
from utils import visualizations as V  # noqa: E402
from utils.csv_io import matrix_to_csv, template_csv  # noqa: E402

st.set_page_config(page_title="DecisionLab — Multi-Criteria Decision Engine",
                   page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

ss = st.session_state
S.init_state()
S.keep_persistent_widgets()

PAGES = ["home", "problem", "criteria", "matrix", "weighting", "ranking", "results"]
PAGE_LABELS = {
    "home": "Home",
    "problem": "1 · Decision problem",
    "criteria": "2 · Criteria",
    "matrix": "3 · Decision matrix",
    "weighting": "4 · Criteria weights",
    "ranking": "5 · Ranking method",
    "results": "6 · Results",
}
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}

st.markdown("""
<style>
.block-container {padding-top: 3.5rem; max-width: 1250px;}
.dl-hero {padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(42,120,214,0.16), rgba(42,120,214,0.03));
  border: 1px solid rgba(42,120,214,0.25);}
.dl-hero h1 {margin: 0 0 .3rem 0; font-size: 2.1rem;}
.dl-hero p {margin: 0; font-size: 1.05rem; opacity: .85;}
.dl-card {padding: 1rem 1.1rem; border-radius: 12px; height: 100%;
  border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.04);}
.dl-card h4 {margin: 0 0 .35rem 0; font-size: 1.02rem;}
.dl-card p {margin: 0; font-size: .92rem; opacity: .85;}
.dl-winner {padding: 1.1rem 1.4rem; border-radius: 14px; margin: .4rem 0 1rem 0;
  background: linear-gradient(135deg, rgba(27,175,122,0.18), rgba(27,175,122,0.04));
  border: 1px solid rgba(27,175,122,0.45);}
.dl-winner .label {font-size: .85rem; text-transform: uppercase; letter-spacing: .06em;
  opacity: .75;}
.dl-winner .name {font-size: 1.7rem; font-weight: 700; margin: .1rem 0;}
.dl-step {font-size: .85rem; text-transform: uppercase; letter-spacing: .08em;
  color: #2a78d6; font-weight: 600; margin-bottom: -.4rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================== helpers

def goto(page: str) -> None:
    ss.p_nav = page


def step_header(number: int | None, title: str, subtitle: str) -> None:
    if number is not None:
        st.markdown(f'<div class="dl-step">Step {number} of 6</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def nav_buttons(back: str | None, forward: str | None, forward_label: str = "Next") -> None:
    st.divider()
    left, _, right = st.columns([1, 3, 1])
    if back:
        left.button(f"← {PAGE_LABELS[back].split('· ')[-1]}", on_click=goto, args=(back,),
                    width="stretch")
    if forward:
        right.button(f"{forward_label} →", on_click=goto, args=(forward,), type="primary",
                     width="stretch")


def persistent_editor(name: str, build: Callable[[], pd.DataFrame], suffix: str = "",
                      **kwargs: Any) -> pd.DataFrame:
    """A data editor whose edits survive navigation between steps.

    The base data is rebuilt from canonical state only when the editor is (re)mounted,
    so edits are never applied twice.
    """
    key = f"{S.editor_key(name)}{suffix}"
    base_key = f"_base_{name}"
    if key not in ss or ss.get(f"_basekey_{name}") != key:
        ss[base_key] = build()
        ss[f"_basekey_{name}"] = key
    return st.data_editor(ss[base_key], key=key, **kwargs)


def show_flash() -> None:
    if ss.get("flash"):
        kind, message = ss.flash
        getattr(st, kind)(message)
        ss.flash = None


def fmt_frame(frame: pd.DataFrame, precision: int = 4) -> Any:
    return frame.style.format(precision=precision, na_rep="—")


def as_fraction(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    frac = Fraction(value).limit_denominator(9)
    return f"1/{frac.denominator}" if frac.numerator == 1 else f"{value:.3f}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ====================================================================== computation

def validated_inputs() -> tuple[pd.DataFrame, list[str]]:
    validate_names(S.alt_names(), "alternative")
    validate_names(S.crit_names(), "criterion")
    types = validate_criteria_types(S.crit_types(), len(ss.crits))
    return validate_decision_matrix(S.named_matrix()), types


def compute_weights(method: str | None = None) -> WeightingResult:
    """Compute weights for the selected method from the current inputs."""
    method = method or ss.get("p_weight_method", "Equal weights")
    names = validate_names(S.crit_names(), "criterion")
    ids = list(ss.crits.index)
    n = len(ids)
    if method == "AHP":
        if n > AHP_MAX_CRITERIA:
            raise MCDMError(f"AHP supports at most {AHP_MAX_CRITERIA} criteria; you have {n}. "
                            "Use BWM, Entropy or CRITIC instead.")
        judgments = {(a, b): S.from_slider(ss.get(S.ahp_key(ids[a], ids[b]), 1))
                     for a in range(n) for b in range(a + 1, n)}
        return ahp_weights(build_pairwise_matrix(n, judgments), names)
    if method == "BWM":
        if n == 1:
            return bwm_weights(0, 0, [1], [1], names)
        best_id, worst_id = ss.get("p_bwm_best"), ss.get("p_bwm_worst")
        if best_id not in ids or worst_id not in ids:
            raise MCDMError("Select the best and worst criteria in Step 4.")
        best, worst = ids.index(best_id), ids.index(worst_id)
        a_b = [1.0 if i == best_id else float(ss.get(f"p_bwm_bo_{i}", 2)) for i in ids]
        a_w = [1.0 if i == worst_id else float(ss.get(f"p_bwm_ow_{i}", 2)) for i in ids]
        a_w[best] = a_b[worst]  # the Best-to-Worst judgment is shared by both vectors
        return bwm_weights(best, worst, a_b, a_w, names)
    if method in ("Entropy", "CRITIC"):
        matrix, types = validated_inputs()
        return entropy_weights(matrix, types) if method == "Entropy" else \
            critic_weights(matrix, types)
    return equal_weights(names)


def cached_weights() -> tuple[WeightingResult | None, str | None]:
    """Weights for the current inputs, memoised on the input signature."""
    sig = S.signature()
    cache = ss.get("_weights_cache")
    if cache and cache[0] == sig:
        return cache[1], cache[2]
    try:
        result, error = compute_weights(), None
    except MCDMError as exc:
        result, error = None, str(exc)
    ss._weights_cache = (sig, result, error)
    return result, error


def rank(matrix: pd.DataFrame, types: list[str], weights: pd.Series,
         method: str) -> RankingResult:
    return topsis(matrix, types, weights) if method == "TOPSIS" else swm(matrix, types, weights)


def run_analysis() -> dict[str, Any]:
    """Validate everything, compute weights and ranking; raise MCDMError on problems."""
    matrix, types = validated_inputs()
    weighting = compute_weights()
    method = ss.get("p_rank_method", "TOPSIS")
    ranking = rank(matrix, types, weighting.weights, method)
    baseline = None
    if ss.get("p_baseline") and weighting.method != "Equal weights":
        base = rank(matrix, types, equal_weights(list(matrix.columns)).weights, method)
        chosen = ranking.ranking.set_index("Alternative")
        equal = base.ranking.set_index("Alternative")
        baseline = pd.DataFrame({
            "Alternative": chosen.index,
            f"Rank ({weighting.method})": chosen["Rank"].values,
            f"Score ({weighting.method})": chosen["Score"].values,
            "Rank (Equal weights)": equal.loc[chosen.index, "Rank"].values,
            "Score (Equal weights)": equal.loc[chosen.index, "Score"].values,
        })
        baseline["Rank change"] = baseline["Rank (Equal weights)"] - \
            baseline[f"Rank ({weighting.method})"]
    criteria = ss.crits[S.CRIT_COLUMNS].reset_index(drop=True).fillna("")
    return {
        "title": ss.get("p_title", "") or "Untitled decision",
        "matrix": matrix, "types": types, "criteria": criteria,
        "weighting": weighting, "ranking": ranking,
        "explanation": explain_ranking(ranking), "baseline": baseline,
        "signature": S.signature(),
    }


def step_status() -> dict[str, bool]:
    names_ok = True
    try:
        validate_names(S.alt_names(), "alternative")
        validate_names(S.crit_names(), "criterion")
    except MCDMError:
        names_ok = False
    types_ok = all(t in ("Benefit", "Cost") for t in S.crit_types())
    matrix_ok = names_ok and not find_invalid_cells(S.named_matrix())
    weights, _ = cached_weights()
    results_ok = ss.get("results") is not None and ss.results["signature"] == S.signature()
    return {
        "problem": names_ok and bool(str(ss.get("p_title", "")).strip()),
        "criteria": names_ok and types_ok,
        "matrix": matrix_ok,
        "weighting": weights is not None,
        "ranking": ss.get("p_rank_method") in RANKING_METHODS,
        "results": results_ok,
    }


# ======================================================================== sidebar

def render_sidebar() -> None:
    status = step_status()
    with st.sidebar:
        st.markdown("## ⚖️ DecisionLab")
        st.caption("Multi-Criteria Decision Engine")
        ss.setdefault("p_nav", "home")

        def label(page: str) -> str:
            if page == "home":
                return "🏠 Home"
            return f"{'✅' if status[page] else '⬜'} {PAGE_LABELS[page]}"

        st.radio("Workflow", PAGES, key="p_nav", format_func=label,
                 help="Follow the steps in order. A check mark means the step is complete.")
        done = sum(status.values())
        st.progress(done / len(status), text=f"{done} of {len(status)} steps complete")
        st.divider()
        st.markdown(f"**Problem:** {ss.get('p_title') or '—'}")
        st.caption(f"{len(ss.alts)} alternatives · {len(ss.crits)} criteria · "
                   f"{ss.get('p_weight_method', '—')} + {ss.get('p_rank_method', '—')}")
        st.divider()
        st.button("📦 Load example problem", on_click=S.load_example, width="stretch",
                  help="Replace the current problem with the supplier-selection example.")
        st.button("🧹 Start a blank problem", on_click=S.start_blank, width="stretch",
                  help="Clear everything and start with 3 alternatives and 3 criteria.")


# ========================================================================== pages

def page_home() -> None:
    st.markdown("""
<div class="dl-hero">
  <h1>⚖️ DecisionLab</h1>
  <p><b>Multi-Criteria Decision Engine</b> — compare options on several criteria at once,
  decide how much each criterion matters, and get a transparent, defensible ranking.</p>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("What does it do?")
        st.markdown(
            "Choosing a supplier, a laptop, a site or a project usually means trading off "
            "several criteria. Price should be low, quality high, delivery fast. No option "
            "is best on everything. **Multi-Criteria Decision Making (MCDM)** makes the trade-off "
            "explicit:\n\n"
            "1. **Weights** say how important each criterion is.\n"
            "2. A **ranking method** combines each option's performance with those weights "
            "into one score.\n\n"
            "Every number DecisionLab produces can be traced back to its formula, so you can "
            "explain the recommendation to others.")
        b1, b2 = st.columns(2)
        b1.button("🚀 Explore the example", type="primary", width="stretch",
                  on_click=lambda: (S.load_example(), goto("problem")),
                  help="A ready-made supplier-selection problem with 5 suppliers and 5 criteria.")
        b2.button("📝 Start from scratch", width="stretch",
                  on_click=lambda: (S.start_blank(), goto("problem")))
    with right:
        st.subheader("How it works")
        for number, (title, text) in enumerate([
            ("Define the problem", "Name your options (alternatives) and criteria."),
            ("Describe the criteria", "Is higher better (benefit) or lower better (cost)?"),
            ("Enter the data", "Type values or upload a CSV."),
            ("Weight the criteria", "AHP, BWM, Entropy, CRITIC or equal weights."),
            ("Pick a ranking method", "TOPSIS or the Simple Weighted Method."),
            ("Get results", "Ranking, charts, explanations and exports."),
        ], start=1):
            st.markdown(f"**{number}. {title}** — {text}")

    st.subheader("Available methods")
    cols = st.columns(len(WEIGHTING_METHODS))
    for col, method in zip(cols, WEIGHTING_METHODS):
        info = WEIGHTING_INFO[method]
        col.markdown(f'<div class="dl-card"><h4>{info["icon"]} {method}</h4>'
                     f'<p><i>{info["kind"]}</i></p></div>', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(2)
    for col, method in zip(cols, RANKING_METHODS):
        info = RANKING_INFO[method]
        col.markdown(f'<div class="dl-card"><h4>{info["icon"]} {method}</h4>'
                     f'<p>{info["summary"]}</p></div>', unsafe_allow_html=True)

    with st.expander("📖 Glossary"):
        st.markdown("""
- **Alternative**: one of the options you are choosing between (a row of the matrix).
- **Criterion**: an aspect you judge the alternatives on (a column of the matrix).
- **Benefit criterion**: higher values are better (quality, reliability).
- **Cost criterion**: lower values are better (price, delivery time).
- **Weight**: the share of importance of a criterion; all weights add up to 100%.
- **Normalisation**: rescaling criteria with different units (€, days, %) to a common scale.
- **Consistency ratio (AHP)**: checks that your pairwise judgments do not contradict each
  other; below 0.10 is considered acceptable.
""")


def page_problem() -> None:
    step_header(1, "Define the decision problem",
                "Give the decision a title and list the options (alternatives) and the "
                "criteria you will judge them on.")
    ss.setdefault("p_title", "My decision")
    st.text_input("Decision problem title", key="p_title",
                  placeholder="e.g. Choosing a new supplier",
                  help="A short name used in the results and exported reports.")
    c1, c2 = st.columns(2)
    c1.number_input("Number of alternatives", min_value=1, max_value=S.MAX_ALTERNATIVES,
                    step=1, key="p_n_alt", on_change=S.resize_alternatives,
                    help="The options you are choosing between. Rows are added or removed "
                         "at the end of the list.")
    c2.number_input("Number of criteria", min_value=1, max_value=S.MAX_CRITERIA, step=1,
                    key="p_n_crit", on_change=S.resize_criteria,
                    help="The aspects you judge the alternatives on.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Alternatives")
        st.caption("Double-click a cell to rename it.")
        edited = persistent_editor(
            "alts", lambda: ss.alts[S.ALT_COLUMNS].copy(), hide_index=True,
            column_config={
                "Alternative": st.column_config.TextColumn("Alternative", required=True,
                                                           help="Unique name of the option"),
                "Description": st.column_config.TextColumn("Description (optional)"),
            })
        ss.alts = edited
    with right:
        st.subheader("Criteria")
        st.caption("Types, units and descriptions are set in the next step.")
        edited = persistent_editor(
            "crits_names", lambda: ss.crits[["Criterion"]].copy(), hide_index=True,
            column_config={"Criterion": st.column_config.TextColumn(
                "Criterion", required=True, help="Unique name of the criterion")})
        ss.crits["Criterion"] = edited["Criterion"]

    try:
        validate_names(S.alt_names(), "alternative")
        validate_names(S.crit_names(), "criterion")
        st.success(f"✔ {len(ss.alts)} alternatives and {len(ss.crits)} criteria defined.")
    except MCDMError as exc:
        st.error(f"✖ {exc}")
    nav_buttons("home", "criteria")


def page_criteria() -> None:
    step_header(2, "Define criteria characteristics",
                "Tell DecisionLab which direction is better for each criterion.")
    info, examples = st.columns([3, 2], gap="large")
    with info:
        st.markdown("- **Benefit** ▲: higher values are better (quality, performance, "
                    "reliability).\n- **Cost** ▼: lower values are better (price, delivery "
                    "time, energy use).")
    with examples:
        st.dataframe(pd.DataFrame({"Example": ["Price", "Performance", "Delivery time",
                                               "Quality"],
                                   "Type": ["Cost ▼", "Benefit ▲", "Cost ▼", "Benefit ▲"]}),
                     hide_index=True)

    edited = persistent_editor(
        "crits", lambda: ss.crits[S.CRIT_COLUMNS].copy(), hide_index=True,
        column_config={
            "Criterion": st.column_config.TextColumn("Criterion", required=True, width="medium"),
            "Type": st.column_config.SelectboxColumn(
                "Type", options=["Benefit", "Cost"], required=True, width="small",
                help="Benefit: higher is better. Cost: lower is better."),
            "Unit": st.column_config.TextColumn("Unit (optional)", width="small",
                                                help="e.g. €, days, %, score 1–10"),
            "Description": st.column_config.TextColumn("Description (optional)", width="large"),
        })
    ss.crits = edited[S.CRIT_COLUMNS]

    types = S.crit_types()
    try:
        validate_names(S.crit_names(), "criterion")
        validate_criteria_types(types, len(types))
        n_benefit = sum(t == "Benefit" for t in types)
        st.success(f"✔ {n_benefit} benefit and {len(types) - n_benefit} cost criteria.")
    except MCDMError as exc:
        st.error(f"✖ {exc}")
    nav_buttons("problem", "matrix")


def handle_upload() -> None:
    key = f"upload_{ss.editor_version.get('upload', 0)}"
    file = ss.get(key)
    if file is None:
        return
    try:
        parsed = S.import_csv(file.getvalue())
    except MCDMError as exc:
        ss.flash = ("error", f"Upload failed: {exc}")
        return
    message = (f"Loaded {parsed.matrix.shape[0]} alternatives × {parsed.matrix.shape[1]} "
               f"criteria from {file.name}.")
    if parsed.types is None:
        message += " No 'type' row found — criterion types were kept or set to Benefit."
    if parsed.invalid_cells:
        message += (f" {len(parsed.invalid_cells)} non-numeric/empty cell(s) were left blank "
                    "for you to fix: " + "; ".join(parsed.invalid_cells[:6]))
        ss.flash = ("warning", message)
    else:
        ss.flash = ("success", message)
    S.bump("upload")


def page_matrix() -> None:
    step_header(3, "Enter the decision matrix",
                "Rows are alternatives, columns are criteria. Type a number in every cell.")
    try:
        validate_names(S.alt_names(), "alternative")
        validate_names(S.crit_names(), "criterion")
    except MCDMError as exc:
        st.error(f"✖ {exc} Fix the names in Step 1 before entering values.")
        nav_buttons("criteria", None)
        return

    t1, t2, t3, t4 = st.columns(4)
    t1.button("➕ Add alternative", on_click=S.add_alternative, width="stretch")
    t2.button("➕ Add criterion", on_click=S.add_criterion, width="stretch")
    with t3.popover("➖ Remove…", width="stretch"):
        alt_ids = list(ss.alts.index)
        rm_alt = st.selectbox("Alternative", alt_ids, format_func=lambda i: ss.alts.at[i, "Alternative"],
                              key="rm_alt")
        st.button("Remove alternative", on_click=S.remove_alternative, args=(rm_alt,),
                  disabled=len(alt_ids) <= 1, width="stretch")
        crit_ids = list(ss.crits.index)
        rm_crit = st.selectbox("Criterion", crit_ids, format_func=S.crit_label, key="rm_crit")
        st.button("Remove criterion", on_click=S.remove_criterion, args=(rm_crit,),
                  disabled=len(crit_ids) <= 1, width="stretch")
    t4.button("🧹 Clear all values", on_click=S.clear_values, width="stretch",
              help="Empty every cell (names and types are kept).")

    names, types = S.crit_names(), S.crit_types()
    column_config = {}
    for name, kind, unit in zip(names, types, ss.crits["Unit"].fillna("")):
        arrow = "▲" if kind == "Benefit" else "▼"
        column_config[name] = st.column_config.NumberColumn(
            f"{name} {arrow}", help=f"{kind} criterion" + (f" · unit: {unit}" if unit else ""))
    label_hash = abs(hash((tuple(S.alt_names()), tuple(names), tuple(types))))
    edited = persistent_editor("matrix", S.named_matrix, suffix=f"_{label_hash}",
                               column_config=column_config)
    ss.decision_values.loc[ss.alts.index, ss.crits.index] = edited.to_numpy(dtype=float)
    st.caption("▲ benefit (higher is better) · ▼ cost (lower is better)")

    matrix = S.named_matrix()
    invalid = find_invalid_cells(matrix)
    v1, v2 = st.columns([1, 3])
    if v1.button("🔍 Validate matrix", width="stretch"):
        st.toast("Matrix is complete ✔" if not invalid else
                 f"{len(invalid)} cell(s) need a value", icon="🔍")
    with v2:
        if invalid:
            listed = ", ".join(f"{a} / {c}" for a, c, _ in invalid[:10])
            more = f" and {len(invalid) - 10} more" if len(invalid) > 10 else ""
            st.error(f"✖ {len(invalid)} missing value(s): {listed}{more}.")
        else:
            notes = [f"✔ Matrix complete: {matrix.shape[0]} × {matrix.shape[1]} values."]
            constant = constant_columns(matrix)
            if constant and matrix.shape[0] > 1:
                notes.append(f"Constant criteria (cannot discriminate): {', '.join(constant)}.")
            if (matrix < 0).any().any():
                notes.append("Negative values present — supported by all methods.")
            if (matrix == 0).any().any():
                notes.append("Zero values present — handled safely.")
            st.success("  \n".join(notes))

    with st.expander("📁 Import / export CSV"):
        st.markdown("**CSV format:** first column = alternative names, header row = criterion "
                    "names, optional second row starting with `type` containing `Benefit` or "
                    "`Cost` for each criterion. Comma or semicolon separators are accepted.")
        st.file_uploader("Upload a decision matrix (CSV)", type=["csv"],
                         key=f"upload_{ss.editor_version.get('upload', 0)}",
                         on_change=handle_upload,
                         help="Replaces the current alternatives, criteria and values.")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download empty template", width="stretch",
                           data=template_csv(S.alt_names(), names, types),
                           file_name="decisionlab_template.csv", mime="text/csv")
        d2.download_button("⬇️ Download current matrix", width="stretch",
                           data=matrix_to_csv(matrix, types),
                           file_name="decisionlab_matrix.csv", mime="text/csv")
    nav_buttons("criteria", "weighting")


# ------------------------------------------------------------------ step 4: weights

def ahp_label(v: int) -> str:
    if v == 1:
        return "1 · Equal"
    return f"◀ {v}" if v > 1 else f"{-v} ▶"


def render_ahp_inputs() -> None:
    ids = list(ss.crits.index)
    n = len(ids)
    if n > AHP_MAX_CRITERIA:
        st.error(f"AHP supports at most {AHP_MAX_CRITERIA} criteria (you have {n}). "
                 "Please choose BWM, Entropy or CRITIC.")
        return
    if n == 1:
        st.info("With a single criterion no comparisons are needed; it gets 100% weight.")
        return
    pairs = [(ids[a], ids[b]) for a in range(n) for b in range(a + 1, n)]
    top = st.columns([3, 1])
    top[0].markdown(f"Compare each pair of criteria (**{len(pairs)} comparisons**). Move the "
                    "slider **towards the more important criterion**; the further, the stronger.")
    top[1].button("↺ Reset all to Equal", width="stretch", on_click=lambda: [
        ss.__setitem__(S.ahp_key(i, j), 1) for i, j in pairs])
    with st.expander("Saaty's 1–9 scale"):
        st.markdown(SAATY_HELP)
    box = st.container(height=520) if len(pairs) > 10 else st.container()
    with box:
        for i, j in pairs:
            key = S.ahp_key(i, j)
            ss.setdefault(key, 1)
            left_name, right_name = S.crit_label(i), S.crit_label(j)
            c1, c2, c3 = st.columns([2, 5, 2], vertical_alignment="center")
            c1.markdown(f"<div style='text-align:right'><b>{left_name}</b></div>",
                        unsafe_allow_html=True)
            v = c2.select_slider(f"{left_name} vs {right_name}", options=S.AHP_OPTIONS,
                                 key=key, format_func=ahp_label, label_visibility="collapsed")
            c3.markdown(f"**{right_name}**")
            if v == 1:
                sentence = f"{left_name} is equally important as {right_name}."
            elif v > 1:
                sentence = f"{left_name} is {INTENSITY_WORDS[v]} {right_name} ({v})."
            else:
                sentence = f"{right_name} is {INTENSITY_WORDS[-v]} {left_name} ({-v})."
            c2.caption(sentence)


def render_bwm_inputs() -> None:
    ids = list(ss.crits.index)
    if len(ids) == 1:
        st.info("With a single criterion no comparisons are needed; it gets 100% weight.")
        return
    if ss.get("p_bwm_best") not in ids:
        ss.p_bwm_best = ids[0]
    c1, c2 = st.columns(2)
    best = c1.selectbox("🥇 Most important criterion (Best)", ids, key="p_bwm_best",
                        format_func=S.crit_label)
    others = [i for i in ids if i != best]
    if ss.get("p_bwm_worst") not in others:
        ss.p_bwm_worst = others[-1]
    worst = c2.selectbox("🥉 Least important criterion (Worst)", others, key="p_bwm_worst",
                         format_func=S.crit_label)
    best_name, worst_name = S.crit_label(best), S.crit_label(worst)
    scale = list(range(1, 10))
    st.caption("Scale: 1 = equally important … 9 = extremely more important.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"**Best-to-Others** — how much more important is **{best_name}** than…")
        st.caption(f"{best_name} vs itself = 1")
        for j in others:
            key = f"p_bwm_bo_{j}"
            ss.setdefault(key, 8 if j == worst else 2)
            st.select_slider(f"…{S.crit_label(j)}" + (" (worst)" if j == worst else ""),
                             options=scale, key=key)
    with right:
        st.markdown(f"**Others-to-Worst** — how much more important than **{worst_name}** is…")
        st.caption(f"{worst_name} vs itself = 1")
        a_bw = ss.get(f"p_bwm_bo_{worst}", 8)
        st.select_slider(f"{best_name} (best) — same as Best-to-Worst", options=scale,
                         value=a_bw, disabled=True, key=f"bwm_bw_display_{a_bw}",
                         help="This judgment is set once, in the Best-to-Others column.")
        for j in ids:
            if j in (best, worst):
                continue
            key = f"p_bwm_ow_{j}"
            ss.setdefault(key, 2)
            st.select_slider(f"{S.crit_label(j)}…", options=scale, key=key)


def render_weight_details(result: WeightingResult, compact: bool = False) -> None:
    """Method-specific diagnostics (consistency, solver status, intermediate tables)."""
    d = result.details
    if result.method == "AHP" and "CR" in d:
        cols = st.columns(4)
        cols[0].metric("λ max", f"{d['lambda_max']:.4f}")
        cols[1].metric("CI", f"{d['CI']:.4f}")
        cols[2].metric("RI", f"{d['RI']:.2f}")
        cols[3].metric("CR", f"{d['CR']:.4f}",
                       delta="consistent" if d["consistent"] else "inconsistent",
                       delta_color="normal" if d["consistent"] else "inverse",
                       help=f"Consistency ratio; acceptable when ≤ {CR_THRESHOLD:.2f}.")
        with st.expander("Pairwise comparison matrix", expanded=not compact):
            st.dataframe(d["pairwise_matrix"].map(as_fraction))
    elif result.method == "BWM":
        cols = st.columns(3)
        cols[0].metric("Optimal ξ*", f"{d['xi']:.4f}", help="Maximum deviation from your "
                       "judgments at the optimum; 0 = perfectly consistent.")
        cols[1].metric("Consistency ratio", f"{d['consistency_ratio']:.4f}",
                       help="ξ* divided by the consistency index for a_BW; close to 0 is best.")
        cols[2].metric("Solver", "✔ converged" if d["solver_success"] else "⚠ check",
                       help=d["solver_status"])
        st.caption(f"Solver status: {d['solver_status']} · Best: **{d['best']}** · "
                   f"Worst: **{d['worst']}**")
        with st.expander("Comparison vectors", expanded=not compact):
            st.dataframe(fmt_frame(d["comparisons"], 0))
    elif result.method in ("Entropy", "CRITIC"):
        with st.expander(f"{result.method} calculation table", expanded=not compact):
            st.dataframe(fmt_frame(d["summary"]))
        if result.method == "CRITIC":
            with st.expander("Correlation matrix between criteria"):
                st.dataframe(fmt_frame(d["correlation_matrix"], 3))
                st.caption("Correlations close to 1 mean two criteria carry the same "
                           "information; CRITIC then lowers both weights.")


def page_weighting() -> None:
    step_header(4, "Select the criteria weighting method",
                "Decide how important each criterion is. Weights always sum to 100%.")
    ss.setdefault("p_weight_method", "AHP")
    method = st.selectbox("Weighting method", WEIGHTING_METHODS, key="p_weight_method",
                          format_func=lambda m: f"{WEIGHTING_INFO[m]['icon']} {m}",
                          help="Subjective methods use your judgments; objective methods "
                               "derive weights from the decision matrix.")
    info = WEIGHTING_INFO[method]
    st.info(f"**{method}** — *{info['kind']}*\n\n{info['summary']}\n\n**When to use:** "
            f"{info['when']}")

    if method == "AHP":
        render_ahp_inputs()
    elif method == "BWM":
        render_bwm_inputs()
    elif method in ("Entropy", "CRITIC"):
        st.markdown("No input is needed: the weights are computed from the decision matrix "
                    "you entered in Step 3 (at least two alternatives are required).")
        if method == "CRITIC":
            st.markdown("**Interpretation:** a criterion scores high when alternatives differ "
                        "a lot on it *and* it is not simply repeating what another criterion "
                        "already says.")

    st.subheader("Weights preview")
    result, error = cached_weights()
    if error:
        st.error(f"✖ {error}")
    elif result is not None:
        for message in result.warnings:
            st.warning(message)
        left, right = st.columns([2, 3], gap="large")
        with left:
            table = X.weights_table(result)
            st.dataframe(table, hide_index=True, column_config={
                "Weight": st.column_config.NumberColumn(format="%.4f"),
                "Weight (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                              max_value=100, width="medium")})
            st.caption(f"Sum of weights = {result.weights.sum():.6f}")
        with right:
            st.plotly_chart(V.weights_bar(result.weights), config=PLOTLY_CONFIG)
        render_weight_details(result, compact=True)
    nav_buttons("matrix", "ranking")


# ------------------------------------------------------------------ step 5: ranking

def page_ranking() -> None:
    step_header(5, "Select the ranking method",
                "Choose how criteria weights and performance values are combined into a score.")
    ss.setdefault("p_rank_method", "TOPSIS")
    ss.setdefault("p_baseline", True)
    st.radio("Ranking method", RANKING_METHODS, key="p_rank_method", horizontal=True,
             format_func=lambda m: f"{RANKING_INFO[m]['icon']} {m}")
    cols = st.columns(2, gap="large")
    for col, method in zip(cols, RANKING_METHODS):
        info = RANKING_INFO[method]
        with col.container(border=True):
            selected = " ✅" if method == ss.p_rank_method else ""
            st.markdown(f"#### {info['icon']} {method}{selected}")
            st.markdown(info["summary"])
            st.caption(f"**When to use:** {info['when']}")
    st.checkbox("Also compute an **equal-weights baseline** for comparison", key="p_baseline",
                help="Shows how the ranking would change if every criterion mattered equally "
                     "— a quick sensitivity check.")
    st.success(f"Selected combination: **{ss.get('p_weight_method')} + {ss.p_rank_method}**")
    nav_buttons("weighting", "results", "Go to results")


# ------------------------------------------------------------------ step 6: results

def page_results() -> None:
    step_header(6, "Calculate and explore the results",
                "Validate all inputs, compute the weights and rank the alternatives.")
    cols = st.columns(4)
    cols[0].metric("Alternatives", len(ss.alts))
    cols[1].metric("Criteria", len(ss.crits))
    cols[2].metric("Weighting", ss.get("p_weight_method", "—"))
    cols[3].metric("Ranking", ss.get("p_rank_method", "—"))

    if st.button("🧮 Calculate Ranking", type="primary", width="stretch"):
        with st.spinner("Calculating…"):
            try:
                ss.results = run_analysis()
                st.toast("Ranking calculated ✔", icon="✅")
            except MCDMError as exc:
                ss.results = None
                st.error(f"✖ Cannot calculate: {exc}")
    results = ss.get("results")
    if results is None:
        st.info("Click **Calculate Ranking** to compute the results.")
        nav_buttons("ranking", None)
        return
    if results["signature"] != S.signature():
        st.warning("⚠ Inputs changed since the last calculation. The results below are "
                   "out of date; click **Calculate Ranking** to update them.")
    render_results(results)
    nav_buttons("ranking", None)


def render_results(res: dict[str, Any]) -> None:
    weighting: WeightingResult = res["weighting"]
    ranking: RankingResult = res["ranking"]
    table = ranking.ranking
    winner = table.iloc[0]
    tied = table[table["Rank"] == 1]["Alternative"].tolist()

    for message in weighting.warnings + ranking.warnings:
        st.warning(message)

    label = "Recommended alternative" if len(tied) == 1 else "Tied for first place"
    st.markdown(f"""
<div class="dl-winner"><div class="label">🏆 {label}</div>
<div class="name">{' · '.join(map(str, tied))}</div>
<div>Score <b>{winner['Score']:.4f}</b> with <b>{weighting.method} + {ranking.method}</b>
for “{res['title']}”</div></div>""", unsafe_allow_html=True)

    m = st.columns(3)
    if len(table) > 1:
        margin = winner["Score"] - table.iloc[1]["Score"]
        m[0].metric("Lead over runner-up", f"{margin:.4f}",
                    help=f"Score difference to {table.iloc[1]['Alternative']}.")
    top_criterion = weighting.weights.idxmax()
    m[1].metric("Most important criterion", str(top_criterion),
                help=f"Receives {weighting.weights.max():.1%} of the total weight.")
    m[1].caption(f"{weighting.weights.max():.1%} of the total weight")
    m[2].metric("Score range", f"{table['Score'].min():.3f} – {table['Score'].max():.3f}")

    tabs = st.tabs(["🏆 Ranking", "⚖️ Weights", "📊 Charts", "🧮 Calculations",
                    "💡 Explanation", "⬇️ Export"])
    with tabs[0]:
        render_ranking_tab(res)
    with tabs[1]:
        render_weights_tab(res)
    with tabs[2]:
        render_charts_tab(res)
    with tabs[3]:
        render_calculations_tab(res)
    with tabs[4]:
        render_explanation_tab(res)
    with tabs[5]:
        render_export_tab(res)


def render_ranking_tab(res: dict[str, Any]) -> None:
    ranking: RankingResult = res["ranking"]
    table = ranking.ranking.copy()
    m = len(table)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    table["Position"] = [f"{medals.get(r, '')} {ordinal(r)} of {m}".strip() for r in table["Rank"]]
    table["Indicator"] = ["🏆 Best choice" if r == 1 else "" for r in table["Rank"]]
    score_name = "Closeness CC" if ranking.method == "TOPSIS" else "Weighted score"
    st.dataframe(table[["Rank", "Position", "Alternative", "Score", "Indicator"]],
                 hide_index=True, column_config={
                     "Rank": st.column_config.NumberColumn(width="small"),
                     "Alternative": st.column_config.TextColumn(width="medium"),
                     "Score": st.column_config.ProgressColumn(
                         score_name, format="%.4f", min_value=0.0, max_value=1.0,
                         width="large", help="0 = worst possible, 1 = best possible")})
    st.plotly_chart(V.scores_bar(table, score_name), config=PLOTLY_CONFIG)

    baseline = res.get("baseline")
    if baseline is not None:
        st.subheader("Comparison with equal weights")
        st.caption("A quick sensitivity check: how the ranking changes if every criterion "
                   "matters equally. Positive rank change = the alternative does better with "
                   "your weights.")
        st.dataframe(fmt_frame(baseline), hide_index=True)
        ranks = baseline.set_index("Alternative")[[c for c in baseline.columns
                                                   if c.startswith("Rank (")]]
        st.plotly_chart(V.comparison_chart(ranks), config=PLOTLY_CONFIG)
        if (baseline["Rank change"] == 0).all():
            st.success("The ranking is identical under equal weights — the result is robust "
                       "to the choice of weights.")


def render_weights_tab(res: dict[str, Any]) -> None:
    weighting: WeightingResult = res["weighting"]
    table = X.weights_table(weighting)
    table.insert(1, "Type", [t.capitalize() for t in res["types"]])
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.dataframe(table, hide_index=True, column_config={
            "Weight": st.column_config.NumberColumn(format="%.4f"),
            "Weight (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                          max_value=100, width="medium")})
        st.caption(f"Method: **{weighting.method}** · sum of weights = "
                   f"{weighting.weights.sum():.6f}")
    with right:
        st.plotly_chart(V.weights_bar(weighting.weights), config=PLOTLY_CONFIG)
    render_weight_details(weighting)


def render_charts_tab(res: dict[str, Any]) -> None:
    ranking: RankingResult = res["ranking"]
    order = ranking.ranking["Alternative"].tolist()
    st.plotly_chart(V.matrix_heatmap(res["matrix"], res["types"],
                                     "Decision matrix (colour = relative performance)"),
                    config=PLOTLY_CONFIG)
    if ranking.method == "SWM":
        st.plotly_chart(V.contributions_chart(ranking.details["weighted_matrix"], order),
                        config=PLOTLY_CONFIG)
    else:
        st.plotly_chart(V.distance_chart(ranking.details["distances"], order),
                        config=PLOTLY_CONFIG)
        st.caption("A good alternative has a short distance to the ideal best (D+) and a long "
                   "distance to the ideal worst (D−).")
    if res["matrix"].shape[1] >= 3:
        st.markdown("**Profile comparison (radar)**")
        chosen = st.multiselect("Alternatives to compare (up to 8)", order,
                                default=order[:min(3, len(order))], max_selections=8,
                                key="radar_choice")
        normalised = min_max_view(res)
        if chosen:
            st.plotly_chart(V.radar_chart(normalised, chosen), config=PLOTLY_CONFIG)
            st.caption("Each axis is rescaled so that 1 = best alternative on that criterion "
                       "and 0 = worst (cost criteria are inverted).")
    else:
        st.caption("The radar chart needs at least three criteria.")


def min_max_view(res: dict[str, Any]) -> pd.DataFrame:
    from core.normalization import min_max_normalize
    return min_max_normalize(res["matrix"], res["types"])


def render_calculations_tab(res: dict[str, Any]) -> None:
    weighting: WeightingResult = res["weighting"]
    ranking: RankingResult = res["ranking"]
    with st.expander(f"📐 Formulas — {weighting.method}", expanded=True):
        for title, formula in FORMULAS[weighting.method]:
            st.markdown(f"**{title}**")
            st.latex(formula)
    with st.expander(f"📐 Formulas — {ranking.method}", expanded=True):
        for title, formula in FORMULAS[ranking.method]:
            st.markdown(f"**{title}**")
            st.latex(formula)
    with st.expander("1️⃣ Original decision matrix"):
        types_row = ["benefit ▲" if t == "benefit" else "cost ▼" for t in res["types"]]
        st.caption("Criterion types: " + ", ".join(f"{c} ({t})" for c, t in
                                                    zip(res["matrix"].columns, types_row)))
        st.dataframe(fmt_frame(res["matrix"]))
    for key, value in weighting.details.items():
        if isinstance(value, pd.DataFrame) and key not in ("summary",):
            with st.expander(f"2️⃣ {weighting.method}: {key.replace('_', ' ')}"):
                shown = value.map(as_fraction) if key == "pairwise_matrix" else fmt_frame(value)
                st.dataframe(shown)
    if "summary" in weighting.details:
        with st.expander(f"2️⃣ {weighting.method}: calculation summary"):
            st.dataframe(fmt_frame(weighting.details["summary"]))
    labels = {"normalised_matrix": "Normalised matrix", "weighted_matrix":
              "Weighted normalised matrix" if ranking.method == "TOPSIS" else
              "Weighted contributions (w_j × r_ij)",
              "ideal_solutions": "Ideal best and ideal worst vectors",
              "distances": "Distances and closeness coefficients"}
    for key, label in labels.items():
        if key in ranking.details:
            with st.expander(f"3️⃣ {ranking.method}: {label}"):
                st.dataframe(fmt_frame(ranking.details[key]))
    with st.expander("4️⃣ Final scores"):
        st.dataframe(fmt_frame(ranking.ranking), hide_index=True)
    with st.expander("📋 Conventions used by DecisionLab"):
        st.markdown(CONVENTIONS)


def render_explanation_tab(res: dict[str, Any]) -> None:
    ranking: RankingResult = res["ranking"]
    weighting: WeightingResult = res["weighting"]
    explanation: pd.DataFrame = res["explanation"]
    first = explanation.iloc[0]
    top_weights = weighting.weights.sort_values(ascending=False)
    st.markdown(
        f"### Why **{first['Alternative']}** ranks first\n"
        f"The criteria that matter most are **{top_weights.index[0]}** "
        f"({top_weights.iloc[0]:.1%})"
        + (f" and **{top_weights.index[1]}** ({top_weights.iloc[1]:.1%})"
           if len(top_weights) > 1 else "")
        + f". {first['Alternative']} performs well on: {first['Main strengths']}"
        + (f"; its weak points are: {first['Main weaknesses']}."
           if first["Main weaknesses"] != "—" else " and has no major weak point.")
        + f"\n\n{first['Explanation']}")
    st.caption("Percentages show how much of the achievable range the alternative reaches on "
               "each criterion (100% = best among the alternatives, 0% = worst). Strengths "
               "and weaknesses are ordered by their weighted impact on the score.")
    st.dataframe(fmt_frame(explanation), hide_index=True, column_config={
        "Explanation": st.column_config.TextColumn(width="large")})
    if ranking.method == "TOPSIS":
        st.info("**Reading TOPSIS:** the closeness coefficient is the share of the "
                "'ideal-worst → ideal-best' distance covered by the alternative. 1 would mean "
                "it is the best on every criterion at once.")
    else:
        st.info("**Reading SWM:** the score is the weighted average of the 0–1 normalised "
                "values. It equals 1 only if the alternative is the best on every criterion.")


def render_export_tab(res: dict[str, Any]) -> None:
    report = X.ReportData(title=res["title"], criteria=res["criteria"], matrix=res["matrix"],
                          weighting=res["weighting"], ranking=res["ranking"],
                          explanation=res["explanation"], baseline=res.get("baseline"))
    slug = "".join(ch if ch.isalnum() else "_" for ch in res["title"]).strip("_")[:40] or \
        "decision"
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Rankings (CSV)", X.ranking_csv(res["ranking"]),
                       file_name=f"{slug}_ranking.csv", mime="text/csv", width="stretch")
    c2.download_button("⬇️ Criteria weights (CSV)", X.weights_csv(res["weighting"]),
                       file_name=f"{slug}_weights.csv", mime="text/csv", width="stretch")
    c3, c4 = st.columns(2)
    try:
        c3.download_button("⬇️ Complete report (Excel)", X.excel_report(report),
                           file_name=f"{slug}_report.xlsx", width="stretch",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet")
    except ImportError:
        c3.warning("Install `openpyxl` to enable the Excel export.")
    if X.pdf_available():
        c4.download_button("⬇️ Summary report (PDF)", X.pdf_report(report),
                           file_name=f"{slug}_summary.pdf", mime="application/pdf",
                           width="stretch")
    else:
        c4.warning("Install `fpdf2` to enable the PDF export.")
    st.caption("The Excel workbook contains the summary, ranking, weights, criteria, "
               "decision matrix, explanations, baseline comparison and every intermediate "
               "matrix. The PDF is a concise printable summary.")


# ============================================================================= main

PAGE_RENDERERS: dict[str, Callable[[], None]] = {
    "home": page_home, "problem": page_problem, "criteria": page_criteria,
    "matrix": page_matrix, "weighting": page_weighting, "ranking": page_ranking,
    "results": page_results,
}

show_flash()
PAGE_RENDERERS[ss.get("p_nav", "home")]()
# Rendered last so the progress checklist reflects edits made during this run.
render_sidebar()
