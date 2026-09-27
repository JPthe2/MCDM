"""Plotly figures for the results dashboard.

Colour roles follow a validated, colour-blind-safe palette:
single-series magnitude → one blue; identity → fixed categorical order (never
cycled, capped at 8 with an "Other" fold); heatmap → one-hue blue ramp.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.normalization import min_max_normalize

PRIMARY = "#2a78d6"
PRIMARY_STRONG = "#184f95"
MUTED = "#9ec5f4"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#a3a29c"
SEQUENTIAL = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
              [0.75, "#1c5cab"], [1.0, "#0d366b"]]
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=10, r=20, t=50, b=10), font=FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12), bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def weights_bar(weights: pd.Series, title: str = "Criteria importance") -> go.Figure:
    """Horizontal bar chart of weights, most important at the top."""
    ordered = weights.sort_values()
    fig = go.Figure(go.Bar(
        x=ordered.values, y=[str(i) for i in ordered.index], orientation="h",
        marker=dict(color=PRIMARY, cornerradius=4),
        text=[f"{v:.1%}" for v in ordered.values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
    ))
    _layout(fig, title, height=max(260, 44 * len(weights) + 90))
    fig.update_xaxes(tickformat=".0%", range=[0, max(ordered.max() * 1.18, 0.05)],
                     title="Weight", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=False)
    return fig


def scores_bar(ranking: pd.DataFrame, score_label: str = "Score",
               title: str = "Alternative scores") -> go.Figure:
    """Bar chart of scores in rank order; the top-ranked alternative is emphasised."""
    top = ranking["Rank"] == ranking["Rank"].min()
    colors = [PRIMARY_STRONG if t else PRIMARY for t in top]
    labels = [f"🏆 {a}" if t else str(a) for a, t in zip(ranking["Alternative"], top)]
    fig = go.Figure(go.Bar(
        x=labels, y=ranking["Score"], marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.3f}" for v in ranking["Score"]], textposition="outside", cliponaxis=False,
        customdata=ranking["Rank"],
        hovertemplate="<b>%{x}</b><br>" + score_label + ": %{y:.4f}<br>Rank: %{customdata}"
                      "<extra></extra>",
    ))
    _layout(fig, title)
    fig.update_yaxes(title=score_label, range=[0, max(ranking["Score"].max() * 1.15, 0.05)])
    return fig


def matrix_heatmap(matrix: pd.DataFrame, types: Sequence[str],
                   title: str = "Decision matrix") -> go.Figure:
    """Heatmap of the raw matrix, coloured by relative performance per criterion.

    Colour = min-max score with direction applied (darker = better), so criteria with
    different units are comparable; the cell text shows the raw value.
    """
    performance = min_max_normalize(matrix, types)
    headers = [f"{c} ({'↑' if t == 'benefit' else '↓'})" for c, t in zip(matrix.columns, types)]
    text = matrix.map(lambda v: f"{v:,.4g}").to_numpy()
    fig = go.Figure(go.Heatmap(
        z=performance.to_numpy(), x=headers, y=[str(i) for i in matrix.index],
        text=text, texttemplate="%{text}", colorscale=SEQUENTIAL, zmin=0, zmax=1,
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="Relative<br>performance", side="top"),
                      tickvals=[0, 1], ticktext=["worst", "best"], thickness=12),
        customdata=matrix.to_numpy(),
        hovertemplate="<b>%{y}</b> · %{x}<br>Value: %{customdata:,.4g}"
                      "<br>Relative performance: %{z:.2f}<extra></extra>",
    ))
    _layout(fig, title, height=max(300, 44 * matrix.shape[0] + 150))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def radar_chart(normalised: pd.DataFrame, alternatives: Sequence[str],
                title: str = "Profile comparison") -> go.Figure:
    """Radar chart of normalised (0–1, direction-aware) values, up to 8 alternatives."""
    categories = [str(c) for c in normalised.columns]
    fig = go.Figure()
    for i, alt in enumerate(list(alternatives)[:len(CATEGORICAL)]):
        values = normalised.loc[alt].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=categories + categories[:1], name=str(alt),
            line=dict(color=CATEGORICAL[i], width=2), marker=dict(size=8),
            fill="toself", opacity=0.85, fillcolor=_rgba(CATEGORICAL[i], 0.08),
            hovertemplate="<b>" + str(alt) + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
        ))
    _layout(fig, title, height=460)
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], tickvals=[0, 0.5, 1],
                                                 gridcolor="rgba(128,128,128,0.25)"),
                                 angularaxis=dict(gridcolor="rgba(128,128,128,0.25)")),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0))
    return fig


def contributions_chart(weighted: pd.DataFrame, order: Sequence[str],
                        title: str = "Score contributions by criterion") -> go.Figure:
    """Stacked horizontal bars: how much each criterion adds to each alternative's score.

    More than 8 criteria are folded: the 7 largest total contributors keep their own
    colour and the rest are grouped as "Other".
    """
    data = weighted.loc[list(order)]
    folded = data.shape[1] > len(CATEGORICAL)
    if folded:
        keep = data.sum().sort_values(ascending=False).index[:len(CATEGORICAL) - 1]
        other = data.drop(columns=keep).sum(axis=1)
        data = data[keep].copy()
        data["Other criteria"] = other
    fig = go.Figure()
    for i, criterion in enumerate(data.columns):
        color = OTHER_COLOR if folded and i == len(data.columns) - 1 else CATEGORICAL[i]
        fig.add_trace(go.Bar(
            y=[str(a) for a in data.index], x=data[criterion], name=str(criterion),
            orientation="h", marker=dict(color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            hovertemplate="<b>%{y}</b><br>" + str(criterion) + ": %{x:.4f}<extra></extra>",
        ))
    _layout(fig, title, height=max(300, 42 * len(data) + 140))
    fig.update_layout(barmode="stack",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Weighted normalised score", showgrid=True,
                     gridcolor="rgba(128,128,128,0.18)")
    return fig


def distance_chart(distances: pd.DataFrame, order: Sequence[str],
                   title: str = "TOPSIS distances to the ideal solutions") -> go.Figure:
    """Grouped bars of D+ (smaller is better) and D− (larger is better)."""
    data = distances.loc[list(order)]
    fig = go.Figure()
    for i, column in enumerate(data.columns[:2]):
        fig.add_trace(go.Bar(
            x=[str(a) for a in data.index], y=data[column], name=str(column),
            marker=dict(color=CATEGORICAL[i], cornerradius=4),
            hovertemplate="<b>%{x}</b><br>" + str(column) + ": %{y:.4f}<extra></extra>",
        ))
    _layout(fig, title)
    fig.update_layout(barmode="group", bargroupgap=0.08,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
    fig.update_yaxes(title="Euclidean distance")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def comparison_chart(ranks: pd.DataFrame, title: str = "Rank comparison") -> go.Figure:
    """Dot plot of the rank of each alternative under several weighting schemes."""
    fig = go.Figure()
    for i, column in enumerate(ranks.columns[:len(CATEGORICAL)]):
        fig.add_trace(go.Scatter(
            x=ranks[column], y=[str(a) for a in ranks.index], mode="markers", name=str(column),
            marker=dict(size=13, color=CATEGORICAL[i], line=dict(width=2, color="white")),
            hovertemplate="<b>%{y}</b><br>" + str(column) + ": rank %{x}<extra></extra>",
        ))
    _layout(fig, title, height=max(260, 40 * len(ranks) + 120))
    max_rank = int(np.nanmax(ranks.to_numpy())) if ranks.size else 1
    fig.update_xaxes(title="Rank (1 = best)", dtick=1, range=[0.5, max_rank + 0.5],
                     showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    return fig
