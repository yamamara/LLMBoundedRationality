from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go

from sandbox.statistics import PRIMARY_METRICS


def empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white")
    return figure


def build_figures(results: dict[str, Any]) -> list[go.Figure]:
    decisions = results.get("decision_rows", [])
    rounds = results.get("round_rows", [])
    agents = results.get("agent_aggregates", [])
    runs = results.get("run_aggregates", [])
    if not decisions:
        return [empty_figure(title) for title in (
            "Bid Distribution", "Bid-to-Value Ratios", "Utilities", "Win Rates",
            "Revenue", "Allocative Efficiency", "Changes Across Rounds and Runs",
        )]

    bid_distribution = px.histogram(
        decisions, x="bid", color="player_id", marginal="box", title="Bid Distribution"
    )
    ratios = px.box(
        decisions, x="player_id", y="bid_to_value_ratio", color="provider",
        points="outliers", title="Bid-to-Value Ratios",
    )
    utilities = px.bar(
        agents, x="player_id", y="total_utility", color="run_id", barmode="group",
        title="Total Utilities by Agent and Run",
    )
    win_rates = px.bar(
        agents, x="player_id", y="win_rate", color="run_id", barmode="group",
        title="Win Rates",
    )
    revenue = px.line(
        rounds, x="round", y="revenue", color="run_id", markers=True,
        title="Revenue Across Rounds",
    )
    efficiency = px.line(
        rounds, x="round", y="allocative_efficiency", color="run_id", markers=True,
        title="Allocative Efficiency Across Rounds",
    )
    changes = px.scatter(
        decisions, x="round", y="cumulative_utility", color="player_id",
        facet_col="run_index" if len(runs) <= 6 else None,
        title="Cumulative Utility Changes Across Rounds and Runs",
    )
    for figure in (bid_distribution, ratios, utilities, win_rates, revenue, efficiency, changes):
        figure.update_layout(template="plotly_white", legend_title_text="")
    return [bid_distribution, ratios, utilities, win_rates, revenue, efficiency, changes]


TREATMENT_DIMENSIONS = {
    "mechanism": "Mechanism",
    "mechanism_description_treatment": "Description",
    "valuation_schedule": "Valuation schedule",
    "num_players": "Players",
}


def _stat_hover(row: dict[str, Any], definition: dict[str, str]) -> str:
    sd = "N/A" if row["sample_sd"] is None else f"{row['sample_sd']:.4g}"
    interval = (
        "N/A"
        if row["ci95_lower"] is None
        else f"[{row['ci95_lower']:.4g}, {row['ci95_upper']:.4g}]"
    )
    return (
        f"Mean: {row['mean']:.4g}<br>SD: {sd}<br>95% CI: {interval}<br>"
        f"Trials: {row['n']}<br>Preferred: {definition['preferred_direction']}<br>"
        f"{definition['definition']}"
    )


def build_primary_figure(
    results: dict[str, Any],
    metric: str,
    error_mode: str = "ci95",
    x_dimension: str = "mechanism",
    color_dimension: str = "mechanism_description_treatment",
    filters: dict[str, Any] | None = None,
) -> go.Figure:
    definition = results.get("metric_definitions", PRIMARY_METRICS).get(
        metric, PRIMARY_METRICS[metric]
    )
    rows = [
        row for row in results.get("treatment_aggregates", []) if row["metric"] == metric
    ]
    filters = filters or {}
    for dimension, selected in filters.items():
        if dimension not in {x_dimension, color_dimension} and selected is not None:
            rows = [row for row in rows if row.get(dimension) == selected]
    if not rows:
        return empty_figure(definition["label"])
    error_key = "two_sd" if error_mode == "two_sd" else "ci95_half_width"
    color_values = list(dict.fromkeys(row[color_dimension] for row in rows))
    figure = go.Figure()
    for color in color_values:
        selected = [row for row in rows if row[color_dimension] == color]
        hover = [_stat_hover(row, definition) for row in selected]
        figure.add_bar(
            name=str(color),
            x=[str(row[x_dimension]) for row in selected],
            y=[row["mean"] for row in selected],
            error_y={
                "type": "data",
                "array": [row[error_key] or 0 for row in selected],
                "visible": True,
            },
            hovertext=hover,
            hoverinfo="text",
        )
    figure.update_layout(
        title=definition["label"],
        template="plotly_white",
        barmode="group",
        xaxis_title=TREATMENT_DIMENSIONS[x_dimension],
        yaxis_title=definition["label"],
        legend_title_text=TREATMENT_DIMENSIONS[color_dimension],
        annotations=[
            {
                "text": f"{definition['definition']} Preferred: {definition['preferred_direction']}.",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.12,
                "showarrow": False,
                "align": "left",
            }
        ],
    )
    return figure


def build_research_figures(
    results: dict[str, Any],
    error_mode: str = "ci95",
    x_dimension: str = "mechanism",
    color_dimension: str = "mechanism_description_treatment",
    filters: dict[str, Any] | None = None,
) -> list[go.Figure]:
    primary = [
        build_primary_figure(
            results, metric, error_mode, x_dimension, color_dimension, filters
        )
        for metric in PRIMARY_METRICS
    ]
    return primary + build_figures(results)
