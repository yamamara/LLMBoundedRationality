from __future__ import annotations

import json
from typing import Any

from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from frontend.api_client import AuctionApiClient
from frontend.cournot_results import (
    cournot_experiment,
    register_cournot_callbacks,
)
from frontend.figures import TREATMENT_DIMENSIONS, build_research_figures
from sandbox.prompts import DEFAULT_AGENT_PROMPT, DEFAULT_SYSTEM_PROMPT


def _number(label, component_id, value, minimum=None, maximum=None, step=1):
    return html.Label(
        [label, dcc.Input(id=component_id, type="number", value=value, min=minimum, max=maximum, step=step)],
        className="control-field",
    )


def _agent_card(index: int):
    prefix = f"agent-{index}"
    return html.Div(
        [
            html.H4(f"Agent P{index}"),
            dcc.Dropdown(id=f"{prefix}-profile", placeholder="Provider profile"),
            dcc.Input(id=f"{prefix}-model", placeholder="Model (blank uses profile default)"),
            _number("Temperature", f"{prefix}-temperature", 0.0, 0, 2, 0.1),
            _number("Max output tokens", f"{prefix}-tokens", 800, 32, 32768),
            _number("Timeout seconds", f"{prefix}-timeout", 60, 1, 600),
            _number("Memory rounds", f"{prefix}-memory", 3, 0, 1000),
            _number("Retries", f"{prefix}-retries", 2, 1, 10),
            html.Label(
                [
                    "Reasoning effort",
                    dcc.Dropdown(
                        id=f"{prefix}-reasoning",
                        options=[{"label": value.title(), "value": value} for value in ("none", "low", "medium", "high")],
                        value="none",
                        clearable=False,
                    ),
                ],
                className="control-field",
            ),
            html.Label(
                [
                    "Provider options",
                    dcc.Textarea(
                        id=f"{prefix}-options",
                        value="",
                        placeholder='Optional JSON, e.g. {"num_ctx": 4096}',
                    ),
                ],
                className="control-field",
            ),
        ],
        id=f"agent-card-{index}",
        className="agent-card",
        style={} if index <= 4 else {"display": "none"},
    )


def create_dash_app(api_client: AuctionApiClient | None = None) -> Dash:
    client = api_client or AuctionApiClient()
    app = Dash(__name__, suppress_callback_exceptions=True, title="Bounded Rationality Lab")
    app.layout = html.Div(
        [
            dcc.Location(id="location"),
            dcc.Store(id="simulation-id"),
            dcc.Store(id="results-store"),
            dcc.Store(id="provider-store"),
            dcc.Store(id="prompt-defaults", data={"system_template": DEFAULT_SYSTEM_PROMPT, "agent_template": DEFAULT_AGENT_PROMPT}),
            dcc.Interval(id="progress-poll", interval=1000, disabled=True),
            html.Header(
                [
                    html.H1("Bounded Rationality Research Lab"),
                    html.P("Auction experiments and Cournot market simulations."),
                ]
            ),
            html.Section(
                [
                    dcc.RadioItems(
                        id="experiment-toggle",
                        options=[
                            {"label": "Auction", "value": "auction"},
                            {"label": "Cournot", "value": "cournot"},
                        ],
                        value="auction",
                        inline=True,
                        className="experiment-toggle",
                    )
                ],
                className="experiment-switcher",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Button("Hyperparameters", id="show-hyperparameters", n_clicks=0, className="section-toggle active"),
                            html.Button("Agent Assignment", id="show-agents", n_clicks=0, className="section-toggle"),
                        ],
                        className="section-toggles",
                    ),
                    html.Div(
                        [
                            _number("Simulations (n)", "simulation-count", 10, 1, 100),
                            _number("Rounds per run", "rounds", 1, 1, 500),
                            _number("Starting budget", "starting-budget", 100, 0),
                            _number("Minimum valuation", "value-min", 1, 0),
                            _number("Maximum valuation", "value-max", 100, 0),
                            html.Label(["Mechanism", dcc.Dropdown(id="mechanism", options=[{"label": "First price", "value": "first_price"}, {"label": "Second price", "value": "second_price"}], value="first_price", clearable=False)], className="control-field"),
                            html.Label(["Players", dcc.Dropdown(id="player-count", options=[{"label": str(value), "value": value} for value in (2, 4, 6, 8)], value=4, clearable=False)], className="control-field"),
                            html.Label(["Mechanism description", dcc.Dropdown(id="description-treatment", options=[{"label": value.replace("_", " ").title(), "value": value} for value in ("name_only", "concise", "full")], value="concise", clearable=False)], className="control-field"),
                            _number("Base seed", "seed", 42),
                            _number("Tie seed", "tie-seed", 42),
                            html.Label(["Institution", dcc.Input(id="institution", value="baseline")], className="control-field"),
                            html.Label(
                                ["Tie rule", dcc.Dropdown(
                                    id="tie-rule",
                                    options=[
                                        {"label": "Seeded random", "value": "seeded_random"},
                                        {"label": "Player priority", "value": "player_priority"},
                                    ],
                                    value="seeded_random", clearable=False,
                                )], className="control-field",
                            ),
                            html.Label(
                                ["Tie priority", dcc.Input(id="tie-priority", value="P1,P2,P3,P4")],
                                className="control-field",
                            ),
                            html.Div(
                                [
                                    html.H3("Editable prompts"),
                                    html.Label(["System prompt", dcc.Textarea(id="system-prompt", value=DEFAULT_SYSTEM_PROMPT)], className="control-field prompt-field"),
                                    html.Label(["System placeholder", dcc.Dropdown(id="system-placeholder")], className="control-field"),
                                    html.Button("Insert system placeholder", id="insert-system-placeholder", n_clicks=0),
                                    html.Label(["Agent prompt", dcc.Textarea(id="agent-prompt", value=DEFAULT_AGENT_PROMPT)], className="control-field prompt-field"),
                                    html.Label(["Agent placeholder", dcc.Dropdown(id="agent-placeholder")], className="control-field"),
                                    html.Button("Insert agent placeholder", id="insert-agent-placeholder", n_clicks=0),
                                    html.Button("Reset prompts", id="reset-prompts", n_clicks=0),
                                    html.Div(id="prompt-validation", className="status-message"),
                                    html.Details([html.Summary("Prompt preview"), html.H4("System"), html.Pre(id="system-prompt-preview"), html.H4("Agent"), html.Pre(id="agent-prompt-preview")]),
                                ],
                                className="prompt-editor",
                            ),
                            html.Div(
                                [
                                    html.H3("Sensitivity matrix"),
                                    dcc.Checklist(id="matrix-enabled", options=[{"label": "Run Cartesian treatment matrix", "value": "enabled"}], value=[]),
                                    html.Label(["Mechanisms", dcc.Dropdown(id="matrix-mechanisms", options=[{"label": "First price", "value": "first_price"}, {"label": "Second price", "value": "second_price"}], value=["first_price"], multi=True)], className="control-field"),
                                    html.Label(["Description treatments", dcc.Dropdown(id="matrix-descriptions", options=[{"label": value.replace("_", " ").title(), "value": value} for value in ("name_only", "concise", "full")], value=["concise"], multi=True)], className="control-field"),
                                    html.Label(["Player counts", dcc.Dropdown(id="matrix-player-counts", options=[{"label": str(value), "value": value} for value in (2, 4, 6, 8)], value=[4], multi=True)], className="control-field"),
                                    html.Label(["Valuation schedules", dcc.Textarea(id="valuation-schedules", value="", placeholder='Optional JSON list, e.g. [{"name":"high-P1","player_ranges":{"P1":{"minimum":80,"maximum":100}},"rotate_across_positions":true}]')], className="control-field"),
                                    html.Div(id="matrix-summary"),
                                ],
                                className="matrix-editor",
                            ),
                        ],
                        id="hyperparameters-panel",
                        className="control-grid tab-content",
                    ),
                    html.Div(
                        [html.Div(id="provider-status", className="provider-status"), *[_agent_card(index) for index in range(1, 9)]],
                        id="agent-assignment-panel",
                        className="agent-grid tab-content",
                        style={"display": "none"},
                    ),
                    html.Button("Run Simulation", id="run-button", n_clicks=0, className="run-button"),
                    html.Div(id="status-message", className="status-message"),
                    html.Progress(id="progress-bar", value=0, max=100),
                ],
                id="auction-configuration",
                className="panel",
            ),
            html.Section(
                [
                    html.H2("Results"),
                    html.Div(id="summary-cards", className="summary-grid"),
                    dash_table.DataTable(
                        id="results-table", data=[],
                        columns=[{"name": "", "id": "_placeholder"}],
                        sort_action="native", filter_action="native",
                        page_action="native", page_size=25, fixed_rows={"headers": True},
                        row_selectable="single",
                        style_table={"overflowX": "auto", "maxHeight": "520px"},
                    ),
                    html.Div(id="prompt-detail"),
                    html.Div(
                        [
                            html.Div([
                                html.Label(["Error bars", dcc.RadioItems(id="error-mode", options=[{"label": "Mean ± 2 SD", "value": "two_sd"}, {"label": "95% t confidence interval", "value": "ci95"}], value="ci95", inline=True)]),
                                html.Label(["X-axis", dcc.Dropdown(id="plot-x", options=[{"label": label, "value": value} for value, label in TREATMENT_DIMENSIONS.items()], value="mechanism", clearable=False)]),
                                html.Label(["Color", dcc.Dropdown(id="plot-color", options=[{"label": label, "value": value} for value, label in TREATMENT_DIMENSIONS.items()], value="mechanism_description_treatment", clearable=False)]),
                                *[html.Label([label, dcc.Dropdown(id=f"filter-{dimension}")], className="control-field") for dimension, label in TREATMENT_DIMENSIONS.items()],
                            ], className="plot-controls"),
                            html.Div(
                                [
                                    html.Button(label, id=f"graph-button-{index}", n_clicks=0, className="graph-button")
                                    for index, label in enumerate(
                                        ("Revenue", "Efficiency", "Payoff", "Truthfulness", "Bid Distribution", "Bid / Value Ratios", "Utilities", "Win Rates", "Revenue by Round", "Efficiency by Round", "Trends"),
                                        1,
                                    )
                                ],
                                className="graph-buttons",
                            ),
                            dcc.Graph(id="active-graph"),
                        ],
                        id="graphs-panel",
                        style={"display": "none"},
                    ),
                ],
                id="auction-results",
                className="panel",
            ),
            html.Section(
                [
                    html.H2("Round Inspector"),
                    dcc.Dropdown(id="inspect-run", placeholder="Select a run"),
                    dcc.Slider(id="round-slider", min=0, max=0, step=1, value=0, marks={0: "0"}),
                    html.Div(id="round-inspector", className="agent-grid"),
                ],
                id="auction-round-inspector",
                className="panel round-panel",
            ),
            cournot_experiment(),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("auction-configuration", "style"),
        Output("auction-results", "style"),
        Output("auction-round-inspector", "style"),
        Output("cournot-experiment", "style"),
        Input("experiment-toggle", "value"),
    )
    def switch_experiment(experiment):
        if experiment == "cournot":
            return {"display": "none"}, {"display": "none"}, {"display": "none"}, {}
        return {}, {}, {}, {"display": "none"}

    @app.callback(
        [Output(f"agent-{index}-profile", "options") for index in range(1, 9)]
        + [Output(f"agent-{index}-profile", "value") for index in range(1, 9)]
        + [Output("provider-store", "data"), Output("provider-status", "children")],
        Input("location", "pathname"),
    )
    def load_profiles(_pathname):
        try:
            profiles = client.providers()["profiles"]
            options = [
                {
                    "label": f"{profile['profile_id']} ({'ready' if profile['configured'] else 'not configured'})",
                    "value": profile["profile_id"],
                    "disabled": not profile["configured"],
                }
                for profile in profiles
            ]
            ready = next((profile["profile_id"] for profile in profiles if profile["configured"]), None)
            statuses = [
                html.Div(f"{profile['profile_id']}: {profile['configuration_status'].replace('_', ' ')}")
                for profile in profiles
            ]
            return [options] * 8 + [ready] * 8 + [profiles, statuses]
        except Exception as exc:
            return [[]] * 8 + [None] * 8 + [[], f"Provider status unavailable: {exc}"]

    @app.callback(
        Output("prompt-defaults", "data"),
        Output("system-placeholder", "options"),
        Output("agent-placeholder", "options"),
        Input("location", "pathname"),
    )
    def load_prompt_configuration(_pathname):
        try:
            config = client.prompt_configuration()
        except Exception:
            config = {
                "system_template": DEFAULT_SYSTEM_PROMPT,
                "agent_template": DEFAULT_AGENT_PROMPT,
                "placeholders": {},
            }
        options = [
            {"label": f"{name} — {description}", "value": name}
            for name, description in config.get("placeholders", {}).items()
        ]
        return config, options, options

    @app.callback(
        [Output(f"agent-card-{index}", "style") for index in range(1, 9)],
        Input("player-count", "value"),
        Input("matrix-enabled", "value"),
        Input("matrix-player-counts", "value"),
    )
    def show_agent_rows(player_count, matrix_enabled, matrix_counts):
        count = max(matrix_counts or [player_count]) if "enabled" in (matrix_enabled or []) else player_count
        return [{} if index <= count else {"display": "none"} for index in range(1, 9)]

    @app.callback(
        Output("hyperparameters-panel", "style"),
        Output("agent-assignment-panel", "style"),
        Output("show-hyperparameters", "className"),
        Output("show-agents", "className"),
        Input("show-hyperparameters", "n_clicks"),
        Input("show-agents", "n_clicks"),
    )
    def toggle_configuration(_hyper_clicks, _agent_clicks):
        if ctx.triggered_id == "show-agents":
            return {"display": "none"}, {}, "section-toggle", "section-toggle active"
        return {}, {"display": "none"}, "section-toggle active", "section-toggle"

    @app.callback(
        Output("system-prompt", "value"),
        Output("agent-prompt", "value"),
        Input("reset-prompts", "n_clicks"),
        Input("insert-system-placeholder", "n_clicks"),
        Input("insert-agent-placeholder", "n_clicks"),
        State("system-prompt", "value"),
        State("agent-prompt", "value"),
        State("system-placeholder", "value"),
        State("agent-placeholder", "value"),
        State("prompt-defaults", "data"),
        prevent_initial_call=True,
    )
    def edit_prompts(_reset, _insert_system, _insert_agent, system, agent, system_name, agent_name, defaults):
        if ctx.triggered_id == "reset-prompts":
            return defaults["system_template"], defaults["agent_template"]
        if ctx.triggered_id == "insert-system-placeholder" and system_name:
            return f"{system or ''} {{{{ {system_name} }}}}", no_update
        if ctx.triggered_id == "insert-agent-placeholder" and agent_name:
            return no_update, f"{agent or ''} {{{{ {agent_name} }}}}"
        return no_update, no_update

    @app.callback(
        Output("system-prompt-preview", "children"),
        Output("agent-prompt-preview", "children"),
        Output("prompt-validation", "children"),
        Input("system-prompt", "value"),
        Input("agent-prompt", "value"),
        Input("mechanism", "value"),
        Input("description-treatment", "value"),
        Input("player-count", "value"),
        Input("rounds", "value"),
        Input("starting-budget", "value"),
        Input("value-min", "value"),
        prevent_initial_call=False,
    )
    def preview_prompts(system, agent, mechanism, description, players, rounds, budget, value_min):
        try:
            rendered = client.preview_prompts({
                "prompts": {"system_template": system or "", "agent_template": agent or ""},
                "mechanism": mechanism,
                "mechanism_description_treatment": description,
                "num_players": players,
                "total_rounds": rounds,
                "private_value": value_min,
                "remaining_budget": budget,
                "maximum_bid": budget,
            })
            return rendered["system_prompt"], rendered["agent_prompt"], "Prompt templates are valid."
        except Exception as exc:
            return "", "", f"Prompt error: {exc}"

    @app.callback(
        Output("matrix-summary", "children"),
        Input("matrix-enabled", "value"),
        Input("matrix-mechanisms", "value"),
        Input("matrix-descriptions", "value"),
        Input("matrix-player-counts", "value"),
        Input("valuation-schedules", "value"),
        Input("simulation-count", "value"),
    )
    def summarize_matrix(enabled, mechanisms, descriptions, counts, schedules_text, trials):
        if "enabled" not in (enabled or []):
            return "Matrix disabled; n is the number of runs."
        try:
            schedules = json.loads(schedules_text) if schedules_text else []
            schedule_cells = sum(
                count if schedule.get("rotate_across_positions") else 1
                for count in (counts or [])
                for schedule in schedules
            ) if schedules else len(counts or [])
            cells = len(mechanisms or []) * len(descriptions or []) * schedule_cells
            return f"{cells} treatment cells × {trials or 0} trials = {cells * (trials or 0)} runs."
        except Exception as exc:
            return f"Invalid valuation schedule JSON: {exc}"

    agent_states = []
    for index in range(1, 9):
        for suffix in ("profile", "model", "temperature", "tokens", "timeout", "memory", "retries", "reasoning", "options"):
            agent_states.append(State(f"agent-{index}-{suffix}", "value"))

    @app.callback(
        Output("simulation-id", "data"),
        Output("status-message", "children", allow_duplicate=True),
        Output("progress-poll", "disabled", allow_duplicate=True),
        Input("run-button", "n_clicks"),
        State("simulation-count", "value"), State("rounds", "value"),
        State("starting-budget", "value"), State("value-min", "value"), State("value-max", "value"),
        State("mechanism", "value"), State("player-count", "value"), State("description-treatment", "value"),
        State("seed", "value"), State("tie-seed", "value"),
        State("institution", "value"), State("tie-rule", "value"), State("tie-priority", "value"),
        State("system-prompt", "value"), State("agent-prompt", "value"),
        State("matrix-enabled", "value"), State("matrix-mechanisms", "value"),
        State("matrix-descriptions", "value"), State("matrix-player-counts", "value"),
        State("valuation-schedules", "value"),
        *agent_states,
        prevent_initial_call=True,
    )
    def start_simulation(n_clicks, n, rounds, budget, value_min, value_max, mechanism,
                        player_count, description, seed, tie_seed, institution, tie_rule,
                        tie_priority, system_prompt, agent_prompt, matrix_enabled,
                        matrix_mechanisms, matrix_descriptions, matrix_counts,
                        schedules_text, *agent_values):
        if not n_clicks:
            return no_update, no_update, no_update
        try:
            matrix_on = "enabled" in (matrix_enabled or [])
            required_agents = max(matrix_counts or [player_count]) if matrix_on else player_count
            agents = []
            for offset, index in enumerate(range(1, required_agents + 1)):
                values = agent_values[offset * 9:(offset + 1) * 9]
                profile, model, temperature, tokens, timeout, memory, retries, reasoning, options = values
                agents.append({
                    "player_id": f"P{index}", "profile_id": profile, "model": model or None,
                    "temperature": temperature, "max_output_tokens": tokens,
                    "timeout_seconds": timeout, "memory_rounds": memory, "max_retries": retries,
                    "reasoning_effort": None if reasoning == "none" else reasoning,
                    "provider_options": json.loads(options or "{}"),
                })
            auction = {
                "rounds": rounds, "starting_budget": budget, "true_value_min": value_min,
                "true_value_max": value_max, "mechanism": mechanism, "num_players": player_count,
                "mechanism_description_treatment": description, "institution": institution,
                "seed": seed, "tie_breaking": tie_rule, "tie_break_seed": tie_seed,
                "tie_break_priority": None,
            }
            if tie_rule == "player_priority":
                priority = [part.strip() for part in (tie_priority or "").split(",") if part.strip()]
                priority.extend(
                    agent["player_id"] for agent in agents if agent["player_id"] not in priority
                )
                auction["tie_break_priority"] = priority[:required_agents]
            payload = {
                "name": "sealed-bid-auction", "n": n, "max_parallel_runs": 4,
                "auction": auction, "agents": agents,
                "prompts": {"system_template": system_prompt, "agent_template": agent_prompt},
            }
            if matrix_on:
                payload["experiment_matrix"] = {
                    "mechanisms": matrix_mechanisms or [],
                    "mechanism_description_treatments": matrix_descriptions or [],
                    "player_counts": matrix_counts or [],
                    "valuation_schedules": json.loads(schedules_text) if schedules_text else [],
                }
            response = client.create_simulation(payload)
            return response["simulation_id"], "Simulation queued.", False
        except Exception as exc:
            return None, f"Error: {exc}", True

    @app.callback(
        Output("status-message", "children"), Output("progress-bar", "value"),
        Output("results-store", "data"), Output("progress-poll", "disabled"),
        Input("progress-poll", "n_intervals"), State("simulation-id", "data"),
        prevent_initial_call=True,
    )
    def poll_status(_interval, simulation_id):
        if not simulation_id:
            return no_update, no_update, no_update, True
        try:
            state = client.status(simulation_id)
            percent = round(state["progress"] * 100, 1)
            message = (
                f"{state['status'].replace('_', ' ').title()}: "
                f"{state['completed_runs']} completed, {state['failed_runs']} failed; "
                f"{state['completed_rounds']}/{state['total_rounds']} rounds."
            )
            if state["status"] in {"completed", "completed_with_errors", "failed"}:
                return message, percent, client.results(simulation_id), True
            return message, percent, no_update, False
        except Exception as exc:
            return f"Error checking progress: {exc}", no_update, no_update, True

    @app.callback(
        Output("summary-cards", "children"), Output("results-table", "data"),
        Output("results-table", "columns"),
        Output("inspect-run", "options"), Output("inspect-run", "value"),
        Output("graphs-panel", "style"),
        Input("results-store", "data"),
    )
    def render_results(results):
        if not results:
            return (
                [],
                [],
                [{"name": "", "id": "_placeholder"}],
                [],
                None,
                {"display": "none"},
            )
        batch = results["batch_aggregates"]
        cards = [
            html.Div([html.Span(label), html.Strong("—" if value is None else f"{value:.3f}" if isinstance(value, float) else value)], className="metric-card")
            for label, value in (
                ("Successful runs", batch["successful_runs"]), ("Failed runs", batch["failed_runs"]),
                ("Mean revenue", batch["mean_revenue"]), ("Mean efficiency", batch["mean_efficiency"]),
                ("Mean payoff", batch.get("mean_payoff")),
                ("Mean truthfulness deviation", batch.get("mean_truthfulness_deviation")),
            )
        ]
        aggregates = {(row["run_id"], row["player_id"]): row for row in results["agent_aggregates"]}
        table_rows = []
        for row in results["decision_rows"]:
            aggregate = aggregates[(row["run_id"], row["player_id"])]
            rendered = row.get("rendered_agent_prompt", "")
            table_rows.append({**row, "prompt_preview": rendered[:120] + ("…" if len(rendered) > 120 else ""), **{f"aggregate_{key}": value for key, value in aggregate.items() if key not in {"run_id", "run_index", "player_id", "provider", "model"}}})
        hidden = {"usage", "system_prompt_template", "agent_prompt_template", "rendered_system_prompt", "rendered_agent_prompt", "accepted_prompt_request", "prompt_attempts"}
        visible = [key for key in table_rows[0] if key not in hidden] if table_rows else []
        columns = [{"name": key.replace("_", " ").title(), "id": key} for key in visible]
        runs = sorted({row["run_id"] for row in results["round_rows"]})
        options = [{"label": run_id, "value": run_id} for run_id in runs]
        return cards, table_rows, columns, options, runs[0] if runs else None, {"display": "block"}

    @app.callback(
        *[Output(f"filter-{dimension}", "options") for dimension in TREATMENT_DIMENSIONS],
        *[Output(f"filter-{dimension}", "value") for dimension in TREATMENT_DIMENSIONS],
        Input("results-store", "data"),
    )
    def configure_plot_filters(results):
        rows = (results or {}).get("treatment_aggregates", [])
        outputs = []
        values = []
        for dimension in TREATMENT_DIMENSIONS:
            unique = list(dict.fromkeys(row[dimension] for row in rows))
            outputs.append([{"label": str(value), "value": value} for value in unique])
            values.append(unique[0] if unique else None)
        return (*outputs, *values)

    @app.callback(
        Output("active-graph", "figure"),
        Input("results-store", "data"),
        *[Input(f"graph-button-{index}", "n_clicks") for index in range(1, 12)],
        Input("error-mode", "value"),
        Input("plot-x", "value"),
        Input("plot-color", "value"),
        *[Input(f"filter-{dimension}", "value") for dimension in TREATMENT_DIMENSIONS],
    )
    def switch_graph(results, *_values):
        error_mode = _values[11]
        x_dimension = _values[12]
        color_dimension = _values[13]
        filter_values = _values[14:18]
        if x_dimension == color_dimension:
            color_dimension = next(value for value in TREATMENT_DIMENSIONS if value != x_dimension)
        filters = dict(zip(TREATMENT_DIMENSIONS, filter_values))
        figures = build_research_figures(
            results or {}, error_mode, x_dimension, color_dimension, filters
        )
        triggered = ctx.triggered_id
        if isinstance(triggered, str) and triggered.startswith("graph-button-"):
            index = int(triggered.rsplit("-", 1)[1]) - 1
        else:
            index = 0
        return figures[index]

    @app.callback(
        Output("prompt-detail", "children"),
        Input("results-table", "selected_rows"),
        State("results-table", "data"),
    )
    def show_prompt_detail(selected_rows, rows):
        if not selected_rows or not rows:
            return html.P("Select a decision row to inspect its complete prompts.")
        row = rows[selected_rows[0]]
        sections = [
            ("System template", row.get("system_prompt_template", "")),
            ("Agent template", row.get("agent_prompt_template", "")),
            ("Rendered system prompt", row.get("rendered_system_prompt", "")),
            ("Rendered agent prompt", row.get("rendered_agent_prompt", "")),
            ("Accepted prompt request", json.dumps(row.get("accepted_prompt_request", {}), indent=2)),
            ("All prompt attempts", json.dumps(row.get("prompt_attempts", []), indent=2)),
        ]
        return html.Details(
            [html.Summary("Complete prompt provenance"), *[html.Div([html.H4(title), html.Pre(value)]) for title, value in sections]],
            open=True,
        )

    @app.callback(
        Output("round-slider", "max"), Output("round-slider", "marks"), Output("round-slider", "value"),
        Input("inspect-run", "value"), State("results-store", "data"),
    )
    def configure_slider(run_id, results):
        rows = [row for row in (results or {}).get("round_rows", []) if row["run_id"] == run_id]
        maximum = max((row["round"] for row in rows), default=0)
        marks = {value: str(value) for value in range(maximum + 1)} if maximum <= 20 else {0: "0", maximum: str(maximum)}
        return maximum, marks, 0

    @app.callback(
        Output("round-inspector", "children"), Input("inspect-run", "value"),
        Input("round-slider", "value"), State("results-store", "data"),
    )
    def inspect_round(run_id, round_number, results):
        rows = [
            row for row in (results or {}).get("decision_rows", [])
            if row["run_id"] == run_id and row["round"] == round_number
        ]
        return [
            html.Div(
                [html.H4(row["player_id"]), html.P(f"Valuation: {row['valuation']}"),
                 html.P(f"Bid: {row['bid']}"), html.P("Winner" if row["winner"] else "Lost"),
                 html.P(f"Payment: {row['payment']}"), html.P(f"Utility: {row['utility']}"),
                 html.P(f"Cumulative utility: {row['cumulative_utility']}")],
                className="agent-card",
            )
            for row in rows
        ]

    register_cournot_callbacks(app, client)
    return app
