from __future__ import annotations

import json

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from frontend.api_client import AuctionApiClient
from sandbox.cournot_prompts import (
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
)

MIN_COURNOT_PLAYERS = 2
DEFAULT_COURNOT_PLAYERS = 4
MAX_COURNOT_PLAYERS = 8
COURNOT_AGENT_FIELDS = (
    "type",
    "policy",
    "initial",
    "profile",
    "model",
    "temperature",
    "memory",
    "retries",
    "system-prompt",
    "agent-prompt",
)


def _number(label, component_id, value, minimum=None, maximum=None, step=1):
    return html.Label(
        [
            label,
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
            ),
        ],
        className="control-field",
    )


def _agent_card(index: int):
    prefix = f"cournot-agent-{index}"
    return html.Div(
        [
            html.H4(f"Player P{index}"),
            html.Label(
                [
                    "Participant type",
                    dcc.Dropdown(
                        id=f"{prefix}-type",
                        options=[
                            {"label": "Script", "value": "script"},
                            {"label": "Model", "value": "model"},
                            {"label": "Human", "value": "human"},
                        ],
                        value="script",
                        clearable=False,
                    ),
                ],
                className="control-field",
            ),
            html.Div(
                [
                    html.Label(
                        [
                            "Policy",
                            dcc.Dropdown(
                                id=f"{prefix}-policy",
                                options=[
                                    {
                                        "label": "Myopic best reply",
                                        "value": "cournot_best_reply",
                                    },
                                    {
                                        "label": "Random quantity (1-100)",
                                        "value": "cournot_random_quantity",
                                    },
                                    {
                                        "label": "Previous average quantity",
                                        "value": "cournot_previous_average",
                                    },
                                ],
                                value="cournot_best_reply",
                                clearable=False,
                            ),
                        ],
                        className="control-field",
                    ),
                    _number(
                        "Initial quantity",
                        f"{prefix}-initial",
                        index * 10,
                        0,
                        100,
                        0.01,
                    ),
                ],
                id=f"{prefix}-script-settings",
                className="agent-settings",
            ),
            html.Div(
                [
                    html.Label(
                        [
                            "Model provider",
                            dcc.Dropdown(
                                id=f"{prefix}-profile",
                                placeholder="Provider profile",
                                clearable=False,
                            ),
                        ],
                        className="control-field",
                    ),
                    html.Label(
                        [
                            "Model",
                            dcc.Input(
                                id=f"{prefix}-model",
                                type="text",
                                value="llama3.1",
                                placeholder="Blank uses the provider default",
                            ),
                        ],
                        className="control-field",
                    ),
                    _number("Temperature", f"{prefix}-temperature", 0.0, 0, 2, 0.1),
                    _number(
                        "Memory rounds (blank = all)",
                        f"{prefix}-memory",
                        None,
                        0,
                        1000,
                    ),
                    _number("Retries", f"{prefix}-retries", 2, 1, 10),
                ],
                id=f"{prefix}-model-settings",
                className="agent-settings",
                style={"display": "none"},
            ),
        ],
        id=f"cournot-agent-{index}-card",
        className="agent-card cournot-agent-card",
        style={} if index <= DEFAULT_COURNOT_PLAYERS else {"display": "none"},
    )


def _agent_prompt_card(index: int):
    prefix = f"cournot-agent-{index}"
    return html.Div(
        [
            html.H4(f"Player P{index}"),
            html.P(
                "Used when this player's participant type is Model.",
                className="field-help",
            ),
            html.Label(
                [
                    "System prompt",
                    dcc.Textarea(
                        id=f"{prefix}-system-prompt",
                        value=DEFAULT_COURNOT_SYSTEM_PROMPT,
                    ),
                ],
                className="control-field prompt-field",
            ),
            html.Label(
                [
                    "Decision prompt",
                    dcc.Textarea(
                        id=f"{prefix}-agent-prompt",
                        value=DEFAULT_COURNOT_AGENT_PROMPT,
                    ),
                ],
                className="control-field prompt-field",
            ),
        ],
        id=f"cournot-agent-{index}-prompt-card",
        className="agent-card cournot-agent-prompt-card",
        style={} if index <= DEFAULT_COURNOT_PLAYERS else {"display": "none"},
    )


def cournot_experiment():
    return html.Div(
        [
            dcc.Store(id="cournot-simulation-id"),
            dcc.Store(id="cournot-human-request"),
            dcc.Store(id="cournot-result-artifacts", data={}),
            dcc.Store(id="cournot-provider-store", data=[]),
            dcc.Store(id="cournot-player-count", data=DEFAULT_COURNOT_PLAYERS),
            dcc.Interval(id="cournot-progress-poll", interval=1000, disabled=True),
            html.Section(
                [
                    html.Div(
                        [
                            html.Button(
                                "Hyperparameters",
                                id="cournot-show-hyperparameters",
                                n_clicks=0,
                                className="section-toggle active",
                            ),
                            html.Button(
                                "Agent Assignment",
                                id="cournot-show-agents",
                                n_clicks=0,
                                className="section-toggle",
                            ),
                            html.Button(
                                "Agent Prompts",
                                id="cournot-show-prompts",
                                n_clicks=0,
                                className="section-toggle",
                            ),
                        ],
                        className="section-toggles",
                    ),
                    html.Div(
                        [
                            html.Label(
                                [
                                    "Treatment",
                                    dcc.Dropdown(
                                        id="cournot-treatment",
                                        options=[
                                            {"label": "BEST", "value": "BEST"},
                                            {"label": "FULL", "value": "FULL"},
                                        ],
                                        value="BEST",
                                        clearable=False,
                                    ),
                                ],
                                className="control-field",
                            ),
                            _number("Rounds", "cournot-rounds", 40, 1, 500),
                            _number("Seed", "cournot-seed", 7),
                            _number(
                                "Revision probability",
                                "cournot-revision-probability",
                                round(2 / 3, 4),
                                0,
                                1,
                                0.0001,
                            ),
                        ],
                        id="cournot-hyperparameters-panel",
                        className="control-grid tab-content",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(
                                        f"{DEFAULT_COURNOT_PLAYERS} players",
                                        id="cournot-player-count-label",
                                    ),
                                    html.Button(
                                        "Remove player",
                                        id="cournot-remove-player",
                                        n_clicks=0,
                                        className="player-count-button",
                                    ),
                                    html.Button(
                                        "Add player",
                                        id="cournot-add-player",
                                        n_clicks=0,
                                        className="player-count-button",
                                    ),
                                ],
                                className="cournot-player-controls",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="cournot-provider-status",
                                        className="provider-status",
                                    ),
                                    *[
                                        _agent_card(index)
                                        for index in range(
                                            1, MAX_COURNOT_PLAYERS + 1
                                        )
                                    ],
                                ],
                                className="agent-grid",
                            ),
                        ],
                        id="cournot-agent-assignment-panel",
                        className="tab-content",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    _agent_prompt_card(index)
                                    for index in range(1, MAX_COURNOT_PLAYERS + 1)
                                ],
                                className="agent-grid",
                            ),
                        ],
                        id="cournot-agent-prompts-panel",
                        className="tab-content",
                        style={"display": "none"},
                    ),
                    html.Button(
                        "Run Cournot Simulation",
                        id="cournot-run-button",
                        n_clicks=0,
                        className="run-button",
                    ),
                    html.Div(id="cournot-status-message", className="status-message"),
                    html.Progress(
                        id="cournot-progress-bar",
                        value=0,
                        max=100,
                        title="Cournot simulation progress",
                        **{"aria-label": "Cournot simulation progress"},
                    ),
                ],
                className="panel",
            ),
            html.Section(
                [
                    html.H2(id="cournot-human-heading"),
                    html.Pre(id="cournot-human-observation"),
                    _number("Quantity", "cournot-human-quantity", 0, 0, 100, 0.01),
                    html.Button(
                        "Submit Quantity",
                        id="cournot-human-submit",
                        n_clicks=0,
                        className="run-button",
                    ),
                    html.Div(id="cournot-human-error", className="status-message"),
                ],
                id="cournot-human-panel",
                className="panel human-decision-panel",
                style={"display": "none"},
            ),
            cournot_results_panel(),
        ],
        id="cournot-experiment",
        style={"display": "none"},
    )


def cournot_results_panel():
    return html.Section(
        [
            html.Div(
                [
                    html.H2("Cournot Results"),
                    html.Div(id="cournot-result-meta", className="cournot-run-status"),
                ],
                className="cournot-panel-heading",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H3("Average Profit by Player"),
                            html.Label(
                                [
                                    "Whiskers",
                                    dcc.RadioItems(
                                        id="cournot-error-mode",
                                        options=[
                                            {"label": "+/- 2 SD", "value": "two_sd"},
                                            {"label": "95% CI", "value": "ci95"},
                                        ],
                                        value="two_sd",
                                        inline=True,
                                    ),
                                ],
                                className="cournot-error-control",
                            ),
                        ],
                        className="cournot-artifact-heading",
                    ),
                    html.Img(id="cournot-score-graph", alt="Cournot player scores"),
                ],
                className="cournot-artifact",
            ),
            html.Section(
                [
                    html.H3("Event Playback"),
                    html.Iframe(
                        id="cournot-playback-frame", title="Cournot event playback"
                    ),
                ],
                className="cournot-artifact",
            ),
        ],
        id="cournot-results-panel",
        className="panel cournot-results-panel",
        style={"display": "none"},
    )


def _profile_model(profile_id, profiles):
    profile = next(
        (item for item in (profiles or []) if item["profile_id"] == profile_id),
        None,
    )
    return profile.get("model", "") if profile else ""


def register_cournot_callbacks(app: Dash, client: AuctionApiClient) -> None:
    @app.callback(
        *[
            Output(f"cournot-agent-{index}-profile", "options")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        *[
            Output(f"cournot-agent-{index}-profile", "value")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        Output("cournot-provider-store", "data"),
        Output("cournot-provider-status", "children"),
        Input("location", "pathname"),
    )
    def load_cournot_profiles(_pathname):
        try:
            profiles = client.providers()["profiles"]
            options = [
                {
                    "label": (
                        f"{profile['profile_id']} "
                        f"({'ready' if profile['configured'] else 'not configured'})"
                    ),
                    "value": profile["profile_id"],
                    "disabled": not profile["configured"],
                }
                for profile in profiles
            ]
            ready = next(
                (
                    profile["profile_id"]
                    for profile in profiles
                    if profile["configured"]
                ),
                None,
            )
            statuses = [
                html.Div(
                    f"{profile['profile_id']}: "
                    f"{profile['configuration_status'].replace('_', ' ')}"
                )
                for profile in profiles
            ]
            return (
                *([options] * MAX_COURNOT_PLAYERS),
                *([ready] * MAX_COURNOT_PLAYERS),
                profiles,
                statuses,
            )
        except Exception as exc:
            return (
                *([[]] * MAX_COURNOT_PLAYERS),
                *([None] * MAX_COURNOT_PLAYERS),
                [],
                f"Provider status unavailable: {exc}",
            )

    for index in range(1, MAX_COURNOT_PLAYERS + 1):
        app.callback(
            Output(f"cournot-agent-{index}-model", "value"),
            Input(f"cournot-agent-{index}-profile", "value"),
            State("cournot-provider-store", "data"),
            prevent_initial_call=True,
        )(_profile_model)

    @app.callback(
        Output("cournot-player-count", "data"),
        Output("cournot-player-count-label", "children"),
        Output("cournot-remove-player", "disabled"),
        Output("cournot-add-player", "disabled"),
        *[
            Output(f"cournot-agent-{index}-card", "style")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        *[
            Output(f"cournot-agent-{index}-prompt-card", "style")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        Input("cournot-remove-player", "n_clicks"),
        Input("cournot-add-player", "n_clicks"),
        State("cournot-player-count", "data"),
    )
    def change_player_count(_remove_clicks, _add_clicks, current_count):
        count = int(current_count or DEFAULT_COURNOT_PLAYERS)
        if ctx.triggered_id == "cournot-remove-player":
            count = max(MIN_COURNOT_PLAYERS, count - 1)
        elif ctx.triggered_id == "cournot-add-player":
            count = min(MAX_COURNOT_PLAYERS, count + 1)
        card_styles = [
            {} if index <= count else {"display": "none"}
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ]
        return (
            count,
            f"{count} players",
            count == MIN_COURNOT_PLAYERS,
            count == MAX_COURNOT_PLAYERS,
            *card_styles,
            *card_styles,
        )

    @app.callback(
        Output("cournot-score-graph", "src"),
        Input("cournot-result-artifacts", "data"),
        Input("cournot-error-mode", "value"),
    )
    def select_score_graph(artifacts, error_mode):
        if not artifacts:
            return ""
        return artifacts.get(error_mode, artifacts.get("two_sd", ""))

    @app.callback(
        Output("cournot-hyperparameters-panel", "style"),
        Output("cournot-agent-assignment-panel", "style"),
        Output("cournot-agent-prompts-panel", "style"),
        Output("cournot-show-hyperparameters", "className"),
        Output("cournot-show-agents", "className"),
        Output("cournot-show-prompts", "className"),
        Input("cournot-show-hyperparameters", "n_clicks"),
        Input("cournot-show-agents", "n_clicks"),
        Input("cournot-show-prompts", "n_clicks"),
    )
    def toggle_configuration(_hyper_clicks, _agent_clicks, _prompt_clicks):
        if ctx.triggered_id == "cournot-show-agents":
            return (
                {"display": "none"}, {}, {"display": "none"},
                "section-toggle", "section-toggle active", "section-toggle",
            )
        if ctx.triggered_id == "cournot-show-prompts":
            return (
                {"display": "none"}, {"display": "none"}, {},
                "section-toggle", "section-toggle", "section-toggle active",
            )
        return (
            {}, {"display": "none"}, {"display": "none"},
            "section-toggle active", "section-toggle", "section-toggle",
        )

    @app.callback(
        *[
            Output(f"cournot-agent-{index}-script-settings", "style")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        *[
            Output(f"cournot-agent-{index}-model-settings", "style")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
        *[
            Input(f"cournot-agent-{index}-type", "value")
            for index in range(1, MAX_COURNOT_PLAYERS + 1)
        ],
    )
    def show_agent_settings(*agent_types):
        script_styles = [
            {} if agent_type == "script" else {"display": "none"}
            for agent_type in agent_types
        ]
        model_styles = [
            {} if agent_type == "model" else {"display": "none"}
            for agent_type in agent_types
        ]
        return (*script_styles, *model_styles)

    agent_states = []
    for index in range(1, MAX_COURNOT_PLAYERS + 1):
        for suffix in COURNOT_AGENT_FIELDS:
            agent_states.append(State(f"cournot-agent-{index}-{suffix}", "value"))

    @app.callback(
        Output("cournot-simulation-id", "data"),
        Output("cournot-status-message", "children"),
        Output("cournot-progress-bar", "value"),
        Output("cournot-progress-poll", "disabled"),
        Output("cournot-result-artifacts", "data"),
        Output("cournot-playback-frame", "src"),
        Output("cournot-results-panel", "style"),
        Output("cournot-result-meta", "children"),
        Output("cournot-human-request", "data"),
        Output("cournot-human-panel", "style"),
        Output("cournot-human-heading", "children"),
        Output("cournot-human-observation", "children"),
        Output("cournot-human-quantity", "value"),
        Output("cournot-human-error", "children"),
        Input("cournot-run-button", "n_clicks"),
        Input("cournot-progress-poll", "n_intervals"),
        Input("cournot-human-submit", "n_clicks"),
        State("cournot-simulation-id", "data"),
        State("cournot-human-request", "data"),
        State("cournot-human-quantity", "value"),
        State("cournot-treatment", "value"),
        State("cournot-rounds", "value"),
        State("cournot-seed", "value"),
        State("cournot-revision-probability", "value"),
        State("cournot-player-count", "data"),
        *agent_states,
        prevent_initial_call=True,
    )
    def run_poll_or_submit(
        _run, _poll, _submit, simulation_id, human_request, human_quantity,
        treatment, rounds, seed, revision_probability, player_count,
        *agent_values,
    ):
        hidden_human = (None, {"display": "none"}, "", "", 0, "")
        if ctx.triggered_id == "cournot-run-button":
            try:
                agents = []
                for offset, index in enumerate(
                    range(1, int(player_count) + 1)
                ):
                    field_count = len(COURNOT_AGENT_FIELDS)
                    values = agent_values[
                        offset * field_count:(offset + 1) * field_count
                    ]
                    (
                        kind,
                        script_policy,
                        initial,
                        profile_id,
                        model,
                        temperature,
                        memory,
                        retries,
                        system_prompt_override,
                        agent_prompt_override,
                    ) = values
                    policy = {
                        "script": script_policy,
                        "model": "cournot_llm",
                        "human": "web_human",
                    }[kind]
                    agent = {
                        "player_id": f"P{index}", "policy": policy,
                        "initial_quantity": initial,
                        "profile_id": profile_id,
                        "model": model or None,
                        "temperature": temperature, "memory_rounds": memory,
                        "max_retries": retries,
                    }
                    if kind == "model":
                        agent.update(
                            {
                                "system_prompt_template": (
                                    system_prompt_override.strip()
                                    if system_prompt_override
                                    else None
                                ),
                                "agent_prompt_template": (
                                    agent_prompt_override.strip()
                                    if agent_prompt_override
                                    else None
                                ),
                            }
                        )
                    agents.append(agent)
                response = client.create_cournot_simulation(
                    {
                        "name": f"huck-cournot-{treatment.lower()}",
                        "cournot": {
                            "rounds": rounds, "treatment": treatment, "seed": seed,
                            "revision_probability": revision_probability,
                        },
                        "agents": agents,
                        "prompts": {
                            "system_template": DEFAULT_COURNOT_SYSTEM_PROMPT,
                            "agent_template": DEFAULT_COURNOT_AGENT_PROMPT,
                        },
                    }
                )
                return (
                    response["simulation_id"], "Cournot simulation queued.", 0, False,
                    {}, "", {"display": "none"}, "", *hidden_human,
                )
            except Exception as exc:
                return (None, f"Error: {exc}", 0, True, {}, "", {"display": "none"}, "", *hidden_human)

        if ctx.triggered_id == "cournot-human-submit":
            if not simulation_id or not human_request:
                return (no_update,) * 13 + ("No human decision is pending.",)
            try:
                client.submit_cournot_human_decision(
                    simulation_id, human_request["request_id"], human_quantity
                )
                return (
                    simulation_id, "Human quantity submitted.", no_update, False,
                    no_update, no_update, {"display": "none"}, no_update,
                    *hidden_human,
                )
            except Exception as exc:
                return (no_update,) * 13 + (str(exc),)

        if not simulation_id:
            return (no_update,) * 14
        try:
            state = client.cournot_status(simulation_id)
            progress = round(state["progress"] * 100, 1)
            if state["status"] == "completed":
                result = client.cournot_results(simulation_id)
                return (
                    simulation_id, "Cournot simulation completed.", 100, True,
                    {
                        "two_sd": result["graph_two_sd_url"],
                        "ci95": result["graph_ci95_url"],
                    },
                    result["playback_url"], {"display": "block"},
                    (
                        f"{result['run_id']} | {result['treatment']} | "
                        f"{result['player_count']} players"
                    ),
                    *hidden_human,
                )
            if state["status"] == "failed":
                return (
                    simulation_id, f"Cournot simulation failed: {state.get('error')}",
                    progress, True, {}, "", {"display": "none"}, "", *hidden_human,
                )
            pending = client.cournot_human_decision(simulation_id).get("pending")
            if pending:
                observation = pending["observation"]
                private = observation["private_state"]
                suggestion = private.get("best_reply_quantity")
                if suggestion is None:
                    suggestion = private.get("previous_quantity") or 0
                return (
                    simulation_id,
                    (
                        "Waiting for a human quantity. "
                        f"{state['completed_rounds']} of {state['total_rounds']} "
                        f"rounds complete ({progress:.0f}%)."
                    ),
                    progress,
                    False,
                    no_update, no_update, {"display": "none"}, "", pending, {},
                    f"{pending['player_id']} - Round {pending['round'] + 1}",
                    json.dumps(observation, indent=2, sort_keys=True), suggestion, "",
                )
            return (
                simulation_id,
                (
                    f"{state['status'].title()}... "
                    f"{state['completed_rounds']} of {state['total_rounds']} "
                    f"rounds complete ({progress:.0f}%)."
                ),
                progress,
                False,
                no_update, no_update, {"display": "none"}, "", *hidden_human,
            )
        except Exception as exc:
            return (
                simulation_id, f"Error checking Cournot progress: {exc}", no_update,
                True, {}, "", {"display": "none"}, "", *hidden_human,
            )
