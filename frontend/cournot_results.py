from __future__ import annotations

import json

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from frontend.api_client import AuctionApiClient
from sandbox.cournot_prompts import (
    COURNOT_PLACEHOLDER_DESCRIPTIONS,
    DEFAULT_COURNOT_AGENT_PROMPT,
    DEFAULT_COURNOT_SYSTEM_PROMPT,
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
                                options=[
                                    {
                                        "label": "Myopic best reply",
                                        "value": "cournot_best_reply",
                                    }
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
                            "Base URL",
                            dcc.Input(
                                id=f"{prefix}-base-url",
                                value="http://localhost:11434/v1",
                            ),
                        ],
                        className="control-field",
                    ),
                    html.Label(
                        [
                            "Model",
                            dcc.Input(id=f"{prefix}-model", value="llama3.1"),
                        ],
                        className="control-field",
                    ),
                    html.Label(
                        [
                            "API key environment variable",
                            dcc.Input(id=f"{prefix}-api-key-env", placeholder="Optional"),
                        ],
                        className="control-field",
                    ),
                    _number("Temperature", f"{prefix}-temperature", 0.0, 0, 2, 0.1),
                    _number("Memory rounds", f"{prefix}-memory", 1, 0, 1000),
                    _number("Retries", f"{prefix}-retries", 2, 1, 10),
                ],
                id=f"{prefix}-model-settings",
                className="agent-settings",
                style={"display": "none"},
            ),
        ],
        className="agent-card cournot-agent-card",
    )


def cournot_experiment():
    placeholder_options = [
        {"label": f"{name} - {description}", "value": name}
        for name, description in COURNOT_PLACEHOLDER_DESCRIPTIONS.items()
    ]
    return html.Div(
        [
            dcc.Store(id="cournot-simulation-id"),
            dcc.Store(id="cournot-human-request"),
            dcc.Store(id="cournot-result-artifacts", data={}),
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
                            html.Div(
                                [
                                    html.H3("Editable Prompts"),
                                    html.Label(
                                        [
                                            "System prompt",
                                            dcc.Textarea(
                                                id="cournot-system-prompt",
                                                value=DEFAULT_COURNOT_SYSTEM_PROMPT,
                                            ),
                                        ],
                                        className="control-field prompt-field",
                                    ),
                                    html.Label(
                                        [
                                            "System placeholder",
                                            dcc.Dropdown(
                                                id="cournot-system-placeholder",
                                                options=placeholder_options,
                                            ),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Button(
                                        "Insert system placeholder",
                                        id="cournot-insert-system-placeholder",
                                        n_clicks=0,
                                    ),
                                    html.Label(
                                        [
                                            "Agent prompt",
                                            dcc.Textarea(
                                                id="cournot-agent-prompt",
                                                value=DEFAULT_COURNOT_AGENT_PROMPT,
                                            ),
                                        ],
                                        className="control-field prompt-field",
                                    ),
                                    html.Label(
                                        [
                                            "Agent placeholder",
                                            dcc.Dropdown(
                                                id="cournot-agent-placeholder",
                                                options=placeholder_options,
                                            ),
                                        ],
                                        className="control-field",
                                    ),
                                    html.Button(
                                        "Insert agent placeholder",
                                        id="cournot-insert-agent-placeholder",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Reset prompts",
                                        id="cournot-reset-prompts",
                                        n_clicks=0,
                                    ),
                                ],
                                className="prompt-editor cournot-prompt-editor",
                            ),
                        ],
                        id="cournot-hyperparameters-panel",
                        className="control-grid tab-content",
                    ),
                    html.Div(
                        [_agent_card(index) for index in range(1, 5)],
                        id="cournot-agent-assignment-panel",
                        className="agent-grid tab-content",
                        style={"display": "none"},
                    ),
                    html.Button(
                        "Run Cournot Simulation",
                        id="cournot-run-button",
                        n_clicks=0,
                        className="run-button",
                    ),
                    html.Div(id="cournot-status-message", className="status-message"),
                    html.Progress(id="cournot-progress-bar", value=0, max=100),
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


def register_cournot_callbacks(app: Dash, client: AuctionApiClient) -> None:
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
        Output("cournot-show-hyperparameters", "className"),
        Output("cournot-show-agents", "className"),
        Input("cournot-show-hyperparameters", "n_clicks"),
        Input("cournot-show-agents", "n_clicks"),
    )
    def toggle_configuration(_hyper_clicks, _agent_clicks):
        if ctx.triggered_id == "cournot-show-agents":
            return {"display": "none"}, {}, "section-toggle", "section-toggle active"
        return {}, {"display": "none"}, "section-toggle active", "section-toggle"

    @app.callback(
        *[
            Output(f"cournot-agent-{index}-script-settings", "style")
            for index in range(1, 5)
        ],
        *[
            Output(f"cournot-agent-{index}-model-settings", "style")
            for index in range(1, 5)
        ],
        *[Input(f"cournot-agent-{index}-type", "value") for index in range(1, 5)],
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

    @app.callback(
        Output("cournot-system-prompt", "value"),
        Output("cournot-agent-prompt", "value"),
        Input("cournot-reset-prompts", "n_clicks"),
        Input("cournot-insert-system-placeholder", "n_clicks"),
        Input("cournot-insert-agent-placeholder", "n_clicks"),
        State("cournot-system-prompt", "value"),
        State("cournot-agent-prompt", "value"),
        State("cournot-system-placeholder", "value"),
        State("cournot-agent-placeholder", "value"),
        prevent_initial_call=True,
    )
    def edit_prompts(_reset, _system_insert, _agent_insert, system, agent, system_name, agent_name):
        if ctx.triggered_id == "cournot-reset-prompts":
            return DEFAULT_COURNOT_SYSTEM_PROMPT, DEFAULT_COURNOT_AGENT_PROMPT
        if ctx.triggered_id == "cournot-insert-system-placeholder" and system_name:
            return f"{system or ''} {{{{ {system_name} }}}}", no_update
        if ctx.triggered_id == "cournot-insert-agent-placeholder" and agent_name:
            return no_update, f"{agent or ''} {{{{ {agent_name} }}}}"
        return no_update, no_update

    agent_states = []
    for index in range(1, 5):
        for suffix in (
            "type", "initial", "base-url", "model", "api-key-env",
            "temperature", "memory", "retries",
        ):
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
        State("cournot-system-prompt", "value"),
        State("cournot-agent-prompt", "value"),
        *agent_states,
        prevent_initial_call=True,
    )
    def run_poll_or_submit(
        _run, _poll, _submit, simulation_id, human_request, human_quantity,
        treatment, rounds, seed, revision_probability, system_prompt, agent_prompt,
        *agent_values,
    ):
        hidden_human = (None, {"display": "none"}, "", "", 0, "")
        if ctx.triggered_id == "cournot-run-button":
            try:
                agents = []
                for offset, index in enumerate(range(1, 5)):
                    values = agent_values[offset * 8:(offset + 1) * 8]
                    kind, initial, base_url, model, key_env, temperature, memory, retries = values
                    policy = {
                        "script": "cournot_best_reply",
                        "model": "cournot_openai_compatible",
                        "human": "web_human",
                    }[kind]
                    agents.append(
                        {
                            "player_id": f"P{index}", "policy": policy,
                            "initial_quantity": initial, "base_url": base_url,
                            "model": model, "api_key_env": key_env or None,
                            "temperature": temperature, "memory_rounds": memory,
                            "max_retries": retries,
                        }
                    )
                response = client.create_cournot_simulation(
                    {
                        "name": f"huck-cournot-{treatment.lower()}",
                        "cournot": {
                            "rounds": rounds, "treatment": treatment, "seed": seed,
                            "revision_probability": revision_probability,
                        },
                        "agents": agents,
                        "prompts": {
                            "system_template": system_prompt,
                            "agent_template": agent_prompt,
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
                    f"{result['run_id']} | {result['treatment']}", *hidden_human,
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
                    simulation_id, "Waiting for a human quantity.", progress, False,
                    no_update, no_update, {"display": "none"}, "", pending, {},
                    f"{pending['player_id']} - Round {pending['round'] + 1}",
                    json.dumps(observation, indent=2, sort_keys=True), suggestion, "",
                )
            return (
                simulation_id, f"{state['status'].title()}...", progress, False,
                no_update, no_update, {"display": "none"}, "", *hidden_human,
            )
        except Exception as exc:
            return (
                simulation_id, f"Error checking Cournot progress: {exc}", no_update,
                True, {}, "", {"display": "none"}, "", *hidden_human,
            )
