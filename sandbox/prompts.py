from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from sandbox.models import Observation


DEFAULT_SYSTEM_PROMPT = (
    "You are player {{ player_id }} in a research auction with {{ num_players }} total "
    "players and {{ num_opponents }} opponents. The mechanism is {{ mechanism_name }}. "
    "{{ mechanism_description }} Maximize your own payoff, use only the information "
    "provided here, and follow the response rules exactly."
)

DEFAULT_AGENT_PROMPT = (
    "Round {{ round_number }} of {{ total_rounds }}.\n"
    "Your private value is {{ private_value }} and your remaining budget is "
    "{{ remaining_budget }}.\n"
    "Submit one integer bid from {{ minimum_bid }} through {{ maximum_bid }}.\n"
    "Public history: {{ public_history }}\n"
    "Return only JSON containing action_type='submit_bid', value={'bid': integer}, "
    "and a brief reasoning summary."
)

PLACEHOLDER_DESCRIPTIONS = {
    "player_id": "Current participant identifier.",
    "num_players": "Total bidders in the auction.",
    "num_opponents": "Number of other bidders.",
    "mechanism_name": "Human-readable auction mechanism name.",
    "mechanism_description": "Selected mechanism-description treatment text.",
    "round_number": "Human-facing, one-based round number.",
    "total_rounds": "Total rounds in the run.",
    "private_value": "This participant's current private value.",
    "remaining_budget": "This participant's current remaining budget.",
    "minimum_bid": "Minimum legal bid.",
    "maximum_bid": "Maximum legal bid.",
    "public_history": "Publicly observable completed-round history as JSON.",
}

_VALID_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_ANY_TEMPLATE_MARKER = re.compile(r"{{|}}|{%|%}|{#|#}")


class PromptTemplateError(ValueError):
    pass


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def template_placeholders(template: str) -> set[str]:
    matched_spans: list[tuple[int, int]] = []
    names: set[str] = set()
    for match in _VALID_PLACEHOLDER.finditer(template):
        matched_spans.append(match.span())
        names.add(match.group(1))
    remainder = list(template)
    for start, end in matched_spans:
        remainder[start:end] = " " * (end - start)
    if _ANY_TEMPLATE_MARKER.search("".join(remainder)):
        raise PromptTemplateError(
            "Only simple {{ placeholder }} expressions are supported"
        )
    unknown = names - PLACEHOLDER_DESCRIPTIONS.keys()
    if unknown:
        raise PromptTemplateError(f"Unknown prompt placeholders: {sorted(unknown)}")
    return names


def render_template(template: str, context: Mapping[str, Any]) -> str:
    if len(template) > 50_000:
        raise PromptTemplateError("Prompt templates may not exceed 50,000 characters")
    names = template_placeholders(template)
    undefined = names - context.keys()
    if undefined:
        raise PromptTemplateError(f"Undefined prompt placeholders: {sorted(undefined)}")
    return _VALID_PLACEHOLDER.sub(lambda match: _stringify(context[match.group(1)]), template)


def prompt_context(observation: Observation) -> dict[str, Any]:
    public = observation.public_state
    private = observation.private_state
    limits = observation.legal_actions.limits
    return {
        "player_id": observation.player_id,
        "num_players": public["num_players"],
        "num_opponents": public["num_opponents"],
        "mechanism_name": public["mechanism_name"],
        "mechanism_description": public["mechanism_description"],
        "round_number": observation.round + 1,
        "total_rounds": public["rounds_total"],
        "private_value": private["true_value"],
        "remaining_budget": private["remaining_budget"],
        "minimum_bid": limits["bid_min"],
        "maximum_bid": limits["bid_max"],
        "public_history": public["completed_rounds"],
    }


@dataclass(frozen=True)
class RenderedPrompts:
    system: str
    agent: str


def render_prompts(
    system_template: str,
    agent_template: str,
    observation: Observation,
) -> RenderedPrompts:
    context = prompt_context(observation)
    return RenderedPrompts(
        render_template(system_template, context),
        render_template(agent_template, context),
    )


def validate_prompt_templates(system_template: str, agent_template: str) -> None:
    template_placeholders(system_template)
    template_placeholders(agent_template)

