from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from sandbox.models import Observation


DEFAULT_COURNOT_SYSTEM_PROMPT = (
    "You are firm {{ player_id }} in a {{ num_players }}-firm repeated Cournot market using the "
    "{{ treatment }} information treatment. Maximize your own profit and follow the "
    "response rules exactly."
)

DEFAULT_COURNOT_AGENT_PROMPT = (
    "Round {{ round_number }} of {{ total_rounds }}.\n"
    "Choose a quantity from {{ quantity_min }} through {{ quantity_max }} in increments "
    "of {{ quantity_step }}.\n"
    "Experiment-visible observation: {{ observation }}\n"
    "Return JSON only with action_type='submit_quantity', "
    "value={'quantity': number}, and a brief reasoning string."
)

COURNOT_PLACEHOLDER_DESCRIPTIONS = {
    "player_id": "Current firm identifier.",
    "num_players": "Number of firms in the market.",
    "round_number": "Human-facing, one-based round number.",
    "total_rounds": "Total periods in the run.",
    "treatment": "BEST or FULL information treatment.",
    "quantity_min": "Minimum legal quantity.",
    "quantity_max": "Maximum legal quantity.",
    "quantity_step": "Legal quantity increment.",
    "previous_quantity": "Firm's quantity in the previous period.",
    "previous_profit": "Firm's profit in the previous period.",
    "opponents_total": "Previous total quantity of the other firms.",
    "best_reply": "Calculated myopic best reply.",
    "public_history": "Experiment-visible completed-period history.",
    "observation": "Complete experiment-visible observation as JSON.",
}

_VALID_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_ANY_TEMPLATE_MARKER = re.compile(r"{{|}}|{%|%}|{#|#}")


def _placeholder_names(template: str) -> set[str]:
    matched_spans = []
    names = set()
    for match in _VALID_PLACEHOLDER.finditer(template):
        matched_spans.append(match.span())
        names.add(match.group(1))
    remainder = list(template)
    for start, end in matched_spans:
        remainder[start:end] = " " * (end - start)
    if _ANY_TEMPLATE_MARKER.search("".join(remainder)):
        raise ValueError("Only simple {{ placeholder }} expressions are supported")
    unknown = names - COURNOT_PLACEHOLDER_DESCRIPTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown Cournot prompt placeholders: {sorted(unknown)}")
    return names


def validate_cournot_prompt_templates(system_template: str, agent_template: str) -> None:
    _placeholder_names(system_template)
    _placeholder_names(agent_template)


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _render(template: str, context: Mapping[str, Any]) -> str:
    if len(template) > 50_000:
        raise ValueError("Cournot prompt templates may not exceed 50,000 characters")
    names = _placeholder_names(template)
    undefined = names - context.keys()
    if undefined:
        raise ValueError(f"Undefined Cournot prompt placeholders: {sorted(undefined)}")
    return _VALID_PLACEHOLDER.sub(
        lambda match: _stringify(context[match.group(1)]), template
    )


@dataclass(frozen=True)
class RenderedCournotPrompts:
    system: str
    agent: str


def render_cournot_prompts(
    system_template: str,
    agent_template: str,
    observation: Observation,
) -> RenderedCournotPrompts:
    private = observation.private_state
    public = observation.public_state
    limits = observation.legal_actions.limits
    context = {
        "player_id": observation.player_id,
        "num_players": public.get("market", {}).get("firms"),
        "round_number": observation.round + 1,
        "total_rounds": public["rounds_total"],
        "treatment": public["treatment"],
        "quantity_min": limits["quantity_min"],
        "quantity_max": limits["quantity_max"],
        "quantity_step": limits["quantity_step"],
        "previous_quantity": private["previous_quantity"],
        "previous_profit": private["previous_profit"],
        "opponents_total": private["opponents_total_last_round"],
        "best_reply": private["best_reply_quantity"],
        "public_history": public["completed_rounds"],
        "observation": observation.to_dict(),
    }
    return RenderedCournotPrompts(
        system=_render(system_template, context),
        agent=_render(agent_template, context),
    )
