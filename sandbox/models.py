from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    player_id: str
    round: int
    public_state: dict[str, Any]
    private_state: dict[str, Any]
    legal_actions: LegalActions

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "round": self.round,
            "public_state": self.public_state,
            "private_state": self.private_state,
            "legal_actions": self.legal_actions.to_dict(),
        }


@dataclass(frozen=True)
class Action:
    action_type: str
    value: dict[str, Any]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "value": self.value,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class LegalActions:
    action_type: str
    limits: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "limits": self.limits,
        }


@dataclass
class AgentDecision:
    action: Action
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    name: str

    def decide(self, observation: Observation) -> AgentDecision:
        ...


@dataclass
class Participant:
    player_id: str
    agent: Agent
    agent_type: str
    role: str = "bidder"
    private_memory: dict[str, Any] = field(default_factory=dict)
