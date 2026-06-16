from __future__ import annotations

from abc import ABC, abstractmethod

from sandbox.models import Action, LegalActions, Observation


# Java Interface equivalent since all environments NEED these methods
class Environment(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def observe(self, player_id: str) -> Observation:
        ...

    @abstractmethod
    def legal_actions(self, player_id: str) -> LegalActions:
        ...

    @abstractmethod
    def step(self, player_id: str, action: Action) -> None:
        ...

    @abstractmethod
    def is_done(self) -> bool:
        ...
