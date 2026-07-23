from __future__ import annotations

import math
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sandbox.models import Action, AgentDecision, Observation


@dataclass
class _HumanSlot:
    condition: threading.Condition
    pending: dict[str, Any] | None = None
    response: float | None = None
    cancelled: bool = False


class HumanDecisionBroker:
    def __init__(self):
        self.lock = threading.RLock()
        self.slots: dict[str, _HumanSlot] = {}

    def request(
        self, job_id: str, observation: Observation, timeout_seconds: float
    ) -> AgentDecision:
        slot = self._slot(job_id)
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        with slot.condition:
            slot.pending = {
                "request_id": request_id,
                "player_id": observation.player_id,
                "round": observation.round,
                "observation": observation.to_dict(),
            }
            slot.response = None
            slot.cancelled = False
            slot.condition.notify_all()
            ready = slot.condition.wait_for(
                lambda: slot.response is not None or slot.cancelled,
                timeout=timeout_seconds,
            )
            if not ready:
                slot.pending = None
                raise TimeoutError(
                    f"Timed out waiting for browser decision from {observation.player_id}"
                )
            if slot.cancelled:
                slot.pending = None
                raise RuntimeError("Browser human decision was cancelled")
            quantity = slot.response
            slot.pending = None
            slot.response = None
        return AgentDecision(
            Action(
                "submit_quantity",
                {"quantity": quantity},
                "Human-entered quantity from the web interface.",
            ),
            {"latency_seconds": time.monotonic() - started, "retry_count": 0},
        )

    def pending(self, job_id: str) -> dict[str, Any] | None:
        slot = self._slot(job_id)
        with slot.condition:
            return deepcopy(slot.pending)

    def submit(self, job_id: str, request_id: str, quantity: float) -> None:
        slot = self._slot(job_id)
        with slot.condition:
            pending = slot.pending
            if pending is None or pending["request_id"] != request_id:
                raise ValueError("Human decision request is no longer pending")
            limits = pending["observation"]["legal_actions"]["limits"]
            if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
                raise ValueError("Quantity must be a number")
            if not math.isfinite(quantity):
                raise ValueError("Quantity must be finite")
            if not limits["quantity_min"] <= quantity <= limits["quantity_max"]:
                raise ValueError(
                    f"Quantity must be between {limits['quantity_min']} and "
                    f"{limits['quantity_max']}"
                )
            steps = round((quantity - limits["quantity_min"]) / limits["quantity_step"])
            grid_value = limits["quantity_min"] + steps * limits["quantity_step"]
            if not math.isclose(quantity, grid_value, abs_tol=1e-9):
                raise ValueError(f"Quantity must use increments of {limits['quantity_step']}")
            slot.response = float(quantity)
            slot.condition.notify_all()

    def finish(self, job_id: str) -> None:
        with self.lock:
            slot = self.slots.pop(job_id, None)
        if slot is not None:
            with slot.condition:
                slot.cancelled = True
                slot.condition.notify_all()

    def _slot(self, job_id: str) -> _HumanSlot:
        with self.lock:
            return self.slots.setdefault(
                job_id, _HumanSlot(condition=threading.Condition())
            )


HUMAN_DECISION_BROKER = HumanDecisionBroker()


class WebHumanAgent:
    name = "web_human"

    def __init__(self, job_id: str, timeout_seconds: float = 3600.0):
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds

    def decide(self, observation: Observation) -> AgentDecision:
        return HUMAN_DECISION_BROKER.request(
            self.job_id, observation, self.timeout_seconds
        )
