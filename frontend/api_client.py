from __future__ import annotations

import os
from typing import Any

import requests


class AuctionApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("INTERNAL_API_URL", "http://127.0.0.1:8000/api/v1")).rstrip("/")
        self.timeout = timeout

    def providers(self) -> dict[str, Any]:
        return self._request("GET", "/providers")

    def prompt_configuration(self) -> dict[str, Any]:
        return self._request("GET", "/prompts")

    def preview_prompts(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/prompts/preview", json=payload)

    def create_simulation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/simulations", json=payload)

    def status(self, simulation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/simulations/{simulation_id}")

    def results(self, simulation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/simulations/{simulation_id}/results")

    def create_cournot_simulation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/cournot-simulations", json=payload)

    def cournot_status(self, simulation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/cournot-simulations/{simulation_id}")

    def cournot_results(self, simulation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/cournot-simulations/{simulation_id}/results")

    def cournot_human_decision(self, simulation_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/cournot-simulations/{simulation_id}/human-decision"
        )

    def submit_cournot_human_decision(
        self, simulation_id: str, request_id: str, quantity: float
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/cournot-simulations/{simulation_id}/human-decision",
            json={"request_id": request_id, "quantity": quantity},
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = requests.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    detail = str(exc.response.json().get("detail", ""))
                except Exception:
                    detail = exc.response.text[:500]
            raise RuntimeError(detail or str(exc)) from exc
        return response.json()
