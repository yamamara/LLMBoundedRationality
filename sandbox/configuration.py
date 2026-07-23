from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    model: str
    endpoint: str | None = None
    api_key_env: str | None = None
    transport: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.model) and (
            self.provider == "local_llama"
            or bool(self.api_key_env and os.environ.get(self.api_key_env))
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("api_key_env", None)
        value["defaults"] = {
            key: option
            for key, option in value.get("defaults", {}).items()
            if not any(marker in key.lower() for marker in ("key", "token", "password", "secret"))
        }
        value["configured"] = self.configured
        if not self.model:
            value["configuration_status"] = "missing_model"
        elif self.provider != "local_llama" and not (
            self.api_key_env and os.environ.get(self.api_key_env)
        ):
            value["configuration_status"] = "missing_api_key"
        else:
            value["configuration_status"] = "configured"
        return value


DEFAULT_PROFILES = [
    ProviderProfile(
        "local-ollama",
        "local_llama",
        "llama3.1",
        "http://localhost:11434",
        transport="ollama",
        defaults={"temperature": 0.0, "num_ctx": 4096, "keep_alive": "5m"},
    ),
    ProviderProfile("openai", "openai", "", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ProviderProfile("anthropic", "anthropic", "", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    ProviderProfile("gemini", "gemini", "", None, "GEMINI_API_KEY"),
]

ENV_OVERRIDES = {
    "openai": ("OPENAI_MODEL", "OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL"),
    "gemini": ("GEMINI_MODEL", "GEMINI_BASE_URL"),
    "local_llama": ("OLLAMA_MODEL", "OLLAMA_BASE_URL"),
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_provider_profiles(path: Path = Path("config/providers.json")) -> dict[str, ProviderProfile]:
    load_dotenv()
    raw_profiles: list[dict[str, Any]]
    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))
        raw_profiles = body.get("profiles", body) if isinstance(body, dict) else body
        profiles = [ProviderProfile(**profile) for profile in raw_profiles]
    else:
        profiles = list(DEFAULT_PROFILES)

    resolved: dict[str, ProviderProfile] = {}
    for profile in profiles:
        model_env, endpoint_env = ENV_OVERRIDES[profile.provider]
        endpoint = os.environ.get(endpoint_env, profile.endpoint)
        if (
            profile.provider == "local_llama"
            and profile.transport == "openai_compatible"
            and endpoint
            and not endpoint.rstrip("/").endswith("/v1")
        ):
            endpoint = endpoint.rstrip("/") + "/v1"
        resolved_profile = ProviderProfile(
            profile_id=profile.profile_id,
            provider=profile.provider,
            model=os.environ.get(model_env, profile.model),
            endpoint=endpoint,
            api_key_env=profile.api_key_env,
            transport=profile.transport,
            defaults=profile.defaults,
        )
        if resolved_profile.profile_id in resolved:
            raise ValueError(f"Duplicate provider profile ID: {resolved_profile.profile_id}")
        resolved[resolved_profile.profile_id] = resolved_profile
    return resolved
