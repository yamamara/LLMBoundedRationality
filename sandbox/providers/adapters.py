from __future__ import annotations

import os
from typing import Any

from sandbox.providers.base import ProviderRequest, ProviderResponse


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key_env: str = "OPENAI_API_KEY", endpoint: str | None = None, client=None):
        self.api_key_env = api_key_env
        self.endpoint = endpoint
        self._client = client

    def _get_client(self, timeout_seconds: float | None = None):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("OpenAI support requires the 'openai' package") from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key environment variable {self.api_key_env}")
            kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
            if self.endpoint:
                kwargs["base_url"] = self.endpoint
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "instructions": request.system_prompt,
            "input": request.messages,
            "temperature": request.options.temperature,
            "max_output_tokens": request.options.max_output_tokens,
            "timeout": request.options.timeout_seconds,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "auction_bid",
                    "strict": True,
                    "schema": request.json_schema,
                }
            },
        }
        if request.options.top_p is not None:
            kwargs["top_p"] = request.options.top_p
        if request.options.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.options.reasoning_effort}
        kwargs.update(request.options.provider_options)
        response = self._get_client().responses.create(**kwargs)
        usage = _dump(getattr(response, "usage", None))
        return ProviderResponse(
            text=response.output_text,
            provider=self.name,
            model=getattr(response, "model", request.model),
            usage=usage,
            request_id=getattr(response, "id", ""),
            finish_reason=getattr(response, "status", ""),
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY", endpoint: str | None = None, client=None):
        self.api_key_env = api_key_env
        self.endpoint = endpoint
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError("Anthropic support requires the 'anthropic' package") from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key environment variable {self.api_key_env}")
            kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
            if self.endpoint:
                kwargs["base_url"] = self.endpoint
            self._client = Anthropic(**kwargs)
        return self._client

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "system": request.system_prompt,
            "messages": request.messages,
            "temperature": request.options.temperature,
            "max_tokens": request.options.max_output_tokens,
            "output_config": {
                "format": {"type": "json_schema", "schema": request.json_schema}
            },
            "timeout": request.options.timeout_seconds,
        }
        if request.options.top_p is not None:
            kwargs["top_p"] = request.options.top_p
        kwargs.update(request.options.provider_options)
        response = self._get_client().messages.create(**kwargs)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return ProviderResponse(
            text=text,
            provider=self.name,
            model=getattr(response, "model", request.model),
            usage=_dump(getattr(response, "usage", None)),
            request_id=getattr(response, "id", ""),
            finish_reason=getattr(response, "stop_reason", ""),
        )


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key_env: str = "GEMINI_API_KEY", endpoint: str | None = None, client=None):
        self.api_key_env = api_key_env
        self.endpoint = endpoint
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError("Gemini support requires the 'google-genai' package") from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key environment variable {self.api_key_env}")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self.endpoint:
                kwargs["http_options"] = types.HttpOptions(base_url=self.endpoint)
            self._client = genai.Client(**kwargs)
        return self._client

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        try:
            from google.genai import types
        except ImportError:
            types = None
        config_values: dict[str, Any] = {
            "system_instruction": request.system_prompt,
            "temperature": request.options.temperature,
            "max_output_tokens": request.options.max_output_tokens,
            "response_mime_type": "application/json",
            "response_json_schema": request.json_schema,
        }
        if types:
            config_values["http_options"] = types.HttpOptions(
                timeout=int(request.options.timeout_seconds * 1000)
            )
        if request.options.top_p is not None:
            config_values["top_p"] = request.options.top_p
        config_values.update(request.options.provider_options)
        config = types.GenerateContentConfig(**config_values) if types else config_values
        contents = "\n".join(message["content"] for message in request.messages)
        response = self._get_client().models.generate_content(
            model=request.model, contents=contents, config=config
        )
        usage = _dump(getattr(response, "usage_metadata", None))
        return ProviderResponse(
            text=response.text,
            provider=self.name,
            model=request.model,
            usage=usage,
            request_id=getattr(response, "response_id", ""),
            finish_reason="",
        )


class LocalLlamaProvider:
    name = "local_llama"

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        transport: str = "ollama",
        api_key_env: str | None = None,
        client=None,
    ):
        if transport not in {"ollama", "openai_compatible"}:
            raise ValueError("Local Llama transport must be ollama or openai_compatible")
        self.endpoint = endpoint
        self.transport = transport
        self.api_key_env = api_key_env
        self._client = client

    def _get_client(self, timeout_seconds: float | None = None):
        if self._client is not None:
            return self._client
        if self.transport == "ollama":
            try:
                from ollama import Client
            except ImportError as exc:
                raise RuntimeError("Ollama support requires the 'ollama' package") from exc
            self._client = Client(host=self.endpoint, timeout=timeout_seconds)
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Compatible local support requires the 'openai' package") from exc
            key = os.environ.get(self.api_key_env, "local") if self.api_key_env else "local"
            self._client = OpenAI(
                api_key=key,
                base_url=self.endpoint,
                max_retries=0,
                timeout=timeout_seconds,
            )
        return self._client

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        messages = [{"role": "system", "content": request.system_prompt}, *request.messages]
        if self.transport == "ollama":
            options = {
                "temperature": request.options.temperature,
                "num_predict": request.options.max_output_tokens,
            }
            if request.options.top_p is not None:
                options["top_p"] = request.options.top_p
            provider_options = dict(request.options.provider_options)
            keep_alive = provider_options.pop("keep_alive", None)
            options.update(provider_options)
            kwargs: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "format": request.json_schema,
                "options": options,
            }
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            response = self._get_client(request.options.timeout_seconds).chat(**kwargs)
            response_dict = _dump(response)
            return ProviderResponse(
                text=response_dict.get("message", {}).get("content", ""),
                provider=self.name,
                model=response_dict.get("model", request.model),
                usage={
                    key: response_dict[key]
                    for key in ("prompt_eval_count", "eval_count", "total_duration")
                    if key in response_dict
                },
                finish_reason=response_dict.get("done_reason", ""),
            )

        kwargs = {
            "model": request.model,
            "messages": messages,
            "temperature": request.options.temperature,
            "max_tokens": request.options.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "auction_bid", "strict": True, "schema": request.json_schema},
            },
            "timeout": request.options.timeout_seconds,
        }
        if request.options.top_p is not None:
            kwargs["top_p"] = request.options.top_p
        kwargs.update(request.options.provider_options)
        response = self._get_client(request.options.timeout_seconds).chat.completions.create(**kwargs)
        choice = response.choices[0]
        return ProviderResponse(
            text=choice.message.content,
            provider=self.name,
            model=getattr(response, "model", request.model),
            usage=_dump(getattr(response, "usage", None)),
            request_id=getattr(response, "id", ""),
            finish_reason=getattr(choice, "finish_reason", ""),
        )


def build_provider(
    provider: str,
    endpoint: str | None = None,
    api_key_env: str | None = None,
    transport: str | None = None,
):
    if provider == "openai":
        return OpenAIProvider(api_key_env or "OPENAI_API_KEY", endpoint)
    if provider == "anthropic":
        return AnthropicProvider(api_key_env or "ANTHROPIC_API_KEY", endpoint)
    if provider == "gemini":
        return GeminiProvider(api_key_env or "GEMINI_API_KEY", endpoint)
    if provider == "local_llama":
        return LocalLlamaProvider(
            endpoint or "http://localhost:11434",
            transport or "ollama",
            api_key_env,
        )
    raise ValueError(f"Unsupported provider: {provider}")
