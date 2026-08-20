from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sandbox.providers.adapters import AnthropicProvider
from sandbox.providers.base import GenerationOptions, ProviderRequest


class ProviderAdapterTests(unittest.TestCase):
    def test_anthropic_sends_temperature_without_top_p(self):
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"quantity": 20}')],
            model="claude-test",
            usage={},
            id="request-1",
            stop_reason="end_turn",
        )
        provider = AnthropicProvider(client=client)
        request = ProviderRequest(
            model="claude-test",
            system_prompt="Choose a quantity.",
            messages=[{"role": "user", "content": "Choose now."}],
            json_schema={"type": "object"},
            options=GenerationOptions(temperature=0.2, top_p=1.0),
        )

        provider.generate(request)

        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("top_p", kwargs)


if __name__ == "__main__":
    unittest.main()
