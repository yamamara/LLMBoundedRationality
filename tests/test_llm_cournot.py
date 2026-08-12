from __future__ import annotations

import unittest

from sandbox.agents.llm_cournot import LLMCournotAgent, LLMCournotAgentConfig
from sandbox.models import LegalActions, Observation
from sandbox.providers.base import ProviderResponse


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ProviderResponse(
            text=(
                '{"action_type":"submit_quantity","value":{"quantity":19.8},'
                '"reasoning":"best response"}'
            ),
            provider=self.name,
            model=request.model,
        )


class LLMCournotAgentTests(unittest.TestCase):
    def test_provider_backed_agent_requests_and_validates_quantity(self):
        provider = FakeProvider()
        agent = LLMCournotAgent(
            provider,
            LLMCournotAgentConfig(
                model="test-model",
                system_prompt_template="Firm {{ player_id }}.",
                agent_prompt_template="Choose in round {{ round_number }}.",
            ),
        )
        observation = Observation(
            player_id="P1",
            round=0,
            public_state={
                "treatment": "BEST",
                "rounds_total": 40,
                "market": {"firms": 4},
                "completed_rounds": [],
            },
            private_state={
                "previous_quantity": 20.0,
                "previous_profit": 0.0,
                "opponents_total_last_round": 60.0,
                "best_reply_quantity": 19.8,
            },
            legal_actions=LegalActions(
                "submit_quantity",
                {
                    "quantity_min": 0.0,
                    "quantity_max": 100.0,
                    "quantity_step": 0.01,
                },
            ),
        )

        decision = agent.decide(observation)

        self.assertEqual(decision.action.value["quantity"], 19.8)
        self.assertEqual(provider.requests[0].model, "test-model")
        self.assertEqual(
            provider.requests[0].json_schema["properties"]["action_type"]["enum"],
            ["submit_quantity"],
        )
        self.assertEqual(decision.metadata["provider"], "fake")


if __name__ == "__main__":
    unittest.main()
