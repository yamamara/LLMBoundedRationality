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
    @staticmethod
    def observation_with_history():
        return Observation(
            player_id="P1",
            round=3,
            public_state={
                "treatment": "BEST",
                "rounds_total": 40,
                "market": {"firms": 4},
                "completed_rounds": [
                    {"round": index, "own_profit": index * 10}
                    for index in range(3)
                ],
            },
            private_state={
                "previous_quantity": 20.0,
                "previous_profit": 20.0,
                "opponents_total_last_round": 60.0,
                "best_reply_quantity": 19.5,
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

    def test_unlimited_memory_includes_all_completed_rounds(self):
        agent = LLMCournotAgent(
            FakeProvider(),
            LLMCournotAgentConfig(model="test-model", memory_rounds=None),
        )

        prompt_observation = agent._prompt_observation(
            self.observation_with_history()
        )

        self.assertEqual(
            [row["round"] for row in prompt_observation.public_state["completed_rounds"]],
            [0, 1, 2],
        )

    def test_numeric_and_zero_memory_still_truncate_history(self):
        observation = self.observation_with_history()

        last_two = LLMCournotAgent(
            FakeProvider(),
            LLMCournotAgentConfig(model="test-model", memory_rounds=2),
        )._prompt_observation(observation)
        none = LLMCournotAgent(
            FakeProvider(),
            LLMCournotAgentConfig(model="test-model", memory_rounds=0),
        )._prompt_observation(observation)

        self.assertEqual(
            [row["round"] for row in last_two.public_state["completed_rounds"]],
            [1, 2],
        )
        self.assertEqual(none.public_state["completed_rounds"], [])


if __name__ == "__main__":
    unittest.main()
