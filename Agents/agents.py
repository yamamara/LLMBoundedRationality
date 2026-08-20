# agent.py
import logging
import json
from typing import List, Dict
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("SimulationEngine.Agent")

class AgentDecision(BaseModel):
    """Schema enforcement for sequential model outputs."""
    chosen_node: int = Field(..., description="The integer index of the server node selected.")
    reasoning: str = Field(..., description="Strategic rationale for this decision.")
    estimated_opponent_behavior: str = Field(..., description="Prediction of competitor actions.")

class StrategicAgent:
    def __init__(self, agent_id: str, model_role: str):
        self.agent_id = agent_id
        self.model_role = model_role
        self.history: List[Dict[str, str]] = []
        
    def construct_system_prompt(self) -> str:
        return (
            f"You are autonomous AI agent {self.agent_id}.\n"
            f"Objective: Maximize individual system utility.\n"
            f"You must return ONLY a raw JSON object matching the requested schema."
        )

    def formulate_decision(self, inference_engine, environment_prompt: str) -> AgentDecision:
        """Processes inference synchronously for this agent's specific turn."""
        messages = [
            {"role": "system", "content": self.construct_system_prompt()},
            *self.history,
            {"role": "user", "content": environment_prompt}
        ]

        # Inject strict Pydantic JSON schema instruction
        schema_json = json.dumps(AgentDecision.model_json_schema())
        messages[-1]["content"] += f"\n\nCRITICAL: Return a raw JSON object matching this schema:\n{schema_json}"

        try:
            # Query the single shared engine sequentially
            raw_output = inference_engine.generate_completion(messages)
            decision = AgentDecision.model_validate_json(raw_output)
            
            # Commit to history
            self.history.append({"role": "user", "content": environment_prompt})
            self.history.append({"role": "assistant", "content": decision.model_dump_json()})
            return decision

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"[{self.agent_id}] Schema parsing failed: {str(e)}. Falling back.")
            return AgentDecision(
                chosen_node=0, 
                reasoning="Fallback activated due to parsing error.", 
                estimated_opponent_behavior="Unknown"
            )