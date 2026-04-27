from pydantic import BaseModel
from src.ai_devs_core.agent import BaseAgent, FAgent
from src.ai_devs_core.utils import count_tokens
from dataclasses import dataclass

class ObservedMemory:
    observation_log: list[str] = []
    compressed_log : str = ""
    current_task : str = ""

    def get_memory_state(self) -> str:
        return f"""
        <memories>
        {self.compressed_log}
        </memories>
        <observations>
        {["\n" + o for o in self.observation_log]}
        </observations>
        <current_task>
        {self.current_task}
        </current_task>
        """

    def add(self, observation: str):
        self.observation_log.append(observation)

    def reflect(self):
        agent = FAgent(model_id="mistral-small-latest")
        response = agent.chat_completion(
            chat_history=[
            {"role": "system", "content": "You are a memory reflector. You'll see observations log. You'll have to compress it into key observations while preserving dates and parts of logs."},
            {"role": "user", "content": f"Logs:{['\n'+o for o in self.observation_log]}"}
            ],
            max_steps=2,
        )
        self.compressed_log += response.choices[0].message.content

