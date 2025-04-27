from typing import List, Optional

class AgentBase:
    def run(self, user_input: str, chat_history: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Each agent must implement the run method.")
