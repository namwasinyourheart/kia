from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import List
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType

_ = load_dotenv()

class LLM:
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        model_provider: str = "google_genai",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        self.chat_model = init_chat_model(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str) -> str:
        message = HumanMessage(content=prompt)
        response = self.chat_model.invoke([message])
        return response.content

    def bind_tools(self, tools: List[BaseTool], agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION):
        """
        Bind LangChain tools to this model and return an AgentExecutor.
        """
        return initialize_agent(
            tools,
            self.chat_model,
            agent=agent_type,
            verbose=False
        )
