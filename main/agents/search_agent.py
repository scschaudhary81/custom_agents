from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_tavily import TavilySearch
from main.custom_response_schemas.SearchResponseAgent import SearchResponseAgent

load_dotenv()


class SearchAgent:
    """Tavily-backed search agent that returns a structured SearchResponseAgent."""

    def __init__(
        self,
        model_name: str,
        *,
        max_retries: int = 10,
        timeout: int = 120,
    ) -> None:
        self.model = init_chat_model(model_name, max_retries=max_retries, timeout=timeout)
        self.tools = [TavilySearch()]
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            response_format=SearchResponseAgent,
        )

    def search(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        messages: list[AnyMessage] = [HumanMessage(content=query)]
        agent_response = self.agent.invoke({"messages": messages})
        result : SearchResponseAgent = agent_response["structured_response"]
        return result.model_dump_json(indent=4)


if __name__ == "__main__":
    agent = SearchAgent(model_name="google_genai:gemini-3.5-flash")
    response = agent.search("Give me 3 Software Engineer job recommendations from linkedin")
    print(response)