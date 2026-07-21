from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from backend.services.image_price_service.product.product_research_agent import (
    ProductResearchAgent,
)


class ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ToolAwareFakeModel":
        return self


@tool
def fake_product_search(query: str) -> str:
    """回傳固定格式的商品搜尋結果，供單元測試使用。"""
    return f"{query} 的搜尋結果"


def test_agent_returns_direct_response_when_no_tool_is_requested() -> None:
    llm = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="product_name: Headphones\nbrand_model: Sony WH-1000XM5"
            )
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_product_search])

    result = agent.run("Sony WH-1000XM5")

    assert result == "product_name: Headphones\nbrand_model: Sony WH-1000XM5"


def test_agent_executes_requested_tool_before_final_response() -> None:
    llm = ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_product_search",
                        "args": {"query": "WH-1000XM5"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="product_name: Headphones\nbrand_model: Sony WH-1000XM5"
            ),
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_product_search])

    result = agent.run("Identify WH-1000XM5")

    assert result == "product_name: Headphones\nbrand_model: Sony WH-1000XM5"
