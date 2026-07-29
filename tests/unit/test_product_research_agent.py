import json

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.product import (
    product_identifier,
)
from backend.services.image_price_service.product.product_identifier import (
    ProductIdentifier,
)
from backend.services.image_price_service.product.product_research_agent import (
    ProductResearchAgent,
)


class _FakeMarketRepository:
    def __init__(
        self,
        result: tuple[str, str, int] = ("未知商品", "未知型號", 0),
    ) -> None:
        self.result = result

    def find_by_text(self, text: str) -> tuple[str, str, int]:
        return self.result


class _FakeResearchAgent:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.response


def test_product_agent_executes_json_ai_request_without_tools() -> None:
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=(
                    "```json\n"
                    '{"product_name":"Sony 耳機","brand_model":"WH-1000XM5"}'
                    "\n```"
                )
            )
        ]
    )
    service = ProductResearchAgent(llm=llm)

    result = service.complete_json(
        system_prompt="只回傳 JSON",
        user_prompt="辨識商品",
    )

    assert result == {
        "product_name": "Sony 耳機",
        "brand_model": "WH-1000XM5",
    }


def test_parse_json_object_accepts_markdown_and_leading_text() -> None:
    result = ProductResearchAgent._parse_json_object(
        "以下是結果：\n```json\n"
        '{"prices":[{"price":9990}]}'
        "\n```\n"
    )

    assert result == {"prices": [{"price": 9990}]}


def test_message_text_reads_structured_content_and_reasoning_fallback() -> None:
    structured_message = AIMessage(
        content=[
            {
                "type": "text",
                "text": '{"prices":[]}',
            }
        ]
    )
    reasoning_message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": '{"prices":[{"price":9990}]}',
        },
    )

    assert (
        ProductResearchAgent._message_text(structured_message)
        == '{"prices":[]}'
    )
    assert (
        ProductResearchAgent._message_text(reasoning_message)
        == '{"prices":[{"price":9990}]}'
    )


def test_product_identifier_normalizes_product_and_builds_search_query(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        product_identifier.pattern_identifier,
        "identify_product",
        lambda text: ProductIdentification(
            product_name="一般商品",
            brand_model="未知型號",
        ),
    )
    research_agent = _FakeResearchAgent(
        {
            "product_name": "Apple iPhone 15",
            "brand_model": "Apple iPhone 15",
            "known_specs": ["256GB", "藍色", "256GB"],
        }
    )
    identifier = ProductIdentifier(
        market_repo=_FakeMarketRepository(),
        research_agent=research_agent,
    )

    result = identifier.identify("iphone 15 256g 藍 NT$25,000")

    assert result.product_name == "Apple iPhone 15"
    assert result.brand_model == "Apple iPhone 15"
    assert result.known_specs == ["256GB", "藍色"]
    assert result.search_query == "Apple iPhone 15 256GB 藍色 價格"
    assert result.market_price == 0
    assert len(research_agent.calls) == 1
    ai_input = json.loads(research_agent.calls[0]["user_prompt"])
    assert ai_input["ocr_text"] == "iphone 15 256g 藍 NT$25,000"


def test_product_identifier_builds_query_for_local_market_match() -> None:
    research_agent = _FakeResearchAgent({})
    identifier = ProductIdentifier(
        market_repo=_FakeMarketRepository(
            ("Sony 無線降噪耳機", "Sony WH-1000XM5", 9990)
        ),
        research_agent=research_agent,
    )

    result = identifier.identify("WH-1000XM5")

    assert result.search_query == "Sony WH-1000XM5 無線降噪耳機 價格"
    assert result.market_price == 9990
    assert research_agent.calls == []
