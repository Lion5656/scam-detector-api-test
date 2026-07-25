import json
import logging

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.product import (
    product_identifier,
    product_research_agent,
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


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    def with_structured_output(
        self,
        schema,
        *,
        method=None,
        include_raw=False,
        **kwargs,
    ):
        assert method == "json_mode"
        assert kwargs["reasoning_format"] == "hidden"
        assert kwargs["reasoning_effort"] == "none"

        def invoke(messages):
            raw = self.invoke(messages)
            try:
                parsed = schema.model_validate_json(str(raw.content))
                parsing_error = None
            except Exception as error:
                parsed = None
                parsing_error = error
            if include_raw:
                return {
                    "raw": raw,
                    "parsed": parsed,
                    "parsing_error": parsing_error,
                }
            if parsing_error is not None:
                raise parsing_error
            return parsed

        return RunnableLambda(invoke)


class _JsonModeFailureModel(_ToolAwareFakeModel):
    def with_structured_output(
        self,
        schema,
        *,
        method=None,
        include_raw=False,
        **kwargs,
    ):
        assert method == "json_mode"
        assert kwargs["reasoning_format"] == "hidden"
        assert kwargs["reasoning_effort"] == "none"

        def invoke(messages):
            raise RuntimeError("json_validate_failed")

        return RunnableLambda(invoke)


class _RateLimitFailureModel(_ToolAwareFakeModel):
    def with_structured_output(
        self,
        schema,
        *,
        method=None,
        include_raw=False,
        **kwargs,
    ):
        def invoke(messages):
            raise RuntimeError(
                "Error code: 429 - rate_limit_exceeded; "
                "Please try again in 10m42.384s"
            )

        return RunnableLambda(invoke)


@pytest.fixture(autouse=True)
def _disable_groq_retry_delay(monkeypatch) -> None:
    monkeypatch.setattr(
        product_research_agent.settings,
        "GROQ_FALLBACK_RETRY_DELAY_SECONDS",
        0.0,
    )


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


def test_product_agent_centralizes_groq_tool_call() -> None:
    @tool
    def fake_search(
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """回傳集中於 ProductAgent 的搜尋結果。"""
        return [
            {
                "title": "Sony WH-1000XM5",
                "link": "https://shop.example/sony",
                "snippet": "特價 NT$9,990",
            }
        ]

    llm = _ToolAwareFakeModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "prices": [
                            {
                                "price": 9990,
                                "currency": "TWD",
                                "url": "https://shop.example/sony",
                                "evidence": "特價 NT$9,990",
                                "condition": "new",
                                "product_match": True,
                            }
                        ]
                    }
                ),
            ),
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_search])

    result = agent.online_price_search(
        system_prompt="先搜尋再回傳價格 JSON",
        user_prompt=json.dumps(
            {
                "product_query": "Sony WH-1000XM5 台灣 價格",
                "max_results": 8,
            }
        ),
        allowed_tool_names=["fake_search"],
    )

    assert result.output["prices"][0]["price"] == 9990
    assert result.tool_results == [
        {
            "title": "Sony WH-1000XM5",
            "link": "https://shop.example/sony",
            "snippet": "特價 NT$9,990",
        }
    ]
    assert result.tool_errors == []


def test_online_price_search_accepts_batch_with_one_low_price() -> None:
    @tool
    def fake_search(
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """Return one invalid low price and one valid market price."""
        return [
            {
                "title": "Low-priced accessory",
                "link": "https://shop.example/accessory",
                "snippet": "NT$150",
            },
            {
                "title": "Sony WH-1000XM5",
                "link": "https://shop.example/sony",
                "snippet": "NT$9,990",
            },
        ]

    llm = _ToolAwareFakeModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "prices": [
                            {
                                "price": 150,
                                "currency": "TWD",
                                "url": "https://shop.example/accessory",
                                "evidence": "NT$150",
                                "condition": "new",
                                "product_match": False,
                            },
                            {
                                "price": 9990,
                                "currency": "TWD",
                                "url": "https://shop.example/sony",
                                "evidence": "NT$9,990",
                                "condition": "new",
                                "product_match": True,
                            },
                        ]
                    }
                )
            )
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_search])

    result = agent.online_price_search(
        system_prompt="Return price evidence as JSON.",
        user_prompt=json.dumps(
            {
                "product_query": "Sony WH-1000XM5",
                "max_results": 8,
            }
        ),
        allowed_tool_names=["fake_search"],
    )

    assert [item["price"] for item in result.output["prices"]] == [150, 9990]


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


def test_online_price_search_logs_invalid_final_json(caplog) -> None:
    @tool
    def fake_search(
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """回傳測試搜尋結果。"""
        return []

    llm = _ToolAwareFakeModel(
        responses=[
            AIMessage(content="沒有可解析的 JSON"),
            AIMessage(content="仍然沒有 JSON"),
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_search])

    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            ValueError,
            match="Groq 未回傳有效的價格結構",
        ):
            agent.online_price_search(
                system_prompt="搜尋後回傳 JSON",
                user_prompt=json.dumps(
                    {
                        "product_query": "商品價格",
                        "max_results": 8,
                    }
                ),
                allowed_tool_names=["fake_search"],
            )

    assert (
        "Groq JSON mode failed tool=fake_search"
    ) in caplog.text
    assert "Groq price output failed tool=fake_search" in caplog.text
    assert "response='仍然沒有 JSON'" in caplog.text


def test_online_price_search_retries_text_mode_after_json_mode_error(
    caplog,
) -> None:
    @tool
    def fake_search(
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """回傳測試搜尋結果。"""
        return [
            {
                "title": "Sony WH-1000XM5",
                "link": "https://shop.example/sony",
                "snippet": "特價 NT$9,990",
            }
        ]

    llm = _JsonModeFailureModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "prices": [
                            {
                                "price": 9990,
                                "currency": "TWD",
                                "url": "https://shop.example/sony",
                                "evidence": "特價 NT$9,990",
                                "condition": "new",
                                "product_match": True,
                            }
                        ]
                    }
                )
            )
        ]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_search])

    with caplog.at_level(logging.WARNING):
        result = agent.online_price_search(
            system_prompt="搜尋後回傳 JSON",
            user_prompt=json.dumps(
                {
                    "product_query": "Sony WH-1000XM5",
                    "max_results": 8,
                }
            ),
            allowed_tool_names=["fake_search"],
        )

    assert result.output["prices"][0]["price"] == 9990
    assert (
        "Groq JSON mode failed tool=fake_search "
        "error=json_validate_failed; retrying text mode"
    ) in caplog.text


def test_online_price_search_retries_text_mode_after_rate_limit(
    caplog,
) -> None:
    @tool
    def fake_search(
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        """回傳測試搜尋結果。"""
        return []

    llm = _RateLimitFailureModel(
        responses=[AIMessage(content='{"prices":[]}')]
    )
    agent = ProductResearchAgent(llm=llm, tools=[fake_search])

    with caplog.at_level(logging.WARNING):
        result = agent.online_price_search(
            system_prompt="搜尋後回傳 JSON",
            user_prompt=json.dumps(
                {
                    "product_query": "商品價格",
                    "max_results": 8,
                }
            ),
            allowed_tool_names=["fake_search"],
        )

    assert result.output == {"prices": []}
    assert (
        "Groq JSON mode failed tool=fake_search "
        "error=Error code: 429 - rate_limit_exceeded; "
        "Please try again in 10m42.384s; retrying text mode"
    ) in caplog.text


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
    assert result.search_query == "Apple iPhone 15 256GB 藍色 台灣 價格"
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

    assert result.search_query == "Sony WH-1000XM5 Sony 無線降噪耳機 台灣 價格"
    assert result.market_price == 9990
    assert research_agent.calls == []
