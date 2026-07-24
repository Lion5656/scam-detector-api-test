import json

import pytest
from pydantic import SecretStr

from backend.services.image_price_service.models import MarketplaceCondition
from backend.services.image_price_service.pricing import (
    online_marketprice_service,
)
from backend.services.image_price_service.pricing.online_marketprice_service import (
    OnlineMarketPriceService,
)
from backend.services.image_price_service.product import product_research_agent
from backend.services.image_price_service.product.product_research_agent import (
    GroqRateLimitError,
    ProductAgentResult,
    search_market_prices_serpapi,
    search_market_prices_tavily,
)


class _FakeResearchAgent:
    def __init__(self, responses: list[ProductAgentResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.system_prompts: list[str] = []

    def online_price_search(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_tool_names: list[str],
    ) -> ProductAgentResult:
        self.system_prompts.append(system_prompt)
        self.calls.append(
            {
                "user_input": json.loads(user_prompt),
                "allowed_tool_names": allowed_tool_names,
            }
        )
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _disable_search_fallback_delay(monkeypatch) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_FALLBACK_DELAY_SECONDS",
        0.0,
    )


def _search_result(
    *,
    url: str,
    price: int,
) -> dict:
    return {
        "title": "商品 A",
        "link": url,
        "snippet": f"特價 NT${price:,}",
    }


def _price_candidate(
    url: str,
    price: int,
    condition: str = "new",
) -> dict:
    return {
        "price": price,
        "currency": "TWD",
        "url": url,
        "evidence": f"特價 NT${price:,}",
        "condition": condition,
        "product_match": True,
    }


def _agent_result(
    results: list[dict],
    condition: str = "new",
) -> ProductAgentResult:
    return ProductAgentResult(
        output={
            "prices": [
                _price_candidate(
                    result["link"],
                    int(
                        result["snippet"]
                        .replace("特價 NT$", "")
                        .replace(",", "")
                    ),
                    condition,
                )
                for result in results
            ]
        },
        tool_results=results,
        tool_errors=[],
    )


def test_serpapi_google_light_tool_is_defined_in_product_agent(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class FakeSerpApiClient:
        def __init__(self, *, api_key: str, timeout: int) -> None:
            assert api_key == "serp-test-key"
            assert timeout == 60

        def search(self, params: dict) -> str:
            calls.append(params)
            return json.dumps(
                {
                    "search_metadata": {
                        "status": "Success",
                    },
                    "organic_results": [
                        {
                            "position": 1,
                            "title": "Apple iPhone 15 256GB",
                            "link": "https://shop.example/iphone",
                            "displayed_link": "shop.example › iphone",
                            "snippet": "特價 NT$30,500",
                            "extensions": ["額外欄位不應保留"],
                        },
                        {
                            "position": 2,
                            "title": "Apple iPhone 15",
                            "link": "https://shop-two.example/iphone",
                            "snippet": "特價 NT$31,500",
                        },
                    ],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        product_research_agent.settings,
        "SERPAPI_API_KEY",
        SecretStr("serp-test-key"),
    )
    monkeypatch.setattr(
        product_research_agent.serpapi,
        "Client",
        FakeSerpApiClient,
    )

    result = search_market_prices_serpapi.invoke(
        {
            "query": "Apple iPhone 15 256GB 台灣 價格",
            "max_results": 8,
        }
    )

    assert calls == [
        {
            "engine": "google_light",
            "q": "Apple iPhone 15 256GB 台灣 價格",
            "google_domain": "google.com.tw",
            "hl": "zh-tw",
            "gl": "tw",
        }
    ]
    assert result == [
        {
            "title": "Apple iPhone 15 256GB",
            "link": "https://shop.example/iphone",
            "snippet": "特價 NT$30,500",
        },
        {
            "title": "Apple iPhone 15",
            "link": "https://shop-two.example/iphone",
            "snippet": "特價 NT$31,500",
        }
    ]


def test_tavily_tool_calls_api_and_normalizes_results(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class FakeTavilyClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "tavily-test-key"

        def search(self, **kwargs) -> dict:
            calls.append(kwargs)
            return {
                "results": [
                    {
                        "title": "Apple iPhone 15 128GB",
                        "url": "https://shop.example/iphone",
                        "content": "全新售價 NT$21,890",
                    },
                    {
                        "title": "缺少網址的結果",
                        "url": "",
                        "content": "NT$20,000",
                    },
                ]
            }

    monkeypatch.setattr(
        product_research_agent.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    monkeypatch.setattr(
        product_research_agent.settings,
        "SEARCH_COUNTRY",
        "taiwan",
    )
    monkeypatch.setattr(
        product_research_agent.settings,
        "SEARCH_DOMAIN",
        ["shop.example"],
    )
    monkeypatch.setattr(
        product_research_agent.settings,
        "EXCLUDE_DOMAIN",
        ["example.invalid"],
    )
    monkeypatch.setattr(
        product_research_agent,
        "TavilyClient",
        FakeTavilyClient,
    )

    result = search_market_prices_tavily.invoke(
        {
            "query": "Apple iPhone 15 台灣 全新 價格",
            "max_results": 3,
        }
    )

    assert calls == [
        {
            "query": "Apple iPhone 15 台灣 全新 價格",
            "max_results": 3,
            "country": "taiwan",
            "include_domains": ["shop.example"],
            "exclude_domains": ["example.invalid"],
        }
    ]
    assert result == [
        {
            "title": "Apple iPhone 15 128GB",
            "link": "https://shop.example/iphone",
            "snippet": "全新售價 NT$21,890",
        }
    ]


def test_online_price_keeps_prompt_and_statistics_without_search_tools(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_PRICE_POINTS",
        2,
    )
    primary_results = [
        _search_result(
            url="https://shop-a.example/1",
            price=30000,
        ),
        _search_result(
            url="https://shop-b.example/2",
            price=31000,
        ),
        _search_result(
            url="https://shop-b.example/3",
            price=32000,
        ),
    ]
    agent = _FakeResearchAgent([_agent_result(primary_results)])
    service = OnlineMarketPriceService(research_agent=agent)

    price, search_tool = service.estimate_price(
        "商品 A 台灣 價格",
        max_results=8,
    )

    assert price == 31000
    assert search_tool == "serp_api"
    assert len(agent.calls) == 1
    assert agent.calls[0]["user_input"]["product_query"] == (
        "商品 A 台灣 價格 全新"
    )
    assert agent.calls[0]["user_input"]["target_condition"] == "new"
    assert agent.calls[0]["allowed_tool_names"] == [
        "search_market_prices_serpapi"
    ]
    assert "只擷取全新商品價格" in agent.system_prompts[0]


def test_online_price_uses_title_used_grade_in_query_and_prompt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_PRICE_POINTS",
        2,
    )
    results = [
        _search_result(url="https://shop-a.example/1", price=12_000),
        _search_result(url="https://shop-b.example/2", price=13_000),
    ]
    agent = _FakeResearchAgent([_agent_result(results, condition="used")])
    service = OnlineMarketPriceService(research_agent=agent)

    price, search_tool = service.estimate_price(
        "Sony PS5 台灣 價格",
        condition=MarketplaceCondition.USED,
        condition_text="Sony PS5 2手 9成新",
    )

    assert (price, search_tool) == (12_500, "serp_api")
    assert agent.calls[0]["user_input"]["product_query"] == (
        "Sony PS5 台灣 價格 2手 9成新"
    )
    assert agent.calls[0]["user_input"]["target_condition"] == "used"
    assert "2手 9成新" in agent.system_prompts[0]
    assert 'condition 必須填 "used"' in agent.system_prompts[0]


def test_online_price_requests_fallback_tools_after_insufficient_primary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        3,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_FALLBACK_DELAY_SECONDS",
        1.5,
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        online_marketprice_service.time,
        "sleep",
        sleep_calls.append,
    )
    primary_results = [
        _search_result(
            url="https://shop-a.example/1",
            price=30000,
        )
    ]
    tavily_results = [
        _search_result(
            url="https://shop-b.example/2",
            price=31000,
        )
    ]
    ddgs_results = [
        _search_result(
            url="https://shop-c.example/3",
            price=32000,
        )
    ]
    agent = _FakeResearchAgent(
        [
            _agent_result(primary_results),
            _agent_result(tavily_results),
            _agent_result(ddgs_results),
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    price, search_tool = service.estimate_price("商品 A 台灣 價格")

    assert price == 31000
    assert search_tool == "ddgs"
    assert [
        call["allowed_tool_names"] for call in agent.calls
    ] == [
        ["search_market_prices_serpapi"],
        ["search_market_prices_tavily"],
        ["search_market_prices_ddgs"],
    ]
    assert sleep_calls == [1.5, 1.5]


def test_online_price_stops_fallbacks_after_groq_rate_limit(
    monkeypatch,
    caplog,
) -> None:
    class RateLimitedResearchAgent:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def online_price_search(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            allowed_tool_names: list[str],
        ) -> ProductAgentResult:
            self.calls.append(allowed_tool_names)
            raise GroqRateLimitError(642.384)

    agent = RateLimitedResearchAgent()
    service = OnlineMarketPriceService(research_agent=agent)

    with caplog.at_level("WARNING"):
        result = service.estimate_price("Apple iPhone 15 台灣 價格")

    assert result == (0, "unused")
    assert agent.calls == [["search_market_prices_serpapi"]]
    assert (
        "price search stopped tool=serp_api "
        "reason=groq_rate_limit retry_after=642.4s"
    ) in caplog.text


def test_online_price_reports_tavily_when_first_fallback_is_sufficient(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_PRICE_POINTS",
        2,
    )
    tavily_results = [
        _search_result(url="https://shop-a.example/1", price=30000),
        _search_result(url="https://shop-b.example/2", price=32000),
    ]
    agent = _FakeResearchAgent(
        [
            _agent_result([]),
            _agent_result(tavily_results),
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    assert service.estimate_price("商品 A 台灣 價格") == (31000, "tavily")
    assert [
        call["allowed_tool_names"] for call in agent.calls
    ] == [
        ["search_market_prices_serpapi"],
        ["search_market_prices_tavily"],
    ]


def test_online_price_reports_unused_when_all_search_tools_fail(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    agent = _FakeResearchAgent(
        [
            _agent_result([]),
            _agent_result([]),
            _agent_result([]),
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    assert service.estimate_price("查無價格的商品") == (0, "unused")


def test_online_price_skips_tavily_when_api_key_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr(""),
    )
    agent = _FakeResearchAgent(
        [
            _agent_result([]),
            _agent_result([]),
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    assert service.estimate_price("查無價格的商品") == (0, "unused")
    assert [
        call["allowed_tool_names"] for call in agent.calls
    ] == [
        ["search_market_prices_serpapi"],
        ["search_market_prices_ddgs"],
    ]


def test_price_validation_discards_only_the_out_of_range_item() -> None:
    low_result = _search_result(
        url="https://shop-a.example/accessory",
        price=150,
    )
    valid_result = _search_result(
        url="https://shop-b.example/product",
        price=3680,
    )
    agent_result = ProductAgentResult(
        output={
            "prices": [
                _price_candidate(low_result["link"], 150),
                _price_candidate(valid_result["link"], 3680),
            ]
        },
        tool_results=[low_result, valid_result],
        tool_errors=[],
    )

    candidates = OnlineMarketPriceService()._validate_agent_prices(
        agent_result
    )

    assert [candidate["price"] for candidate in candidates] == [3680]


def test_aggregate_from_site_prices_requires_multiple_sites(
    monkeypatch,
) -> None:
    service = OnlineMarketPriceService()
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "ONLINE_PRICE_MIN_SITES",
        2,
    )

    price = service._aggregate_from_site_prices(
        {"shop-a.example": [30000, 30500, 31000]}
    )

    assert price == 0
