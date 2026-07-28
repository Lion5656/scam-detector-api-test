import json

import pytest
from pydantic import SecretStr

from backend.services.dto.price_analysis import MarketPriceCandidateEvidence
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
    PriceRiskPolicy,
)
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
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr(""),
    )


def _search_result(
    *,
    url: str,
    product: str,
    price: int,
    condition_text: str,
    snippet_prefix: str = "售價",
) -> dict:
    return {
        "title": f"{product} {condition_text}".strip(),
        "link": url,
        "snippet": f"{snippet_prefix} NT${price:,}",
    }


def _price_candidate(
    result: dict,
    price: int,
    *,
    condition: str,
) -> dict:
    return {
        "price": price,
        "currency": "TWD",
        "url": result["link"],
        "evidence": result["snippet"],
        "condition": condition,
        "product_match": True,
    }


def _agent_result(
    results: list[dict],
    *,
    conditions: list[str],
    prices: list[int] | None = None,
) -> ProductAgentResult:
    resolved_prices = prices or [
        int(
            result["snippet"]
            .split("NT$", maxsplit=1)[1]
            .replace(",", "")
        )
        for result in results
    ]
    return ProductAgentResult(
        output={
            "prices": [
                _price_candidate(
                    result,
                    price,
                    condition=condition,
                )
                for result, price, condition in zip(
                    results,
                    resolved_prices,
                    conditions,
                    strict=True,
                )
            ]
        },
        tool_results=results,
        tool_errors=[],
    )


def _evidence_candidate(
    price: int,
    index: int,
    *,
    condition: MarketplaceCondition = MarketplaceCondition.NEW,
    site: str | None = None,
) -> MarketPriceCandidateEvidence:
    host = site or f"shop-{index}.example"
    return MarketPriceCandidateEvidence(
        candidate_id=f"candidate-{index}",
        title=f"Apple iPhone 15 256GB {condition.value}",
        price=price,
        condition=condition,
        url=f"https://{host}/items/{index}",
        evidence=f"售價 NT${price:,}",
    )


def _policy(**updates) -> PriceRiskPolicy:
    return PriceRiskPolicy(
        **{
            **DEFAULT_PRICE_RISK_POLICY.model_dump(),
            **updates,
        }
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
                    "search_metadata": {"status": "Success"},
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
        "SERP_API_KEY",
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
    assert len(result) == 2
    assert set(result[0]) == {"title", "link", "snippet"}


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

    assert calls[0]["query"] == "Apple iPhone 15 台灣 全新 價格"
    assert result == [
        {
            "title": "Apple iPhone 15 128GB",
            "link": "https://shop.example/iphone",
            "snippet": "全新售價 NT$21,890",
        }
    ]


def test_known_new_condition_only_accepts_new_prices() -> None:
    results = [
        _search_result(
            url=f"https://shop-{index}.example/item",
            product="Apple iPhone 15 256GB",
            price=price,
            condition_text=condition_text,
        )
        for index, (price, condition_text) in enumerate(
            [
                (30_000, "全新"),
                (31_000, "全新未拆封"),
                (32_000, "新品"),
                (20_000, "二手・良好"),
            ],
            start=1,
        )
    ]
    agent = _FakeResearchAgent(
        [_agent_result(results, conditions=["new", "new", "new", "used"])]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 台灣 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.reference_mode == "median_low_sample"
    assert estimate.median_price == 31_000
    assert {candidate.condition for candidate in estimate.candidates} == {
        MarketplaceCondition.NEW
    }
    assert agent.calls[0]["user_input"]["target_condition"] == "new"


def test_known_used_condition_only_accepts_matching_used_grade() -> None:
    results = [
        _search_result(
            url=f"https://used-{index}.example/item",
            product="Sony PS5",
            price=price,
            condition_text=condition_text,
        )
        for index, (price, condition_text) in enumerate(
            [
                (12_000, "二手・近全新"),
                (12_500, "近全新"),
                (13_000, "二手 近全新"),
                (10_000, "二手・良好"),
                (15_000, "全新"),
            ],
            start=1,
        )
    ]
    agent = _FakeResearchAgent(
        [_agent_result(results, conditions=["used"] * 4 + ["new"])]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    estimate = service.estimate_price(
        "Sony PS5 台灣 價格",
        condition=MarketplaceCondition.USED,
        condition_text="二手・近全新",
    )

    assert estimate.status == "success"
    assert estimate.median_price == 12_500
    assert [candidate.price for candidate in estimate.candidates] == [
        12_000,
        12_500,
        13_000,
    ]
    assert "二手 近全新" in agent.calls[0]["user_input"]["product_query"]


def test_unknown_condition_forces_separate_new_and_used_searches() -> None:
    new_results = [
        _search_result(
            url=f"https://new-{index}.example/item",
            product="Apple iPhone 15 256GB",
            price=30_000 + index * 500,
            condition_text="全新",
        )
        for index in range(1, 4)
    ]
    used_results = [
        _search_result(
            url=f"https://used-{index}.example/item",
            product="Apple iPhone 15 256GB",
            price=20_000 + index * 500,
            condition_text="二手",
        )
        for index in range(1, 4)
    ]
    agent = _FakeResearchAgent(
        [
            _agent_result(new_results, conditions=["new"] * 3),
            _agent_result(used_results, conditions=["used"] * 3),
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    estimates = service.estimate_prices(
        "Apple iPhone 15 256GB 台灣 價格",
        condition=MarketplaceCondition.UNKNOWN,
    )

    assert tuple(estimate.condition for estimate in estimates) == (
        MarketplaceCondition.NEW,
        MarketplaceCondition.USED,
    )
    assert all(estimate.status == "success" for estimate in estimates)
    assert [call["user_input"]["target_condition"] for call in agent.calls] == [
        "new",
        "used",
    ]
    assert "全新" in agent.calls[0]["user_input"]["product_query"]
    assert "二手" in agent.calls[1]["user_input"]["product_query"]


def test_validation_excludes_other_model_version_capacity_accessory_and_installment():
    rows = [
        ("Apple iPhone 15 Pro 256GB", 30_000, "全新", "售價"),
        ("Apple iPhone 14 Pro 256GB", 28_000, "全新", "售價"),
        ("Apple iPhone 15 Pro 128GB", 27_000, "全新", "售價"),
        ("Apple iPhone 15 Pro Max 256GB", 35_000, "全新", "售價"),
        ("Apple iPhone 15 Pro 256GB 手機殼", 999, "全新", "售價"),
        ("Apple iPhone 15 Pro 256GB", 1_500, "全新", "分期每期"),
        ("Apple iPhone 15 Pro 256GB 2024 年款", 2_024, "全新", "售價"),
        ("Samsung Galaxy S24 Ultra 256GB", 32_000, "全新", "售價"),
    ]
    results = [
        _search_result(
            url=f"https://shop-{index}.example/item",
            product=product,
            price=price,
            condition_text=condition_text,
            snippet_prefix=snippet_prefix,
        )
        for index, (product, price, condition_text, snippet_prefix) in enumerate(
            rows,
            start=1,
        )
    ]
    agent_result = _agent_result(results, conditions=["new"] * len(results))

    candidates = OnlineMarketPriceService()._validate_agent_prices(
        agent_result,
        product_query="Apple iPhone 15 Pro 256GB 台灣 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert [candidate.price for candidate in candidates] == [30_000]


def test_out_of_range_candidate_is_removed_before_sample_threshold_check():
    results = [
        _search_result(
            url=f"https://shop-{index}.example/item",
            product="Apple iPhone 15 256GB",
            price=price,
            condition_text="全新",
        )
        for index, price in enumerate(
            [30_000, 31_000, 120_000, 0],
            start=1,
        )
    ]
    validated = OnlineMarketPriceService()._validate_agent_prices(
        _agent_result(results, conditions=["new"] * 4),
        product_query="Apple iPhone 15 256GB 台灣 價格",
        condition=MarketplaceCondition.NEW,
    )

    estimate = OnlineMarketPriceService()._aggregate_candidates(
        validated,
        MarketplaceCondition.NEW,
    )

    assert [candidate.price for candidate in validated] == [30_000, 31_000]
    assert estimate.status == "insufficient"
    assert estimate.sample_count == 2


def test_not_found_and_insufficient_are_distinct_statuses() -> None:
    service = OnlineMarketPriceService()
    insufficient_candidates = [
        _evidence_candidate(30_000, 1),
        _evidence_candidate(31_000, 2),
    ]

    not_found = service._aggregate_candidates([], MarketplaceCondition.NEW)
    insufficient = service._aggregate_candidates(
        insufficient_candidates,
        MarketplaceCondition.NEW,
    )

    assert not_found.status == "not_found"
    assert not_found.sample_count == 0
    assert insufficient.status == "insufficient"
    assert insufficient.sample_count == 2


def test_independent_source_count_is_checked_by_injected_policy() -> None:
    service = OnlineMarketPriceService()
    candidates = [
        _evidence_candidate(30_000 + index * 500, index, site="same.example")
        for index in range(3)
    ]

    estimate = service._aggregate_candidates(
        candidates,
        MarketplaceCondition.NEW,
    )

    assert estimate.status == "insufficient"
    assert estimate.sample_count == 3
    assert estimate.site_count == 1


@pytest.mark.parametrize(
    ("prices", "expected_median"),
    [
        ([10_000, 20_000, 90_000], 20_000),
        ([10_000, 20_000, 30_000, 90_000], 25_000),
    ],
)
def test_three_or_four_samples_use_median_without_iqr(
    monkeypatch,
    prices,
    expected_median,
) -> None:
    service = OnlineMarketPriceService()
    percentile_calls: list[float] = []
    original_percentile = service._percentile

    def track_percentile(values, quantile):
        percentile_calls.append(quantile)
        return original_percentile(values, quantile)

    monkeypatch.setattr(service, "_percentile", track_percentile)
    candidates = [
        _evidence_candidate(price, index)
        for index, price in enumerate(prices, start=1)
    ]

    estimate = service._aggregate_candidates(
        candidates,
        MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.reference_mode == "median_low_sample"
    assert estimate.median_price == expected_median
    assert estimate.low_price == round(expected_median * 0.75)
    assert estimate.high_price == round(expected_median * 1.25)
    assert 0.25 not in percentile_calls
    assert 0.75 not in percentile_calls


def test_five_or_more_samples_use_linear_percentiles_and_iqr() -> None:
    service = OnlineMarketPriceService()
    candidates = [
        _evidence_candidate(price, index)
        for index, price in enumerate(
            [10_000, 11_000, 12_000, 13_000, 14_000, 15_000],
            start=1,
        )
    ]

    estimate = service._aggregate_candidates(
        candidates,
        MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.reference_mode == "iqr"
    assert estimate.median_price == 12_500
    assert estimate.low_price == 11_250
    assert estimate.high_price == 13_750
    assert estimate.sample_count == 6
    assert estimate.site_count == 6
    assert estimate.confidence == 1.0


def test_post_iqr_three_samples_downgrade_to_median_low_sample() -> None:
    service = OnlineMarketPriceService()
    candidates = [
        _evidence_candidate(price, index)
        for index, price in enumerate(
            [1, 10_000, 10_000, 10_000, 100_000],
            start=1,
        )
    ]

    estimate = service._aggregate_candidates(
        candidates,
        MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.reference_mode == "median_low_sample"
    assert estimate.sample_count == 3
    assert [candidate.price for candidate in estimate.candidates] == [
        10_000,
        10_000,
        10_000,
    ]


def test_post_iqr_below_policy_minimum_is_insufficient() -> None:
    service = OnlineMarketPriceService(
        policy=_policy(minimum_market_samples=4)
    )
    candidates = [
        _evidence_candidate(price, index)
        for index, price in enumerate(
            [1, 10_000, 10_000, 10_000, 100_000],
            start=1,
        )
    ]

    estimate = service._aggregate_candidates(
        candidates,
        MarketplaceCondition.NEW,
    )

    assert estimate.status == "insufficient"
    assert estimate.reference_mode == "median_low_sample"
    assert estimate.sample_count == 3


def test_fallback_searches_accumulate_candidates_until_policy_is_met(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    results = [
        _search_result(
            url=f"https://shop-{index}.example/item",
            product="Apple iPhone 15 256GB",
            price=30_000 + index * 1_000,
            condition_text="全新",
        )
        for index in range(1, 4)
    ]
    agent = _FakeResearchAgent(
        [
            _agent_result([result], conditions=["new"])
            for result in results
        ]
    )
    service = OnlineMarketPriceService(research_agent=agent)

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 台灣 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.sample_count == 3
    assert [
        call["allowed_tool_names"] for call in agent.calls
    ] == [
        ["search_market_prices_serpapi"],
        ["search_market_prices_tavily"],
        ["search_market_prices_ddgs"],
    ]


def test_rate_limit_returns_structured_not_found_result(caplog) -> None:
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
        estimate = service.estimate_price(
            "Apple iPhone 15 256GB 台灣 價格"
        )

    assert estimate.status == "not_found"
    assert estimate.median_price == 0
    assert agent.calls == [["search_market_prices_serpapi"]]
    assert "reason=groq_rate_limit retry_after=642.4s" in caplog.text
