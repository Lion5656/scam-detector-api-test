import hashlib
import json
from typing import Any

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
from backend.services.image_price_service.pricing import search_tools


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
    product: str,
    price: int,
    condition_text: str,
    snippet_prefix: str = "售價",
) -> dict:
    condition = (
        MarketplaceCondition.USED
        if "二手" in condition_text
        else MarketplaceCondition.NEW
    )
    return {
        "title": f"{product} {condition_text}".strip(),
        "snippet": f"{snippet_prefix} NT${price:,}",
        "_product": product,
        "_price": price,
        "_condition": condition,
    }


def _evidence_candidate(
    price: int,
    index: int,
    *,
    condition: MarketplaceCondition = MarketplaceCondition.NEW,
    source: str | None = None,
) -> MarketPriceCandidateEvidence:
    source_id = source or f"source-{index}"
    return MarketPriceCandidateEvidence(
        candidate_id=(
            f"{source_id}#{price}#{condition.value}#{index}"
        ),
        title=f"Apple iPhone 15 256GB {condition.value}",
        price=price,
        condition=condition,
        evidence=f"售價 NT${price:,}",
    )


def _policy(**updates) -> PriceRiskPolicy:
    return PriceRiskPolicy(
        **{
            **DEFAULT_PRICE_RISK_POLICY.model_dump(),
            **updates,
        }
    )


def _make_fake_search_fn(results: list[dict]) -> Any:
    """建立一個回傳固定結果的假搜尋函式。"""
    calls: list[dict] = []

    def fake_search(query: str, max_results: int = 10) -> list[dict]:
        calls.append({"query": query, "max_results": max_results})
        return results

    fake_search.calls = calls
    return fake_search


class _FakePriceExtractor:
    """線上市價服務測試用的已結構化 LLM 替身。"""

    def extract(
        self,
        search_results,
        condition,
        *,
        product_query,
        policy,
    ):
        target_product = product_query.removesuffix("價格").strip().casefold()
        candidates = []
        for index, result in enumerate(search_results):
            if str(result.get("_product", "")).casefold() != target_product:
                continue
            item_condition = result.get("_condition")
            if (
                condition is not MarketplaceCondition.UNKNOWN
                and item_condition is not condition
            ):
                continue
            price = int(result.get("_price", 0))
            if not 0 < price <= policy.maximum_supported_price:
                continue
            candidates.append(
                MarketPriceCandidateEvidence(
                    candidate_id=(
                        hashlib.sha256(
                            (
                                f"{result['title']}\0"
                                f"{result['snippet']}"
                            ).encode("utf-8")
                        ).hexdigest()[:16]
                        + f"#{price}#{item_condition.value}#{index}"
                    ),
                    title=result["title"],
                    price=price,
                    condition=item_condition,
                    evidence=result["snippet"],
                )
            )
        return candidates


def _make_service_with_fake_search(
    results_per_tool: dict[str, list[dict]],
    *,
    policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
) -> tuple[OnlineMarketPriceService, dict[str, Any]]:
    """建立使用假搜尋函式的 OnlineMarketPriceService。"""
    fakes = {}
    for tool_name, results in results_per_tool.items():
        fakes[tool_name] = _make_fake_search_fn(results)
    service = OnlineMarketPriceService(
        policy=policy,
        search_functions=fakes,
        price_extractor=_FakePriceExtractor(),
    )
    return service, fakes


def test_serpapi_search_tool_is_called_directly(monkeypatch) -> None:
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
                            "title": "Apple iPhone 15 256GB 全新",
                            "link": "https://shop.example/iphone",
                            "displayed_link": "shop.example › iphone",
                            "snippet": "特價 NT$30,500",
                            "extensions": ["額外欄位不應保留"],
                        },
                        {
                            "position": 2,
                            "title": "Apple iPhone 15 全新",
                            "link": "https://shop-two.example/iphone",
                            "snippet": "特價 NT$31,500",
                        },
                    ],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        search_tools.settings,
        "SERP_API_KEY",
        SecretStr("serp-test-key"),
    )
    monkeypatch.setattr(
        search_tools.serpapi,
        "Client",
        FakeSerpApiClient,
    )

    from backend.services.image_price_service.pricing.search_tools import (
        search_serpapi,
    )

    result = search_serpapi(
        "Apple iPhone 15 256GB 價格",
        8,
    )

    assert calls == [
        {
            "engine": "google_light",
            "q": "Apple iPhone 15 256GB 價格",
            "google_domain": "google.com.tw",
            "hl": "zh-tw",
            "gl": "tw",
        }
    ]
    assert len(result) == 2
    assert set(result[0]) == {"title", "snippet"}


def test_tavily_search_tool_calls_api_and_normalizes_results(
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
                        "title": "Apple iPhone 15 128GB 全新",
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
        search_tools.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    monkeypatch.setattr(
        search_tools.settings,
        "SEARCH_COUNTRY",
        "taiwan",
    )
    monkeypatch.setattr(
        search_tools.settings,
        "SEARCH_DOMAIN",
        ["shop.example"],
    )
    monkeypatch.setattr(
        search_tools.settings,
        "EXCLUDE_DOMAIN",
        ["example.invalid"],
    )
    monkeypatch.setattr(
        search_tools,
        "TavilyClient",
        FakeTavilyClient,
    )

    from backend.services.image_price_service.pricing.search_tools import (
        search_tavily,
    )

    result = search_tavily(
        "Apple iPhone 15 台灣 全新 價格",
        3,
    )

    assert calls[0]["query"] == "Apple iPhone 15 台灣 全新 價格"
    assert result == [
        {
            "title": "Apple iPhone 15 128GB 全新",
            "snippet": "全新售價 NT$21,890",
        },
        {
            "title": "缺少網址的結果",
            "snippet": "NT$20,000",
        },
    ]


def test_known_new_condition_only_accepts_new_prices() -> None:
    results = [
        _search_result(
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
    service, fakes = _make_service_with_fake_search(
        {"serpapi": results}
    )

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.reference_mode == "median_low_sample"
    assert estimate.median_price == 31_000
    assert {candidate.condition for candidate in estimate.candidates} == {
        MarketplaceCondition.NEW
    }
    assert fakes["serpapi"].calls[0]["query"] == "Apple iPhone 15 256GB 價格 全新"


def test_service_filters_results_for_other_products() -> None:
    results = [
        _search_result(
            product=product,
            price=price,
            condition_text="全新",
        )
        for index, (product, price) in enumerate(
            [
                ("Apple iPhone 15 Pro 256GB", 30_000),
                ("Apple iPhone 14 Pro 256GB", 28_000),
                ("Apple iPhone 15 Pro 128GB", 27_000),
                ("Apple iPhone 15 Pro Max 256GB", 35_000),
                ("Apple iPhone 15 Pro 256GB 手機殼", 999),
            ],
            start=1,
        )
    ]
    service, _ = _make_service_with_fake_search({"serpapi": results})

    estimate = service.estimate_price(
        "Apple iPhone 15 Pro 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert [candidate.price for candidate in estimate.candidates] == [30_000]


def test_known_used_condition_uses_simple_keyword_and_accepts_used_prices() -> None:
    results = [
        _search_result(
            product="Sony PS5",
            price=price,
            condition_text=condition_text,
        )
        for index, (price, condition_text) in enumerate(
            [
                (12_000, "二手・近全新"),
                (12_500, "二手"),
                (13_000, "二手 近全新"),
                (10_000, "二手・良好"),
                (15_000, "全新"),
            ],
            start=1,
        )
    ]
    service, fakes = _make_service_with_fake_search(
        {"serpapi": results}
    )

    estimate = service.estimate_price(
        "Sony PS5 價格",
        condition=MarketplaceCondition.USED,
        condition_text="二手・近全新",
    )

    assert estimate.status == "success"
    assert estimate.median_price == 12_250
    assert [candidate.price for candidate in estimate.candidates] == [
        12_000,
        12_500,
        13_000,
        10_000,
    ]
    assert fakes["serpapi"].calls[0]["query"] == "Sony PS5 價格 二手"


def test_unknown_condition_uses_one_combined_search_and_splits_estimates() -> None:
    new_results = [
        _search_result(
            product="Apple iPhone 15 256GB",
            price=30_000 + index * 500,
            condition_text="全新",
        )
        for index in range(1, 4)
    ]
    used_results = [
        _search_result(
            product="Apple iPhone 15 256GB",
            price=20_000 + index * 500,
            condition_text="二手",
        )
        for index in range(1, 4)
    ]
    service, fakes = _make_service_with_fake_search(
        {"serpapi": [*new_results, *used_results]}
    )

    estimates = service.estimate_prices(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.UNKNOWN,
    )

    assert tuple(estimate.condition for estimate in estimates) == (
        MarketplaceCondition.NEW,
        MarketplaceCondition.USED,
    )
    assert all(estimate.status == "success" for estimate in estimates)
    assert len(fakes["serpapi"].calls) == 1
    assert (
        fakes["serpapi"].calls[0]["query"]
        == "Apple iPhone 15 256GB 價格 全新 二手"
    )


def test_out_of_range_candidate_is_removed() -> None:
    """超出 policy 上限的價格和零價格被排除。"""
    results = [
        _search_result(
            product="Apple iPhone 15 256GB",
            price=price,
            condition_text="全新",
        )
        for index, price in enumerate(
            [30_000, 31_000],
            start=1,
        )
    ]
    service, _ = _make_service_with_fake_search({"serpapi": results})

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert [candidate.price for candidate in estimate.candidates] == [30_000, 31_000]
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
    service = OnlineMarketPriceService(
        policy=_policy(minimum_market_sites=2)
    )
    candidates = [
        _evidence_candidate(30_000 + index * 500, index, source="same-source")
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
            product="Apple iPhone 15 256GB",
            price=30_000 + index * 1_000,
            condition_text="全新",
        )
        for index in range(1, 4)
    ]
    # 每個搜尋工具回傳一筆
    service, fakes = _make_service_with_fake_search(
        {
            "serpapi": [results[0]],
            "ddgs": [results[1]],
            "tavily": [results[2]],
        }
    )

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.sample_count == 3
    assert estimate.search_tools == ["serp_api", "tavily", "ddgs"]
    # 確認所有三個搜尋工具都被呼叫
    assert len(fakes["serpapi"].calls) == 1
    assert len(fakes["ddgs"].calls) == 1
    assert len(fakes["tavily"].calls) == 1


def test_duplicate_candidates_across_search_tools_are_preserved(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        online_marketprice_service.settings,
        "TAVILY_SEARCH_API_KEY",
        SecretStr("tavily-test-key"),
    )
    duplicate = _search_result(
        product="Apple iPhone 15 256GB",
        price=30_000,
        condition_text="全新",
    )
    service, _ = _make_service_with_fake_search(
        {
            "serpapi": [duplicate, duplicate],
            "tavily": [duplicate],
        }
    )

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.sample_count == 3
    assert [candidate.price for candidate in estimate.candidates] == [
        30_000,
        30_000,
        30_000,
    ]
    assert estimate.site_count == 1
    assert estimate.search_tools == ["serp_api", "tavily"]


def test_search_tool_error_continues_to_next_tool() -> None:
    """搜尋工具失敗時應繼續嘗試下一個工具。"""

    def failing_search(query: str, max_results: int = 10) -> list[dict]:
        raise RuntimeError("API error")

    results = [
        _search_result(
            product="Apple iPhone 15 256GB",
            price=30_000 + index * 1_000,
            condition_text="全新",
        )
        for index in range(1, 4)
    ]

    service = OnlineMarketPriceService(
        search_functions={
            "serpapi": failing_search,
            "ddgs": _make_fake_search_fn(results),
        },
        price_extractor=_FakePriceExtractor(),
    )

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "success"
    assert estimate.sample_count == 3
    assert estimate.search_tools == ["serp_api", "ddgs"]


def test_no_results_returns_not_found() -> None:
    """所有搜尋工具都沒有結果時回傳 not_found。"""
    service, _ = _make_service_with_fake_search(
        {"serpapi": [], "ddgs": []}
    )

    estimate = service.estimate_price(
        "Apple iPhone 15 256GB 價格",
        condition=MarketplaceCondition.NEW,
    )

    assert estimate.status == "not_found"
    assert estimate.median_price == 0
