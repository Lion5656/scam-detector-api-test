"""LLM 搜尋結果價格整理器的單元測試。"""

from typing import Any

from backend.services.image_price_service.domain.models import \
    MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY, PriceRiskPolicy)
from backend.services.image_price_service.pricing.search_result_price_extractor import (
    ExtractedSearchPrice, SearchPriceExtraction, SearchResultPriceExtractor,
    _MAX_LLM_TEXT_CHARS, _MAX_RESULTS_FOR_LLM, _prepare_llm_results,
    _sanitize_llm_text, _strict_response_format,
    extract_prices_from_search_results)


class FakeStructuredLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[Any] = []

    def invoke(self, prompt: Any) -> Any:
        self.calls.append(prompt)
        return self.response


def _extractor(
    response: SearchPriceExtraction | dict[str, Any],
) -> tuple[SearchResultPriceExtractor, FakeStructuredLLM]:
    llm = FakeStructuredLLM(response)
    return (
        SearchResultPriceExtractor(
            api_key="",
            model_name="test-model",
            structured_llm=llm,
        ),
        llm,
    )


def test_irregular_multi_item_snippet_is_structured_in_one_llm_call() -> None:
    evidence = "超戰士列傳景品七龍珠海賊王航海王二手拆擺七支合售. $840"
    results = [
        {
            "title": "二手動漫景品",
            "snippet": (
                f"{evidence} ; 二手拆擺《七龍珠》一番賞經典對戰組合"
                "代理版A賞超級賽亞人悟飯. $500 ; 七龍珠正版超造集天津飯"
                "與餃子已絕版僅 ..."
            ),
        }
    ]
    extractor, llm = _extractor(
        SearchPriceExtraction(
            candidates=[
                ExtractedSearchPrice(
                    result_index=0,
                    price=840,
                    condition=MarketplaceCondition.USED,
                    evidence=evidence,
                )
            ]
        )
    )

    candidates = extract_prices_from_search_results(
        results,
        MarketplaceCondition.USED,
        product_query="超戰士列傳景品 七支合售",
        extractor=extractor,
    )

    assert len(llm.calls) == 1
    assert [candidate.price for candidate in candidates] == [840]
    assert candidates[0].evidence == evidence
    assert llm.calls[0][0][0] == "system"
    assert "價格位於固定位置" in llm.calls[0][0][1]
    assert '"candidates"' in llm.calls[0][0][1]
    assert llm.calls[0][1][0] == "human"
    assert "$500" in llm.calls[0][1][1]


def test_llm_can_return_multiple_prices_from_different_results() -> None:
    results = [
        {
            "title": "Apple iPhone 15 256GB 全新",
            "snippet": "限時供應，現在 NT$30,500",
        },
        {
            "title": "全新現貨",
            "snippet": "NT$31,000 Apple iPhone 15 256GB",
        },
    ]
    extractor, _ = _extractor(
        {
            "candidates": [
                {
                    "result_index": 0,
                    "price": 30_500,
                    "condition": "new",
                    "evidence": "Apple iPhone 15 256GB 全新",
                },
                {
                    "result_index": 1,
                    "price": 31_000,
                    "condition": "new",
                    "evidence": "NT$31,000 Apple iPhone 15 256GB",
                },
            ]
        }
    )

    candidates = extractor.extract(
        results,
        MarketplaceCondition.NEW,
        product_query="Apple iPhone 15 256GB",
    )

    assert [candidate.price for candidate in candidates] == [30_500, 31_000]


def test_duplicate_prices_from_same_result_are_preserved() -> None:
    evidence = "二手七龍珠公仔售價 NT$500"
    results = [
        {
            "title": "二手七龍珠公仔",
            "snippet": evidence,
        }
    ]
    extractor, _ = _extractor(
        {
            "candidates": [
                {
                    "result_index": 0,
                    "price": 500,
                    "condition": "used",
                    "evidence": evidence,
                },
                {
                    "result_index": 0,
                    "price": 500,
                    "condition": "used",
                    "evidence": evidence,
                },
            ]
        }
    )

    candidates = extractor.extract(
        results,
        MarketplaceCondition.USED,
        product_query="七龍珠公仔",
    )

    assert [candidate.price for candidate in candidates] == [500, 500]
    assert len({candidate.candidate_id for candidate in candidates}) == 2


def test_code_rejects_hallucinated_evidence_wrong_condition_and_bad_index() -> None:
    results = [
        {
            "title": "Sony PS5 二手",
            "snippet": "售價為 12,000 元",
        }
    ]
    extractor, _ = _extractor(
        {
            "candidates": [
                {
                    "result_index": 0,
                    "price": 12_000,
                    "condition": "new",
                    "evidence": "Sony PS5 二手",
                },
                {
                    "result_index": 0,
                    "price": 12_000,
                    "condition": "used",
                    "evidence": "不存在的原文",
                },
                {
                    "result_index": 9,
                    "price": 12_000,
                    "condition": "used",
                    "evidence": "Sony PS5 二手",
                },
            ]
        }
    )

    candidates = extractor.extract(
        results,
        MarketplaceCondition.USED,
        product_query="Sony PS5",
    )

    assert candidates == []


def test_code_keeps_policy_boundary_and_rejects_larger_price() -> None:
    policy = DEFAULT_PRICE_RISK_POLICY
    maximum = policy.maximum_supported_price
    results = [
        {
            "title": "目標商品 全新",
            "snippet": f"價格 {maximum} 元",
        },
        {
            "title": "目標商品 全新",
            "snippet": f"價格 {maximum + 1} 元",
        },
    ]
    extractor, _ = _extractor(
        {
            "candidates": [
                {
                    "result_index": 0,
                    "price": maximum,
                    "condition": "new",
                    "evidence": f"價格 {maximum} 元",
                },
                {
                    "result_index": 1,
                    "price": maximum + 1,
                    "condition": "new",
                    "evidence": f"價格 {maximum + 1} 元",
                },
            ]
        }
    )

    candidates = extractor.extract(
        results,
        MarketplaceCondition.NEW,
        product_query="目標商品",
        policy=policy,
    )

    assert [candidate.price for candidate in candidates] == [maximum]


def test_results_do_not_need_urls_to_be_sent_to_llm() -> None:
    extractor, llm = _extractor({"candidates": []})

    candidates = extractor.extract(
        [
            {
                "title": "商品",
                "snippet": "500 元",
            }
        ],
        MarketplaceCondition.NEW,
        product_query="商品",
    )

    assert candidates == []
    assert len(llm.calls) == 1


def test_prompt_uses_system_role_and_defines_empty_output() -> None:
    extractor, llm = _extractor({"candidates": []})

    extractor.extract(
        [
            {
                "title": "Apple iPhone 15 256GB 二手",
                "snippet": "售價 NT$20,000",
            }
        ],
        MarketplaceCondition.USED,
        product_query="Apple iPhone 15 256GB",
    )

    messages = llm.calls[0]
    assert messages[0][0] == "system"
    assert "json" in messages[0][1]
    assert '{"candidates": []}' in messages[0][1]
    assert "只負責從每筆 title 與 snippet 擷取" in messages[0][1]
    assert messages[1][0] == "human"
    assert '"results"' in messages[1][1]
    assert '"target_product"' not in messages[1][1]


def test_strict_response_format_requires_schema_compliance() -> None:
    response_format = _strict_response_format()

    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["name"] == "SearchPriceExtraction"
    assert json_schema["schema"]["required"] == ["candidates"]
    assert json_schema["schema"]["additionalProperties"] is False
    candidate_schema = json_schema["schema"]["properties"]["candidates"]["items"]
    assert candidate_schema["additionalProperties"] is False
    assert candidate_schema["required"] == [
        "result_index",
        "price",
        "condition",
        "evidence",
    ]


def test_llm_search_results_respect_request_size_budget() -> None:
    results = [
        {
            "title": f"商品 {index} " + "標" * 300,
            "snippet": "價格 NT$20,000 " + "說明" * 1_000,
        }
        for index in range(15)
    ]

    prepared = _prepare_llm_results(results)

    assert len(prepared) == _MAX_RESULTS_FOR_LLM
    assert all(set(result) == {"title", "snippet"} for result in prepared)
    assert (
        sum(
            len(str(result["title"])) + len(str(result["snippet"]))
            for result in prepared
        )
        <= _MAX_LLM_TEXT_CHARS
    )


def test_sanitize_llm_text_removes_invisible_and_decorative_characters() -> None:
    raw = "🔥\u200b尼卡\u200c魯夫✨ NT$８４０\u200d｜全新★\ufeff"

    assert _sanitize_llm_text(raw) == "尼卡魯夫 NT$840 全新"


def test_only_sanitized_search_text_is_sent_to_llm() -> None:
    extractor, llm = _extractor({"candidates": []})

    extractor.extract(
        [
            {
                "title": "🔥尼卡\u200b魯夫✨",
                "snippet": "售價 NT$８４０★",
            }
        ],
        MarketplaceCondition.NEW,
        product_query="尼卡魯夫",
    )

    assert len(llm.calls) == 1
    human_prompt = llm.calls[0][1][1]
    assert "🔥" not in human_prompt
    assert "✨" not in human_prompt
    assert "★" not in human_prompt
    assert "\u200b" not in human_prompt
    assert "尼卡魯夫" in human_prompt
    assert "NT$840" in human_prompt
