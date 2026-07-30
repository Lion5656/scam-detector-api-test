import inspect
from enum import Enum
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from backend.services.dto.price_analysis import MarketPriceEstimate
from backend.services.image_price_service.domain.models import \
    MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY, PriceRiskPolicy)
from backend.services.image_price_service.image_price_analyzer import (
    default_decision_engine, default_online_price_service)
from backend.services.image_price_service.pricing.online_marketprice_service import \
    OnlineMarketPriceService
from backend.services.image_price_service.risk.fusion_decision_engine import \
    FusionDecisionEngine


def _policy_data(**updates):
    payload = DEFAULT_PRICE_RISK_POLICY.model_dump()
    payload.update(updates)
    return payload


def test_default_price_risk_policy_matches_documented_thresholds():
    policy = DEFAULT_PRICE_RISK_POLICY

    assert policy.underprice_relative_bands == (
        (0.10, 20),
        (0.20, 35),
        (0.35, 55),
        (0.50, 75),
        (float("inf"), 95),
    )
    assert policy.overprice_relative_bands == (
        (0.15, 20),
        (0.30, 35),
        (0.60, 55),
        (1.00, 70),
        (float("inf"), 95),
    )
    assert policy.absolute_gap_bands == (
        (499, 5),
        (1_999, 10),
        (4_999, 20),
        (9_999, 30),
        (100_000, 50),
    )
    assert policy.model_dump(
        exclude={
            "underprice_relative_bands",
            "overprice_relative_bands",
            "absolute_gap_bands",
        }
    ) == {
        "maximum_supported_price": 100_000,
        "low_score_max": 39,
        "medium_score_max": 79,
        "maximum_score": 100,
        "overprice_score_cap": 80,
        "minimum_market_confidence": 0.60,
        "minimum_market_samples": 3,
        "minimum_market_sites": 1,
        "minimum_iqr_samples": 5,
        "small_sample_relative_tolerance": 0.25,
        "small_sample_score_cap": 79,
        "llm_review_min_confidence": 0.80,
        "condition_llm_correction_max_confidence": 0.80,
    }


def test_price_risk_policy_is_frozen_and_forbids_extra_fields():
    assert issubclass(PriceRiskPolicy, BaseModel)
    assert PriceRiskPolicy.model_config["frozen"] is True
    assert PriceRiskPolicy.model_config["extra"] == "forbid"

    with pytest.raises(ValidationError):
        DEFAULT_PRICE_RISK_POLICY.low_score_max = 20

    with pytest.raises(ValidationError):
        PriceRiskPolicy(**_policy_data(enable_fallback=True))


def test_price_risk_policy_has_no_behavior_switch_or_enum_fields():
    forbidden_names = {
        "enable_dual_path",
        "enable_fallback",
        "enable_llm",
        "fallback_strategy",
        "merge_strategy",
    }

    assert forbidden_names.isdisjoint(PriceRiskPolicy.model_fields)
    for field in PriceRiskPolicy.model_fields.values():
        annotation = field.annotation
        assert annotation is not bool
        assert not (
            isinstance(annotation, type)
            and issubclass(annotation, Enum)
        )
        assert get_origin(annotation) is not None or annotation in {int, float}
        assert bool not in get_args(annotation)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "underprice_relative_bands": (
                (0.20, 10),
                (0.10, 25),
                (float("inf"), 85),
            )
        },
        {
            "overprice_relative_bands": (
                (0.30, 10),
                (0.15, 20),
                (float("inf"), 70),
            )
        },
        {"absolute_gap_bands": ((1_999, 5), (499, 0), (100_000, 30))},
        {"absolute_gap_bands": ((499, 0), (99_999, 30))},
        {
            "underprice_relative_bands": (
                (0.10, 101),
                (float("inf"), 85),
            )
        },
        {"maximum_score": 101},
        {"low_score_max": 79},
        {"medium_score_max": 100},
        {"maximum_supported_price": 0},
        {"minimum_iqr_samples": 4},
        {"minimum_iqr_samples": 5, "minimum_market_samples": 6},
        {"small_sample_score_cap": 80},
        {"llm_review_min_confidence": 1.01},
        {"condition_llm_correction_max_confidence": -0.01},
    ],
)
def test_price_risk_policy_rejects_invalid_thresholds(updates):
    with pytest.raises(ValidationError):
        PriceRiskPolicy(**_policy_data(**updates))


def test_market_service_and_decision_engine_share_injected_policy():
    policy = PriceRiskPolicy(
        **_policy_data(
            minimum_market_sites=2,
            minimum_market_samples=4,
            minimum_iqr_samples=6,
        )
    )
    market_service = OnlineMarketPriceService(policy=policy)
    decision_engine = FusionDecisionEngine(policy=policy)

    assert market_service.policy is policy
    assert decision_engine.policy is policy


def test_application_assembly_and_fallback_use_the_shared_policy():
    assert (
        default_online_price_service.policy
        is default_decision_engine.policy
        is DEFAULT_PRICE_RISK_POLICY
    )

    policy = PriceRiskPolicy(
        **_policy_data(
            low_score_max=50,
        )
    )
    market_service = OnlineMarketPriceService(policy=policy)
    decision_engine = FusionDecisionEngine(policy=policy)
    fallback = decision_engine._alt_deep_result(
        {
            "risk_score": 55.0,
            "risk_label": "MEDIUM",
            "reason": "價格規則結果",
            "evidence": [],
            "confidence": 0.9,
        },
    )

    assert market_service.policy is decision_engine.policy is policy
    assert fallback["risk_label"] == "MEDIUM"


def test_custom_policy_changes_price_boundary_label_and_deep_analysis(
    monkeypatch,
):
    policy = PriceRiskPolicy(
        **_policy_data(
            low_score_max=0,
            medium_score_max=10,
            overprice_score_cap=10,
            small_sample_score_cap=10,
        )
    )
    default_calls: list[dict] = []
    custom_calls: list[dict] = []
    default_engine = FusionDecisionEngine(
        condition_reviewer=lambda context: default_calls.append(context),
    )
    custom_engine = FusionDecisionEngine(
        policy=policy,
        condition_reviewer=lambda context: custom_calls.append(context),
    )
    estimate = MarketPriceEstimate(
        status="success",
        condition=MarketplaceCondition.NEW,
        reference_mode="iqr",
        median_price=100,
        low_price=100,
        high_price=100,
        sample_count=5,
        site_count=3,
        source="online",
        confidence=0.9,
    )
    arguments = {
        "product_name": "測試商品",
        "selling_price": 90,
        "market_estimates": (estimate,),
        "condition": MarketplaceCondition.NEW,
        "condition_detail": "全新",
        "condition_source_text": "狀況 全新",
        "condition_extraction_confidence": 0.8,
    }

    default_result = default_engine.evaluate(**arguments)
    custom_result = custom_engine.evaluate(**arguments)

    assert default_result["risk_label"] == "LOW"
    assert default_result["decision_layer"] == "fast"
    assert default_calls == []
    assert custom_result["risk_label"] == "HIGH"
    assert custom_result["decision_layer"] == "llm_simulated"
    assert len(custom_calls) == 1


def test_custom_policy_changes_supported_market_price_boundary():
    policy = PriceRiskPolicy(
        **_policy_data(
            absolute_gap_bands=((499, 0), (500, 30)),
            maximum_supported_price=500,
        )
    )
    from backend.services.image_price_service.domain.models import \
        MarketplaceCondition
    from backend.services.image_price_service.pricing.search_result_price_extractor import (
        ExtractedSearchPrice, SearchPriceExtraction,
        SearchResultPriceExtractor, extract_prices_from_search_results)

    class FakeStructuredLLM:
        def __init__(self, price):
            self.price = price

        def invoke(self, prompt):
            return SearchPriceExtraction(
                candidates=[
                    ExtractedSearchPrice(
                        result_index=0,
                        price=self.price,
                        condition=MarketplaceCondition.NEW,
                        evidence=f"售價 NT${self.price}",
                    )
                ]
            )

    def extractor_for(price):
        return SearchResultPriceExtractor(
            api_key="",
            model_name="test",
            structured_llm=FakeStructuredLLM(price),
        )

    within_range = extract_prices_from_search_results(
        [{"title": "商品 全新", "snippet": "售價 NT$500"}],
        MarketplaceCondition.NEW,
        product_query="商品",
        policy=policy,
        extractor=extractor_for(500),
    )
    out_of_range = extract_prices_from_search_results(
        [{"title": "商品 全新", "snippet": "售價 NT$501"}],
        MarketplaceCondition.NEW,
        product_query="商品",
        policy=policy,
        extractor=extractor_for(501),
    )
    assert len(within_range) == 1
    assert within_range[0].price == 500
    assert len(out_of_range) == 0


def test_decision_engine_has_no_legacy_magic_price_thresholds():
    source = inspect.getsource(FusionDecisionEngine)

    for legacy_threshold in (
        "market_price * 0.5",
        "market_price * 2",
        "90.0",
        "score >= 80",
        "score >= 40",
    ):
        assert legacy_threshold not in source
