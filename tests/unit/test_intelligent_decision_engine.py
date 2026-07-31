import inspect

import pytest

from backend.config import Settings
from backend.services.dto.price_analysis import (
    DeepAnalysisReview,
    MarketPriceEstimate,
)
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.risk.fusion_decision_engine import (
    FusionDecisionEngine,
)


def _estimate(
    condition: MarketplaceCondition,
    *,
    low: int,
    median: int,
    high: int,
    status: str = "success",
    reference_mode: str = "iqr",
    confidence: float = 0.9,
    sample_count: int = 5,
    site_count: int = 3,
    source: str = "online",
) -> MarketPriceEstimate:
    return MarketPriceEstimate(
        status=status,
        condition=condition,
        reference_mode=reference_mode,
        median_price=median,
        low_price=low,
        high_price=high,
        sample_count=sample_count,
        site_count=site_count,
        source=source,
        confidence=confidence,
        candidates=(),
    )


def _new_estimate(
    *,
    low: int = 25_000,
    median: int = 27_500,
    high: int = 30_000,
    **overrides,
) -> MarketPriceEstimate:
    return _estimate(
        MarketplaceCondition.NEW,
        low=low,
        median=median,
        high=high,
        **overrides,
    )


def _used_estimate(
    *,
    low: int = 15_000,
    median: int = 17_500,
    high: int = 20_000,
    **overrides,
) -> MarketPriceEstimate:
    return _estimate(
        MarketplaceCondition.USED,
        low=low,
        median=median,
        high=high,
        **overrides,
    )


def _unknown_estimate(
    *,
    low: int = 15_000,
    median: int = 22_500,
    high: int = 30_000,
    sample_count: int = 6,
    **overrides,
) -> MarketPriceEstimate:
    return _estimate(
        MarketplaceCondition.UNKNOWN,
        low=low,
        median=median,
        high=high,
        sample_count=sample_count,
        **overrides,
    )


def _evaluate(engine: FusionDecisionEngine, **overrides):
    values = {
        "product_name": "Apple iPhone 15 256GB",
        "selling_price": 27_000,
        "market_estimates": (_new_estimate(),),
        "condition": MarketplaceCondition.NEW,
        "condition_detail": "全新",
        "condition_source_text": "狀況 全新",
        "condition_extraction_confidence": 0.97,
    }
    values.update(overrides)
    return engine.evaluate(**values)


def test_evaluate_integration_known_condition_without_deep_review():
    reviewer_calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: reviewer_calls.append(context)
    )

    result = _evaluate(
        engine,
        condition=MarketplaceCondition.NEW,
        condition_extraction_confidence=0.97,
        market_estimates=(_new_estimate(),),
    )

    assert result["risk_label"] == "LOW"
    assert result["risk_score"] == 0.0
    assert result["condition"] is MarketplaceCondition.NEW
    assert result["decision_layer"] == "fast"
    assert reviewer_calls == []


def test_evaluate_integration_unknown_condition_without_reviewable_evidence():
    reviewer_calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: reviewer_calls.append(context)
    )

    result = _evaluate(
        engine,
        selling_price=18_000,
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
        condition_extraction_confidence=0.0,
        text="",
        market_estimates=(_unknown_estimate(),),
    )

    assert result["risk_label"] == "LOW"
    assert result["condition"] is MarketplaceCondition.UNKNOWN
    assert result["decision_layer"] == "fast"
    assert reviewer_calls == []
    assert "15000～30000" in result["reason"]


def test_evaluate_integration_deep_review_confirms_known_condition():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.NEW,
        condition_detail="全新",
        condition_evidence="狀況 全新",
        reason="原文已明確標示為全新",
        review_confidence=0.9,
    )
    reviewer_calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: (
            reviewer_calls.append(context) or review
        )
    )
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    result = _evaluate(
        engine,
        selling_price=90_000,
        condition=MarketplaceCondition.NEW,
        condition_detail="全新",
        condition_source_text="狀況 全新",
        condition_extraction_confidence=0.8,
        market_estimates=(estimate,),
        reprice=lambda *_: pytest.fail("品況未修正時不應重新查價"),
    )

    assert result["risk_label"] == "MEDIUM"
    assert result["risk_score"] == 70.0
    assert result["condition"] is MarketplaceCondition.NEW
    assert result["condition_detail"] == "全新"
    assert result["decision_layer"] == "llm"
    assert len(reviewer_calls) == 1
    assert "狀況 全新" in result["evidence"]


def test_evaluate_integration_deep_review_corrects_unknown_condition():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="良好",
        condition_evidence="二手・良好",
        reason="原文明確標示為二手良好",
        review_confidence=0.9,
    )
    reviewer_calls: list[dict[str, object]] = []
    reprice_calls: list[tuple[MarketplaceCondition, str]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: (
            reviewer_calls.append(context) or review
        )
    )

    def reprice(
        condition: MarketplaceCondition,
        condition_detail: str,
    ) -> tuple[MarketPriceEstimate, ...]:
        reprice_calls.append((condition, condition_detail))
        return (
            _used_estimate(
                low=9_000,
                median=10_500,
                high=12_000,
            ),
        )

    result = _evaluate(
        engine,
        selling_price=10_000,
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="商品狀況：二手・良好",
        condition_extraction_confidence=0.0,
        market_estimates=(_unknown_estimate(),),
        reprice=reprice,
    )

    assert result["risk_label"] == "LOW"
    assert result["risk_score"] == 0.0
    assert result["condition"] is MarketplaceCondition.USED
    assert result["condition_detail"] == "良好"
    assert result["decision_layer"] == "llm"
    assert len(reviewer_calls) == 1
    assert reprice_calls == [(MarketplaceCondition.USED, "良好")]


def test_market_interval_returns_fast_low():
    result = _evaluate(FusionDecisionEngine())

    assert result["risk_label"] == "LOW"
    assert result["risk_score"] == 0.0
    assert result["decision_layer"] == "fast"
    assert result["market_price_source"] == "online"


def test_success_result_explains_interval_gaps_and_condition():
    result = _evaluate(FusionDecisionEngine())

    assert "25000～30000 市場區間" in result["reason"]
    assert "相對差距 0.00%" in result["reason"]
    assert "絕對差額 0" in result["reason"]
    assert any("new 市場區間" in item for item in result["evidence"])


def test_unknown_result_explains_combined_market_interval():
    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=18_000,
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
        condition_extraction_confidence=0.0,
        market_estimates=(_unknown_estimate(),),
    )

    assert result["risk_label"] == "LOW"
    assert "相對差距" in result["reason"]
    assert "絕對差額" in result["reason"]
    assert any("unknown 市場區間" in item for item in result["evidence"])


def test_underprice_gap_uses_violated_low_boundary_not_median():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=10_000,
        median=15_000,
        high=20_000,
    )

    components = engine._price_components(5_000, estimate)

    assert components["boundary"] == 10_000
    assert components["relative_gap"] == 0.5
    assert components["relative_score"] == 75
    assert components["absolute_bonus"] == 30
    assert components["score"] == 100.0


def test_overprice_gap_uses_violated_high_boundary_not_median():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=8_000,
        median=9_000,
        high=10_000,
    )

    components = engine._price_components(15_000, estimate)

    assert components["boundary"] == 10_000
    assert components["relative_gap"] == 0.5
    assert components["relative_score"] == 55
    assert components["absolute_bonus"] == 30
    assert components["score"] == 80.0


@pytest.mark.parametrize(
    ("relative_gap", "expected_score"),
    [
        (0.10, 20),
        (0.20, 35),
        (0.35, 55),
        (0.50, 75),
    ],
)
def test_underprice_relative_score_inclusive_boundaries(
    relative_gap,
    expected_score,
):
    engine = FusionDecisionEngine()
    boundary = 100_000
    estimate = _new_estimate(
        low=boundary,
        median=boundary,
        high=boundary,
    )
    listed_price = round(boundary * (1 - relative_gap))

    components = engine._price_components(listed_price, estimate)

    assert components["relative_gap"] == pytest.approx(relative_gap)
    assert components["relative_score"] == expected_score


@pytest.mark.parametrize(
    ("boundary", "relative_gap", "expected_score"),
    [
        (50_000, 0.15, 20),
        (50_000, 0.30, 35),
        (50_000, 0.60, 55),
        (40_000, 1.00, 70),
    ],
)
def test_overprice_relative_score_inclusive_boundaries(
    boundary,
    relative_gap,
    expected_score,
):
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=boundary,
        median=boundary,
        high=boundary,
    )
    listed_price = round(boundary * (1 + relative_gap))

    components = engine._price_components(listed_price, estimate)

    assert components["relative_gap"] == pytest.approx(relative_gap)
    assert components["relative_score"] == expected_score


@pytest.mark.parametrize(
    ("absolute_gap", "expected_bonus"),
    [
        (500, 10),
        (2_000, 20),
        (5_000, 30),
        (10_000, 50),
    ],
)
def test_absolute_gap_inclusive_lower_boundaries(
    absolute_gap,
    expected_bonus,
):
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    components = engine._price_components(
        100_000 - absolute_gap,
        estimate,
    )

    assert components["absolute_gap"] == absolute_gap
    assert components["absolute_bonus"] == expected_bonus


def test_listing_maximum_is_supported_and_next_value_is_decision_error():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=90_000,
        median=95_000,
        high=100_000,
    )

    supported = _evaluate(
        engine,
        selling_price=100_000,
        market_estimates=(estimate,),
    )
    unsupported = _evaluate(
        engine,
        selling_price=100_001,
        market_estimates=(estimate,),
    )

    assert supported["risk_label"] == "LOW"
    assert unsupported["risk_label"] == "UNKNOWN"
    assert unsupported["decision_layer"] == "decision_error"
    assert unsupported["error_code"] == "PRICE_OUT_OF_SUPPORTED_RANGE"


def test_100000_boundary_and_90000_listing_scores_exactly_medium():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    components = engine._price_components(90_000, estimate)
    result = _evaluate(
        engine,
        selling_price=90_000,
        market_estimates=(estimate,),
    )

    assert components["relative_score"] == 20
    assert components["absolute_bonus"] == 50
    assert components["score"] == 70.0
    assert result["risk_score"] == 70.0
    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "fast"


def test_overprice_pure_price_result_can_reach_high():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=9_000,
        median=9_500,
        high=10_000,
    )

    result = _evaluate(
        engine,
        selling_price=100_000,
        market_estimates=(estimate,),
    )

    assert result["risk_score"] == engine.policy.overprice_score_cap
    assert result["risk_label"] == "HIGH"


def test_small_sample_price_score_uses_policy_cap():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=10_000,
        median=15_000,
        high=20_000,
        reference_mode="median_low_sample",
        confidence=0.6,
        sample_count=3,
    )

    result = _evaluate(
        engine,
        selling_price=1,
        market_estimates=(estimate,),
    )

    assert result["risk_score"] == engine.policy.small_sample_score_cap
    assert result["risk_label"] == "MEDIUM"


def test_unknown_condition_combined_interval_is_low():
    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=18_000,
        market_estimates=(_unknown_estimate(),),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
        condition_extraction_confidence=0.0,
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "fast"


def test_unknown_condition_combined_interval_can_return_high():
    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=5_000,
        market_estimates=(_unknown_estimate(),),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
        condition_extraction_confidence=0.0,
    )

    assert result["risk_label"] == "HIGH"
    assert result["decision_layer"] == "fast"


def test_unknown_condition_uses_combined_sample_count():
    combined_estimate = _unknown_estimate(
        confidence=0.8,
        sample_count=4,
        site_count=2,
    )

    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=18_000,
        market_estimates=(combined_estimate,),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "fast"


def test_unknown_condition_requires_more_than_three_combined_samples():
    combined_estimate = _unknown_estimate(
        confidence=0.6,
        sample_count=3,
        site_count=2,
    )

    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=18_000,
        market_estimates=(combined_estimate,),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="",
    )

    assert result["risk_label"] == "UNKNOWN"
    assert result["decision_layer"] == "decision_error"
    assert result["error_code"] == "MARKET_PRICE_INSUFFICIENT"


def test_risk_score_is_monotonic_as_underprice_gap_increases():
    engine = FusionDecisionEngine()
    estimate = _new_estimate(
        low=20_000,
        median=22_500,
        high=25_000,
    )
    scores = [
        engine._calculate_price_score(price, estimate)
        for price in [19_000, 17_000, 14_000, 10_000, 1]
    ]

    assert scores == sorted(scores)
    assert all(0 <= score <= engine.policy.maximum_score for score in scores)


def test_low_base_score_never_calls_llm():
    calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: calls.append(context)
    )

    result = _evaluate(
        engine,
        condition_extraction_confidence=0.1,
        condition_source_text="狀況可能為二手",
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "fast"
    assert calls == []


def test_medium_base_with_high_confidence_condition_does_not_call_llm():
    calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: calls.append(context)
    )
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    result = _evaluate(
        engine,
        selling_price=90_000,
        market_estimates=(estimate,),
        condition_extraction_confidence=0.97,
    )

    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "fast"
    assert calls == []


@pytest.mark.parametrize(
    ("condition", "confidence", "has_conflict"),
    [
        (MarketplaceCondition.UNKNOWN, 0.97, False),
        (MarketplaceCondition.NEW, 0.8, False),
        (MarketplaceCondition.NEW, 0.97, True),
    ],
)
def test_medium_or_high_with_reviewable_condition_evidence_calls_llm(
    condition,
    confidence,
    has_conflict,
):
    calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: calls.append(context)
    )
    estimates = (
        (_unknown_estimate(),)
        if condition == MarketplaceCondition.UNKNOWN
        else (_new_estimate(),)
    )

    result = _evaluate(
        engine,
        selling_price=10_000,
        market_estimates=estimates,
        condition=condition,
        condition_detail="",
        condition_source_text="狀況可能為二手",
        condition_extraction_confidence=confidence,
        condition_has_conflict=has_conflict,
    )

    assert result["decision_layer"] == "llm_simulated"
    assert len(calls) == 1
    assert set(calls[0]) == {
        "product_name",
        "text",
        "condition",
        "condition_detail",
        "condition_source_text",
        "condition_extraction_confidence",
    }


def test_condition_review_can_use_listing_text_and_detail_together():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="近全新",
        condition_evidence="二手・近全新",
        reason="完整文字已標示二手品況",
        review_confidence=0.9,
    )
    contexts: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: (
            contexts.append(context) or review
        )
    )

    result = _evaluate(
        engine,
        selling_price=10_000,
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="近全新",
        condition_source_text="",
        condition_extraction_confidence=0.0,
        text="商品說明：二手・近全新，功能正常",
        market_estimates=(_unknown_estimate(),),
        reprice=lambda *_: (_used_estimate(),),
    )

    assert result["condition"] is MarketplaceCondition.USED
    assert result["condition_detail"] == "近全新"
    assert contexts[0]["text"] == "商品說明：二手・近全新，功能正常"
    assert contexts[0]["condition_detail"] == "近全新"


def test_missing_condition_source_never_calls_llm_and_keeps_combined_result():
    calls: list[dict[str, object]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: calls.append(context)
    )

    result = _evaluate(
        engine,
        selling_price=10_000,
        market_estimates=(_unknown_estimate(),),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text=" ",
        condition_extraction_confidence=0.0,
    )

    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "fast"
    assert calls == []


def test_llm_low_confidence_condition_correction_reprices_once_and_can_be_low():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="良好",
        condition_evidence="二手・良好",
        reason="原文明確標示二手良好",
        review_confidence=0.9,
    )
    contexts: list[dict[str, object]] = []
    reprice_calls: list[tuple[MarketplaceCondition, str]] = []
    engine = FusionDecisionEngine(
        condition_reviewer=lambda context: (
            contexts.append(context) or review
        )
    )
    original_new_market = _new_estimate(
        low=40_000,
        median=45_000,
        high=50_000,
    )

    def reprice(condition, detail):
        reprice_calls.append((condition, detail))
        return (_used_estimate(),)

    assert (
        engine._calculate_price_score(18_000, original_new_market)
        > engine.policy.medium_score_max
    )
    result = _evaluate(
        engine,
        selling_price=18_000,
        market_estimates=(original_new_market,),
        condition=MarketplaceCondition.NEW,
        condition_detail="全新",
        condition_source_text="狀況可能為二手・良好",
        condition_extraction_confidence=0.7,
        reprice=reprice,
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "llm"
    assert result["condition"] == MarketplaceCondition.USED
    assert reprice_calls == [(MarketplaceCondition.USED, "良好")]
    assert len(contexts) == 1


@pytest.mark.parametrize(
    ("condition_evidence", "review_confidence"),
    [
        ("不在原始狀態文字中的證據", 0.9),
        ("狀況 全新", 0.79),
    ],
)
def test_untraceable_or_low_confidence_llm_review_uses_fallback(
    condition_evidence,
    review_confidence,
):
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.NEW,
        condition_detail="全新",
        condition_evidence=condition_evidence,
        reason="確認為全新",
        review_confidence=review_confidence,
    )
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)

    result = _evaluate(
        engine,
        selling_price=90_000,
        market_estimates=(estimate,),
        condition_source_text="狀況 全新",
        condition_extraction_confidence=0.8,
    )

    assert result["risk_score"] == 70.0
    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "llm_simulated"


def test_llm_cannot_override_high_confidence_extractor_condition():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="良好",
        condition_evidence="二手・良好",
        reason="判斷為二手",
        review_confidence=0.95,
    )
    reprice_calls: list[tuple[MarketplaceCondition, str]] = []
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)

    result = _evaluate(
        engine,
        selling_price=18_000,
        condition_source_text="標題全新，但描述寫二手・良好",
        condition_extraction_confidence=0.97,
        condition_has_conflict=True,
        reprice=lambda condition, detail: (
            reprice_calls.append((condition, detail)) or (_used_estimate(),)
        ),
    )

    assert result["decision_layer"] == "llm_simulated"
    assert result["condition"] == MarketplaceCondition.NEW
    assert reprice_calls == []


def test_llm_same_condition_and_detail_does_not_reprice():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.NEW,
        condition_detail="全新",
        condition_evidence="狀況 全新",
        reason="確認為全新",
        review_confidence=0.9,
    )
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    result = _evaluate(
        engine,
        selling_price=90_000,
        market_estimates=(estimate,),
        condition_extraction_confidence=0.8,
        reprice=lambda *_: pytest.fail("相同狀態不得重新查價"),
    )

    assert result["decision_layer"] == "llm"
    assert result["risk_score"] == 70.0


def test_unknown_reviewed_as_unknown_keeps_combined_result_without_reprice():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_evidence="品況無法確認",
        reason="證據不足以確認品況",
        review_confidence=0.9,
    )
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)

    result = _evaluate(
        engine,
        selling_price=10_000,
        market_estimates=(_unknown_estimate(),),
        condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_source_text="品況無法確認",
        condition_extraction_confidence=0.0,
        reprice=lambda *_: pytest.fail("UNKNOWN 未變更不得重新查價"),
    )

    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "llm"


def test_accepted_change_to_unknown_reprices_combined_market_once():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.UNKNOWN,
        condition_detail="",
        condition_evidence="無法確認是否全新",
        reason="原文不足以確認全新",
        review_confidence=0.9,
    )
    calls: list[tuple[MarketplaceCondition, str]] = []
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)

    def reprice(condition, detail):
        calls.append((condition, detail))
        return (_unknown_estimate(),)

    result = _evaluate(
        engine,
        selling_price=18_000,
        condition_source_text="無法確認是否全新",
        condition_extraction_confidence=0.7,
        reprice=reprice,
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "llm"
    assert calls == [(MarketplaceCondition.UNKNOWN, "")]


def test_accepted_correction_with_insufficient_reprice_is_decision_error():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="良好",
        condition_evidence="二手良好",
        reason="確認為二手",
        review_confidence=0.9,
    )
    insufficient = _used_estimate(
        status="insufficient",
        confidence=0.4,
        sample_count=2,
        site_count=2,
    )
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)

    result = _evaluate(
        engine,
        selling_price=18_000,
        condition_source_text="二手良好",
        condition_extraction_confidence=0.7,
        reprice=lambda *_: (insufficient,),
    )

    assert result["risk_label"] == "UNKNOWN"
    assert result["decision_layer"] == "decision_error"
    assert result["error_code"] == "MARKET_PRICE_INSUFFICIENT"


def test_llm_extra_risk_fields_are_invalid_and_cannot_control_final_score():
    engine = FusionDecisionEngine(
        condition_reviewer=lambda _: {
            "reviewed_condition": "new",
            "condition_detail": "全新",
            "condition_evidence": "狀況 全新",
            "reason": "嘗試直接指定分數",
            "review_confidence": 0.9,
            "risk_score": 0,
            "risk_label": "LOW",
        }
    )
    estimate = _new_estimate(
        low=100_000,
        median=100_000,
        high=100_000,
    )

    result = _evaluate(
        engine,
        selling_price=90_000,
        market_estimates=(estimate,),
        condition_extraction_confidence=0.8,
    )

    assert result["risk_score"] == 70.0
    assert result["risk_label"] == "MEDIUM"
    assert result["decision_layer"] == "llm_simulated"


def test_alt_deep_result_rejects_low_and_preserves_medium_or_high():
    engine = FusionDecisionEngine()
    with pytest.raises(ValueError, match="MEDIUM／HIGH"):
        engine._alt_deep_result(
            {
                "risk_score": 0.0,
                "risk_label": "LOW",
            }
        )

    base_result = {
        "risk_score": 65.0,
        "risk_label": "MEDIUM",
        "reason": "價格規則結果",
        "evidence": ["價格證據"],
        "confidence": 0.9,
        "condition": MarketplaceCondition.NEW,
    }
    fallback = engine._alt_deep_result(base_result)

    assert fallback["risk_score"] == 65.0
    assert fallback["risk_label"] == "MEDIUM"
    assert fallback["condition"] == MarketplaceCondition.NEW
    assert fallback["decision_layer"] == "llm_simulated"


def test_enforce_price_risk_only_applies_to_final_high_confidence_iqr_high():
    engine = FusionDecisionEngine()
    full_iqr = _new_estimate(
        low=20_000,
        median=22_500,
        high=25_000,
    )
    enforced = engine._enforce_price_risk(
        {
            "risk_label": "LOW",
            "risk_score": 0.0,
            "evidence": [],
        },
        1,
        full_iqr,
    )
    small_sample = full_iqr.model_copy(
        update={
            "reference_mode": "median_low_sample",
            "sample_count": 3,
            "confidence": 0.6,
        }
    )
    unchanged = engine._enforce_price_risk(
        {
            "risk_label": "LOW",
            "risk_score": 0.0,
            "evidence": [],
        },
        1,
        small_sample,
    )

    assert enforced["risk_label"] == "HIGH"
    assert enforced["risk_score"] > engine.policy.medium_score_max
    assert unchanged["risk_label"] == "LOW"
    assert unchanged["risk_score"] == 0.0


def test_image_price_engine_contains_no_blacklist_or_direct_settings_dependency():
    source = inspect.getsource(FusionDecisionEngine).casefold()

    assert "blacklist" not in source
    assert "_load_blacklist_terms" not in source
    assert "_run_blacklist_hit" not in source
    assert "deep_analysis_score_min" not in source
    assert "deep_analysis_score_max" not in source
    assert "settings." not in source
    assert "BLACKLIST_TERMS_PATH" not in Settings.model_fields


def test_fallback_local_reference_uses_same_price_and_condition_rules():
    fallback = _new_estimate(source="fallback_local")

    result = _evaluate(
        FusionDecisionEngine(),
        selling_price=27_000,
        market_estimates=(fallback,),
    )

    assert result["risk_label"] == "LOW"
    assert result["decision_layer"] == "fast"
    assert result["market_price_source"] == "fallback_local"
