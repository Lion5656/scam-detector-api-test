import pytest
from pydantic import ValidationError

from backend.services.dto.price_analysis import (
    DeepAnalysisReview,
    ImagePriceAnalysisResult,
    MarketPriceCandidateEvidence,
    MarketPriceEstimate,
)
from backend.services.image_price_service.domain.models import (
    MainPriceExtractionResult,
    MarketplaceCondition,
    MarketplaceLayout,
)


def test_deep_analysis_review_only_accepts_condition_review_fields():
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="近全新",
        condition_evidence="狀況 二手・近全新",
        reason="原文已明確標示二手品況",
        review_confidence=0.9,
    )

    assert set(DeepAnalysisReview.model_fields) == {
        "reviewed_condition",
        "condition_detail",
        "condition_evidence",
        "reason",
        "review_confidence",
    }
    with pytest.raises(ValidationError):
        DeepAnalysisReview(
            **review.model_dump(),
            review_status="confirmed",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "reviewed_condition": MarketplaceCondition.NEW,
                "condition_evidence": " ",
            },
            "確認商品狀態時必須提供可追溯的狀態原文證據",
        ),
        (
            {
                "reviewed_condition": MarketplaceCondition.UNKNOWN,
                "condition_detail": "近全新",
            },
            "商品狀態為 UNKNOWN 時不得提供已確認的狀態細節",
        ),
    ],
)
def test_deep_analysis_review_rejects_inconsistent_condition(
    payload,
    message,
):
    with pytest.raises(ValidationError, match=message):
        DeepAnalysisReview(
            reason="無法確認",
            review_confidence=0.5,
            **payload,
        )


def test_market_price_estimate_preserves_interval_and_evidence():
    candidate = MarketPriceCandidateEvidence(
        candidate_id="candidate-1",
        title="iPhone 15 128GB 二手",
        price=20_000,
        condition=MarketplaceCondition.USED,
        url="https://example.com/item/1",
        evidence="售價 NT$20,000",
    )
    estimate = MarketPriceEstimate(
        status="success",
        condition=MarketplaceCondition.USED,
        reference_mode="iqr",
        median_price=20_000,
        low_price=18_000,
        high_price=22_000,
        sample_count=5,
        site_count=3,
        source="online",
        confidence=0.9,
        candidates=(candidate,),
    )

    assert estimate.low_price == 18_000
    assert estimate.high_price == 22_000
    assert estimate.sample_count == 5
    assert estimate.site_count == 3
    assert estimate.confidence == 0.9


def test_condition_extraction_confidence_has_unambiguous_field_name():
    result = MainPriceExtractionResult(
        price=10_000,
        currency="TWD",
        confidence=0.9,
        source_text="NT$10,000",
        layout=MarketplaceLayout.MOBILE,
        condition=MarketplaceCondition.USED,
        condition_detail="良好",
        condition_source_text="商品狀況 良好",
        condition_extraction_confidence=0.8,
    )

    assert result.condition_extraction_confidence == 0.8
    assert "condition_confidence" not in MainPriceExtractionResult.model_fields


def test_out_of_range_listing_price_uses_explicit_error_code():
    extraction = MainPriceExtractionResult(
        price=100_001,
        currency="TWD",
        confidence=0.9,
        source_text="NT$100,001",
        layout=MarketplaceLayout.MOBILE,
    )
    assert extraction.error_code == "PRICE_OUT_OF_SUPPORTED_RANGE"

    with pytest.raises(ValidationError, match="PRICE_OUT_OF_SUPPORTED_RANGE"):
        ImagePriceAnalysisResult(
            filename="listing.png",
            content_type="image/png",
            extracted_text="NT$100,001",
            listed_price=100_001,
        )


def test_unknown_result_accepts_decision_error_and_explicit_error_code():
    result = ImagePriceAnalysisResult(
        filename="listing.png",
        content_type="image/png",
        extracted_text="",
        risk_label="UNKNOWN",
        decision_layer="decision_error",
        error_code="MARKET_PRICE_NOT_FOUND",
    )

    assert result.error_code == "MARKET_PRICE_NOT_FOUND"
    with pytest.raises(ValidationError):
        ImagePriceAnalysisResult(
            filename="listing.png",
            content_type="image/png",
            extracted_text="",
            decision_layer="source_validation",
        )
