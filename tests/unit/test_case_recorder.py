from backend.services.dto.price_analysis import (
    ImagePriceAnalysisResult,
    MarketPriceCandidateEvidence,
    MarketPriceEstimate,
)
from backend.services.image_price_service.domain.models import (
    MarketplaceCondition,
)
from backend.services.image_price_service.case_recorder import record_case


class _FakeCaseRepository:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.payload = None

    def save(self, payload: dict) -> str:
        if self.error is not None:
            raise self.error
        self.payload = payload
        return "case-1"


def _result() -> ImagePriceAnalysisResult:
    return ImagePriceAnalysisResult(
        filename="ad.png",
        content_type="image/png",
        extracted_text="iphone 15",
        product_name="Apple iPhone 15",
        brand_model="Apple iPhone 15",
        listed_price=25_000,
        market_price=31_234,
        market_price_source="online",
        market_price_estimates=(
            MarketPriceEstimate(
                status="success",
                condition=MarketplaceCondition.USED,
                reference_mode="iqr",
                median_price=31_234,
                low_price=29_000,
                high_price=33_000,
                sample_count=5,
                site_count=3,
                source="online",
                confidence=0.8,
                candidates=(
                    MarketPriceCandidateEvidence(
                        candidate_id="candidate-1",
                        title="Apple iPhone 15",
                        price=31_234,
                        condition=MarketplaceCondition.USED,
                        url="https://example.com/iphone",
                        evidence="二手售價 31,234",
                    ),
                ),
            ),
        ),
        condition=MarketplaceCondition.USED,
        condition_detail="近全新",
        condition_source_text="狀況 二手・近全新",
        condition_extraction_confidence=0.97,
        evidence=["used 市場區間 29,000～33,000"],
    )


def test_record_case_preserves_existing_payload_fields(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        True,
    )
    repository = _FakeCaseRepository()

    assert record_case(repository, _result()) == "case-1"
    assert repository.payload is not None
    assert repository.payload["selling_price"] == 25_000
    assert repository.payload["market_price_source"] == "online"
    assert repository.payload["market_price"] == 31_234
    assert repository.payload["condition_detail"] == "近全新"
    assert repository.payload["condition_source_text"] == "狀況 二手・近全新"
    assert repository.payload["condition_extraction_confidence"] == 0.97
    estimate = repository.payload["market_price_estimates"][0]
    assert estimate["low_price"] == 29_000
    assert "candidates" not in estimate
    assert "listed_price" not in repository.payload


def test_record_case_skips_repository_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    repository = _FakeCaseRepository()

    assert record_case(repository, _result()) is None
    assert repository.payload is None


def test_record_case_keeps_analysis_successful_when_repository_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        True,
    )
    repository = _FakeCaseRepository(error=OSError("disk unavailable"))

    assert record_case(repository, _result()) is None
