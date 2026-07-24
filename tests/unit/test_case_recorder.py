from backend.services.dto.price_analysis import ImagePriceAnalysisResult
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
