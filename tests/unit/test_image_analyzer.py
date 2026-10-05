from backend.services.dto.price_analysis import (
    DeepAnalysisReview,
    MarketPriceCandidateEvidence,
    MarketPriceEstimate,
    ProductIdentification,
)
from backend.services.image_price_service.domain.models import (
    MainPriceExtractionResult,
    MarketplaceCondition,
    MarketplaceLayout,
    OCRDocument,
)
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
)
from backend.services.image_price_service.image_price_analyzer import (
    ImagePriceAnalyzer,
)
from backend.services.image_price_service.risk.fusion_decision_engine import (
    FusionDecisionEngine,
)


class _FakeOCR:
    def __init__(self, text: str):
        self.text = text

    def extract_document(self, data: bytes) -> OCRDocument:
        return OCRDocument(text=self.text)


class _FakeProductIdentifier:
    def __init__(self, product: ProductIdentification):
        self.product = product
        self.received = None

    def identify(self, text: str) -> ProductIdentification:
        self.received = text
        return self.product


def _estimate(
    condition: MarketplaceCondition,
    value: int,
) -> MarketPriceEstimate:
    if value <= 0:
        return MarketPriceEstimate(
            status="not_found",
            condition=condition,
            reference_mode="median_low_sample",
            median_price=0,
            low_price=0,
            high_price=0,
            sample_count=0,
            site_count=0,
            source="online",
            confidence=0.0,
        )
    sample_count = 4 if condition is MarketplaceCondition.UNKNOWN else 3
    confidence = 0.8 if condition is MarketplaceCondition.UNKNOWN else 0.6
    return MarketPriceEstimate(
        status="success",
        condition=condition,
        reference_mode="median_low_sample",
        median_price=value,
        low_price=round(value * 0.75),
        high_price=round(value * 1.25),
        sample_count=sample_count,
        site_count=3,
        source="online",
        confidence=confidence,
        candidates=(
            MarketPriceCandidateEvidence(
                candidate_id=f"{condition.value}-1",
                title="候選商品",
                price=value,
                condition=condition,
                evidence=f"售價 {value}",
            ),
        ),
    )


def test_search_tools_preserve_call_order_and_remove_duplicates() -> None:
    first = _estimate(MarketplaceCondition.NEW, 30_000).model_copy(
        update={"search_tools": ["serp_api", "tavily"]}
    )
    second = _estimate(MarketplaceCondition.USED, 20_000).model_copy(
        update={"search_tools": ["tavily", "ddgs"]}
    )

    assert ImagePriceAnalyzer._search_tools((first, second)) == [
        "serp_api",
        "tavily",
        "ddgs",
    ]


class _FakeOnlinePriceService:
    policy = DEFAULT_PRICE_RISK_POLICY

    def __init__(
        self,
        value: int,
        *,
        values_by_condition: dict[MarketplaceCondition, int] | None = None,
    ):
        self.value = value
        self.values_by_condition = values_by_condition or {}
        self.calls: list[dict] = []

    def estimate_prices(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[MarketPriceEstimate, ...]:
        self.calls.append(
            {
                "product_query": product_query,
                "max_results": max_results,
                "condition": condition,
                "condition_text": condition_text,
            }
        )
        return (
            _estimate(
                condition,
                self.values_by_condition.get(condition, self.value),
            ),
        )


class _FakeDecisionEngine:
    def __init__(self, *, call_reprice_twice: bool = False):
        self.received = None
        self.call_reprice_twice = call_reprice_twice
        self.reprice_results = []

    def evaluate(self, **kwargs):
        self.received = kwargs
        if self.call_reprice_twice:
            callback = kwargs["reprice"]
            self.reprice_results.append(
                callback(MarketplaceCondition.NEW, "全新")
            )
            self.reprice_results.append(
                callback(MarketplaceCondition.USED, "二手")
            )
        return {
            "risk_label": "LOW",
            "risk_score": 25.0,
            "reason": "商品資料分析完成",
            "evidence": [],
            "confidence": 0.9,
            "decision_layer": "fast",
            "market_price_source": "online",
            "condition": kwargs["condition"],
            "condition_detail": kwargs["condition_detail"],
        }


class _FakeCaseRepository:
    def __init__(self):
        self.payload = None

    def save(self, payload: dict) -> str:
        self.payload = payload
        return "case-1"


class _FakePriceExtractor:
    def __init__(self, result: MainPriceExtractionResult):
        self.result = result

    def extract(self, document, detection) -> MainPriceExtractionResult:
        return self.result


def _build_analyzer(
    text: str,
    market_price: int,
    case_repo=None,
    *,
    decision_engine=None,
    values_by_condition=None,
):
    decision = (
        decision_engine
        if decision_engine is not None
        else _FakeDecisionEngine()
    )
    identifier = _FakeProductIdentifier(
        ProductIdentification(
            product_name="Apple iPhone 15",
            brand_model="Apple iPhone 15",
            market_price=27_900,
            search_query="Apple iPhone 15 價格",
        )
    )
    price_service = _FakeOnlinePriceService(
        market_price,
        values_by_condition=values_by_condition,
    )
    analyzer = ImagePriceAnalyzer(
        ocr_service=_FakeOCR(text),
        product_identifier=identifier,
        online_price_service=price_service,
        decision_engine=decision,
        case_repo=(
            case_repo
            if case_repo is not None
            else _FakeCaseRepository()
        ),
    )
    return analyzer, decision, identifier, price_service


_USED_LISTING = """iphone 15 256GB
NT$25,000
發送訊息給賣家
說明
賣家
詳細內容
狀況 二手・近全新"""


def test_image_price_detector_passes_only_structured_decision_input(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, identifier, _ = _build_analyzer(
        _USED_LISTING,
        market_price=27_900,
    )

    result = analyzer.image_price_detector(
        b"image",
        filename="ad.png",
        content_type="image/png",
    )

    assert result.filename == "ad.png"
    assert result.product_name == "iphone 15 256GB"
    assert result.risk_label == "LOW"
    assert result.score == 25.0
    assert result.success is True
    assert result.marketplace_layout == "mobile"
    assert result.price_source_text == "NT$25,000"
    assert result.condition is MarketplaceCondition.USED
    assert result.condition_detail == "近全新"
    assert result.market_price_source == "online"
    assert result.search_tools == []
    assert result.market_price == 27_900
    assert identifier.received == "iphone 15 256GB"

    received = decision.received
    assert received["selling_price"] == 25_000
    assert received["condition"] is MarketplaceCondition.USED
    assert received["condition_detail"] == "近全新"
    assert received["condition_source_text"] == "狀況 二手・近全新"
    assert received["text"] == _USED_LISTING
    assert received["condition_extraction_confidence"] == 0.97
    assert received["market_estimates"][0].candidates == ()
    assert result.market_price_estimates[0].candidates == ()
    assert {
        "brand_model",
        "market_price",
        "market_price_source",
    }.isdisjoint(received)


def test_successful_analysis_records_sanitized_case(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        True,
    )
    repository = _FakeCaseRepository()
    analyzer, _, _, _ = _build_analyzer(
        _USED_LISTING,
        market_price=27_900,
        case_repo=repository,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.success is True
    assert repository.payload is not None
    assert repository.payload["selling_price"] == 25_000
    assert repository.payload["condition_detail"] == "近全新"
    estimate = repository.payload["market_price_estimates"][0]
    assert estimate["low_price"] == 20_925
    assert "candidates" not in estimate


def test_detector_uses_fusion_decision_without_overriding(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, _, _, _ = _build_analyzer(
        _USED_LISTING.replace("NT$25,000", "NT$10,000"),
        market_price=27_900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.risk_label == "LOW"
    assert result.score == 25.0
    assert result.evidence == []


def test_detector_delegates_insufficient_market_data_to_decision_engine(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, _, _, _ = _build_analyzer(
        _USED_LISTING,
        market_price=0,
        decision_engine=FusionDecisionEngine(),
    )

    result = analyzer.image_price_detector(b"image")

    assert result.market_price == 0
    assert result.market_price_source == "not_evaluated"
    assert result.condition is MarketplaceCondition.USED
    assert result.risk_label == "UNKNOWN"
    assert result.score == "未知"
    assert result.error_code == "MARKET_PRICE_NOT_FOUND"
    assert result.decision_layer == "decision_error"


def test_unknown_condition_keeps_combined_market_estimate(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    listing = _USED_LISTING.replace("狀況 二手・近全新", "未提供品況")
    analyzer, _, _, service = _build_analyzer(
        listing,
        market_price=0,
        decision_engine=FusionDecisionEngine(),
        values_by_condition={
            MarketplaceCondition.UNKNOWN: 25_000,
        },
    )

    result = analyzer.image_price_detector(b"image")

    assert service.calls[0]["condition"] is MarketplaceCondition.UNKNOWN
    assert len(result.market_price_estimates) == 1
    assert result.market_price == 25_000
    assert result.market_price_source == "online"
    assert result.condition is MarketplaceCondition.UNKNOWN
    assert result.risk_label == "LOW"
    assert result.decision_layer == "fast"


def test_reprice_callback_can_resolve_market_price_only_once(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    decision = _FakeDecisionEngine(call_reprice_twice=True)
    analyzer, _, _, service = _build_analyzer(
        _USED_LISTING,
        market_price=27_900,
        decision_engine=decision,
    )

    analyzer.image_price_detector(b"image")

    assert len(service.calls) == 2
    assert service.calls[1]["condition"] is MarketplaceCondition.NEW
    assert decision.reprice_results[0] is decision.reprice_results[1]


def test_llm_condition_correction_reprices_once_and_uses_final_interval(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    review = DeepAnalysisReview(
        reviewed_condition=MarketplaceCondition.USED,
        condition_detail="近全新",
        condition_evidence="狀況 全新",
        reason="狀態原文需要修正為二手",
        review_confidence=0.9,
    )
    engine = FusionDecisionEngine(condition_reviewer=lambda _: review)
    analyzer, _, _, service = _build_analyzer(
        _USED_LISTING.replace("NT$25,000", "NT$10,000"),
        market_price=0,
        decision_engine=engine,
        values_by_condition={
            MarketplaceCondition.NEW: 30_000,
            MarketplaceCondition.USED: 10_000,
        },
    )
    analyzer._price_extractor = _FakePriceExtractor(
        MainPriceExtractionResult(
            price=10_000,
            currency="TWD",
            confidence=0.95,
            source_text="NT$10,000",
            layout=MarketplaceLayout.MOBILE,
            product_name="iphone 15 256GB",
            condition=MarketplaceCondition.NEW,
            condition_detail="全新",
            condition_source_text="狀況 全新",
            condition_extraction_confidence=0.7,
        )
    )

    result = analyzer.image_price_detector(b"image")

    assert len(service.calls) == 2
    assert [call["condition"] for call in service.calls] == [
        MarketplaceCondition.NEW,
        MarketplaceCondition.USED,
    ]
    assert result.condition is MarketplaceCondition.USED
    assert result.condition_detail == "近全新"
    assert result.market_price == 10_000
    assert result.market_price_estimates[0].condition is MarketplaceCondition.USED
    assert result.risk_label == "LOW"
    assert result.decision_layer == "llm"


def test_empty_ocr_is_rejected_before_decision(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _, _ = _build_analyzer("", market_price=0)

    result = analyzer.image_price_detector(b"image")

    assert result.risk_label == "UNKNOWN"
    assert result.score == "未知"
    assert result.success is False
    assert result.error_code == "INVALID_IMAGE_SOURCE"
    assert result.message == "圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖"
    assert decision.received is None


def test_non_marketplace_does_not_call_product_or_online_price(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _, service = _build_analyzer(
        "一般購物網站 商品特價 NT$9,999 加入購物車",
        market_price=27_900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.error_code == "INVALID_IMAGE_SOURCE"
    assert result.market_price_source == "not_evaluated"
    assert decision.received is None
    assert service.calls == []


def test_valid_marketplace_with_offer_range_only_stops_before_decision(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.case_recorder.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _, service = _build_analyzer(
        """iphone 14 pro max 256g 紫
6 offers from NT$8,000 to NT$13,000
發送訊息給賣家
說明
賣家
詳細內容
狀況 二手・近全新""",
        market_price=27_900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.success is False
    assert result.error_code == "MAIN_PRICE_NOT_FOUND"
    assert result.listed_price is None
    assert decision.received is None
    assert service.calls == []
