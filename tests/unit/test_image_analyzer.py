from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.category.registry import CategoryRegistry
from backend.services.image_price_service.image_price_analyzer import ImagePriceAnalyzer
from backend.services.image_price_service.models import MarketplaceCondition


class _FakeOCR:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self, data: bytes) -> str:
        return self.text


class _FakeProductIdentifier:
    def __init__(self, product: ProductIdentification):
        self.product = product
        self.received = None

    def identify(self, text: str) -> ProductIdentification:
        self.received = text
        return self.product


class _FakeOnlinePriceService:
    def __init__(self, value: int):
        self.value = value

    def estimate_taiwan_market_price(self, product_query: str, max_results: int = 8) -> int:
        return self.value


class _FakeDecisionEngine:
    def __init__(self):
        self.received = None

    def evaluate(self, **kwargs):
        self.received = kwargs
        return {
            "risk_label": "低風險",
            "risk_score": 25.0,
            "reason": "商品資料分析完成",
            "evidence": [],
            "confidence": 0.9,
            "decision_layer": "fast",
            "tool_observations": {},
        }


class _FakeCaseRepository:
    def append_case(self, payload: dict) -> str:
        return "case-1"


class _FakeCategoryHandler:
    name = "phone"

    def supports(self, product_name: str, brand_model: str) -> bool:
        return "iphone" in f"{product_name} {brand_model}".lower()


def _build_analyzer(text: str, market_price: int):
    decision = _FakeDecisionEngine()
    identifier = _FakeProductIdentifier(
        ProductIdentification(
            product_name="Apple iPhone 15",
            brand_model="Apple iPhone 15",
            market_price=27900,
        )
    )
    analyzer = ImagePriceAnalyzer(
        ocr_service=_FakeOCR(text),
        product_identifier=identifier,
        online_price_service=_FakeOnlinePriceService(market_price),
        decision_engine=decision,
        case_repo=_FakeCaseRepository(),
    )
    return analyzer, decision, identifier


def test_price_risk_rule_includes_prices_at_least_twice_market():
    assert ImagePriceAnalyzer._has_price_risk(55_800, 27_900) is True
    assert ImagePriceAnalyzer._has_price_risk(60_000, 27_900) is True
    assert ImagePriceAnalyzer._has_price_risk(55_799, 27_900) is False


def test_high_price_rule_adds_matching_reason_and_evidence():
    decision = {
        "risk_label": "低風險",
        "risk_score": 25.0,
        "reason": "原始判定",
        "evidence": [],
    }

    result = ImagePriceAnalyzer._enforce_price_risk(
        decision,
        selling_price=55_800,
        market_price=27_900,
        is_high_risk=True,
    )

    assert result["risk_label"] == "高風險"
    assert result["risk_score"] == 90.0
    assert "2 倍以上" in result["reason"]
    assert "高於行情 2 倍規則觸發" in result["evidence"]


def test_image_price_detector_returns_complete_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, identifier = _build_analyzer(
        """iphone 15 256GB
NT$25,000
發送訊息給賣家
說明
賣家
詳細內容
狀況 二手・近全新""",
        market_price=27900,
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
    assert identifier.received == "iphone 15 256GB"
    assert decision.received["selling_price"] == 25000


def test_low_price_rule_is_enforced_after_decision(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, _, _ = _build_analyzer(
        """iphone 15 256GB
NT$10,000
發送訊息給賣家
說明
賣家
詳細內容
狀況 二手・近全新""",
        market_price=27900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.is_high_risk_below_market is True
    assert result.risk_label == "HIGH"
    assert isinstance(result.score, float) and result.score >= 90.0
    assert "低於行情 50% 規則觸發" in result.evidence


def test_empty_ocr_is_rejected_before_decision(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _ = _build_analyzer("", market_price=0)

    result = analyzer.image_price_detector(b"image")

    assert result.risk_label == "UNKNOWN"
    assert result.score == "未知"
    assert result.success is False
    assert result.error_code == "INVALID_IMAGE_SOURCE"
    assert result.message == "圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖"
    assert decision.received is None


def test_non_marketplace_does_not_call_product_or_online_price(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _ = _build_analyzer(
        "一般購物網站 商品特價 NT$9,999 加入購物車",
        market_price=27900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.error_code == "INVALID_IMAGE_SOURCE"
    assert result.market_price_source == "not_evaluated"
    assert decision.received is None


def test_valid_marketplace_with_offer_range_only_stops_before_decision(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.CASE_MEMORY_ENABLED",
        False,
    )
    analyzer, decision, _ = _build_analyzer(
        """iphone 14 pro max 256g 紫
6 offers from NT$8,000 to NT$13,000
發送訊息給賣家
說明
賣家
詳細內容
狀況 二手・近全新""",
        market_price=27900,
    )

    result = analyzer.image_price_detector(b"image")

    assert result.success is False
    assert result.error_code == "MAIN_PRICE_NOT_FOUND"
    assert result.listed_price is None
    assert decision.received is None


def test_market_price_prefers_online_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    analyzer = ImagePriceAnalyzer(
        online_price_service=_FakeOnlinePriceService(31234),
    )

    assert analyzer._resolve_market_price(
        "Apple iPhone 15",
        "Apple iPhone 15",
        27900,
    ) == (31234, "online")


def test_market_price_falls_back_to_local_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.image_price_analyzer.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    analyzer = ImagePriceAnalyzer(
        online_price_service=_FakeOnlinePriceService(0),
    )

    assert analyzer._resolve_market_price(
        "Apple iPhone 15",
        "Apple iPhone 15",
        27900,
    ) == (27900, "fallback_local")


def test_category_registry_is_ready_for_future_handlers():
    registry = CategoryRegistry()
    handler = _FakeCategoryHandler()

    registry.register(handler)

    assert registry.get("phone") is handler
    assert registry.resolve("Apple iPhone 15", "Apple A3090") is handler
    assert registry.resolve("Dyson 吹風機", "Dyson HD15") is None
