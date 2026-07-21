import pytest

from backend.services.image_price_service.models import (
    MainPriceExtractionError,
    MarketplaceCondition,
    MarketplaceDetectionResult,
    OCRDocument,
    OCRTextBlock,
)
from backend.services.image_price_service.platform.fb_marketplace.fb_marketplace_detector import (
    FBMarketplaceDetector,
)
from backend.services.image_price_service.platform.fb_marketplace.fb_marketplace_extractor import (
    FBMarketplacePriceExtractor,
)

MOBILE_TEXT = """iphone 14 pro max 256g 紫
NT$13,000
6 offers from NT$8,000 to NT$13,000
發送訊息給賣家
說明
限面交
賣家
詳細內容
狀況 二手・近全新"""

DESKTOP_TEXT = """17Promax 銀256GB 台中面交 二手機可折優惠
NT$39,500 · 有存貨
台中市, 台灣
公開面交
詳細資料
狀況 二手・近全新"""

MACBOOK_TEXT = """MacBook Pro 16 2019 i7 16/512
NT$13,000
附近・6公里
有人感興趣・1 次出價
發送訊息給賣家
傳送出價
分享
儲存
說明
末代intel，可使用雙系統
2020全新品購入，良好使用，無修無拆無換。
賣家
Wei-Cheng Fang
2014加入 Facebook
詳細內容"""


def test_mobile_uses_single_price_above_offer_range():
    detector = FBMarketplaceDetector()
    extractor = FBMarketplacePriceExtractor()
    document = OCRDocument(text=MOBILE_TEXT)

    detection = detector.detect(document)
    result = extractor.extract(document, detection)

    assert detection.is_marketplace is True
    assert detection.layout == "mobile"
    assert result.price == 13000
    assert result.currency == "TWD"
    assert result.source_text == "NT$13,000"
    assert result.error_code is None
    assert result.product_name == "iphone 14 pro max 256g 紫"
    assert result.reason == "商品標題正下方的第一個 NT$ 單一價格"
    assert {item.amount for item in result.rejected_candidates} == {8000, 13000}
    assert all(item.section == "offer_range" for item in result.rejected_candidates)


def test_desktop_extracts_only_number_from_inventory_line():
    detector = FBMarketplaceDetector()
    extractor = FBMarketplacePriceExtractor()
    document = OCRDocument(text=DESKTOP_TEXT)

    detection = detector.detect(document)
    result = extractor.extract(document, detection)

    assert detection.is_marketplace is True
    assert detection.layout == "desktop"
    assert result.price == 39500
    assert result.source_text == "NT$39,500"
    assert result.product_name == "17Promax 銀256GB 台中面交 二手機可折優惠"
    assert result.reason == "桌面版右側商品標題下方 NT$ 價格列"


def test_extracts_reference_listing_title_seller_and_used_condition_from_description():
    detector = FBMarketplaceDetector()
    document = OCRDocument(text=MACBOOK_TEXT)

    result = FBMarketplacePriceExtractor().extract(document, detector.detect(document))

    assert result.product_name == "MacBook Pro 16 2019 i7 16/512"
    assert result.price == 13000
    assert result.source_text == "NT$13,000"
    assert result.seller_name == "Wei-Cheng Fang"
    assert result.condition is MarketplaceCondition.USED
    assert result.condition_confidence == 0.82
    assert result.warnings == []


def test_explicit_detail_condition_has_priority_over_description():
    text = MACBOOK_TEXT.replace(
        "2020全新品購入，良好使用，無修無拆無換。",
        "全新未拆封",
    ) + "\n狀況 二手・近全新"
    detection = MarketplaceDetectionResult(
        is_marketplace=True,
        layout="mobile",
        confidence=0.9,
        evidence=[],
    )

    result = FBMarketplacePriceExtractor().extract(OCRDocument(text=text), detection)

    assert result.condition is MarketplaceCondition.USED
    assert result.condition_confidence == 0.97


def test_missing_condition_defaults_to_new_with_lower_confidence_warning():
    text = MACBOOK_TEXT.replace(
        "2020全新品購入，良好使用，無修無拆無換。",
        "歡迎私訊了解商品",
    )
    detection = MarketplaceDetectionResult(
        is_marketplace=True,
        layout="mobile",
        confidence=0.9,
        evidence=[],
    )

    result = FBMarketplacePriceExtractor().extract(OCRDocument(text=text), detection)

    assert result.condition is MarketplaceCondition.NEW
    assert result.condition_confidence == 0.35
    assert result.warnings == ["未找到明確商品狀況，依規則預設為全新"]


def test_non_marketplace_source_is_rejected():
    detection = FBMarketplaceDetector().detect(
        OCRDocument(text="購物網站夏日特賣\n耳機\nNT$1,299\n加入購物車")
    )

    assert detection.is_marketplace is False
    assert detection.confidence < 0.65


def test_basic_listing_fields_are_enough_without_facebook_ui_markers():
    detection = FBMarketplaceDetector().detect(OCRDocument(text="""筆記型電腦 16GB
NT$18,000
詳細內容
狀況 二手
賣家
王小明"""))

    assert detection.is_marketplace is True
    assert detection.layout == "mobile"
    assert detection.confidence == 0.95
    assert detection.evidence == [
        "價格附近有商品標題",
        "偵測到商品價格",
        "偵測到詳細資料區段",
        "偵測到商品狀況",
        "偵測到賣家資訊",
    ]


def test_title_and_price_without_basic_detail_sections_are_not_enough():
    detection = FBMarketplaceDetector().detect(
        OCRDocument(text="筆記型電腦 16GB\nNT$18,000")
    )

    assert detection.is_marketplace is False
    assert detection.confidence == 0.5


def test_product_photo_without_marketplace_ui_is_rejected():
    detection = FBMarketplaceDetector().detect(OCRDocument(text="iphone 14 pro max"))

    assert detection.is_marketplace is False
    assert detection.layout == "unknown"


def test_offer_range_only_never_becomes_main_price():
    with pytest.raises(MainPriceExtractionError) as exc_info:
        FBMarketplacePriceExtractor().extract(
            OCRDocument(text="6 offers from NT$8,000 to NT$13,000"),
            MarketplaceDetectionResult(
                is_marketplace=True,
                layout="mobile",
                confidence=0.8,
                evidence=["測試指定 mobile 版型"],
            ),
        )

    extraction = exc_info.value.result
    assert extraction.price is None
    assert exc_info.value.error_code == "MAIN_PRICE_NOT_FOUND"
    assert extraction.error_code == "MAIN_PRICE_NOT_FOUND"
    assert {item.amount for item in extraction.rejected_candidates} == {8000, 13000}


def test_non_nt_price_is_not_accepted_as_marketplace_main_price():
    document = OCRDocument(text="""MacBook Pro 16 2019 i7 16/512
$13,000
發送訊息給賣家
說明
賣家
詳細內容""")
    detection = MarketplaceDetectionResult(
        is_marketplace=True,
        layout="mobile",
        confidence=0.9,
        evidence=[],
    )

    with pytest.raises(MainPriceExtractionError) as exc_info:
        FBMarketplacePriceExtractor().extract(document, detection)

    result = exc_info.value.result
    assert result.price is None
    assert result.error_code == "MAIN_PRICE_NOT_FOUND"
    assert result.rejected_candidates[0].reject_reason == "商品主價格必須以 NT$ 開頭"


def test_desktop_bbox_rejects_price_outside_right_information_panel():
    document = OCRDocument(
        text=DESKTOP_TEXT,
        width=1848,
        height=878,
        blocks=[
            OCRTextBlock("17Promax 銀256GB 台中面交 二手機可折優惠", x=1200, y=80, width=500, height=35),
            OCRTextBlock("NT$39,500 · 有存貨", x=100, y=125, width=220, height=30),
            OCRTextBlock("公開面交", x=1200, y=175, width=100, height=25),
            OCRTextBlock("詳細資料", x=1200, y=270, width=100, height=25),
            OCRTextBlock("狀況 二手・近全新", x=1200, y=320, width=250, height=25),
        ],
    )
    detection = MarketplaceDetectionResult(
        is_marketplace=True,
        layout="desktop",
        confidence=0.9,
        evidence=[],
    )

    with pytest.raises(MainPriceExtractionError) as exc_info:
        FBMarketplacePriceExtractor().extract(document, detection)

    result = exc_info.value.result
    assert result.price is None
    assert result.error_code == "MAIN_PRICE_NOT_FOUND"
    assert result.rejected_candidates[0].reject_reason == "價格不在商品標題下方同一資訊欄"


def test_low_confidence_main_price_raises_specific_exception():
    document = OCRDocument(text="""MacBook Pro 16 2019 i7 16/512
分享
NT$13,000""")
    detection = MarketplaceDetectionResult(
        is_marketplace=True,
        layout="mobile",
        confidence=0.8,
        evidence=[],
    )

    with pytest.raises(MainPriceExtractionError) as exc_info:
        FBMarketplacePriceExtractor().extract(document, detection)

    assert exc_info.value.error_code == "LOW_CONFIDENCE_PRICE_EXTRACTION"
    assert exc_info.value.result.price is None
    assert exc_info.value.result.candidates[0].confidence < 0.65
