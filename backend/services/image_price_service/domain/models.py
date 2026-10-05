"""定義 OCR、商品頁驗證與刊登欄位抽取共用的資料模型。"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
)


class MarketplaceCondition(str, Enum):
    """刊登商品的使用狀況。"""

    NEW = "new"
    USED = "used"
    UNKNOWN = "unknown"


class MarketplaceLayout(str, Enum):
    """商品頁截圖的版面類型。"""

    MOBILE = "mobile"
    DESKTOP = "desktop"
    UNKNOWN = "unknown"


class PriceSection(str, Enum):
    """價格候選值所在的商品頁區段。"""

    MAIN_PRICE = "main_price"
    OFFER_RANGE = "offer_range"
    DESCRIPTION = "description"
    SELLER_INFO = "seller_info"
    MESSAGE_BOX = "message_box"
    DETAIL = "detail"
    UNKNOWN = "unknown"


class OCRTextBlock(BaseModel):
    """保存單一 OCR 文字區塊及其座標。"""

    text: str
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    confidence: float | None = None


class OCRDocument(BaseModel):
    """保存 OCR 全文、文字區塊與頁面尺寸。"""

    text: str
    blocks: list[OCRTextBlock] = Field(default_factory=list)
    width: float | None = None
    height: float | None = None

    @property
    def has_coordinates(self) -> bool:
        """判斷所有文字區塊是否都有座標。"""
        return bool(self.blocks) and all(
            block.x is not None and block.y is not None
            for block in self.blocks
        )


class ProductAgentResult(BaseModel):
    """保存模型輸出、搜尋結果與工具錯誤。"""

    output: dict[str, Any]
    tool_results: list[dict[str, Any]]
    tool_errors: list[str]

class DetectionResult(BaseModel):
    """商品頁來源驗證與版型推定結果。"""

    is_marketplace: bool
    layout: MarketplaceLayout
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    reason: str | None = None


class PriceCandidate(BaseModel):
    """從 OCR 文字擷取的單一價格候選及其判定資訊。"""

    amount: int
    currency: str
    source_text: str
    block_index: int
    x: float | None
    y: float | None
    context_before: str | None
    context_after: str | None
    section: PriceSection
    confidence: float
    reject_reason: str | None = None


class MainPriceExtractionResult(BaseModel):
    """保存 Marketplace 刊登欄位的抽取結果。"""

    price: int | None
    currency: str | None
    confidence: float
    source_text: str | None
    layout: MarketplaceLayout
    candidates: list[PriceCandidate] = Field(default_factory=list)
    rejected_candidates: list[PriceCandidate] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None
    reason: str | None = None
    product_name: str | None = None
    seller_name: str | None = None
    condition: MarketplaceCondition = MarketplaceCondition.UNKNOWN
    condition_detail: str = ""
    condition_source_text: str = ""
    condition_extraction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "只代表 extractor 擷取商品狀態的可信度，不代表商品匹配、"
            "市場資料或價格風險可信度"
        ),
    )
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_supported_listing_price(self) -> "MainPriceExtractionResult":
        """將超出服務上限的刊登價格標記為明確的擷取錯誤。"""
        if (
            self.price is not None
            and self.price
            > DEFAULT_PRICE_RISK_POLICY.maximum_supported_price
        ):
            self.error_code = "PRICE_OUT_OF_SUPPORTED_RANGE"
            self.message = "刊登價格超出商品價格驗證服務支援範圍"
        return self

class MainPriceExtractionError(ValueError):
    """表示主價格不存在或擷取信心不足。"""

    def __init__(self, result: MainPriceExtractionResult):
        """建立錯誤並保留原始擷取結果。"""
        if not result.error_code:
            raise ValueError("MainPriceExtractionError 必須包含 error_code")
        self.result = result
        self.error_code = result.error_code
        super().__init__(result.message or result.error_code)
