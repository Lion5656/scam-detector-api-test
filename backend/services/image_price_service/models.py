"""定義 OCR、商品頁驗證與刊登欄位抽取共用的資料模型。"""

from enum import Enum

from pydantic import BaseModel, Field


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
    """保存單一 OCR 文字區塊及其選用的版面座標與信心分數。"""

    text: str
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    confidence: float | None = None


class OCRDocument(BaseModel):
    """保存 OCR 全文、文字區塊及選用的頁面尺寸。"""

    text: str
    blocks: list[OCRTextBlock] = Field(default_factory=list)
    width: float | None = None
    height: float | None = None

    @property
    def has_coordinates(self) -> bool:
        """所有文字區塊皆有水平與垂直座標時回傳真值。"""
        return bool(self.blocks) and all(
            block.x is not None and block.y is not None
            for block in self.blocks
        )


class MarketplaceDetectionResult(BaseModel):
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
    """刊登主價格、商品欄位及候選價格的完整抽取結果。"""

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
    condition_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class MainPriceExtractionError(ValueError):
    """表示主價格不存在或信心不足，並保留完整抽取結果。"""

    def __init__(self, result: MainPriceExtractionResult):
        if not result.error_code:
            raise ValueError("MainPriceExtractionError 必須包含 error_code")
        self.result = result
        self.error_code = result.error_code
        super().__init__(result.message or result.error_code)
