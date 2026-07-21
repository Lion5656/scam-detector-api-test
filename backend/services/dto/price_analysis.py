from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.services.image_price_service.models import (
    MarketplaceCondition,
    MarketplaceLayout 
)
RiskLabel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
DecisionLayer = Literal["fast", "llm", "llm_simulated", "source_validation"]


class ImagePriceAnalysisResult(BaseModel):
    """圖片價格分析的內部完整結果"""

    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: str
    extracted_text: str
    success: bool = True
    error_code: str | None = None
    message: str | None = None
    product_name: str | None = None
    brand_model: str | None = None
    listed_price: int | None = Field(default=None, ge=0)
    market_price: int = Field(default=0, ge=0)
    risk_label: RiskLabel = "UNKNOWN"
    has_risk: bool | None = None
    score: str | float | None = None
    reason: str | None = None
    evidence: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    decision_layer: DecisionLayer = "fast"
    marketplace_layout: MarketplaceLayout = MarketplaceLayout.UNKNOWN
    marketplace_confidence: float = Field(default=0.0, ge=0, le=1)
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    price_source_text: str | None = None
    price_extraction_reason: str | None = None
    seller_name: str | None = None
    condition: MarketplaceCondition = MarketplaceCondition.UNKNOWN
    extraction_warnings: list[str] = Field(default_factory=list)


class ProductIdentification(BaseModel):
    """商品辨識階段輸出的商品名稱、品牌型號與本地參考價格。"""

    product_name: str
    brand_model: str
    market_price: int = Field(default=0, ge=0)
