from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.services.dto.price_analysis import (
    DecisionLayer,
    MarketPriceReferenceMode,
    MarketPriceSource,
    MarketPriceStatus,
    SearchTool,
)
from backend.services.image_price_service.domain.models import MarketplaceCondition


class ImageUploadResponse(BaseModel):
    filename: str = Field(..., description="上傳檔名")
    content_type: str = Field(..., description="檔案 MIME 類型")
    size_bytes: int = Field(..., ge=1, description="檔案大小（byte）")
    message: str = Field(..., description="處理結果訊息")


class MarketPriceEstimateResponse(BaseModel):
    """對外回傳且不含市場候選明細的價格區間。"""

    model_config = ConfigDict(extra="forbid")

    status: MarketPriceStatus
    condition: MarketplaceCondition
    reference_mode: MarketPriceReferenceMode
    median_price: int = Field(ge=0)
    low_price: int = Field(ge=0)
    high_price: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    site_count: int = Field(ge=0)
    source: MarketPriceSource
    confidence: float = Field(ge=0, le=1)


class ImagePriceDebugInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_tools: list[SearchTool] = Field(default_factory=list)
    market_price_source: MarketPriceSource = "not_evaluated"
    market_price_estimates: tuple[MarketPriceEstimateResponse, ...] = ()
    condition_source_text: str = ""
    condition_extraction_confidence: float = Field(default=0.0, ge=0, le=1)
    price_source_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ImagePriceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str | None = None
    condition: MarketplaceCondition | None = None
    condition_detail: str = ""
    listed_price: int | None = Field(default=None, ge=0)
    market_price: int | None = Field(default=None, ge=0)
    seller_name: str | None = None
    risk_label: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    risk_score: str | float | None = None
    decision_layer: DecisionLayer
    error_code: str | None = None
    result: str | None = None
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    debug: ImagePriceDebugInfo | None = None
