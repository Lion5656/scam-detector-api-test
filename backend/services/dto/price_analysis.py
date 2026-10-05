from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.services.image_price_service.domain.models import (
    MarketplaceCondition,
    MarketplaceLayout,
)
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
)

RiskLabel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
DecisionLayer = Literal["fast", "llm", "llm_simulated", "decision_error"]
MarketPriceSource = Literal["online", "fallback_local", "not_evaluated"]
SearchTool = Literal["serp_api", "tavily", "ddgs"]
MarketPriceStatus = Literal["success", "insufficient", "not_found"]
MarketPriceReferenceMode = Literal["iqr", "median_low_sample"]


class MarketPriceCandidateEvidence(BaseModel):
    """保留已驗證市場候選的來源與價格證據。"""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    title: str
    price: int = Field(gt=0)
    condition: MarketplaceCondition
    evidence: str


class MarketPriceEstimate(BaseModel):
    """保留市場價格區間、資料量、來源與可信度。"""

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
    confidence: float = Field(ge=0.0, le=1.0)
    search_tools: list[SearchTool] = Field(default_factory=list)
    candidates: tuple[MarketPriceCandidateEvidence, ...] = ()


class DeepAnalysisReview(BaseModel):
    """LLM 對有限商品狀態原文所做的結構化複核。"""

    model_config = ConfigDict(extra="forbid")

    reviewed_condition: MarketplaceCondition
    condition_detail: str = ""
    condition_evidence: str = ""
    reason: str
    review_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_review_consistency(self) -> "DeepAnalysisReview":
        """拒絕缺少證據或與 UNKNOWN 狀態矛盾的複核結果。"""
        if (
            self.reviewed_condition != MarketplaceCondition.UNKNOWN
            and not self.condition_evidence.strip()
        ):
            raise ValueError("確認商品狀態時必須提供可追溯的狀態原文證據")

        if (
            self.reviewed_condition == MarketplaceCondition.UNKNOWN
            and self.condition_detail.strip()
        ):
            raise ValueError("商品狀態為 UNKNOWN 時不得提供已確認的狀態細節")

        return self


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
    market_price_source: MarketPriceSource = "not_evaluated"
    market_price_estimates: tuple[MarketPriceEstimate, ...] = ()
    risk_label: RiskLabel = "UNKNOWN"
    score: str | float | None = None
    reason: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    decision_layer: DecisionLayer = "fast"
    search_tools: list[SearchTool] = Field(default_factory=list)
    marketplace_layout: MarketplaceLayout = MarketplaceLayout.UNKNOWN
    marketplace_confidence: float = Field(default=0.0, ge=0, le=1)
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    price_source_text: str | None = None
    price_extraction_reason: str | None = None
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
    extraction_warnings: list[str] = Field(default_factory=list)

    @field_validator("listed_price")
    @classmethod
    def validate_supported_listing_price(cls, value: int | None) -> int | None:
        """拒絕超出服務支援上限的刊登價格。"""
        if (
            value is not None
            and value > DEFAULT_PRICE_RISK_POLICY.maximum_supported_price
        ):
            raise ValueError(
                "PRICE_OUT_OF_SUPPORTED_RANGE：刊登價格超出支援範圍"
            )
        return value


class ProductIdentification(BaseModel):
    """商品辨識階段輸出的正規化資訊、查價搜尋詞與本地參考價格。"""

    product_name: str
    brand_model: str
    known_specs: list[str] = Field(default_factory=list)
    search_query: str = ""
    market_price: int = Field(default=0, ge=0)


class OCRResponse(BaseModel):
    extracted_text: str = Field(..., description="OCR 萃取文字")
