from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.services.image_price_service.models import MarketplaceCondition


class ImageUploadResponse(BaseModel):
    filename: str = Field(..., description="上傳檔名")
    content_type: str = Field(..., description="檔案 MIME 類型")
    size_bytes: int = Field(..., ge=1, description="檔案大小（byte）")
    message: str = Field(..., description="處理結果訊息")


class ImagePriceDebugInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: MarketplaceCondition | None = None
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    price_source_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ImagePriceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str | None = None
    listed_price: int | None = Field(default=None, ge=0)
    seller_name: str | None = None
    risk_label: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    result: str
    debug: ImagePriceDebugInfo | None = None


class OCRResponse(BaseModel):
    extracted_text: str = Field(..., description="OCR 萃取文字")
