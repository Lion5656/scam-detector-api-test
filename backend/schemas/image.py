from pydantic import BaseModel, Field


class ImageUploadResponse(BaseModel):
    filename: str = Field(..., description="上傳檔名")
    content_type: str = Field(..., description="檔案 MIME 類型")
    size_bytes: int = Field(..., ge=1, description="檔案大小（byte）")
    message: str = Field(..., description="處理結果訊息")


class ImageScamResponse(BaseModel):
    filename: str = Field(..., description="上傳檔名")
    content_type: str = Field(..., description="檔案 MIME 類型")
    size_bytes: int = Field(..., ge=1, description="檔案大小（byte）")
    product_name: str = Field(..., description="主要商品名稱")
    brand_model: str = Field(..., description="辨識到的品牌與型號")
    selling_price: int = Field(..., ge=0, description="廣告販售價格，找不到則為 0")
    market_price: int = Field(..., ge=0, description="估算正常市價")
    market_price_source: str = Field(..., description="市價來源：online 或 fallback_local")
    is_high_risk_below_market: bool = Field(..., description="是否低於行情 50% 高風險")
    label: str = Field(..., description="最終詐騙風險標籤")
    score: str | float | None = Field(default=None, description="最終風險分數")
    reason: str | None = Field(default=None, description="判定原因")
    cls_model_confidence: float | None = Field(default=None, description="文字模型信心度")
    extracted_text: str = Field(..., description="OCR 萃取文字")
    evidence: list[str] = Field(default_factory=list, description="結構化證據清單")
    confidence: float | None = Field(default=None, description="最終決策信心度")
    decision_layer: str = Field(default="fast", description="決策層級：fast / llm / llm_simulated")
    tool_observations: dict = Field(default_factory=dict, description="工具輸出摘要")
    case_id: str | None = Field(default=None, description="案例記錄 ID")
