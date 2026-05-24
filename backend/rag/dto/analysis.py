from pydantic import BaseModel, Field

class RagEvidence(BaseModel):
    used: bool
    label: str | None = None
    score: float | None = None
    reason: str | None = None
    raw_response: str | None = None


class RagResponse(BaseModel):
    urgency: float = Field(ge=0, le=1, description="催促語氣")
    money_related: float = Field(ge=0, le=1, description="金錢獲利")
    baiting: float = Field(ge=0, le=1, description="異常誘導")
    asks_for_personal_info: float = Field(ge=0, le=1, description="索取個資")
    reputation: float = Field(ge=0, le=1, description="不可信度")
    is_anti_fraud: float = Field(ge=0, le=1, description="是否為反詐騙")
    social_engineering: float = Field(default=0.0, ge=0, le=1, description="社交工程和情緒操控")
    reason: str = Field(..., min_length=50, max_length=300, description="原因")
    suggestion: str = Field(..., min_length=10, max_length=30, description="建議")
