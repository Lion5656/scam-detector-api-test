from pydantic import BaseModel, Field


class AnalysisInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class AnalysisResult(BaseModel):
    label: str
    score: str | float | None = None
    reason: str | None = None


class BaseEvidence(BaseModel):
    text: str
    rule_score: int
    rule_reason: str
    rule_hits: list[str]
    model_label: str
    model_confidence: float | None
    model_margin: float | None
    chunk_consistent: bool


class RagEvidence(BaseModel):
    used: bool
    label: str | None = None
    score: float | None = None
    reason: str | None = None
    raw_response: str | None = None


class RagResponse(BaseModel):
    urgency: float = Field(ge=0, le=1, description="催促語氣")
    money_related: float = Field(ge=0, le=1, description="金錢獲利")
    bating: float = Field(ge=0, le=1, description="異常操作")
    asks_for_personal_info: float = Field(ge=0, le=1, description="索取個資")
    reputation: float = Field(ge=0, le=1, description="不可信度")
    reason: str = Field(..., min_length=50, max_length=300, description="原因")
