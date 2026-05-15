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