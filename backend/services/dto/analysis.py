from pydantic import BaseModel, Field


class AnalysisInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    
class AnalysisResult(BaseModel):
    label: str
    score: str | float | None = None
    confidence_score: float | None = None
    reason: str | None = None