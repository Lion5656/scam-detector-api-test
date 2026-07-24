from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class TextResponse(BaseModel):
    label: str
    score: str | float | None = None
    reason: str | None = None
    cls_model_confidence: float | None = None
    # decision_source: str | None = None
    # llm_used: bool | None = None
    # route_reason: str | None = None