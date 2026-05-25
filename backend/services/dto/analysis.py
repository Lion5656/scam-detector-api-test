from pydantic import BaseModel

class BaseEvidence(BaseModel):
    text: str
    rule_score: int
    rule_reason: str
    rule_hits: list[str]
    model_label: str
    model_confidence: float | None
    model_margin: float | None
    chunk_consistent: bool