# 請求的datamodel
from pydantic import BaseModel, Field


# 請求的datamodel
class Request(BaseModel):
    text: str = Field(..., min_length=1, max_length=512) # 文字必填, 大小 1~512

# 回傳的datamodel
class Response(BaseModel):
    label: str
    score: str | float | None = None
    confidence_score: float | None = None
    reason: str | None = None