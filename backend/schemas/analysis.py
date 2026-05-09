from pydantic import BaseModel, Field, HttpUrl


# 請求的datamodel
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000) # 文字必填, 大小 1~10000

class URLRequest(BaseModel):
    url: HttpUrl

# 回傳的datamodel
class Response(BaseModel):
    label: str
    score: str | float | None = None
    reason: str | None = None
