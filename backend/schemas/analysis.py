from pydantic import BaseModel, Field, HttpUrl


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class URLRequest(BaseModel):
    url: HttpUrl


class Response(BaseModel):
    label: str
    score: str | float | None = None
    reason: str | None = None
    decision_source: str | None = None
    llm_used: bool | None = None
    route_reason: str | None = None
