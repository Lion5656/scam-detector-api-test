from pydantic import BaseModel, HttpUrl


class UrlRequest(BaseModel):
    url: HttpUrl

class UrlResponse(BaseModel):
    label: str
    score: str | float | None = None
    reason: str 