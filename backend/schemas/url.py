from pydantic import BaseModel, Field, HttpUrl

class UrlRequest(BaseModel):
    url: HttpUrl

class UrlResponse(BaseModel):
    label: str
    score: str | float | None = None