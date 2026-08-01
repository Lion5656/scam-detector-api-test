from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class TextResponse(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )   
    label: str
    score: str | float | None = None
    reason: str | None = None
    cls_model_confidence: float | None = None
    # decision_source: str | None = None
    # llm_used: bool | None = None
    # route_reason: str | None = None