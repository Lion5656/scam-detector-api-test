"""建立後端統一回傳的格式"""
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

ResponseData = TypeVar("ResponseData")

class ApiResponse(BaseModel, Generic[ResponseData]):
    """統一回傳格式"""
    success: bool = True
    version: str = "1.0"
    data: ResponseData | None = None

class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(
            alias_generator=to_camel,
            populate_by_name=True,
            serialize_by_alias=True,
    )
       
    """統一回傳錯誤格式"""
    success: bool = False
    version: str = "1.0"
    error_code: int
    error_message: object | None = None
    