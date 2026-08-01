"""建立後端統一回傳的格式"""
from typing import Generic, TypeVar

from pydantic import BaseModel

ResponseData = TypeVar("ResponseData")

class ApiResponse(BaseModel, Generic[ResponseData]):
    """統一回傳格式"""
    success: bool = True
    version: str = "1.0"
    data: ResponseData | None = None

class ApiErrorResponse(BaseModel):
    """統一回傳錯誤格式"""
    success: bool = False
    version: str = "1.0"
    error_code: int
    error_message: object | None = None
    