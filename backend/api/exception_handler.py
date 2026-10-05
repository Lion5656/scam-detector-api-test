import logging
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ExceptionHandler

from backend.api.response import ApiErrorResponse

logger = logging.getLogger(__name__)

def make_error_response(status_code: int, message: object | None = None) -> JSONResponse:
    """建立統一的錯誤回傳格式"""
    response_content = ApiErrorResponse(
        success=False,
        version="1.0",
        error_code=status_code,
        error_message=message
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(response_content))

async def handle_global_exception(req: Request, exc: Exception) -> JSONResponse:
    """處理全域錯誤"""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return make_error_response(status_code=500, message="Internal Server Error")

async def handle_http_exception(req: Request, exc: HTTPException) -> JSONResponse:
    """處理 HTTP 錯誤"""
    logger.error(f"HTTP Exception: {str(exc.detail)}")
    return make_error_response(status_code=exc.status_code, message=exc.detail)

async def handle_validation_exception(req: Request, exc: RequestValidationError) -> JSONResponse:
    """處理資料驗證錯誤"""
    logger.error(f"Validation Error: {exc.errors()}")
    return make_error_response(status_code=422, message=exc.errors())

def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Exception, handle_global_exception)
    app.add_exception_handler(HTTPException, cast(ExceptionHandler, handle_http_exception))
    app.add_exception_handler(RequestValidationError, cast(ExceptionHandler, handle_validation_exception))
