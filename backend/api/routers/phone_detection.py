from fastapi import APIRouter, HTTPException

from backend.api.response import ApiResponse
from backend.api.schemas.phone import (PhoneQueryRequest, PhoneQueryResponse,
                                       PhoneReportError, PhoneReportRequest,
                                       PhoneReportResponse)
from backend.services.phone_service import phone_service

router = APIRouter(prefix="/v1", tags=["phone"])


@router.post(
    "/phones/search",
    response_model=ApiResponse[PhoneQueryResponse],
    summary="電話號碼查詢",
    description="查詢電話號碼是否已存在於黑白名單。若結果為 unknown，可提示使用者進行回報。",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "phone_number": "0422255141"
                    }
                }
            }
        }
    },
)
async def query_phone(req: PhoneQueryRequest) -> ApiResponse[PhoneQueryResponse]:
    try:
        result = phone_service.query_phone(req.phone_number)
        data = PhoneQueryResponse(**result)
        return ApiResponse(data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/phones/report",
    response_model=ApiResponse[PhoneReportResponse],
    summary="可疑號碼回報",
    description=(
        "當查詢結果為 unknown 時，可回報該號碼為可疑名單。"
        " 若選擇「其他」，請一併填寫 other_type 欄位。"
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "standard": {
                            "summary": "回報常見類別",
                            "value": {
                                "phone_number": "0987654321",
                                "phone_type": "詐騙"
                            }
                        },
                        "other": {
                            "summary": "回報其他類別",
                            "value": {
                                "phone_number": "0987654321",
                                "phone_type": "其他",
                                "other_type": "疑似詐騙廣告"
                            }
                        }
                    }
                }
            }
        }
    },
)
async def report_phone(req: PhoneReportRequest) -> ApiResponse[PhoneReportResponse]:
    try:
        result = phone_service.report_suspicious(req.phone_number, req.phone_type, req.other_type)
        data = PhoneReportResponse(**result)
        return ApiResponse(data=data)
    except PhoneReportError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
