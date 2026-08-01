from fastapi import APIRouter, HTTPException

from backend.api.response import ApiResponse
from backend.api.schemas.url import UrlRequest, UrlResponse
from backend.core.config import settings
from backend.services.url_service.url_analyzer import detector

router = APIRouter(prefix="/v1", tags=["url-detector"])

# url網址分析
@router.post("/url/analyze", response_model=ApiResponse[UrlResponse], summary="執行url分析")
def analyze_url(req: UrlRequest) -> ApiResponse[UrlResponse]:
    try:
        url = str(req.url)
        result = detector.url_detector(url)
        data = UrlResponse(**result)
        return ApiResponse(data=data)
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
    
