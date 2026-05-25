from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.url import UrlResponse, UrlRequest
from backend.services.url_service.url_analyzer import detector

router = APIRouter(prefix="/v1", tags=["url-detector"])

# url網址分析
@router.post("/anaylze/url", response_model=UrlResponse, summary="執行url分析")
def analyze_url(req: UrlRequest) -> UrlResponse:
    try:
        url = str(req.url)
        result = detector.url_detector(url)

        return UrlResponse(**result)
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
    
