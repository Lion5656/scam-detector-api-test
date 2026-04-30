from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.analysis import Response, URLRequest
from backend.services.url_analyzer import detector

router = APIRouter(prefix="/v1", tags=["url-detector"])

# url網址分析
@router.post("/anaylze/url", response_model=Response, summary="執行url分析")
async def analyze_url(req: URLRequest) -> Response:
    try:
        url = str(req.url)
        result = await detector.url_detector(url)

        return Response(**result)
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail={e})
        raise HTTPException(status_code=500, detail="Internal Error")
    