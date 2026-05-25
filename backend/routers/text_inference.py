from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.text import TextResponse, TextRequest
from backend.services.text_service.text_analyzer import text_analyzer
from backend.utils.text_cleaner import normalize_text

router = APIRouter(prefix="/v1", tags=["text-inference"])


@router.post("/analyze/text/hybrid", response_model=TextResponse, summary="文字詐騙風險分析", description="使用規則、多模型做混合判斷")
async def analyze_text(req: TextRequest) -> TextResponse:
    try:
        text = normalize_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 不可為空")

        result = await text_analyzer.hybrid_detector(text)
        return TextResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")


@router.post("/analyze/text/model", response_model=TextResponse, summary="純模型文字風險分析", description="僅使用文字分類模型")
def model_analyze_text(req: TextRequest) -> TextResponse:
    try:
        text = normalize_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 不可為空")

        result = text_analyzer.model_detector(text)
        return TextResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
