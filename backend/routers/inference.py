from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.analysis import Request, Response
from backend.services.text_analyzer import inference_engine
from backend.utils.text_cleaner import normalize_text

# 定義路由群組
router = APIRouter(prefix="/v1", tags=["inference"]) # 加上v1前綴，UI加上inference群組

# regex + model 分析
@router.post("/analyze/text", response_model=Response, summary="執行文本風險分析", description="結合正則模糊比對和模型預測")
async def analyze_text(req: Request) -> Response:
    try:
        text = normalize_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 不能為空白")
        
        result = await inference_engine.cascaded_detector(text)
        
        return Response(**result)
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
    
# model only 分析
@router.post("/analyze/text/model", response_model=Response, summary="執行文本風險分析", description="純模型預測")
async def model_analyze_text(req: Request) -> Response:
    try:
        text = normalize_text(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 不能為空白")
        
        result = await inference_engine.model_detector(text)

        return Response(**result)
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")



