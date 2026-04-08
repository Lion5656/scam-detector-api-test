from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.analysis import Request, Response
from backend.services.ai_engine import inference_engine

# 定義路由群組
router = APIRouter(prefix="/v1", tags=["inference"]) # 加上v1前綴，UI加上inference群組

# regex + model 分析
@router.post("/analyze", response_model=Response, summary="執行文本風險分析", description="結合正則模糊比對和模型預測")
async def anaylze_text(req: Request):
    try:
        result = await inference_engine.cascaded_detector(req)
        
        return result
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
    
# model only 分析
@router.post("/model_anaylze", response_model=Response, summary="執行文本風險分析", description="純模型預測")
async def model_analyze_text(req: Request):
    try:
        result = await inference_engine.model_detector(req)

        return result
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")



