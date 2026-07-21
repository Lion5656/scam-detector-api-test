"""商品圖片上傳、價格風險分析與 Google OCR 驗證 API。

正式價格端點只負責 HTTP 輸入驗證、呼叫 image_price_service，並將完整 Service
結果封裝成 ImagePriceResponse；OCR、商品辨識、價格統計與風險決策不在本模組。
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.schemas.image import (
    ImagePriceDebugInfo,
    ImagePriceResponse,
    ImageUploadResponse,
)
from backend.services.image_price_service.image_price_analyzer import \
    image_price_analyzer

router = APIRouter(prefix="/v1", tags=["image-inference"])


@router.post(
    "/analyze/image/upload",
    response_model=ImageUploadResponse,
    summary="圖片上傳測試",
    description="用於測試圖片上傳是否成功。",
)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    """驗證圖片 MIME type、非空內容與大小，回傳上傳資訊但不執行分析。"""
    content_type = file.content_type or ""
    if content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="僅支援 JPG、PNG、WEBP 圖片格式",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上傳檔案不可為空")

    if len(data) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="圖片大小不可超過 20MB")

    return ImageUploadResponse(
        filename=file.filename or "unknown",
        content_type=content_type,
        size_bytes=len(data),
        message="圖片已成功上傳，可進入下一步辨識流程",
    )


@router.post(
    "/analyze/price",
    response_model=ImagePriceResponse,
    summary="價格驗證風險分析",
    description="使用 OCR 分析圖片商品價格",
)
async def analyze_image_price(file: UploadFile = File(...)) -> ImagePriceResponse:
    """驗證商品圖片 HTTP 輸入並封裝 Service 分析結果。

    Router 僅驗證 MIME type、非空內容與大小，呼叫 image_price_detector 後以
    DTO 欄位建立 ImagePriceResponse，不計算或覆寫任何風險結果。
    """
    try:
        content_type = file.content_type or ""
        if content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="僅支援 JPG、PNG、WEBP 圖片格式")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上傳檔案不可為空")
        if len(data) > settings.MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="圖片大小不可超過 20MB")

        result = image_price_analyzer.image_price_detector(
            data=data,
            filename=file.filename or "unknown",
            content_type=content_type,
        )
        warnings = list(result.extraction_warnings)
        if not result.success and result.message and result.message not in warnings:
            warnings.append(result.message)
        return ImagePriceResponse(
            product_name=result.product_name if result.success else None,
            listed_price=result.listed_price if result.success else None,
            seller_name=result.seller_name if result.success else None,
            risk_label=result.risk_label,
            result=(
                "完成商品價格風險檢測。"
                if result.success
                else result.message or "商品價格風險檢測失敗。"
            ),
            debug=ImagePriceDebugInfo(
                condition=result.condition,
                extraction_confidence=result.extraction_confidence,
                price_source_text=result.price_source_text,
                warnings=warnings,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
    
    
