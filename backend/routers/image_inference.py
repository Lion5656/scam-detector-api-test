from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.repository.case_repository import case_repository
from backend.schemas.image import ImageScamResponse, ImageUploadResponse
from backend.services.image_service.image_analyzer import image_analyzer
from backend.services.image_service.intelligent_decision_engine import intelligent_decision_engine
from backend.services.text_service.text_analyzer import text_analyzer

router = APIRouter(prefix="/v1", tags=["image-inference"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB


@router.post(
    "/analyze/image/upload",
    response_model=ImageUploadResponse,
    summary="圖片上傳測試",
    description="用於 Swagger 測試圖片上傳是否成功。",
)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="僅支援 JPG、PNG、WEBP 圖片格式",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上傳檔案不可為空")

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="圖片大小不可超過 20MB")

    return ImageUploadResponse(
        filename=file.filename or "unknown",
        content_type=content_type,
        size_bytes=len(data),
        message="圖片已成功上傳，可進入下一步辨識流程",
    )


@router.post(
    "/analyze/image/scam",
    response_model=ImageScamResponse,
    summary="圖片詐騙風險分析",
    description="OCR 文字抽取後串接既有文字詐騙分析，並加入低於行情 50% 的高風險規則。",
)
async def analyze_image_scam(file: UploadFile = File(...)) -> ImageScamResponse:
    try:
        content_type = file.content_type or ""
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="僅支援 JPG、PNG、WEBP 圖片格式")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上傳檔案不可為空")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="圖片大小不可超過 20MB")

        insight = image_analyzer.analyze_image_bytes(data)
        if not insight.extracted_text:
            return ImageScamResponse(
                filename=file.filename or "unknown",
                content_type=content_type,
                size_bytes=len(data),
                product_name=insight.product_name,
                brand_model=insight.brand_model,
                selling_price=insight.selling_price,
                market_price=insight.market_price,
                market_price_source=insight.market_price_source,
                is_high_risk_below_market=insight.is_high_risk_below_market,
                label="未知風險",
                score="未知",
                reason="圖片文字辨識不足，無法判斷風險",
                cls_model_confidence=0.0,
                extracted_text="",
                evidence=["OCR 無有效文字"],
                confidence=0.0,
                decision_layer="fast",
                tool_observations={},
                case_id=None,
            )

        text_result = await text_analyzer.hybrid_detector(insight.extracted_text)
        intelligent = intelligent_decision_engine.evaluate(
            text_result=text_result,
            product_name=insight.product_name,
            brand_model=insight.brand_model,
            text=insight.extracted_text,
        )

        label = intelligent.get("risk_label", text_result.get("label", "未知風險"))
        score = intelligent.get("risk_score", text_result.get("score"))
        reason = str(intelligent.get("reason") or text_result.get("reason") or "")
        evidence = [str(item) for item in intelligent.get("evidence", [])]
        decision_layer = str(intelligent.get("decision_layer") or "fast")
        confidence = float(intelligent.get("confidence") or text_result.get("cls_model_confidence") or 0.0)
        tool_observations = dict(intelligent.get("tool_observations") or {})

        if insight.is_high_risk_below_market:
            label = "高風險"
            if isinstance(score, (int, float)):
                score = max(float(score), 90.0)
            else:
                score = 90.0

            price_reason = (
                f"販售價格 {insight.selling_price} 低於正常市價 {insight.market_price} 的 50%，"
                "判定為高風險低於行情"
            )
            reason = f"{reason}；{price_reason}" if reason else price_reason
            evidence.append("低於行情 50% 規則觸發")

        case_id = None
        if settings.CASE_MEMORY_ENABLED:
            try:
                case_payload = {
                    "filename": file.filename or "unknown",
                    "product_name": insight.product_name,
                    "brand_model": insight.brand_model,
                    "selling_price": insight.selling_price,
                    "market_price": insight.market_price,
                    "market_price_source": insight.market_price_source,
                    "risk_label": str(label),
                    "risk_score": score,
                    "reason": reason,
                    "evidence": evidence,
                    "confidence": confidence,
                    "decision_layer": decision_layer,
                    "tool_observations": tool_observations,
                    "extracted_text": insight.extracted_text,
                }
                case_id = case_repository.append_case(case_payload)
            except Exception:
                case_id = None

        return ImageScamResponse(
            filename=file.filename or "unknown",
            content_type=content_type,
            size_bytes=len(data),
            product_name=insight.product_name,
            brand_model=insight.brand_model,
            selling_price=insight.selling_price,
            market_price=insight.market_price,
            market_price_source=insight.market_price_source,
            is_high_risk_below_market=insight.is_high_risk_below_market,
            label=str(label),
            score=score,
            reason=reason,
            cls_model_confidence=text_result.get("cls_model_confidence"),
            extracted_text=insight.extracted_text,
            evidence=evidence,
            confidence=confidence,
            decision_layer=decision_layer,
            tool_observations=tool_observations,
            case_id=case_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Error")
