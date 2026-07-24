"""將圖片價格分析結果轉成案例資料並以 best-effort 方式保存。"""

import logging
from typing import Any

from backend.config import settings
from backend.services.dto.price_analysis import ImagePriceAnalysisResult

logger = logging.getLogger(__name__)


def record_case(
    case_repo: Any,
    result: ImagePriceAnalysisResult,
) -> str | None:
    """依設定保存分析案例；寫入失敗時維持原分析結果。"""
    if not settings.CASE_MEMORY_ENABLED:
        return None

    try:
        return case_repo.save(
            {
                "filename": result.filename,
                "content_type": result.content_type,
                "product_name": result.product_name,
                "brand_model": result.brand_model,
                "selling_price": result.listed_price,
                "market_price": result.market_price,
                "market_price_source": result.market_price_source,
                "risk_label": result.risk_label,
                "risk_score": result.score,
                "reason": result.reason,
                "confidence": result.confidence,
                "decision_layer": result.decision_layer,
                "extracted_text": result.extracted_text,
                "marketplace_layout": result.marketplace_layout,
                "marketplace_confidence": result.marketplace_confidence,
                "price_source_text": result.price_source_text,
                "price_extraction_reason": result.price_extraction_reason,
                "seller_name": result.seller_name,
                "condition": result.condition,
                "extraction_warnings": result.extraction_warnings,
            }
        )
    except Exception:
        logger.warning("圖片價格分析案例保存失敗", exc_info=True)
        return None
