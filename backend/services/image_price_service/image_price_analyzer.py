"""協調商品圖片的來源驗證、價格抽取與風險分析流程。"""

import logging
from typing import Any

from backend.config import settings
from backend.repository.case_repository import case_repository
from backend.services.dto.price_analysis import ImagePriceAnalysisResult
from backend.services.image_price_service.case_recorder import record_case
from backend.services.image_price_service.models import (
    MainPriceExtractionError,
    MainPriceExtractionResult,
    MarketplaceCondition,
    MarketplaceLayout,
)
from backend.services.image_price_service.ocr.ocr_service import (
    extract_ocr_document,
)
from backend.services.image_price_service.ocr.ocr_service import (
    ocr_service as default_ocr_service,
)
from backend.services.image_price_service.page_analysis.fb_marketplace.fb_marketplace_detector import (
    fb_marketplace_detector,
)
from backend.services.image_price_service.page_analysis.fb_marketplace.fb_marketplace_extractor import (
    fb_marketplace_price_extractor,
)
from backend.services.image_price_service.pricing.market_price_resolver import (
    resolve_market_price,
)
from backend.services.image_price_service.pricing.online_marketprice_service import (
    online_marketprice_service as default_online_price_service,
)
from backend.services.image_price_service.product.product_identifier import (
    product_identifier as default_product_identifier,
)
from backend.services.image_price_service.risk.fusion_decision_engine import (
    fusion_decision_engine as default_decision_engine,
)

logger = logging.getLogger(__name__)


class ImagePriceAnalyzer:
    """協調圖片 OCR、商品查價與風險判定流程。"""

    def __init__(
        self,
        *,
        ocr_service: Any | None = None,
        product_identifier: Any | None = None,
        online_price_service: Any | None = None,
        decision_engine: Any | None = None,
        case_repo: Any | None = None,
        price_extractor: Any | None = None,
        marketplace_detector: Any | None = None,
    ) -> None:
        """建立分析器並注入各階段使用的服務。"""
        self._ocr_service = (
            ocr_service if ocr_service is not None else default_ocr_service
        )
        self._product_identifier = (
            product_identifier
            if product_identifier is not None
            else default_product_identifier
        )
        self._online_price_service = (
            online_price_service
            if online_price_service is not None
            else default_online_price_service
        )
        self._decision_engine = (
            decision_engine
            if decision_engine is not None
            else default_decision_engine
        )
        self._case_repo = case_repo if case_repo is not None else case_repository
        self._price_extractor = (
            price_extractor
            if price_extractor is not None
            else fb_marketplace_price_extractor
        )
        self._marketplace_detector = (
            marketplace_detector
            if marketplace_detector is not None
            else fb_marketplace_detector
        )

    @staticmethod
    def _error_result(
        *,
        filename: str,
        content_type: str,
        text: str,
        error_code: str,
        message: str,
        layout: MarketplaceLayout | Any,
        evidence: list[str],
        marketplace_confidence: float = 0.0,
        extraction: MainPriceExtractionResult | None = None 
    ) -> ImagePriceAnalysisResult:
        """建立圖片分析失敗時的統一回應。"""
        return ImagePriceAnalysisResult(
            filename=filename or "unknown",
            content_type=content_type,
            success=False,
            error_code=error_code,
            message=message,
            extracted_text=text,
            product_name=None,
            brand_model=None,
            listed_price=None,
            market_price=0,
            market_price_source="not_evaluated",
            risk_label="UNKNOWN",
            score="未知",
            reason=message,
            confidence=0.0,
            evidence=evidence,
            decision_layer="source_validation",
            search_tool="unused",
            marketplace_layout=layout,
            marketplace_confidence=marketplace_confidence,
            extraction_confidence=extraction.confidence if extraction else None,
            price_source_text=extraction.source_text if extraction else None,
            price_extraction_reason=extraction.reason if extraction else None,
            seller_name=extraction.seller_name if extraction else None,
            condition=(
                extraction.condition
                if extraction
                else MarketplaceCondition.UNKNOWN
            ),
            extraction_warnings=extraction.warnings if extraction else [],
        )

    def image_price_detector(
        self,
        data: bytes,
        filename: str = "unknown",
        content_type: str = "",
    ) -> ImagePriceAnalysisResult:
        """分析 Marketplace 截圖並回傳價格風險結果。"""
        document = extract_ocr_document(self._ocr_service, data)
        text = document.text
        detection = self._marketplace_detector.detect(document)
        if not detection.is_marketplace:
            error_code = (
                "UNKNOWN_LAYOUT"
                if detection.confidence >= self._marketplace_detector.threshold
                and detection.layout == "unknown"
                else "INVALID_IMAGE_SOURCE"
            )
            message = (
                "無法辨識 FB Marketplace 商品頁版型"
                if error_code == "UNKNOWN_LAYOUT"
                else "圖片格式錯誤，來源需為 FB Marketplace 商品頁截圖"
            )
            return self._error_result(
                filename=filename,
                content_type=content_type,
                text=text,
                error_code=error_code,
                message=message,
                layout=detection.layout,
                marketplace_confidence=detection.confidence,
                evidence=detection.evidence,
            )

        try:
            extraction = self._price_extractor.extract(document, detection)
            # 相容以回傳值表示失敗、而非直接拋出例外的舊版或自訂抽取器。
            if extraction.price is None or extraction.error_code:
                extraction.error_code = extraction.error_code or "MAIN_PRICE_NOT_FOUND"
                extraction.message = extraction.message or "找不到 FB Marketplace 商品主價格"
                raise MainPriceExtractionError(extraction)
        except MainPriceExtractionError as e:
            extraction = e.result
            logger.warning(
                "主價格抽取失敗 error_code=%s confidence=%s candidates=%s rejected=%s",
                e.error_code,
                extraction.confidence,
                [
                    {
                        "amount": item.amount,
                        "source_text": item.source_text,
                        "confidence": item.confidence,
                    }
                    for item in extraction.candidates
                ],
                [
                    {
                        "amount": item.amount,
                        "source_text": item.source_text,
                        "section": item.section,
                        "confidence": item.confidence,
                        "reject_reason": item.reject_reason,
                    }
                    for item in extraction.rejected_candidates
                ],
            )
            return self._error_result(
                filename=filename,
                content_type=content_type,
                text=text,
                error_code=e.error_code,
                message=str(e),
                layout=detection.layout,
                marketplace_confidence=detection.confidence,
                evidence=detection.evidence,
                extraction=extraction,
            )

        if not extraction.product_name:
            extraction.error_code = "LOW_CONFIDENCE_PRODUCT_EXTRACTION"
            extraction.message = "找不到商品主價格上方的 Marketplace 商品標題"
            return self._error_result(
                filename=filename,
                content_type=content_type,
                text=text,
                error_code=extraction.error_code,
                message=extraction.message,
                layout=detection.layout,
                marketplace_confidence=detection.confidence,
                evidence=detection.evidence,
                extraction=extraction,
            )

        product_name = extraction.product_name
        product = self._product_identifier.identify(product_name)
        market_price, market_price_source, search_tool = resolve_market_price(
            self._online_price_service,
            product_name,
            product.brand_model,
            product.market_price,
            product.search_query,
            condition=extraction.condition,
            condition_text=product_name,
        )
        selling_price = extraction.price

        insufficient_search_results = (
            settings.ONLINE_PRICE_ENABLED and search_tool == "unused"
        )
        if insufficient_search_results:
            decision = {
                "risk_label": "UNKNOWN",
                "risk_score": "未知",
                "reason": "未知商品，搜索結果過少",
                "evidence": ["線上查價結果不足"],
                "confidence": 0.0,
                "decision_layer": "source_validation",
            }
            resolved_condition = MarketplaceCondition.UNKNOWN
        else:
            decision = self._decision_engine.evaluate(
                product_name=product_name,
                brand_model=product.brand_model,
                text=text,
                selling_price=selling_price,
                market_price=market_price,
                market_price_source=market_price_source,
            )
            resolved_condition = extraction.condition
        decision_evidence = decision.get("evidence")
        evidence = (
            [str(item) for item in decision_evidence]
            if isinstance(decision_evidence, list)
            else []
        )
        result = ImagePriceAnalysisResult(
            filename=filename or "unknown",
            content_type=content_type,
            extracted_text=text,
            success=True,
            error_code=None,
            message=None,
            product_name=product_name,
            brand_model=product.brand_model,
            listed_price=selling_price,
            market_price=market_price,
            market_price_source=market_price_source,
            risk_label=decision.get("risk_label", "UNKNOWN"),
            score=decision.get("risk_score"),
            reason=str(decision.get("reason") or ""),
            evidence=evidence,
            confidence=float(decision.get("confidence") or 0.0),
            decision_layer=decision.get("decision_layer", "fast"),
            search_tool=search_tool,
            marketplace_layout=detection.layout,
            marketplace_confidence=detection.confidence,
            extraction_confidence=extraction.confidence,
            price_source_text=extraction.source_text,
            price_extraction_reason=extraction.reason,
            seller_name=extraction.seller_name,
            condition=resolved_condition,
            extraction_warnings=extraction.warnings,
        )

        record_case(self._case_repo, result)
        return result


image_price_analyzer = ImagePriceAnalyzer()
