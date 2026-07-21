"""協調商品圖片的來源驗證、價格抽取與風險分析流程。"""

import logging
from typing import Any, cast

from backend.config import settings
from backend.repository.case_repository import case_repository
from backend.services.dto.price_analysis import (
    DecisionLayer,
    ImagePriceAnalysisResult,
    RiskLabel,
)
from backend.services.image_price_service.models import (
    MainPriceExtractionError,
    MainPriceExtractionResult,
    MarketplaceCondition,
    MarketplaceLayout,
    OCRDocument,
)
from backend.services.image_price_service.platform.fb_marketplace.fb_marketplace_detector import (
    fb_marketplace_detector,
)
from backend.services.image_price_service.platform.fb_marketplace.fb_marketplace_extractor import (
    fb_marketplace_price_extractor,
)
from backend.services.image_price_service.pricing.online_marketprice_service import (
    online_marketprice_service,
)
from backend.services.image_price_service.product.product_identifier import (
    product_identifier,
)
from backend.services.image_price_service.risk.fusion_decision_engine import fusion_decision_engine
from backend.services.image_price_service.ocr.google_vision_ocr_service import google_vision_ocr_service

logger = logging.getLogger(__name__)


class ImagePriceAnalyzer:
    """執行商品頁截圖從 OCR 到風險決策的完整分析流程。

    路由層傳入圖片內容、檔名與媒體類型後，即可將回傳的資料傳輸物件映射為
    API 回應。OCR 文字僅供商品辨識、價格抽取、黑名單檢查及商品風險分析使用，
    不會送入一般文字詐騙偵測流程。
    """

    def __init__(
        self,
        *,
        ocr_service = google_vision_ocr_service,
        online_price_service = online_marketprice_service
    ):
        self._ocr_service = ocr_service or google_vision_ocr_service
        self._online_price_service = (
            online_price_service or online_marketprice_service
        )

        self._product_identifier = product_identifier
        self._price_extractor = fb_marketplace_price_extractor
        self._marketplace_detector = fb_marketplace_detector
        self._decision_engine = fusion_decision_engine
        self._case_repo = case_repository

    def _resolve_market_price(
        self,
        product_name: str,
        brand_model: str,
        fallback_price: int,
    ) -> int:
        """取得線上市場價；查價停用或無有效結果時回傳本地參考價。"""
        if settings.ONLINE_PRICE_ENABLED:
            query = brand_model if brand_model != "未知型號" else product_name
            online_price = self._online_price_service.estimate_taiwan_market_price(
                query,
                max_results=settings.ONLINE_PRICE_MAX_RESULTS,
            )
            if online_price > 0:
                return online_price

        return fallback_price

    @staticmethod
    def _has_price_risk(selling_price: int, market_price: int) -> bool:
        """判斷正數售價是否低於市價五成，或達到市價兩倍以上。"""
        return (
            selling_price > 0
            and market_price > 0
            and (
                selling_price < market_price * 0.5
                or selling_price >= market_price * 2
            )
        )

    @staticmethod
    def _enforce_price_risk(
        decision: dict[str, Any],
        selling_price: int,
        market_price: int,
        is_high_risk: bool,
    ) -> dict[str, Any]:
        """將已觸發的價格異常規則套用至最終決策。

        當售價低於市價五成或達到市價兩倍以上時，將標籤設為高風險、分數
        提高至至少 90，並補上對應原因與證據，避免深度分析覆蓋硬性規則。
        """
        if not is_high_risk:
            return decision

        score = decision.get("risk_score")
        enforced_score = max(float(score), 90.0) if isinstance(score, (int, float)) else 90.0
        reason = str(decision.get("reason") or "")
        if selling_price < market_price * 0.5:
            price_reason = (
                f"販售價格 {selling_price} 低於正常市價 {market_price} 的 50%，"
                "判定為高風險低於行情"
            )
            price_evidence = "低於行情 50% 規則觸發"
        else:
            price_reason = (
                f"販售價格 {selling_price} 達正常市價 {market_price} 的 2 倍以上，"
                "判定為高風險高於行情"
            )
            price_evidence = "高於行情 2 倍規則觸發"

        evidence = [str(item) for item in decision.get("evidence") or []]
        if price_evidence not in evidence:
            evidence.append(price_evidence)

        return {
            **decision,
            "risk_label": "高風險",
            "risk_score": enforced_score,
            "reason": f"{reason}；{price_reason}" if reason else price_reason,
            "evidence": evidence,
        }

    @staticmethod
    def _normalize_risk_label(label: object) -> RiskLabel:
        """將中英文風險標籤轉成 API 使用的標準值。"""
        value = str(label or "").strip()
        normalized = value.upper()
        if normalized == "HIGH" or "高" in value:
            return "HIGH"
        if normalized == "MEDIUM" or "中" in value:
            return "MEDIUM"
        if normalized == "LOW" or "低" in value:
            return "LOW"
        return "UNKNOWN"

    @staticmethod
    def _normalize_decision_layer(layer: object) -> DecisionLayer:
        """驗證決策層名稱，未知值一律降級為快速決策層。"""
        value = str(layer or "fast").strip().lower()
        if value in {"fast", "llm", "llm_simulated", "source_validation"}:
            return cast(DecisionLayer, value)
        return "fast"

    def _store_case(self, result: ImagePriceAnalysisResult) -> str | None:
        """依設定保存分析案例；寫入失敗時維持原分析結果。"""
        if not settings.CASE_MEMORY_ENABLED:
            return None

        try:
            return self._case_repo.append_case(
                {
                    "filename": result.filename,
                    "content_type": result.content_type,
                    "product_name": result.product_name,
                    "brand_model": result.brand_model,
                    "selling_price": result.listed_price,
                    "market_price": result.market_price,
                    "risk_label": result.risk_label,
                    "has_risk": result.has_risk,
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
            pass

    @staticmethod
    def _error_result(
        *,
        filename: str,
        content_type: str,
        text: str,
        error_code: str,
        message: str,
        layout: MarketplaceLayout,
        evidence: list[str],
        marketplace_confidence: float = 0.0,
        extraction: MainPriceExtractionResult | None = None 
    ) -> ImagePriceAnalysisResult:
        """建立來源驗證或欄位抽取失敗時使用的統一回應。"""
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
            risk_label="UNKNOWN",
            score="未知",
            reason=message,
            confidence=0.0,
            evidence=evidence,
            decision_layer="source_validation",
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

    def _extract_ocr_document(self, data: bytes) -> OCRDocument:
        """優先取得含座標的 OCR 文件，並相容僅能回傳全文的服務。"""
        extract_document = self._ocr_service.extract_document
        if callable(extract_document):
            return extract_document(data)
        return OCRDocument(text=self._ocr_service.extract_text(data))

    def image_price_detector(
        self,
        data: bytes,
        filename: str = "unknown",
        content_type: str = "",
    ) -> ImagePriceAnalysisResult:
        """分析商品頁截圖並回傳路由層可直接映射的結果。

        流程依序執行 OCR、來源版型驗證、刊登欄位抽取、商品辨識、市場查價、
        價格異常檢查與風險決策；成功結果會依設定寫入案例記憶。
        """
        document = self._extract_ocr_document(data)
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
        except MainPriceExtractionError as exc:
            extraction = exc.result
            logger.warning(
                "主價格抽取失敗 error_code=%s confidence=%s candidates=%s rejected=%s",
                exc.error_code,
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
                error_code=exc.error_code,
                message=str(exc),
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
        market_price = self._resolve_market_price(
            product_name,
            product.brand_model,
            product.market_price,
        )
        selling_price = extraction.price

        has_risk = self._has_price_risk(selling_price, market_price)

        decision = self._decision_engine.evaluate(
            product_name=product_name,
            brand_model=product.brand_model,
            text=text,
            selling_price=selling_price,
            market_price=market_price,
            has_risk=has_risk,
        )
        decision = self._enforce_price_risk(
            decision,
            selling_price,
            market_price,
            has_risk,
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
            has_risk=has_risk,
            risk_label=self._normalize_risk_label(decision.get("risk_label")),
            score=decision.get("risk_score"),
            reason=str(decision.get("reason") or ""),
            confidence=float(decision.get("confidence") or 0.0),
            decision_layer=self._normalize_decision_layer(decision.get("decision_layer")),
            marketplace_layout=detection.layout,
            marketplace_confidence=detection.confidence,
            extraction_confidence=extraction.confidence,
            price_source_text=extraction.source_text,
            price_extraction_reason=extraction.reason,
            seller_name=extraction.seller_name,
            condition=extraction.condition,
            extraction_warnings=extraction.warnings,
        )

        self._store_case(result)
        return result


image_price_analyzer = ImagePriceAnalyzer()
