"""協調商品圖片的來源驗證、價格抽取與風險分析流程。"""

import logging
from typing import Any, cast

from backend.config import settings
from backend.repository.case_repository import case_repository
from backend.services.dto.price_analysis import (
    ImagePriceAnalysisResult,
    MarketPriceEstimate,
    SearchTool,
)
from backend.services.image_price_service.case_recorder import record_case
from backend.services.image_price_service.domain.models import (
    MainPriceExtractionError,
    MainPriceExtractionResult,
    MarketplaceCondition,
    MarketplaceLayout,
)
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
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
    OnlineMarketPriceService,
)
from backend.services.image_price_service.product.product_identifier import (
    product_identifier as default_product_identifier,
)
from backend.services.image_price_service.risk.condition_reviewer import (
    GroqConditionReviewer,
)
from backend.services.image_price_service.risk.fusion_decision_engine import (
    FusionDecisionEngine,
)

logger = logging.getLogger(__name__)

default_online_price_service = OnlineMarketPriceService(
    policy=DEFAULT_PRICE_RISK_POLICY,
)
default_decision_engine = FusionDecisionEngine(
    policy=DEFAULT_PRICE_RISK_POLICY,
    condition_reviewer=GroqConditionReviewer(
        api_key=settings.GROQ_API_KEY.get_secret_value(),
        model_name=settings.PRODUCT_MODEL_NAME,
    ),
)


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
            decision_layer="decision_error",
            search_tools=[],
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
            condition_detail=extraction.condition_detail if extraction else "",
            condition_source_text=(
                extraction.condition_source_text if extraction else ""
            ),
            condition_extraction_confidence=(
                extraction.condition_extraction_confidence
                if extraction
                else 0.0
            ),
            extraction_warnings=extraction.warnings if extraction else [],
        )

    @staticmethod
    def _without_market_candidates(
        estimates: tuple[MarketPriceEstimate, ...],
    ) -> tuple[MarketPriceEstimate, ...]:
        """將候選明細留在查價服務邊界內。"""
        return tuple(
            estimate.model_copy(update={"candidates": ()})
            for estimate in estimates
        )

    @staticmethod
    def _search_tools(
        estimates: tuple[MarketPriceEstimate, ...],
    ) -> list[SearchTool]:
        """依實際呼叫順序彙整查價使用過的搜尋工具。"""
        search_tools: list[SearchTool] = []
        for estimate in estimates:
            for search_tool in estimate.search_tools:
                if search_tool not in search_tools:
                    search_tools.append(search_tool)
        return search_tools

    @staticmethod
    def _reference_market_price(
        estimates: tuple[MarketPriceEstimate, ...],
    ) -> int:
        """取得單一成功市場區間的中位數，供介面顯示參考。"""
        if len(estimates) != 1 or estimates[0].status != "success":
            return 0
        return estimates[0].median_price

    def image_price_detector(
        self,
        data: bytes,
        filename: str = "unknown",
        content_type: str = "",
    ) -> ImagePriceAnalysisResult:
        """分析 Marketplace 截圖並回傳價格風險結果。"""
        document = self._ocr_service.extract_document(data)
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
        resolved_market_estimates = resolve_market_price(
            self._online_price_service,
            product_name,
            product.brand_model,
            product.market_price,
            product.search_query,
            condition=extraction.condition,
            condition_text=extraction.condition_detail,
        )
        search_tools = self._search_tools(resolved_market_estimates)
        active_market_estimates = self._without_market_candidates(
            resolved_market_estimates
        )
        reprice_performed = False

        def reprice_once(
            condition: MarketplaceCondition,
            condition_detail: str,
        ) -> tuple[MarketPriceEstimate, ...]:
            nonlocal active_market_estimates, reprice_performed, search_tools
            if reprice_performed:
                return active_market_estimates
            reprice_performed = True
            repriced_estimates = resolve_market_price(
                self._online_price_service,
                product_name,
                product.brand_model,
                product.market_price,
                product.search_query,
                condition=condition,
                condition_text=condition_detail,
            )
            for search_tool in self._search_tools(repriced_estimates):
                if search_tool not in search_tools:
                    search_tools.append(search_tool)
            active_market_estimates = self._without_market_candidates(
                repriced_estimates
            )
            return active_market_estimates

        selling_price = cast(int, extraction.price)
        decision = self._decision_engine.evaluate(
            product_name=product_name,
            selling_price=selling_price,
            market_estimates=active_market_estimates,
            condition=extraction.condition,
            condition_detail=extraction.condition_detail,
            condition_source_text=extraction.condition_source_text,
            condition_extraction_confidence=(
                extraction.condition_extraction_confidence
            ),
            text=text,
            reprice=reprice_once,
        )
        decision_error_code = decision.get("error_code")
        if not isinstance(decision_error_code, str):
            decision_error_code = None

        decision_condition = decision.get("condition", extraction.condition)
        if isinstance(decision_condition, MarketplaceCondition):
            resolved_condition = decision_condition
        elif isinstance(decision_condition, str):
            try:
                resolved_condition = MarketplaceCondition(decision_condition)
            except ValueError:
                resolved_condition = extraction.condition
        else:
            resolved_condition = extraction.condition
        resolved_condition_detail = str(
            decision.get(
                "condition_detail",
                extraction.condition_detail,
            )
            or ""
        )
        market_price = self._reference_market_price(
            active_market_estimates,
        )
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
            error_code=decision_error_code,
            message=None,
            product_name=product_name,
            brand_model=product.brand_model,
            listed_price=selling_price,
            market_price=market_price,
            market_price_source=decision.get(
                "market_price_source",
                "not_evaluated",
            ),
            market_price_estimates=active_market_estimates,
            risk_label=decision.get("risk_label", "UNKNOWN"),
            score=decision.get("risk_score"),
            reason=str(decision.get("reason") or ""),
            evidence=evidence,
            confidence=float(decision.get("confidence") or 0.0),
            decision_layer=decision.get("decision_layer", "fast"),
            search_tools=search_tools,
            marketplace_layout=detection.layout,
            marketplace_confidence=detection.confidence,
            extraction_confidence=extraction.confidence,
            price_source_text=extraction.source_text,
            price_extraction_reason=extraction.reason,
            seller_name=extraction.seller_name,
            condition=resolved_condition,
            condition_detail=resolved_condition_detail,
            condition_source_text=extraction.condition_source_text,
            condition_extraction_confidence=(
                extraction.condition_extraction_confidence
            ),
            extraction_warnings=extraction.warnings,
        )

        record_case(self._case_repo, result)
        return result


image_price_analyzer = ImagePriceAnalyzer()
