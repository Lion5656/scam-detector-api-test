"""抽取 Facebook Marketplace 刊登頁的商品資訊。"""

import re
from typing import NoReturn

from backend.services.image_price_service.domain.models import (
    DetectionResult, MainPriceExtractionError, MainPriceExtractionResult,
    MarketplaceLayout, OCRDocument, OCRTextBlock, PriceCandidate, PriceSection)
from backend.services.image_price_service.ocr.ocr_text_utils import (
    expand_blocks, normalized_lines)
from backend.services.image_price_service.page_analysis.common.price_candidate_selector import (
    CandidateCollection, PriceCandidateSelector, candidate_horizontal_position)
from backend.services.image_price_service.page_analysis.fb_marketplace.fb_marketplace_rules import (
    NT_MAIN_PRICE_RE, OFFER_TERMS, PRICE_RE, FBMarketplaceRules)
from backend.services.image_price_service.page_analysis.fb_marketplace.layout_extractors import (
    DesktopLayoutExtractor, LayoutTitleExtractor, MobileLayoutExtractor)


class FBMarketplacePriceExtractor:
    """協調 FB Marketplace 版型、規則及價格候選元件。"""

    minimum_confidence = 0.65

    def __init__(
        self,
        *,
        rules: FBMarketplaceRules | None = None,
        mobile_layout: LayoutTitleExtractor | None = None,
        desktop_layout: LayoutTitleExtractor | None = None,
        candidate_selector: PriceCandidateSelector | None = None,
    ) -> None:
        """建立可替換版型策略與共用候選選擇器的抽取器。"""
        self._rules: FBMarketplaceRules = (
            rules
            if rules is not None
            else FBMarketplaceRules()
        )
        self._mobile_layout = (
            mobile_layout
            or MobileLayoutExtractor(self._rules)
        )
        self._desktop_layout = (
            desktop_layout
            or DesktopLayoutExtractor(self._rules)
        )
        self._candidate_selector = (
            candidate_selector
            or PriceCandidateSelector(self.minimum_confidence)
        )

    def _collect_candidates(
        self,
        document: OCRDocument,
        detection: DetectionResult,
        blocks: list[OCRTextBlock],
        layout_extractor: LayoutTitleExtractor,
    ) -> CandidateCollection:
        """依指定版型策略建立 FB Marketplace 價格候選。"""
        collection = CandidateCollection()
        current_section: PriceSection = PriceSection.UNKNOWN

        for index, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue
            line_section = self._rules.section_for(
                text,
                current_section,
            )
            matches = list(PRICE_RE.finditer(text))
            is_range = any(
                term in text.lower()
                for term in OFFER_TERMS
            )

            for match in matches:
                amount = int(
                    re.sub(r"\D", "", match.group("amount"))
                )
                prior = (
                    blocks[index - 1].text
                    if index > 0
                    else None
                )
                following = (
                    blocks[index + 1].text
                    if index + 1 < len(blocks)
                    else None
                )
                section = (
                    PriceSection.OFFER_RANGE
                    if is_range
                    else line_section
                )
                reject_reason = self._rules.initial_rejection(
                    match,
                    amount,
                    section,
                    is_range,
                )
                title, title_text, title_distance = (
                    layout_extractor.find_title(
                        document,
                        blocks,
                        index,
                        block,
                        text[:match.start()].strip(),
                    )
                )

                confidence = 0.22
                if title is None and reject_reason is None:
                    reject_reason = (
                        "價格附近找不到 Marketplace 商品標題"
                    )
                else:
                    confidence += (
                        self._rules.title_confidence_bonus(
                            detection.layout,
                            title_distance,
                        )
                    )

                position_bonus, position_reject = (
                    self._rules.position_score(
                        detection.layout,
                        document,
                        title,
                        block,
                    )
                )
                confidence += position_bonus
                if (
                    position_reject
                    and reject_reason is None
                ):
                    reject_reason = position_reject
                confidence += (
                    0.10 if document.has_coordinates else -0.03
                )
                confidence = max(0.0, min(confidence, 1.0))

                candidate = PriceCandidate(
                    amount=amount,
                    currency="TWD",
                    source_text=(
                        f"NT${amount:,}"
                        if NT_MAIN_PRICE_RE.fullmatch(
                            match.group("currency")
                        )
                        else match.group(0)
                    ),
                    block_index=index,
                    x=block.x,
                    y=block.y,
                    context_before=prior,
                    context_after=following,
                    section=(
                        PriceSection.MAIN_PRICE
                        if reject_reason is None
                        else section
                    ),
                    confidence=confidence,
                    reject_reason=reject_reason,
                )
                if title_text:
                    collection.titles[id(candidate)] = title_text
                collection.positions[id(candidate)] = (
                    candidate_horizontal_position(
                        block,
                        text,
                        match,
                    ),
                    block.y,
                    block.height,
                )
                target = (
                    collection.rejected
                    if reject_reason
                    else collection.candidates
                )
                target.append(candidate)

            # 出價範圍只套用於命中關鍵字的當行，避免英文「from」讓
            # 後續主價格持續被歸類為出價區段。
            if line_section != "offer_range":
                current_section = line_section

        return collection

    def _extract_mobile_page(
        self,
        document: OCRDocument,
        detection: DetectionResult,
        blocks: list[OCRTextBlock],
        lines: list[str],
    ) -> MainPriceExtractionResult:
        """抽取 mobile Marketplace 頁面商品資訊。"""
        collection = self._collect_candidates(
            document,
            detection,
            blocks,
            self._mobile_layout,
        )
        return self._build_result(
            document,
            detection,
            lines,
            collection,
        )

    def _extract_desktop_page(
        self,
        document: OCRDocument,
        detection: DetectionResult,
        blocks: list[OCRTextBlock],
        lines: list[str],
    ) -> MainPriceExtractionResult:
        """抽取 desktop Marketplace 頁面商品資訊。"""
        collection = self._collect_candidates(
            document,
            detection,
            blocks,
            self._desktop_layout,
        )
        return self._build_result(
            document,
            detection,
            lines,
            collection,
        )

    def _build_result(
        self,
        document: OCRDocument,
        detection: DetectionResult,
        lines: list[str],
        collection: CandidateCollection,
    ) -> MainPriceExtractionResult:
        """由候選資料組裝成功結果或主價格抽取錯誤。"""
        seller_name, _ = self._rules.extract_seller_name(lines)
        selected, has_multiple_prices_on_row = (
            self._candidate_selector.select(
                document,
                collection,
                minimum_confidence=self.minimum_confidence,
            )
        )

        if selected is not None:
            return self._build_success_result(
                detection,
                lines,
                collection,
                selected,
                has_multiple_prices_on_row,
                seller_name,
            )
        self._raise_failure_result(
            detection,
            lines,
            collection,
            seller_name,
        )

    def _build_success_result(
        self,
        detection: DetectionResult,
        lines: list[str],
        collection: CandidateCollection,
        selected: PriceCandidate,
        has_multiple_prices_on_row: bool,
        seller_name: str | None,
    ) -> MainPriceExtractionResult:
        """組裝成功的 Marketplace 商品資訊。"""
        product_name = collection.titles.get(id(selected))
        (
            condition,
            condition_detail,
            condition_source_text,
            condition_confidence,
            warnings,
        ) = (
            self._rules.extract_condition(
                lines,
                product_name,
            )
        )
        self._rules.append_missing_required_field_warnings(
            warnings,
            product_name=product_name,
            price=selected.amount,
        )
        title_confidence = 0.95 if product_name else 0.0
        overall_confidence = min(
            1.0,
            (
                selected.confidence * 0.60
                + title_confidence * 0.20
                + condition_confidence * 0.20
            ),
        )
        if has_multiple_prices_on_row:
            reason = "同列出現多個 NT$ 價格，採用最左側價格"
        else:
            reason = (
                "商品標題正下方的第一個 NT$ 單一價格"
                if detection.layout == MarketplaceLayout.MOBILE
                else "桌面版右側商品標題下方 NT$ 價格列"
            )
        return MainPriceExtractionResult(
            price=selected.amount,
            currency=selected.currency,
            confidence=overall_confidence,
            source_text=selected.source_text,
            layout=detection.layout,
            candidates=collection.candidates,
            rejected_candidates=collection.rejected,
            reason=reason,
            product_name=product_name,
            seller_name=seller_name,
            condition=condition,
            condition_detail=condition_detail,
            condition_source_text=condition_source_text,
            condition_extraction_confidence=condition_confidence,
            warnings=warnings,
        )

    def _raise_failure_result(
        self,
        detection: DetectionResult,
        lines: list[str],
        collection: CandidateCollection,
        seller_name: str | None,
    ) -> NoReturn:
        """組裝失敗結果並中止後續商品分析。"""
        (
            condition,
            condition_detail,
            condition_source_text,
            condition_confidence,
            warnings,
        ) = (
            self._rules.extract_condition(lines)
        )
        self._rules.append_missing_required_field_warnings(
            warnings,
            product_name=None,
            price=None,
        )
        low_confidence = bool(collection.candidates)
        result = MainPriceExtractionResult(
            price=None,
            currency=None,
            confidence=max(
                (
                    item.confidence
                    for item in collection.candidates
                ),
                default=0.0,
            ),
            source_text=None,
            layout=detection.layout,
            candidates=collection.candidates,
            rejected_candidates=collection.rejected,
            error_code=(
                "LOW_CONFIDENCE_PRICE_EXTRACTION"
                if low_confidence
                else "MAIN_PRICE_NOT_FOUND"
            ),
            message=(
                "商品主價格候選信心不足，未進行價格驗證"
                if low_confidence
                else "找不到 FB Marketplace 商品主價格"
            ),
            seller_name=seller_name,
            condition=condition,
            condition_detail=condition_detail,
            condition_source_text=condition_source_text,
            condition_extraction_confidence=condition_confidence,
            warnings=warnings,
        )
        raise MainPriceExtractionError(result)

    def extract(
        self,
        document: OCRDocument,
        detection: DetectionResult,
    ) -> MainPriceExtractionResult:
        """依偵測版型分派 mobile 或 desktop 商品資訊抽取。"""
        blocks = expand_blocks(document)
        lines = normalized_lines(document)
        if detection.layout == MarketplaceLayout.DESKTOP:
            return self._extract_desktop_page(
                document,
                detection,
                blocks,
                lines,
            )
        return self._extract_mobile_page(
            document,
            detection,
            blocks,
            lines,
        )


fb_marketplace_price_extractor = FBMarketplacePriceExtractor()
