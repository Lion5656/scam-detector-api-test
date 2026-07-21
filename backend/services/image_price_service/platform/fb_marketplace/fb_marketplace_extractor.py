"""抽取 Facebook Marketplace 刊登頁的標題、主價格、狀況與賣家。"""

import re

from backend.services.image_price_service.models import (
    MainPriceExtractionError,
    MainPriceExtractionResult,
    MarketplaceDetectionResult,
    MarketplaceCondition,
    MarketplaceLayout,
    OCRDocument,
    OCRTextBlock,
    PriceCandidate,
    PriceSection,
)


_PRICE_RE = re.compile(
    r"(?P<currency>NT\s*\$|NTD|TWD|台幣|\$)\s*"
    r"(?P<amount>[1-9]\d{0,6}(?:\s*[,，]\s*\d{3})*)",
    flags=re.IGNORECASE,
)
_NT_MAIN_PRICE_RE = re.compile(r"^NT\s*\$$", flags=re.IGNORECASE)
_OFFER_TERMS = ("offers from", "offer from", "from", " to ", "傳送出價", "出價", "議價")
_DETAIL_TERMS = ("詳細內容", "詳細資料", "狀況", "描述", "說明")
_SELLER_TERMS = ("賣家資訊", "賣家詳細資料", "賣家")
_MESSAGE_TERMS = ("發送訊息給賣家", "傳送訊息給賣家", "傳訊息給賣家", "留言", "訊息輸入")
_RECOMMENDATION_TERMS = ("推薦商品", "相關商品", "你可能也喜歡", "更多商品")
_LOCATION_TERMS = ("地點", "地圖", "大致位置")
_UI_TERMS = (
    *_OFFER_TERMS,
    *_DETAIL_TERMS,
    *_SELLER_TERMS,
    *_MESSAGE_TERMS,
    *_RECOMMENDATION_TERMS,
    *_LOCATION_TERMS,
    "分享",
    "儲存",
    "公開面交",
    "有存貨",
)
_USED_TERMS = (
    "二手",
    "中古",
    "近全新",
    "使用過",
    "已使用",
    "良好使用",
    "使用痕跡",
    "無修無拆",
)
_NEW_TERMS = ("全新", "未使用", "未拆封", "全新品")


class FBMarketplacePriceExtractor:
    """從商品標題附近選出以 NT$ 開頭的單一主價格與其他刊登欄位。"""

    minimum_confidence = 0.65

    @staticmethod
    def _blocks(document: OCRDocument) -> list[OCRTextBlock]:
        """將多行 OCR 區塊展開成單行，無區塊時以全文各行建立替代區塊。"""
        if document.blocks:
            expanded: list[OCRTextBlock] = []
            for block in document.blocks:
                lines = [line.strip() for line in block.text.splitlines() if line.strip()]
                if len(lines) <= 1:
                    expanded.append(block)
                    continue
                for line in lines:
                    expanded.append(OCRTextBlock(
                        text=line,
                        x=block.x,
                        y=block.y,
                        width=block.width,
                        height=block.height,
                        confidence=block.confidence,
                    ))
            if expanded:
                return expanded
        return [OCRTextBlock(text=line.strip()) for line in document.text.splitlines() if line.strip()]

    @staticmethod
    def _lines(document: OCRDocument) -> list[str]:
        """回傳移除多餘空白與空行的 OCR 全文行。"""
        return [re.sub(r"\s+", " ", line).strip() for line in document.text.splitlines() if line.strip()]

    @staticmethod
    def _normalize_display_text(text: str) -> str:
        """整理顯示文字中的空白、斜線與連字號。"""
        text = re.sub(r"\s+", " ", text).strip(" ·|\t\r\n")
        text = re.sub(r"\s*/\s*", "/", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text

    @staticmethod
    def _section_for(text: str, current: PriceSection) -> PriceSection:
        """依區段關鍵字更新目前價格候選所屬的商品頁區段。"""
        lowered = text.lower()
        if any(term in lowered for term in _OFFER_TERMS):
            return PriceSection.OFFER_RANGE
        if any(term in lowered for term in _MESSAGE_TERMS):
            return PriceSection.MESSAGE_BOX
        if any(term in lowered for term in _SELLER_TERMS):
            return PriceSection.SELLER_INFO
        if any(term in lowered for term in _DETAIL_TERMS):
            return PriceSection.DETAIL
        if any(term in lowered for term in _RECOMMENDATION_TERMS):
            return PriceSection.UNKNOWN
        return current

    @staticmethod
    def _looks_like_title(text: str) -> bool:
        """排除價格與介面用語，判斷文字是否可能為商品標題。"""
        lowered = text.lower().strip()
        return (
            len(lowered) >= 4
            and not _PRICE_RE.search(lowered)
            and not any(term in lowered for term in _UI_TERMS)
        )

    @staticmethod
    def _position_score(
        layout: str,
        document: OCRDocument,
        title: OCRTextBlock | None,
        price: OCRTextBlock,
    ) -> tuple[float, str | None]:
        """依標題與價格的相對座標加權，並回傳不合理位置的原因。"""
        if not title or title.x is None or title.y is None or price.x is None or price.y is None:
            return 0.0, None
        if title is price:
            return 0.10, None
        title_bottom = title.y + (title.height or 0.0)
        vertical_gap = price.y - title_bottom
        same_column = abs(price.x - title.x) <= max(
            title.width or 0.0,
            price.width or 0.0,
            120.0,
        )
        if vertical_gap < -5 or not same_column:
            return -0.30, "價格不在商品標題下方同一資訊欄"
        if document.height and vertical_gap > document.height * 0.12:
            return -0.20, "價格與商品標題距離過遠"
        if layout == "desktop" and document.width and price.x < document.width * 0.55:
            return -0.35, "桌面版價格不在右側商品資訊欄"
        return 0.16, None

    @staticmethod
    def _is_section_header(text: str, header: str) -> bool:
        """忽略空白與裝飾字元後，比對完整區段標題。"""
        normalized = re.sub(r"\s+", "", text).strip(">›·:")
        return normalized == header

    @classmethod
    def _condition_from_text(cls, text: str) -> MarketplaceCondition:
        """依新舊狀況關鍵字判定商品狀況。"""
        compact = re.sub(r"\s+", "", text).lower()
        # 二手關鍵字優先，避免「全新品購入，良好使用」被誤判為目前仍是新品。
        if any(term in compact for term in _USED_TERMS):
            return MarketplaceCondition.USED
        if any(term in compact for term in _NEW_TERMS):
            return MarketplaceCondition.NEW
        return MarketplaceCondition.UNKNOWN

    @classmethod
    def _extract_condition(cls, lines: list[str]) -> tuple[MarketplaceCondition, float, list[str]]:
        """先讀取詳細資料的狀況欄，再以說明文字補充判定。"""
        detail_start = next(
            (index for index, line in enumerate(lines) if cls._is_section_header(line, "詳細內容")
             or cls._is_section_header(line, "詳細資料")),
            None,
        )
        if detail_start is not None:
            detail_lines = lines[detail_start + 1:detail_start + 12]
            for index, line in enumerate(detail_lines):
                if "狀況" not in re.sub(r"\s+", "", line):
                    continue
                value = line
                if (
                    cls._condition_from_text(value) is MarketplaceCondition.UNKNOWN
                    and index + 1 < len(detail_lines)
                ):
                    value = f"{value} {detail_lines[index + 1]}"
                condition = cls._condition_from_text(value)
                if condition is not MarketplaceCondition.UNKNOWN:
                    return condition, 0.97, []

        description_start = next(
            (index for index, line in enumerate(lines) if cls._is_section_header(line, "說明")),
            None,
        )
        if description_start is not None:
            description_lines: list[str] = []
            for line in lines[description_start + 1:]:
                if cls._is_section_header(line, "賣家") or cls._is_section_header(line, "賣家資訊"):
                    break
                description_lines.append(line)
            condition = cls._condition_from_text(" ".join(description_lines))
            if condition is not MarketplaceCondition.UNKNOWN:
                return condition, 0.82, []

        return MarketplaceCondition.NEW, 0.35, ["未找到明確商品狀況，依規則預設為全新"]

    @classmethod
    def _extract_seller_name(cls, lines: list[str]) -> tuple[str | None, float]:
        """從賣家區段擷取不含數字的中英文顯示名稱。"""
        seller_start = next(
            (index for index, line in enumerate(lines) if cls._is_section_header(line, "賣家")
             or cls._is_section_header(line, "賣家資訊")),
            None,
        )
        if seller_start is None:
            return None, 0.0

        for line in lines[seller_start + 1:seller_start + 7]:
            candidate = cls._normalize_display_text(line)
            lowered = candidate.lower()
            if (
                not candidate
                or any(term in lowered for term in ("facebook", "追蹤", "加入", "詳細", "發訊息", "傳訊息"))
                or any(char.isdigit() for char in candidate)
            ):
                continue
            english_name = re.fullmatch(
                r"[A-Za-z][A-Za-z'’-]*(?:-[A-Za-z][A-Za-z'’-]*)?"
                r"(?:\s+[A-Za-z][A-Za-z'’-]*(?:-[A-Za-z][A-Za-z'’-]*)?){1,4}",
                candidate,
            )
            chinese_name = re.fullmatch(r"[\u4e00-\u9fff]{2,5}", candidate.replace(" ", ""))
            if chinese_name:
                return candidate.replace(" ", ""), 0.92
            if english_name:
                return candidate, 0.92
        return None, 0.0

    def extract(
        self,
        document: OCRDocument,
        detection: MarketplaceDetectionResult,
    ) -> MainPriceExtractionResult:
        """選出可信的刊登主價格，並一併回傳標題、賣家與商品狀況。"""
        blocks = self._blocks(document)
        lines = self._lines(document)
        candidates: list[PriceCandidate] = []
        rejected: list[PriceCandidate] = []
        candidate_titles: dict[int, str] = {}
        current_section: PriceSection = PriceSection.UNKNOWN

        for index, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue
            line_section = self._section_for(text, current_section)
            matches = list(_PRICE_RE.finditer(text))
            lowered = text.lower()
            is_range = len(matches) > 1 or any(term in lowered for term in _OFFER_TERMS)

            for match in matches:
                amount = int(re.sub(r"\D", "", match.group("amount")))
                prior = blocks[index - 1].text if index > 0 else None
                following = blocks[index + 1].text if index + 1 < len(blocks) else None
                section: PriceSection = PriceSection.OFFER_RANGE if is_range else line_section
                reject_reason: str | None = None
                confidence = 0.22

                if not _NT_MAIN_PRICE_RE.fullmatch(match.group("currency")):
                    reject_reason = "商品主價格必須以 NT$ 開頭"
                elif amount < 100 or amount > 2_000_000:
                    reject_reason = "價格超出支援範圍"
                elif is_range:
                    reject_reason = "出價或多價格範圍不可作為商品主價格"
                elif section != "unknown":
                    reject_reason = f"價格位於非主價格區塊：{section}"

                title: OCRTextBlock | None = None
                title_text: str | None = None
                title_distance: int | None = None
                prefix = text[:match.start()].strip()
                if self._looks_like_title(prefix):
                    title = block
                    title_text = self._normalize_display_text(prefix)
                    title_distance = 0
                else:
                    for prior_index in range(index - 1, max(-1, index - 3), -1):
                        if self._looks_like_title(blocks[prior_index].text):
                            title = blocks[prior_index]
                            title_text = self._normalize_display_text(title.text)
                            title_distance = index - prior_index
                            break

                if title is None and reject_reason is None:
                    reject_reason = "價格附近找不到 Marketplace 商品標題"
                elif title_distance is not None:
                    if title_distance == 0:
                        confidence += 0.50
                    elif detection.layout ==  MarketplaceLayout.DESKTOP and title_distance <= 2:
                        confidence += 0.48
                    elif detection.layout == MarketplaceLayout.MOBILE and title_distance == 1:
                        confidence += 0.48
                    elif detection.layout == MarketplaceLayout.MOBILE and title_distance <= 2:
                        confidence += 0.34
                    else:
                        confidence += 0.15

                position_bonus, position_reject = self._position_score(
                    detection.layout,
                    document,
                    title,
                    block,
                )
                confidence += position_bonus
                if position_reject and reject_reason is None:
                    reject_reason = position_reject
                confidence += 0.10 if document.has_coordinates else -0.03
                confidence = max(0.0, min(confidence, 1.0))

                candidate = PriceCandidate(
                    amount=amount,
                    currency="TWD",
                    source_text=f"NT${amount:,}" if _NT_MAIN_PRICE_RE.fullmatch(match.group("currency")) else match.group(0),
                    block_index=index,
                    x=block.x,
                    y=block.y,
                    context_before=prior,
                    context_after=following,
                    section= PriceSection.MAIN_PRICE if reject_reason is None else section,
                    confidence=confidence,
                    reject_reason=reject_reason,
                )
                if title_text:
                    candidate_titles[id(candidate)] = title_text
                (rejected if reject_reason else candidates).append(candidate)

            # 出價範圍只套用於命中關鍵字的當行，避免圖片中的英文「from」讓
            # 後續真正的商品標題與主價格持續被歸類為出價區段。
            if line_section != "offer_range":
                current_section = line_section

        accepted = [candidate for candidate in candidates if candidate.confidence >= self.minimum_confidence]
        seller_name, seller_confidence = self._extract_seller_name(lines)
        condition, condition_confidence, warnings = self._extract_condition(lines)
        if seller_name is None:
            warnings.append("找不到賣家區塊中的顯示名稱")

        if accepted:
            accepted.sort(key=lambda item: (-item.confidence, item.block_index))
            selected = accepted[0]
            product_name = candidate_titles.get(id(selected))
            if not product_name:
                warnings.append("找不到主價格上方的 Marketplace 商品標題")
            title_confidence = 0.95 if product_name else 0.0
            overall_confidence = min(1.0, (
                selected.confidence * 0.55
                + title_confidence * 0.15
                + seller_confidence * 0.15
                + condition_confidence * 0.15
            ))
            reason = (
                "商品標題正下方的第一個 NT$ 單一價格"
                if detection.layout == "mobile"
                else "桌面版右側商品標題下方 NT$ 價格列"
            )
            return MainPriceExtractionResult(
                price=selected.amount,
                currency=selected.currency,
                confidence=overall_confidence,
                source_text=selected.source_text,
                layout=detection.layout,
                candidates=candidates,
                rejected_candidates=rejected,
                reason=reason,
                product_name=product_name,
                seller_name=seller_name,
                condition=condition,
                condition_confidence=condition_confidence,
                warnings=warnings,
            )

        low_confidence = bool(candidates)
        error_result = MainPriceExtractionResult(
            price=None,
            currency=None,
            confidence=max((item.confidence for item in candidates), default=0.0),
            source_text=None,
            layout=detection.layout,
            candidates=candidates,
            rejected_candidates=rejected,
            error_code="LOW_CONFIDENCE_PRICE_EXTRACTION" if low_confidence else "MAIN_PRICE_NOT_FOUND",
            message=(
                "商品主價格候選信心不足，未進行價格驗證"
                if low_confidence
                else "找不到 FB Marketplace 商品主價格"
            ),
            seller_name=seller_name,
            condition=condition,
            condition_confidence=condition_confidence,
            warnings=warnings,
        )
        raise MainPriceExtractionError(error_result)


fb_marketplace_price_extractor = FBMarketplacePriceExtractor()
