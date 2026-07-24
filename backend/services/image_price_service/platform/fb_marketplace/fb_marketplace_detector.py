"""驗證 OCR 內容是否具備 Facebook Marketplace 商品頁的必要欄位。"""

import re

from backend.services.image_price_service.models import (
    DetectionResult,
    MarketplaceLayout,
    OCRDocument,
)

_PRICE_RE = re.compile(
    r"(?:NT\s*\$|NTD|TWD|台幣|\$)\s*[1-9]\d{0,6}(?:\s*[,，]\s*\d{3})*",
    flags=re.IGNORECASE,
)


class FBMarketplaceDetector:
    """驗證 Marketplace 商品頁並判斷版型。"""

    threshold = 0.60

    @staticmethod
    def _compact_line(text: str) -> str:
        """移除空白及常見區段標題裝飾字元。"""
        return re.sub(r"\s+", "", text).strip("·:：>›|")

    @staticmethod
    def _has_title_near_price(document: OCRDocument) -> bool:
        """檢查價格同一行或前兩行是否存在可能的商品標題。"""
        lines = [line.strip() for line in document.text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            match = _PRICE_RE.search(line)
            if not match:
                continue

            # OCR 可能將商品標題和價格合併在同一行或同一文字區塊。
            if len(line[:match.start()].strip()) >= 4:
                return True

            for prior in lines[max(0, index - 2):index]:
                if len(prior) >= 4 and not _PRICE_RE.search(prior):
                    return True

        for block in document.blocks:
            match = _PRICE_RE.search(block.text)
            if match and len(block.text[:match.start()].strip()) >= 4:
                return True
        return False

    @classmethod
    def _basic_sections(cls, document: OCRDocument) -> tuple[bool, bool, bool]:
        """檢查詳細資料、商品狀況與賣家資訊區段是否存在。"""
        compact_lines = [
            cls._compact_line(line)
            for line in document.text.splitlines()
            if line.strip()
        ]
        has_detail = any(line in {"詳細內容", "詳細資料"} for line in compact_lines)
        has_condition = any("狀況" in line for line in compact_lines)
        has_seller = any(
            line in {"賣家", "賣家資訊", "賣家詳細資料"}
            for line in compact_lines
        )
        return has_detail, has_condition, has_seller

    @staticmethod
    def _infer_layout(
        document: OCRDocument,
        has_detail: bool,
        has_seller: bool,
    ) -> MarketplaceLayout:
        """優先依頁面比例判斷版型，缺少尺寸時再使用區段名稱推定。"""
        if document.width and document.height:
            if document.width > document.height * 1.25:
                return MarketplaceLayout.DESKTOP
            if document.height > document.width * 1.15:
                return MarketplaceLayout.MOBILE

        compact = re.sub(r"\s+", "", document.text)
        if "詳細資料" in compact or "賣家資訊" in compact:
            return MarketplaceLayout.DESKTOP
        if has_detail or has_seller or "狀況" in compact:
            return MarketplaceLayout.MOBILE
        return MarketplaceLayout.UNKNOWN

    def detect(self, document: OCRDocument) -> DetectionResult:
        """判斷 OCR 文件是否為 Marketplace 商品頁。"""
        text = document.text.strip()
        if not text:
            return DetectionResult(
                is_marketplace=False,
                layout=MarketplaceLayout.UNKNOWN,
                confidence=0.0,
                reason="OCR 未辨識到足以分析的商品頁文字",
            )

        has_price = bool(_PRICE_RE.search(text))
        has_title = self._has_title_near_price(document)
        has_detail, has_condition, has_seller = self._basic_sections(document)
        has_structured_section = has_detail or has_condition or has_seller
        layout = self._infer_layout(document, has_detail, has_seller)

        confidence = 0.0
        evidence: list[str] = []
        if has_title:
            confidence += 0.25
            evidence.append("價格附近有商品標題")
        if has_price:
            confidence += 0.25
            evidence.append("偵測到商品價格")
        if has_detail:
            confidence += 0.15
            evidence.append("偵測到詳細資料區段")
        if has_condition:
            confidence += 0.15
            evidence.append("偵測到商品狀況")
        if has_seller:
            confidence += 0.15
            evidence.append("偵測到賣家資訊")
        confidence = round(min(confidence, 1.0), 2)

        is_marketplace = (
            has_title
            and has_price
            and has_structured_section
            and confidence >= self.threshold
        )
        reason = (
            "商品標題、價格與基本詳細資訊足夠，可進行價格分析"
            if is_marketplace
            else "商品頁缺少標題、價格或詳細資料／狀況／賣家資訊"
        )

        return DetectionResult(
            is_marketplace=is_marketplace,
            layout=layout if is_marketplace else MarketplaceLayout.UNKNOWN,
            confidence=confidence,
            evidence=evidence,
            reason=reason,
        )


fb_marketplace_detector = FBMarketplaceDetector()
