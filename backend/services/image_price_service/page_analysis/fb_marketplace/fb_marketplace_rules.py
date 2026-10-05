"""Facebook Marketplace 特有的文字、欄位與價格判斷規則。"""

import re

from backend.services.image_price_service.domain.models import (
    MarketplaceCondition,
    MarketplaceLayout,
    OCRDocument,
    OCRTextBlock,
    PriceSection,
)
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
)
from backend.services.image_price_service.ocr.ocr_text_utils import (
    normalize_display_text,
)

PRICE_RE = re.compile(
    r"(?P<currency>NT\s*\$|NTD|TWD|台幣|\$)\s*"
    r"(?P<amount>[1-9]\d{0,6}(?:\s*[,，]\s*\d{3})*)",
    flags=re.IGNORECASE,
)
NT_MAIN_PRICE_RE = re.compile(r"^NT\s*\$$", flags=re.IGNORECASE)
OFFER_TERMS = (
    "offers from",
    "offer from",
    "from",
    " to ",
    "傳送出價",
    "出價",
    "議價",
)
DETAIL_TERMS = ("詳細內容", "詳細資料", "狀況", "描述", "說明")
SELLER_TERMS = ("賣家資訊", "賣家詳細資料", "賣家")
MESSAGE_TERMS = (
    "發送訊息給賣家",
    "傳送訊息給賣家",
    "傳訊息給賣家",
    "留言",
    "訊息輸入",
)
RECOMMENDATION_TERMS = (
    "推薦商品",
    "相關商品",
    "你可能也喜歡",
    "更多商品",
)
LOCATION_TERMS = ("地點", "地圖", "大致位置")
UI_TERMS = (
    *OFFER_TERMS,
    *DETAIL_TERMS,
    *SELLER_TERMS,
    *MESSAGE_TERMS,
    *RECOMMENDATION_TERMS,
    *LOCATION_TERMS,
    "分享",
    "儲存",
    "公開面交",
    "有存貨",
)
USED_TERMS = (
    "二手",
    "2手",
    "中古",
    "近全新",
    "使用過",
    "已使用",
    "良好使用",
    "狀況良好",
    "狀態良好",
    "尚可",
    "使用痕跡",
    "無修無拆",
)
NEW_TERMS = ("全新", "未使用", "未拆封", "全新品")
USED_GRADE_RE = re.compile(
    r"(?:[1-9](?:\.\d)?|10|[一二三四五六七八九十])成新",
    flags=re.IGNORECASE,
)
MISSING_TITLE_WARNING = "找不到主價格上方的 Marketplace 商品標題"
MISSING_PRICE_WARNING = "找不到 Marketplace 商品價格"


class FBMarketplaceRules:
    """集中管理 FB Marketplace 特有的欄位與候選規則。"""

    @staticmethod
    def section_for(
        text: str,
        current: PriceSection,
    ) -> PriceSection:
        """依區段關鍵字更新目前價格候選所屬區段。"""
        lowered = text.lower()
        if any(term in lowered for term in OFFER_TERMS):
            return PriceSection.OFFER_RANGE
        if any(term in lowered for term in MESSAGE_TERMS):
            return PriceSection.MESSAGE_BOX
        if any(term in lowered for term in SELLER_TERMS):
            return PriceSection.SELLER_INFO
        if any(term in lowered for term in DETAIL_TERMS):
            return PriceSection.DETAIL
        if any(term in lowered for term in RECOMMENDATION_TERMS):
            return PriceSection.UNKNOWN
        return current

    @staticmethod
    def looks_like_title(text: str) -> bool:
        """排除價格與介面用語，判斷文字是否可能為商品標題。"""
        lowered = text.lower().strip()
        return (
            len(lowered) >= 4
            and not PRICE_RE.search(lowered)
            and not any(term in lowered for term in UI_TERMS)
        )

    @staticmethod
    def append_missing_required_field_warnings(
        warnings: list[str],
        *,
        product_name: str | None,
        price: int | None,
    ) -> None:
        """將缺少的必要商品欄位警告依序且不重複加入。"""
        missing_warnings = (
            (MISSING_TITLE_WARNING, not product_name),
            (MISSING_PRICE_WARNING, price is None),
        )
        for warning, is_missing in missing_warnings:
            if is_missing and warning not in warnings:
                warnings.append(warning)

    @staticmethod
    def position_score(
        layout: MarketplaceLayout,
        document: OCRDocument,
        title: OCRTextBlock | None,
        price: OCRTextBlock,
    ) -> tuple[float, str | None]:
        """依標題與價格位置計算候選分數。"""
        if (
            not title
            or title.x is None
            or title.y is None
            or price.x is None
            or price.y is None
        ):
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
        if (
            document.height
            and vertical_gap > document.height * 0.12
        ):
            return -0.20, "價格與商品標題距離過遠"
        if (
            layout == MarketplaceLayout.DESKTOP
            and document.width
            and price.x < document.width * 0.55
        ):
            return -0.35, "桌面版價格不在右側商品資訊欄"
        return 0.16, None

    @staticmethod
    def is_section_header(text: str, header: str) -> bool:
        """忽略空白與裝飾字元後，比對完整區段標題。"""
        normalized = re.sub(r"\s+", "", text).strip(">›·:")
        return normalized == header

    @classmethod
    def condition_from_text(
        cls,
        text: str,
    ) -> MarketplaceCondition:
        """依新舊狀況關鍵字判定商品狀況。"""
        compact = re.sub(r"\s+", "", text).lower()
        if (
            any(term in compact for term in USED_TERMS)
            or USED_GRADE_RE.search(compact)
        ):
            return MarketplaceCondition.USED
        if any(term in compact for term in NEW_TERMS):
            return MarketplaceCondition.NEW
        return MarketplaceCondition.UNKNOWN

    @classmethod
    def extract_condition(
        cls,
        lines: list[str],
        product_title: str | None = None,
    ) -> tuple[
        MarketplaceCondition,
        str,
        str,
        float,
        list[str],
    ]:
        """依詳細資料、標題、說明的優先序擷取有限狀態證據。"""
        detail_start = next(
            (
                index
                for index, line in enumerate(lines)
                if cls.is_section_header(line, "詳細內容")
                or cls.is_section_header(line, "詳細資料")
            ),
            None,
        )
        if detail_start is not None:
            detail_lines = lines[detail_start + 1:detail_start + 12]
            for index, line in enumerate(detail_lines):
                compact_line = re.sub(r"\s+", "", line)
                if "狀況" not in compact_line and "狀態" not in compact_line:
                    continue
                value = line
                if (
                    cls.condition_from_text(value)
                    is MarketplaceCondition.UNKNOWN
                    and index + 1 < len(detail_lines)
                ):
                    value = f"{value} {detail_lines[index + 1]}"
                condition = cls.condition_from_text(value)
                if condition is not MarketplaceCondition.UNKNOWN:
                    source_text = normalize_display_text(value)
                    return (
                        condition,
                        cls._condition_detail(source_text, condition),
                        source_text,
                        0.97,
                        [],
                    )

        if product_title:
            title_condition = cls.condition_from_text(product_title)
            if title_condition is not MarketplaceCondition.UNKNOWN:
                source_text = normalize_display_text(product_title)
                return (
                    title_condition,
                    cls._condition_detail(source_text, title_condition),
                    source_text,
                    0.99,
                    [],
                )

        description_start = next(
            (
                index
                for index, line in enumerate(lines)
                if cls.is_section_header(line, "說明")
            ),
            None,
        )
        if description_start is not None:
            description_lines: list[str] = []
            for line in lines[description_start + 1:]:
                if (
                    cls.is_section_header(line, "賣家")
                    or cls.is_section_header(line, "賣家資訊")
                ):
                    break
                description_lines.append(line)
            for line in description_lines:
                condition = cls.condition_from_text(line)
                if condition is MarketplaceCondition.UNKNOWN:
                    continue
                source_text = normalize_display_text(line)
                return (
                    condition,
                    cls._condition_detail(source_text, condition),
                    source_text,
                    0.82,
                    [],
                )

        return (
            MarketplaceCondition.UNKNOWN,
            "",
            "",
            0.0,
            ["未找到明確商品狀況"],
        )

    @staticmethod
    def _condition_detail(
        source_text: str,
        condition: MarketplaceCondition,
    ) -> str:
        """從有限狀態原文保留可供查價使用的詳細描述。"""
        detail = re.sub(
            r"^(?:商品)?(?:狀況|狀態)\s*[:：・·-]?\s*",
            "",
            normalize_display_text(source_text),
            flags=re.IGNORECASE,
        ).strip()
        if condition is MarketplaceCondition.USED:
            without_generic_label = re.sub(
                r"^(?:二\s*手|2\s*手|中古)\s*[:：・·-]?\s*",
                "",
                detail,
                flags=re.IGNORECASE,
            ).strip()
            return without_generic_label or detail
        return detail

    @classmethod
    def extract_seller_name(
        cls,
        lines: list[str],
    ) -> tuple[str | None, float]:
        """從賣家區段擷取不含數字的中英文顯示名稱。"""
        seller_start = next(
            (
                index
                for index, line in enumerate(lines)
                if cls.is_section_header(line, "賣家")
                or cls.is_section_header(line, "賣家資訊")
            ),
            None,
        )
        if seller_start is None:
            return None, 0.0

        for line in lines[seller_start + 1:seller_start + 7]:
            candidate = normalize_display_text(line)
            lowered = candidate.lower()
            if (
                not candidate
                or any(
                    term in lowered
                    for term in (
                        "facebook",
                        "追蹤",
                        "加入",
                        "詳細",
                        "發訊息",
                        "傳訊息",
                    )
                )
                or any(char.isdigit() for char in candidate)
            ):
                continue
            english_name = re.fullmatch(
                r"[A-Za-z][A-Za-z'’-]*"
                r"(?:-[A-Za-z][A-Za-z'’-]*)?"
                r"(?:\s+[A-Za-z][A-Za-z'’-]*"
                r"(?:-[A-Za-z][A-Za-z'’-]*)?){1,4}",
                candidate,
            )
            chinese_name = re.fullmatch(
                r"[\u4e00-\u9fff]{2,5}",
                candidate.replace(" ", ""),
            )
            if chinese_name:
                return candidate.replace(" ", ""), 0.92
            if english_name:
                return candidate, 0.92
        return None, 0.0

    @staticmethod
    def initial_rejection(
        match: re.Match[str],
        amount: int,
        section: PriceSection,
        is_range: bool,
    ) -> str | None:
        """套用 FB Marketplace 的價格候選排除規則。"""
        if not NT_MAIN_PRICE_RE.fullmatch(match.group("currency")):
            return "商品主價格必須以 NT$ 開頭"
        if (
            amount < 1
            or amount
            > DEFAULT_PRICE_RISK_POLICY.maximum_supported_price
        ):
            return "價格超出支援範圍"
        if is_range:
            return "出價或多價格範圍不可作為商品主價格"
        if section != "unknown":
            return f"價格位於非主價格區塊：{section}"
        return None

    @staticmethod
    def title_confidence_bonus(
        layout: MarketplaceLayout,
        title_distance: int | None,
    ) -> float:
        """依版型與標題距離回傳候選信心加分。"""
        if title_distance is None:
            return 0.0
        if title_distance == 0:
            return 0.50
        if (
            layout == MarketplaceLayout.DESKTOP
            and title_distance <= 2
        ):
            return 0.48
        if (
            layout == MarketplaceLayout.MOBILE
            and title_distance == 1
        ):
            return 0.48
        if (
            layout == MarketplaceLayout.MOBILE
            and title_distance <= 2
        ):
            return 0.34
        return 0.15
