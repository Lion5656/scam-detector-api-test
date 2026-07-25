"""FB Marketplace mobile 與 desktop 標題版型策略。"""

from typing import Protocol

from backend.services.image_price_service.models import (
    OCRDocument,
    OCRTextBlock,
)
from backend.services.image_price_service.ocr.ocr_text_utils import (
    are_adjacent_title_lines,
    normalize_display_text,
)
from backend.services.image_price_service.page_analysis.fb_marketplace.fb_marketplace_rules import (
    FBMarketplaceRules,
)

_MAX_TITLE_LINES = 4
_DESKTOP_INFO_PANEL_LEFT_RATIO = 0.48
_DESKTOP_TITLE_LOOKBACK_RATIO = 0.20

TitleMatch = tuple[OCRTextBlock | None, str | None, int | None]


class LayoutTitleExtractor(Protocol):
    """定義版型標題抽取策略介面。"""

    def find_title(
        self,
        document: OCRDocument,
        blocks: list[OCRTextBlock],
        price_index: int,
        price: OCRTextBlock,
        prefix: str,
    ) -> TitleMatch:
        """尋找價格對應的商品標題。"""
        ...


def _adjacent_title_above_price(
    rules: FBMarketplaceRules,
    document: OCRDocument,
    blocks: list[OCRTextBlock],
    price_index: int,
    price: OCRTextBlock,
    prefix: str,
) -> TitleMatch:
    """從價格同區塊前綴及相鄰 OCR 行組合商品標題。"""
    title: OCRTextBlock | None = None
    title_distance: int | None = None
    title_fragments: list[str] = []
    if rules.looks_like_title(prefix):
        title = price
        title_distance = 0
        title_fragments.append(normalize_display_text(prefix))

    found_title = title is not None
    adjacent_block = title or price
    for prior_index in range(
        price_index - 1,
        max(-1, price_index - _MAX_TITLE_LINES - 1),
        -1,
    ):
        prior_block = blocks[prior_index]
        if not rules.looks_like_title(prior_block.text):
            if found_title:
                break
            continue
        if found_title and not are_adjacent_title_lines(
            document,
            prior_block,
            adjacent_block,
        ):
            break

        title_fragments.append(
            normalize_display_text(prior_block.text)
        )
        if title is None:
            title = prior_block
            title_distance = price_index - prior_index
        found_title = True
        adjacent_block = prior_block

    title_text = (
        normalize_display_text(
            " ".join(reversed(title_fragments))
        )
        if title_fragments
        else None
    )
    return title, title_text, title_distance


class MobileLayoutExtractor:
    """依 mobile 相鄰 OCR 行規則擷取商品標題。"""

    def __init__(self, rules: FBMarketplaceRules) -> None:
        self._rules = rules

    def find_title(
        self,
        document: OCRDocument,
        blocks: list[OCRTextBlock],
        price_index: int,
        price: OCRTextBlock,
        prefix: str,
    ) -> TitleMatch:
        return _adjacent_title_above_price(
            self._rules,
            document,
            blocks,
            price_index,
            price,
            prefix,
        )


class DesktopLayoutExtractor:
    """依 desktop 右側資訊欄規則擷取商品標題。"""

    def __init__(self, rules: FBMarketplaceRules) -> None:
        self._rules = rules

    def _title_above_price(
        self,
        document: OCRDocument,
        blocks: list[OCRTextBlock],
        price: OCRTextBlock,
    ) -> TitleMatch:
        """在右側資訊欄的加大範圍內尋找主價格上方標題。"""
        if (
            not document.width
            or not document.height
            or price.x is None
            or price.y is None
        ):
            return None, None, None

        panel_left = (
            document.width * _DESKTOP_INFO_PANEL_LEFT_RATIO
        )
        lookback = max(
            document.height * _DESKTOP_TITLE_LOOKBACK_RATIO,
            (price.height or 0.0) * 6,
            80.0,
        )
        same_row_tolerance = max(
            document.height * 0.006,
            (price.height or 0.0) * 0.25,
            6.0,
        )
        horizontal_tolerance = document.width * 0.12
        spatial_matches: list[tuple[int, OCRTextBlock]] = []

        for block_index, candidate in enumerate(blocks):
            if (
                candidate is price
                or candidate.x is None
                or candidate.y is None
                or candidate.x < panel_left
                or not self._rules.looks_like_title(candidate.text)
            ):
                continue
            if not (
                price.y - lookback
                <= candidate.y
                <= price.y + same_row_tolerance
            ):
                continue

            candidate_right = (
                candidate.x + (candidate.width or 0.0)
            )
            price_right = price.x + (price.width or 0.0)
            if (
                candidate_right
                < price.x - horizontal_tolerance
                or candidate.x
                > price_right + horizontal_tolerance
            ):
                continue
            spatial_matches.append((block_index, candidate))

        if not spatial_matches:
            return None, None, None

        spatial_matches.sort(
            key=lambda item: (
                item[1].y or 0.0,
                item[1].x or 0.0,
                item[0],
            )
        )
        selected_matches = spatial_matches[-_MAX_TITLE_LINES:]
        title_text = normalize_display_text(
            " ".join(
                block.text
                for _, block in selected_matches
            )
        )
        _, nearest_block = max(
            selected_matches,
            key=lambda item: (
                item[1].y or 0.0,
                item[0],
            ),
        )
        # 桌面 OCR 閱讀順序可能被左側圖片與導覽列打散；
        # 通過空間驗證後仍視為緊鄰主價格。
        return nearest_block, title_text, 1

    def find_title(
        self,
        document: OCRDocument,
        blocks: list[OCRTextBlock],
        price_index: int,
        price: OCRTextBlock,
        prefix: str,
    ) -> TitleMatch:
        if self._rules.looks_like_title(prefix):
            return _adjacent_title_above_price(
                self._rules,
                document,
                blocks,
                price_index,
                price,
                prefix,
            )

        spatial_title = self._title_above_price(
            document,
            blocks,
            price,
        )
        if spatial_title[1] is not None:
            return spatial_title
        return _adjacent_title_above_price(
            self._rules,
            document,
            blocks,
            price_index,
            price,
            prefix,
        )
