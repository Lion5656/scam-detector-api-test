"""價格候選資料與跨頁面分析的主價格選擇邏輯。"""

import re
from dataclasses import dataclass, field

from backend.services.image_price_service.models import (
    OCRDocument,
    OCRTextBlock,
    PriceCandidate,
)


@dataclass
class CandidateCollection:
    """保存價格候選及選擇主價格所需的附加資訊。"""

    candidates: list[PriceCandidate] = field(default_factory=list)
    rejected: list[PriceCandidate] = field(default_factory=list)
    titles: dict[int, str] = field(default_factory=dict)
    positions: dict[
        int,
        tuple[float, float | None, float | None],
    ] = field(default_factory=dict)


def candidate_horizontal_position(
    block: OCRTextBlock,
    text: str,
    match: re.Match[str],
) -> float:
    """計算同列多價格比較時使用的水平位置。"""
    if block.x is None:
        return float(match.start())
    horizontal_position = block.x
    if block.width and text:
        horizontal_position += (
            block.width * match.start() / len(text)
        )
    return horizontal_position


class PriceCandidateSelector:
    """依信心與同列最左價格規則選出商品主價格。"""

    def __init__(self, minimum_confidence: float) -> None:
        self.minimum_confidence = minimum_confidence

    @staticmethod
    def _same_row_candidates(
        document: OCRDocument,
        selected: PriceCandidate,
        accepted: list[PriceCandidate],
        positions: dict[
            int,
            tuple[float, float | None, float | None],
        ],
    ) -> list[PriceCandidate]:
        """找出與最高信心候選位於同一列的所有價格。"""
        _, selected_y, selected_height = positions[id(selected)]
        same_row: list[PriceCandidate] = []
        for candidate in accepted:
            _, candidate_y, candidate_height = positions[id(candidate)]
            if candidate.block_index == selected.block_index:
                same_row.append(candidate)
                continue
            if selected_y is None or candidate_y is None:
                continue
            row_tolerance = max(
                selected_height or 0.0,
                candidate_height or 0.0,
                (document.height or 0.0) * 0.01,
                8.0,
            )
            if abs(candidate_y - selected_y) <= row_tolerance:
                same_row.append(candidate)
        return same_row

    def select(
        self,
        document: OCRDocument,
        collection: CandidateCollection,
        *,
        minimum_confidence: float | None = None,
    ) -> tuple[PriceCandidate | None, bool]:
        """回傳主價格候選及是否同列存在多個價格。"""
        threshold = (
            self.minimum_confidence
            if minimum_confidence is None
            else minimum_confidence
        )
        accepted = [
            candidate
            for candidate in collection.candidates
            if candidate.confidence >= threshold
        ]
        if not accepted:
            return None, False

        accepted.sort(
            key=lambda item: (-item.confidence, item.block_index)
        )
        selected = accepted[0]
        same_row = self._same_row_candidates(
            document,
            selected,
            accepted,
            collection.positions,
        )
        has_multiple_prices_on_row = len(same_row) > 1
        if has_multiple_prices_on_row:
            selected = min(
                same_row,
                key=lambda item: (
                    collection.positions[id(item)][0],
                    item.block_index,
                ),
            )
        return selected, has_multiple_prices_on_row
