"""OCR 文字區塊與座標處理的共用函式。"""

import re

from backend.services.image_price_service.models import (
    OCRDocument,
    OCRTextBlock,
)


def expand_blocks(document: OCRDocument) -> list[OCRTextBlock]:
    """將 OCR 內容整理成單行文字區塊。"""
    if document.blocks:
        expanded: list[OCRTextBlock] = []
        for block in document.blocks:
            lines = [
                line.strip()
                for line in block.text.splitlines()
                if line.strip()
            ]
            if len(lines) <= 1:
                expanded.append(block)
                continue
            for line in lines:
                expanded.append(
                    OCRTextBlock(
                        text=line,
                        x=block.x,
                        y=block.y,
                        width=block.width,
                        height=block.height,
                        confidence=block.confidence,
                    )
                )
        if expanded:
            return expanded
    return [
        OCRTextBlock(text=line.strip())
        for line in document.text.splitlines()
        if line.strip()
    ]


def normalized_lines(document: OCRDocument) -> list[str]:
    """回傳移除多餘空白與空行的 OCR 全文行。"""
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in document.text.splitlines()
        if line.strip()
    ]


def normalize_display_text(text: str) -> str:
    """整理顯示文字中的空白、斜線與連字號。"""
    text = re.sub(r"\s+", " ", text).strip(" ·|\t\r\n")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+([,，.。;；:：!?！？)])", r"\1", text)
    text = re.sub(r"([(（])\s+", r"\1", text)
    return text


def are_adjacent_title_lines(
    document: OCRDocument,
    earlier: OCRTextBlock,
    later: OCRTextBlock,
) -> bool:
    """判斷兩個 OCR 文字區塊是否為相鄰標題行。"""
    if (
        earlier.x is None
        or earlier.y is None
        or later.x is None
        or later.y is None
    ):
        return True

    horizontal_tolerance = max(
        earlier.width or 0.0,
        later.width or 0.0,
        120.0,
    )
    if abs(earlier.x - later.x) > horizontal_tolerance:
        return False

    if earlier.y > later.y + max(later.height or 0.0, 8.0):
        return False

    vertical_gap = later.y - (
        earlier.y + (earlier.height or 0.0)
    )
    maximum_gap = max(
        (document.height or 0.0) * 0.04,
        (earlier.height or 0.0) * 2,
        (later.height or 0.0) * 2,
        36.0,
    )
    return vertical_gap <= maximum_gap
