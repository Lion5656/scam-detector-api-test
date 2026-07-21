"""依版面位置與商品文字特徵排序電商頁面的 OCR 區塊。"""

from backend.utils.pattern.image_text import (DISCOUNT_HINTS, MODEL_TOKEN_RE,
                                              PRICE_TOKEN_RE)


class EcommerceBlockRanker:
    """以版面位置和內容特徵計算電商 OCR 區塊的優先度。

    位於頁面右側的標題或價格區，以及包含價格、折扣、型號或較長內容的區塊
    會取得較高分。結果會選取高分區塊，再補入完整 OCR 全文並移除重複行。
    """
    def prioritize_blocks_for_ecommerce(
        self,
        blocks: list[dict[str, float | str]],
        width: float,
        height: float,
        fallback_text: str,
    ) -> str:
        """排序並合併 OCR 區塊；無可用區塊時回傳完整 OCR 全文。"""
        if not blocks:
            return fallback_text

        scored: list[tuple[float, str, float, float]] = []
        for item in blocks:
            text = str(item["text"]).strip()
            if not text:
                continue

            x0 = float(item["x0"])
            y0 = float(item["y0"])
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2

            score = 0.0

            # 常見桌面版電商頁將商品資訊放在右側，因此提高右側區塊權重。
            if cx >= width * 0.35:
                score += 2.0
            if cx >= width * 0.35 and cy <= height * 0.28:
                score += 2.0  # 頁面右上方通常是商品標題。
            if cx >= width * 0.35 and height * 0.20 <= cy <= height * 0.60:
                score += 2.0  # 頁面右側中段通常是價格與折扣資訊。

            lowered = text.lower()
            if PRICE_TOKEN_RE.search(lowered):
                score += 4.0
            if any(hint in text for hint in DISCOUNT_HINTS):
                score += 2.0
            if MODEL_TOKEN_RE.search(lowered):
                score += 2.0
            if len(text) >= 10:
                score += 1.0

            scored.append((score, text, y0, x0))

        if not scored:
            return fallback_text

        scored.sort(key=lambda x: (-x[0], x[2], x[3]))

        selected = [text for score, text, _, _ in scored if score >= 3.0][:14]
        if not selected:
            selected = [text for _, text, _, _ in scored[:8]]

        # 補入未入選的完整 OCR 文字，避免遺漏低分但有用的內容。
        merged_lines = selected + [line.strip() for line in fallback_text.splitlines() if line.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for line in merged_lines:
            if line in seen:
                continue
            seen.add(line)
            deduped.append(line)

        return " ".join(deduped)

ecommerce_block_ranker = EcommerceBlockRanker()
