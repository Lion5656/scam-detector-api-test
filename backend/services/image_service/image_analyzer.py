import re
import unicodedata
from dataclasses import dataclass

from backend.config import settings
from backend.repository.market_price_repository import MarketPriceRepository
from backend.services.image_service.online_market_price_service import OnlineMarketPriceService


@dataclass
class ImageTextInsight:
    extracted_text: str
    product_name: str
    brand_model: str
    selling_price: int
    market_price: int
    market_price_source: str
    is_high_risk_below_market: bool


class ImageAnalyzer:
    def __init__(
        self,
        market_repo: MarketPriceRepository | None = None,
        online_price_service: OnlineMarketPriceService | None = None,
    ):
        self._market_repo = market_repo or MarketPriceRepository()
        self._online_price_service = online_price_service or OnlineMarketPriceService()

    _SALE_HINTS = ("售價", "特價", "優惠價", "限時", "只要", "現在", "原價", "下殺", "折扣")
    _DISCOUNT_HINTS = ("折", "折扣", "回饋", "免運", "優惠券", "現折", "下殺")
    _PRICE_TOKEN_RE = re.compile(r"(?:nt\$|twd|\$|售價|特價|優惠價|\d{3,7}\s*元)", flags=re.IGNORECASE)
    _MODEL_TOKEN_RE = re.compile(r"[a-z]{1,6}\s*-?\s*[a-z]{0,4}\s*\d{2,4}", flags=re.IGNORECASE)
    _GENERIC_MODEL_RE = re.compile(r"\b([a-z]{1,6}[\-\s]?[a-z0-9]{1,8}\d{2,5}[a-z0-9]{0,4})\b", flags=re.IGNORECASE)
    _KNOWN_BRANDS = {
        "panasonic": "Panasonic",
        "國際牌": "Panasonic",
        "apple": "Apple",
        "iphone": "Apple",
        "ipad": "Apple",
        "macbook": "Apple",
        "airpods": "Apple",
        "samsung": "Samsung",
        "galaxy": "Samsung",
        "sony": "Sony",
        "playstation": "Sony",
        "nintendo": "Nintendo",
        "dyson": "Dyson",
        "gopro": "GoPro",
        "xiaomi": "Xiaomi",
        "redmi": "Xiaomi",
        "oppo": "OPPO",
        "vivo": "vivo",
        "realme": "realme",
        "asus": "ASUS",
        "acer": "Acer",
        "hp": "HP",
        "dell": "Dell",
        "lenovo": "Lenovo",
    }
    _GENERIC_STOPWORDS = (
        "商城",
        "直送",
        "運送",
        "評價",
        "已售出",
        "加入購物車",
        "直接購買",
        "優惠",
        "折",
        "免運",
        "蝦皮",
        "momo",
        "pchome",
    )

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\r\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 常見 OCR 錯字修正（主要針對英數型號）
        replacements = {
            "iph0ne": "iphone",
            "ga1axy": "galaxy",
            "airp0ds": "airpods",
            "rnacbook": "macbook",
        }
        lowered = text.lower()
        for src, dst in replacements.items():
            lowered = lowered.replace(src, dst)
        return lowered

    @staticmethod
    def _extract_block_text(block) -> str:
        parts: list[str] = []
        for paragraph in block.paragraphs:
            paragraph_text = []
            for word in paragraph.words:
                word_text = "".join(symbol.text for symbol in word.symbols)
                if word_text:
                    paragraph_text.append(word_text)
            if paragraph_text:
                parts.append(" ".join(paragraph_text))
        return " ".join(parts).strip()

    @staticmethod
    def _bbox_from_vertices(vertices) -> tuple[float, float, float, float]:
        xs = [float(v.x or 0) for v in vertices]
        ys = [float(v.y or 0) for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def _prioritize_blocks_for_ecommerce(
        self,
        blocks: list[dict[str, float | str]],
        width: float,
        height: float,
        fallback_text: str,
    ) -> str:
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

            # 蝦皮商品頁常見資訊區在右半側，優先右側區塊。
            if cx >= width * 0.35:
                score += 2.0
            if cx >= width * 0.35 and cy <= height * 0.28:
                score += 2.0  # 標題區
            if cx >= width * 0.35 and height * 0.20 <= cy <= height * 0.60:
                score += 2.0  # 價格與折扣區

            lowered = text.lower()
            if self._PRICE_TOKEN_RE.search(lowered):
                score += 4.0
            if any(hint in text for hint in self._DISCOUNT_HINTS):
                score += 2.0
            if self._MODEL_TOKEN_RE.search(lowered):
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

        # 再補上一輪全圖 OCR 文字，避免漏掉關鍵詞。
        merged_lines = selected + [line.strip() for line in fallback_text.splitlines() if line.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for line in merged_lines:
            if line in seen:
                continue
            seen.add(line)
            deduped.append(line)

        return " ".join(deduped)

    def _extract_text_with_google_vision(self, data: bytes) -> str:
        try:
            from google.cloud import vision
        except ImportError as exc:
            raise RuntimeError("Google Cloud Vision 套件未安裝，請安裝 google-cloud-vision") from exc

        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=data)
            language_hints = [h.strip() for h in settings.GCV_LANGUAGE_HINTS.split(",") if h.strip()]
            image_context = vision.ImageContext(language_hints=language_hints)

            # 先嘗試 document_text_detection，對廣告圖與多行文字穩定性通常較好。
            doc_response = client.document_text_detection(image=image, image_context=image_context)
            if doc_response.error.message:
                raise RuntimeError(f"Google Vision OCR 錯誤: {doc_response.error.message}")
            doc_text = (doc_response.full_text_annotation.text or "").strip()
            if doc_text:
                pages = doc_response.full_text_annotation.pages
                blocks: list[dict[str, float | str]] = []
                page_width = 0.0
                page_height = 0.0

                for page in pages:
                    page_width = max(page_width, float(page.width or 0))
                    page_height = max(page_height, float(page.height or 0))
                    for block in page.blocks:
                        block_text = self._extract_block_text(block)
                        if not block_text:
                            continue
                        x0, y0, x1, y1 = self._bbox_from_vertices(block.bounding_box.vertices)
                        blocks.append({"text": block_text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

                prioritized = self._prioritize_blocks_for_ecommerce(
                    blocks,
                    width=page_width or 1.0,
                    height=page_height or 1.0,
                    fallback_text=doc_text,
                )
                return self._clean_ocr_text(prioritized)

            response = client.text_detection(image=image, image_context=image_context)
        except Exception as exc:
            raise RuntimeError(f"Google Vision OCR 執行失敗: {exc}") from exc

        if response.error.message:
            raise RuntimeError(f"Google Vision OCR 錯誤: {response.error.message}")

        if not response.text_annotations:
            return ""

        return self._clean_ocr_text(response.text_annotations[0].description)

    def _extract_text(self, data: bytes) -> str:
        provider = settings.OCR_PROVIDER.strip().lower()
        if provider not in {"google", "google_vision", "gcv"}:
            raise RuntimeError("OCR_PROVIDER 僅支援 google_vision")

        return self._extract_text_with_google_vision(data)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _extract_brand(self, text: str) -> str | None:
        lowered = text.lower()
        for token, canonical in self._KNOWN_BRANDS.items():
            if token in lowered or token in text:
                return canonical
        return None

    def _extract_generic_model(self, text: str) -> str | None:
        candidates: list[str] = []
        for m in self._GENERIC_MODEL_RE.finditer(text):
            token = m.group(1).strip()
            token_norm = re.sub(r"\s+", "", token).upper()

            if len(token_norm) < 4:
                continue
            if token_norm.isdigit():
                continue
            if token_norm.startswith(("NT", "TWD")):
                continue
            if re.fullmatch(r"\d+[A-Z]?", token_norm):
                continue

            candidates.append(token_norm)

        if not candidates:
            return None

        # 優先包含連字號或英數混合且含數字的型號。
        candidates.sort(key=lambda t: ("-" not in t, not any(ch.isdigit() for ch in t), len(t)))
        return candidates[0]

    def _extract_generic_product_name(self, text: str, brand: str | None) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return f"{brand} 商品" if brand else "一般商品"

        lines = [line.strip() for line in re.split(r"[\n\r]+", cleaned) if line.strip()]
        for line in lines[:6]:
            if any(stop in line for stop in self._GENERIC_STOPWORDS):
                continue

            # 去掉價格/促銷尾巴，保留較像標題的前段。
            line = re.split(r"\$|nt\$|twd|售價|特價|優惠價", line, flags=re.IGNORECASE)[0].strip()
            if len(line) >= 6:
                if brand and brand.lower() not in line.lower():
                    return f"{brand} {line[:40]}"
                return line[:40]

        return f"{brand} 商品" if brand else "一般商品"

    def _identify_product(self, text: str) -> tuple[str, str, int]:
        """
        使用 Agent 解析商品與品牌型號。
        """
        # 快取或常見商品秒召回
        product_name, brand_model, market_price = self._market_repo.find_by_text(text)
        if market_price > 0:
            return product_name, brand_model, market_price

        from backend.services.image_service.product_identifier_agent import create_product_identifier_agent
        
        agent = create_product_identifier_agent()
        
        prompt = f"""
        請從以下文字中，找出商品名稱以及品牌與型號。
        提供的文字是從網路購物平台的圖片 OCR 辨識出來的。
        
        如果你覺得文字中的商品名稱縮寫或型號不完整，
        請使用你的 Search 工具去網路上搜尋，找出該商品最有可能的完整品牌與確切型號。
        
        請以這兩個欄位回傳：
        1. product_name: (完整的商品名稱)
        2. brand_model: (品牌名稱 加上 確切型號)
        
        OCR 文字如下：
        {text}
        
        如果真的找不到任何商品，請回傳：
        product_name: 未知商品
        brand_model: unknown
        """
        try:
            response = agent.run(prompt)
            
            product_name = "未知商品"
            brand_model = "unknown"
            
            for line in response.split('\n'):
                line = line.strip()
                if line.lower().startswith('product_name:'):
                    product_name = line.split(':', 1)[1].strip()
                elif line.lower().startswith('brand_model:'):
                    brand_model = line.split(':', 1)[1].strip()
                    
            return product_name, brand_model, 0
        except Exception as e:
            print(f"Product Identifier Agent Failed: {e}")
            return "未知商品", "unknown", 0

    def _extract_selling_price(self, text: str) -> int:
        clean = unicodedata.normalize("NFKC", text).replace(",", "")
        candidates: list[tuple[int, int]] = []

        for m in re.finditer(r"(?:NT\$?|TWD|台幣)?\s*([1-9]\d{2,6})(?:\s*元|塊)?", clean, flags=re.IGNORECASE):
            value = int(m.group(1))
            if value < 100 or value > 2_000_000:
                continue

            start = max(0, m.start() - 8)
            end = min(len(clean), m.end() + 8)
            context = clean[start:end]
            priority = 1 if any(hint in context for hint in self._SALE_HINTS) else 0
            candidates.append((priority, value))

        if not candidates:
            return 0

        # 有售價語境優先；否則取最小值以貼近廣告主打價格。
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]

    def _resolve_market_price(self, product_name: str, brand_model: str, fallback_price: int) -> tuple[int, str]:
        if settings.ONLINE_PRICE_ENABLED:
            query = brand_model if brand_model != "未知型號" else product_name
            online_price = self._online_price_service.estimate_taiwan_market_price(
                query,
                max_results=settings.ONLINE_PRICE_MAX_RESULTS,
            )
            if online_price > 0:
                return online_price, "online"

        return fallback_price, "fallback_local"

    def analyze_image_bytes(self, data: bytes) -> ImageTextInsight:
        text = self._extract_text(data)
        product_name, brand_model, fallback_market_price = self._identify_product(text)
        market_price, market_price_source = self._resolve_market_price(
            product_name,
            brand_model,
            fallback_market_price,
        )
        selling_price = self._extract_selling_price(text)

        is_high_risk = selling_price > 0 and market_price > 0 and selling_price < market_price * 0.5

        return ImageTextInsight(
            extracted_text=text,
            product_name=product_name,
            brand_model=brand_model,
            selling_price=selling_price,
            market_price=market_price,
            market_price_source=market_price_source,
            is_high_risk_below_market=is_high_risk,
        )


image_analyzer = ImageAnalyzer()
