"""使用品牌對照表、型號正則表達式與標題規則辨識商品。"""

import logging
import re

from backend.services.dto.price_analysis import ProductIdentification

logger = logging.getLogger(__name__)

_GENERIC_MODEL_RE = re.compile(
    (
        r"(?<![a-z0-9])("
        r"[a-z]{1,6}[\-\s]?[a-z0-9]{1,8}\d{2,5}[a-z0-9]{0,4}"
        r"|"
        r"\d{1,6}[\-\s]?[a-z][a-z0-9]{0,10}"
        r")(?![a-z0-9])"
    ),
    flags=re.IGNORECASE,
)
_KNOWN_BRANDS = {
    # 消費電子與家電
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
    "hitachi": "Hitachi",
    "日立": "Hitachi",
    "sanyo": "SANYO",
    "三洋": "SANYO",
    # 服飾、鞋類與運動品牌
    "new balance": "New Balance",
    "newbalance": "New Balance",
    "air jordan": "Air Jordan",
    "airjordan": "Air Jordan",
    "jordan": "Air Jordan",
    "nike": "Nike",
    "耐吉": "Nike",
    "adidas": "adidas",
    "愛迪達": "adidas",
    "fila": "FILA",
    "斐樂": "FILA",
    "puma": "PUMA",
    "彪馬": "PUMA",
    "uniqlo": "UNIQLO",
    "優衣庫": "UNIQLO",
    "under armour": "Under Armour",
    "underarmour": "Under Armour",
    "安德瑪": "Under Armour",
    "converse": "Converse",
    "匡威": "Converse",
    "vans": "Vans",
    "reebok": "Reebok",
    "銳步": "Reebok",
    "the north face": "The North Face",
    "thenorthface": "The North Face",
    "北臉": "The North Face",
    "columbia": "Columbia",
    "哥倫比亞": "Columbia",
    "patagonia": "Patagonia",
    "zara": "ZARA",
    "h&m": "H&M",
    # 家具與居家品牌
    "ikea": "IKEA",
    "宜家家居": "IKEA",
    "nitori": "宜得利 NITORI",
    "宜得利": "宜得利 NITORI",
    "muji": "MUJI 無印良品",
    "無印良品": "MUJI 無印良品",
    "hola": "HOLA",
    "特力和樂": "HOLA",
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
_MODEL_PREFIX_STOPWORDS = ("sale", "price", "discount", "off", "nt", "twd")


class PatternIdentifier:
    """以本地規則辨識商品名稱、品牌與型號。"""
    def _extract_brand(self, text: str, model: str | None) -> str | None:
        """依已知品牌關鍵字回傳標準化品牌名稱。"""
        lowered = text.lower()
        for token, canonical in _KNOWN_BRANDS.items():
            if token in lowered or token in text:
                return canonical

        text = re.sub(r"\s+", "", text)
        model_match = re.search(model, text, flags=re.IGNORECASE) if model else None
        logger.info(f"Model match found: {model_match}, {model}, {text}")
        text_before_model = text[:model_match.start()] if model_match else text

        return text_before_model

    def _extract_generic_model(self, text: str) -> str | None:
        """擷取並排序可能的英數型號，排除價格與一般數字。"""
        model_text = text
        for brand_token in _KNOWN_BRANDS:
            token_pattern = (
                rf"(?<![a-z0-9])"
                rf"{re.escape(brand_token)}"
                rf"(?![a-z0-9])"
            )
            model_text = re.sub(
                token_pattern,
                " ",
                model_text,
                flags=re.IGNORECASE,
            )

        candidates: list[str] = []

        for m in _GENERIC_MODEL_RE.finditer(model_text):
            token = m.group(1).strip()
            token_norm = re.sub(r"\s+", "", token).upper()

            if len(token_norm) < 4:
                continue
            if token_norm.isdigit():
                continue
            if token_norm.startswith(("NT", "TWD")):
                continue
            if token_norm.lower().startswith(_MODEL_PREFIX_STOPWORDS):
                continue

            candidates.append(token_norm)

        if not candidates:
            return None

        # 優先選擇含連字號、含數字且長度較短的候選值。
        candidates.sort(key=lambda t: ("-" not in t, not any(ch.isdigit() for ch in t), len(t)))
        return candidates[0]

    def _extract_generic_product_name(self, text: str, brand: str | None) -> str:
        """從 OCR 文字選出可能的商品標題，必要時補上品牌名稱。"""
        lines = [
            re.sub(r"[^\S\r\n]+", " ", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]
        if not lines:
            return f"{brand} 商品" if brand else "一般商品"

        for line in lines[:6]:
            if any(stop in line for stop in _GENERIC_STOPWORDS):
                continue

            # 截斷價格與促銷欄位，只保留前方較接近商品標題的文字。
            line = re.split(r"\$|nt\$|twd|售價|特價|優惠價", line, flags=re.IGNORECASE)[0].strip()
            if len(line) >= 6:
                if brand and brand.lower() not in line.lower():
                    return f"{brand} {line[:40]}"
                return line[:40]

        return f"{brand} 商品" if brand else "一般商品"

    def identify_product(self, text: str) -> ProductIdentification:
        """將 OCR 文字轉成基本商品資訊。"""
        model = self._extract_generic_model(text)
        brand = self._extract_brand(text, model)
        product_name = self._extract_generic_product_name(text, brand)

        if brand and model:
            brand_model = f"{brand} {model}"
        elif brand:
            brand_model = f"{brand}"
        elif model:
            brand_model = model
        else:
            brand_model = "未知型號"

        return ProductIdentification(product_name=product_name, brand_model=brand_model, market_price=0)
pattern_identifier = PatternIdentifier()
