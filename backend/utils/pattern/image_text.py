import re

SALE_HINTS = ("售價", "特價", "優惠價", "限時", "只要", "現在", "原價", "下殺", "折扣")
DISCOUNT_HINTS = ("折", "折扣", "回饋", "免運", "優惠券", "現折", "下殺")
PRICE_TOKEN_RE = re.compile(r"(?:nt\$|twd|\$|售價|特價|優惠價|\d{3,7}\s*元)", flags=re.IGNORECASE)
MODEL_TOKEN_RE = re.compile(r"[a-z]{1,6}\s*-?\s*[a-z]{0,4}\s*\d{2,4}", flags=re.IGNORECASE)
GENERIC_MODEL_RE = re.compile(r"\b([a-z]{1,6}[\-\s]?[a-z0-9]{1,8}\d{2,5}[a-z0-9]{0,4})\b", flags=re.IGNORECASE)
KNOWN_BRANDS = {
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
GENERIC_STOPWORDS = (
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
MODEL_PREFIX_STOPWORDS = ("sale", "price", "discount", "off", "nt", "twd")
