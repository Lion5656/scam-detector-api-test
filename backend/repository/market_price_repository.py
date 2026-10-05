import json
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from rapidfuzz import fuzz

from backend.core.config import settings


class MarketPriceRecord(TypedDict):
    aliases: list[str]
    product_name: str
    brand_model: str
    market_price: int


DEFAULT_MARKET_PRICE_RECORDS: list[MarketPriceRecord] = [
    {
        "aliases": [
            "Panasonic EH-NE11",
            "國際牌 EH-NE11",
            "EH-NE11",
            "Panasonic 國際牌 1200W 負離子速乾型冷熱吹風機",
        ],
        "product_name": "Panasonic 國際牌 1200W 負離子速乾型冷熱吹風機",
        "brand_model": "Panasonic EH-NE11",
        "market_price": 1290,
    },
    {
        "aliases": ["iPhone 15", "Apple iPhone 15"],
        "product_name": "Apple iPhone 15",
        "brand_model": "Apple iPhone 15",
        "market_price": 27900,
    },
]


class MarketPriceRepository:
    def __init__(self, source_path: str | None = None):
        self._source_path = Path(source_path or settings.MARKET_PRICE_DB_PATH)
        self._records = self._load_records()

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)

    @staticmethod
    def _normalize_for_fuzzy(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load_records(self) -> list[MarketPriceRecord]:
        if not self._source_path.exists():
            return DEFAULT_MARKET_PRICE_RECORDS.copy()

        with self._source_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        records: list[MarketPriceRecord] = []
        for row in raw:
            aliases = [str(a) for a in row.get("aliases", []) if str(a).strip()]
            if not aliases:
                continue

            market_price = int(row.get("market_price", 0) or 0)
            records.append(
                {
                    "aliases": aliases,
                    "product_name": str(row.get("product_name", "未知商品")),
                    "brand_model": str(row.get("brand_model", "未知型號")),
                    "market_price": market_price,
                }
            )

        return records

    def reload(self) -> None:
        self._records = self._load_records()

    def find_by_text(self, text: str) -> tuple[str, str, int]:
        normalized = self._normalize(text)
        for item in self._records:
            for alias in item["aliases"]:
                if self._normalize(alias) in normalized:
                    return item["product_name"], item["brand_model"], item["market_price"]

        # OCR 容錯：若有輕微辨識錯字，使用模糊比對兜底。
        fuzzy_text = self._normalize_for_fuzzy(text)
        best_score = 0.0
        best_item: MarketPriceRecord | None = None

        for item in self._records:
            for alias in item["aliases"]:
                alias_fuzzy = self._normalize_for_fuzzy(alias)
                if not alias_fuzzy:
                    continue

                score = max(
                    float(fuzz.partial_ratio(alias_fuzzy, fuzzy_text)),
                    float(fuzz.token_set_ratio(alias_fuzzy, fuzzy_text)),
                )
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item and best_score >= settings.PRODUCT_MATCH_FUZZY_MIN_SCORE:
            return best_item["product_name"], best_item["brand_model"], best_item["market_price"]

        return "未知商品", "未知型號", 0
    
market_price_repository = MarketPriceRepository()
