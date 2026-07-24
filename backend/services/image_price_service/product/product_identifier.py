"""協調本地資料、規則辨識與 AI 正規化，並建立台灣市場查價搜尋詞。"""

import json
import logging
from typing import Any

from groq import APIError

from backend.repository.market_price_repository import (
    MarketPriceRepository,
    market_price_repository,
)
from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.product.patterm_identifier import (
    pattern_identifier,
)
from backend.services.image_price_service.product.product_research_agent import (
    ProductResearchAgent,
    create_product_research_agent,
)

logging = logging.getLogger(__name__)

PRODUCT_NORMALIZATION_SYSTEM_PROMPT = """
你是商品資訊正規化助手。請針對購物平台 OCR 文字進行一次審查，整理可確認的
品牌、品名、型號、版本、尺寸、容量、品況與其他有助於搜尋的規格。

規則：
1. 修正明顯 OCR 錯字，但不可捏造無法確認的型號或規格。
2. 刊登價格、運費、折扣與分期金額都不是商品規格，不得放入 known_specs。
3. product_name 使用一般消費者可理解的標準商品名稱。
4. brand_model 優先保留品牌與確切型號；無法確認時填「未知型號」。
5. 不要建立搜尋詞，也不要查詢價格。

只回傳以下 JSON，不要加入 Markdown 或額外說明：
{
  "product_name": "標準商品名稱",
  "brand_model": "品牌與型號",
  "known_specs": ["已確認的規格"]
}
""".strip()


class ProductIdentifier:
    """辨識商品資訊並建立查價搜尋詞。"""

    def __init__(
        self,
        market_repo: MarketPriceRepository | None = None,
        research_agent: ProductResearchAgent | None = None,
    ) -> None:
        """建立商品辨識器。"""
        self._market_repo = market_repo or market_price_repository
        self._research_agent = research_agent

    def identify(self, text: str) -> ProductIdentification:
        """辨識商品並回傳正規化資訊。"""
        product_name, brand_model, market_price = self._market_repo.find_by_text(
            text
        )
        if market_price > 0:
            return self._build_identification(
                product_name=product_name,
                brand_model=brand_model,
                market_price=market_price,
            )

        pattern_product = pattern_identifier.identify_product(text)
        user_prompt = json.dumps(
            {
                "ocr_text": text,
                "rule_hint": {
                    "product_name": pattern_product.product_name,
                    "brand_model": pattern_product.brand_model,
                },
            },
            ensure_ascii=False,
        )

        try:
            normalized = self._get_research_agent().complete_json(
                system_prompt=PRODUCT_NORMALIZATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            normalized_name = str(
                normalized.get("product_name", pattern_product.product_name)
            ).strip()
            normalized_model = str(
                normalized.get("brand_model", pattern_product.brand_model)
            ).strip()
            known_specs = self._normalize_specs(
                normalized.get("known_specs", [])
            )
            return self._build_identification(
                product_name=normalized_name or pattern_product.product_name,
                brand_model=normalized_model or pattern_product.brand_model,
                known_specs=known_specs,
            )
        except (APIError, ValueError, TypeError) as e:
            logging.warning(
                "商品正規化失敗，使用 Pattern 規則辨識結果。錯誤: %s", e
            )
            return self._build_identification(
                product_name=pattern_product.product_name,
                brand_model=pattern_product.brand_model,
            )

    def _get_research_agent(self) -> ProductResearchAgent:
        """取得或建立商品研究代理。"""
        if self._research_agent is None:
            self._research_agent = create_product_research_agent()
        return self._research_agent

    def _build_identification(
        self,
        *,
        product_name: str,
        brand_model: str,
        market_price: int = 0,
        known_specs: list[str] | None = None,
    ) -> ProductIdentification:
        """建立商品辨識結果與搜尋詞。"""
        specs = known_specs or []
        return ProductIdentification(
            product_name=product_name,
            brand_model=brand_model,
            known_specs=specs,
            search_query=self._build_price_search_query(
                product_name,
                brand_model,
                specs,
            ),
            market_price=market_price,
        )

    @staticmethod
    def _normalize_specs(value: Any) -> list[str]:
        """整理規格並移除重複值。"""
        if not isinstance(value, list):
            return []

        specs: list[str] = []
        seen: set[str] = set()
        for item in value:
            spec = str(item).strip()
            normalized = spec.casefold()
            if not spec or normalized in seen:
                continue
            seen.add(normalized)
            specs.append(spec)
        return specs

    @staticmethod
    def _build_price_search_query(
        product_name: str,
        brand_model: str,
        known_specs: list[str],
    ) -> str:
        """建立台灣市場價格搜尋詞。"""
        parts: list[str] = []
        seen: set[str] = set()
        unknown_values = {
            "",
            "一般商品",
            "未知商品",
            "未知型號",
            "未知品牌型號",
        }
        for value in [brand_model, product_name, *known_specs]:
            cleaned = str(value).strip()
            normalized = cleaned.casefold()
            if cleaned in unknown_values or normalized in seen:
                continue
            seen.add(normalized)
            parts.append(cleaned)

        if not parts:
            return ""
        return f"{' '.join(parts)} 台灣 價格"


product_identifier = ProductIdentifier()
