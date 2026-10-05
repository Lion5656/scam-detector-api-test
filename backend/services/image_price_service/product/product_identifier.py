"""協調本地資料、規則辨識與 AI 正規化，並建立台灣市場查價搜尋詞。"""

import json
import logging
from typing import Any

from groq import APIError

from backend.repository.market_price_repository import (
    MarketPriceRepository, market_price_repository)
from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.product.patterm_identifier import \
    pattern_identifier
from backend.services.image_price_service.product.product_info_extractor import (
    ProductInfoExtractor, create_product_info_extractor)

logging = logging.getLogger(__name__)

PRODUCT_NORMALIZATION_SYSTEM_PROMPT = """
你是商品資訊正規化助手。請根據購物平台 OCR 文字，辨識並整理商品資訊。

規則：
1. 修正明顯 OCR 錯字，但不可捏造無法確認的型號或規格。
2. 刊登價格、運費、折扣與分期金額都不是商品規格，不得放入 known_specs。
3. product_name 使用一般消費者可理解的標準商品名稱。
4. brand_model 必須使用「品牌 + 型號」；無法確認時填「未知型號」。
5. known_specs 放入容量、尺寸、解析度、顏色、版本、技術規格等已確認資訊
6. 不要建立搜尋詞，也不要查詢價格。

只回傳以下 JSON，不要加入 Markdown 或額外說明：
{
  "product_name": "標準商品名稱",
  "brand_model": "品牌 型號",
  "known_specs": ["已確認的規格"]
}
""".strip()


class ProductIdentifier:
    """辨識商品資訊並建立查價搜尋詞。"""

    def __init__(
        self,
        market_repo: MarketPriceRepository | None = None,
        info_extractor: ProductInfoExtractor | None = None,
    ) -> None:
        """建立商品辨識器。"""
        self._market_repo = market_repo or market_price_repository
        self._info_extractor = info_extractor

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
            normalized = self._get_info_extractor().complete_json(
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

    def _get_info_extractor(self) -> ProductInfoExtractor:
        """取得或建立商品資訊擷取器。"""
        if self._info_extractor is None:
            self._info_extractor = create_product_info_extractor()
        return self._info_extractor

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
        """使用完整品名與少量關鍵規格建立搜尋詞。"""
        unknown_values = {
            "",
            "一般商品",
            "未知商品",
            "未知型號",
            "未知品牌型號",
        }
        unknown_model_markers = ("未知型號", "未知品牌型號")
        normalized_name = " ".join(str(product_name).split())
        normalized_model = " ".join(str(brand_model).split())

        name_is_known = normalized_name not in unknown_values
        model_is_known = (
            normalized_model not in unknown_values
            and not any(
                marker in normalized_model
                for marker in unknown_model_markers
            )
        )

        if name_is_known and model_is_known:
            if normalized_model.casefold() in normalized_name.casefold():
                search_name = normalized_name
            else:
                model_tokens = {
                    token.casefold()
                    for token in normalized_model.split()
                }
                description_tokens = [
                    token
                    for token in normalized_name.split()
                    if token.casefold() not in model_tokens
                ]
                search_name = " ".join(
                    [normalized_model, *description_tokens]
                )
        elif name_is_known:
            search_name = normalized_name
        elif model_is_known:
            search_name = normalized_model
        else:
            return ""

        additional_specs: list[str] = []
        normalized_search_name = search_name.casefold()
        for value in known_specs:
            spec = " ".join(str(value).split())
            if not spec or spec.casefold() in normalized_search_name:
                continue
            additional_specs.append(spec)

        if additional_specs:
            search_name = f"{search_name} {' '.join(additional_specs[:2])}"

        return f"{search_name} 價格"


product_identifier = ProductIdentifier()
