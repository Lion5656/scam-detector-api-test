"""解析商品市場價格，並在線上查價失敗時套用本地參考價。"""

from typing import Any

from backend.config import settings
from backend.services.dto.price_analysis import MarketPriceSource, SearchTool
from backend.services.image_price_service.models import MarketplaceCondition


def resolve_market_price(
    online_price_service: Any,
    product_name: str,
    brand_model: str,
    fallback_price: int,
    search_query: str = "",
    condition: MarketplaceCondition = MarketplaceCondition.NEW,
    condition_text: str = "",
) -> tuple[int, MarketPriceSource, SearchTool]:
    """取得線上市價，失敗時改用本地參考價。"""
    if settings.ONLINE_PRICE_ENABLED:
        query = (
            search_query
            or (
                brand_model
                if brand_model != "未知型號"
                else product_name
            )
        )
        online_price, search_tool = online_price_service.estimate_price(
            query,
            max_results=settings.ONLINE_PRICE_MAX_RESULTS,
            condition=condition,
            condition_text=condition_text,
        )
        if online_price > 0:
            return online_price, "online", search_tool

    return fallback_price, "fallback_local", "unused"
