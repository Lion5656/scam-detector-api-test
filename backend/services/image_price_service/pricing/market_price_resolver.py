"""解析結構化市場價格。"""

from typing import Any

from backend.config import settings
from backend.services.dto.price_analysis import MarketPriceEstimate
from backend.services.image_price_service.domain.models import MarketplaceCondition


def resolve_market_price(
    online_price_service: Any,
    product_name: str,
    brand_model: str,
    fallback_price: int,
    search_query: str = "",
    condition: MarketplaceCondition = MarketplaceCondition.NEW,
    condition_text: str = "",
) -> tuple[MarketPriceEstimate, ...]:
    """取得市場估計。"""
    if settings.ONLINE_PRICE_ENABLED:
        query = (
            search_query
            or (
                brand_model
                if brand_model != "未知型號"
                else product_name
            )
        )
        return online_price_service.estimate_prices(
            query,
            max_results=settings.ONLINE_PRICE_MAX_RESULTS,
            condition=condition,
            condition_text=condition_text,
        )

    supported_fallback = max(
        0,
        min(
            int(fallback_price),
            online_price_service.policy.maximum_supported_price,
        ),
    )
    return (
        MarketPriceEstimate(
            status="success" if supported_fallback > 0 else "not_found",
            condition=condition,
            reference_mode="median_low_sample",
            median_price=supported_fallback,
            low_price=supported_fallback,
            high_price=supported_fallback,
            sample_count=1 if supported_fallback > 0 else 0,
            site_count=1 if supported_fallback > 0 else 0,
            source="fallback_local",
            confidence=0.0,
            candidates=(),
        ),
    )
