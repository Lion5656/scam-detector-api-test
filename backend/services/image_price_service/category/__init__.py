"""公開商品類別處理器的介面與註冊表。"""

from backend.services.image_price_service.category.base import CategoryHandler
from backend.services.image_price_service.category.registry import (
    CategoryRegistry,
    category_registry,
)

__all__ = ["CategoryHandler", "CategoryRegistry", "category_registry"]
