"""管理商品類別處理器的註冊與選用。"""

from backend.services.image_price_service.category.base import CategoryHandler


class CategoryRegistry:
    """保存已註冊的處理器，並依商品資訊選出適用實作。"""

    def __init__(self) -> None:
        self._handlers: dict[str, CategoryHandler] = {}

    def register(self, handler: CategoryHandler) -> None:
        """依處理器名稱新增或取代註冊項目。"""
        name = handler.name.strip()
        if not name:
            raise ValueError("商品種類名稱不可為空")
        self._handlers[name] = handler

    def get(self, name: str) -> CategoryHandler | None:
        """依名稱取得已註冊的處理器。"""
        return self._handlers.get(name)

    def resolve(
        self,
        product_name: str,
        brand_model: str,
    ) -> CategoryHandler | None:
        """依註冊順序選出第一個支援指定商品的處理器。"""
        return next(
            (
                handler
                for handler in self._handlers.values()
                if handler.supports(product_name, brand_model)
            ),
            None,
        )


category_registry = CategoryRegistry()
