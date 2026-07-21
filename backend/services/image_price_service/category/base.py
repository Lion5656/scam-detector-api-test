"""定義商品類別處理器必須實作的介面。"""

from typing import Protocol


class CategoryHandler(Protocol):
    """供各商品類別實作的結構化介面。"""

    name: str

    def supports(self, product_name: str, brand_model: str) -> bool:
        """判斷此處理器是否支援指定的商品名稱與品牌型號。"""
        ...
