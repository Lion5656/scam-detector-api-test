"""協調本地商品資料、規則辨識與 LLM 代理的商品辨識流程。"""

from backend.repository.market_price_repository import (
    MarketPriceRepository,
    market_price_repository
)
from backend.services.dto.price_analysis import ProductIdentification
from backend.services.image_price_service.product.patterm_identifier import (
    pattern_identifier
)
from backend.services.image_price_service.product.product_research_agent import (
    create_product_identifier_agent
)

class ProductIdentifier:
    """依成本由低至高的順序辨識商品名稱、品牌與型號。

    服務會先查詢本地市場價資料，再使用品牌與型號規則；只有規則結果不完整時
    才呼叫商品研究代理。代理執行失敗時回傳原規則結果，避免中斷圖片分析流程。
    """

    def __init__(self, market_repo: MarketPriceRepository | None = None):
        """注入市場價資料庫；未提供時使用模組共用實例。"""
        self._market_repo = market_repo or market_price_repository

    def identify(self, text: str) -> ProductIdentification:
        """依本地資料、規則與商品研究代理的順序辨識 OCR 文字。"""
        # 本地資料命中時直接回傳，並沿用資料庫中的市場參考價。
        product_name, brand_model, market_price = self._market_repo.find_by_text(text)
        if market_price > 0:
            return ProductIdentification(product_name=product_name, brand_model=brand_model, market_price=market_price)

        product = pattern_identifier.identify_product(text)
        if (
            product.product_name not in {"一般商品", "未知商品"}
            and "未知型號" not in product.brand_model
        ):
            return ProductIdentification(product_name=product.product_name, brand_model=product.brand_model, market_price=0)

        prompt = f"""
        請從以下文字中，找出商品名稱以及品牌與型號。
        提供的文字是從網路購物平台的圖片 OCR 辨識出來的。
        
        如果你覺得文字中的商品名稱縮寫或型號不完整，
        請使用你的 Search 工具去網路上搜尋，找出該商品最有可能的完整品牌與確切型號。
        
        請以這兩個欄位回傳：
        1. product_name: (完整的商品名稱)
        2. brand_model: (品牌名稱 加上 確切型號)
        
        OCR 文字如下：
        {text}
        
        如果真的找不到任何商品，請回傳：
        product_name: 未知商品
        brand_model: 未知型號
        """
        try:
            agent = create_product_identifier_agent()
            response = agent.run(prompt)
            
            product_name = "未知商品"
            brand_model= "未知型號"
            
            for line in response.split('\n'):
                line = line.strip()
                if line.lower().startswith('product_name:'):
                    product_name = line.split(':', 1)[1].strip()
                elif line.lower().startswith('brand_model:'):
                    brand_model = line.split(':', 1)[1].strip()
                    
            return ProductIdentification(product_name=product_name, brand_model=brand_model, market_price=0)
        except Exception as e:
            print(f"Product Identifier Agent Failed: {e}")
            return pattern_identifier.identify_product(text)
        
product_identifier = ProductIdentifier()

