from backend.repository.market_price_repository import MarketPriceRepository
from backend.services.image_service.image_analyzer import ImageAnalyzer


class _FakeOnlinePriceService:
    def __init__(self, value: int):
        self.value = value

    def estimate_taiwan_market_price(self, product_query: str, max_results: int = 8) -> int:
        return self.value


def test_identify_product_from_json_market_data(tmp_path):
    db_path = tmp_path / "market.json"
    db_path.write_text(
        """
[
  {
    "aliases": ["iphone 15"],
    "product_name": "Apple iPhone 15",
    "brand_model": "Apple iPhone 15",
    "market_price": 27900
  }
]
""".strip(),
        encoding="utf-8",
    )

    repo = MarketPriceRepository(str(db_path))
    analyzer = ImageAnalyzer(repo)

    product_name, brand_model, market_price = analyzer._identify_product("iPhone 15 限時優惠")

    assert product_name == "Apple iPhone 15"
    assert brand_model == "Apple iPhone 15"
    assert market_price == 27900


def test_identify_product_with_ocr_typo_fuzzy_match(tmp_path):
        db_path = tmp_path / "market.json"
        db_path.write_text(
                """
[
    {
        "aliases": ["iphone 15"],
        "product_name": "Apple iPhone 15",
        "brand_model": "Apple iPhone 15",
        "market_price": 27900
    }
]
""".strip(),
                encoding="utf-8",
        )

        repo = MarketPriceRepository(str(db_path))
        analyzer = ImageAnalyzer(repo)

        product_name, brand_model, market_price = analyzer._identify_product("iph0ne 15 限時下殺")

        assert product_name == "Apple iPhone 15"
        assert brand_model == "Apple iPhone 15"
        assert market_price == 27900


def test_extract_selling_price_pick_sale_hint_priority():
    analyzer = ImageAnalyzer()
    text = "原價 38900 元，今天限時特價 12999 元，分期另計"

    selling_price = analyzer._extract_selling_price(text)

    assert selling_price == 12999


def test_extract_selling_price_not_found_returns_zero():
    analyzer = ImageAnalyzer()
    text = "歡迎私訊了解更多詳情，數量有限"

    selling_price = analyzer._extract_selling_price(text)

    assert selling_price == 0


def test_extract_selling_price_with_fullwidth_digits():
    analyzer = ImageAnalyzer()
    text = "限時特價 １２９９９ 元，錯過不再"

    selling_price = analyzer._extract_selling_price(text)

    assert selling_price == 12999


def test_identify_panasonic_hair_dryer_model():
    analyzer = ImageAnalyzer()
    text = "Panasonic 國際牌 1200W 負離子速乾型冷熱吹風機 EH-NE11"

    product_name, brand_model, market_price = analyzer._identify_product(text)

    assert "Panasonic" in product_name
    assert brand_model == "Panasonic EH-NE11"
    assert market_price >= 1000


from unittest.mock import patch

def test_identify_product_generic_fallback_not_unknown():
    analyzer = ImageAnalyzer()
    text = "小米 Xiaomi 掃拖機器人 X10+ 旗艦版 限時優惠"

    with patch('backend.services.image_service.product_identifier_agent.ChatGroq') as MockChat:
        mock_instance = MockChat.return_value
        # Mock zero-shot return value containing the expected product info
        mock_instance.invoke.return_value.content = "product_name: 掃拖機器人 X10+\nbrand_model: Xiaomi X10+"
        mock_instance.bind_tools.return_value = mock_instance
        
        product_name, brand_model, market_price = analyzer._identify_product(text)

    assert product_name != "未知商品"
    assert brand_model != "未知型號"
    assert "Xiaomi" in brand_model or "X10" in brand_model.upper()
    assert market_price == 0


def test_prioritize_blocks_for_shopee_layout_prefers_price_and_title():
    analyzer = ImageAnalyzer()
    blocks = [
        {"text": "Panasonic 國際牌 1200W 負離子速乾型冷熱吹風機 EH-NE11", "x0": 600, "y0": 20, "x1": 1400, "y1": 120},
        {"text": "$675 $1,290 5.2折", "x0": 620, "y0": 160, "x1": 980, "y1": 260},
        {"text": "運送 查看配送資訊", "x0": 620, "y0": 360, "x1": 980, "y1": 430},
        {"text": "Panasonic", "x0": 40, "y0": 30, "x1": 280, "y1": 100},
    ]
    fallback = "Panasonic 國際牌 1200W 負離子速乾型冷熱吹風機 EH-NE11\n$675"

    merged = analyzer._prioritize_blocks_for_ecommerce(blocks, width=1467, height=680, fallback_text=fallback)

    assert "eh-ne11" in merged.lower()
    assert "$675" in merged
    assert "5.2折" in merged


def test_resolve_market_price_prefers_online_result():
    analyzer = ImageAnalyzer(online_price_service=_FakeOnlinePriceService(31234))

    price, source = analyzer._resolve_market_price("Apple iPhone 15", "Apple iPhone 15", 27900)

    assert price == 31234
    assert source == "online"


def test_resolve_market_price_fallback_to_local_when_online_empty():
    analyzer = ImageAnalyzer(online_price_service=_FakeOnlinePriceService(0))

    price, source = analyzer._resolve_market_price("Apple iPhone 15", "Apple iPhone 15", 27900)

    assert price == 27900
    assert source == "fallback_local"
