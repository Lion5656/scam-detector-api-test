import pytest

from backend.services.image_price_service.product.patterm_identifier import (
    PatternIdentifier,
)


def test_multiline_title_is_preserved_before_price_and_ui_text() -> None:
    result = PatternIdentifier().identify_product(
        "Apple iPhone 15 256GB\nNT$25,000\n加入購物車"
    )

    assert result.product_name == "Apple iPhone 15 256GB"


@pytest.mark.parametrize(
    ("text", "expected_brand"),
    [
        ("New Balance M2002RXD-D 防水休閒鞋", "New Balance"),
        ("NIKE Air Max DN8 男鞋", "Nike"),
        ("adidas Samba OG", "adidas"),
        ("FILA Disruptor 厚底鞋", "FILA"),
        ("Air Jordan 1 Retro High", "Air Jordan"),
        ("Nike Air Jordan 1 Low", "Air Jordan"),
        ("PUMA Speedcat OG", "PUMA"),
        ("UNIQLO 羽絨外套", "UNIQLO"),
        ("日立 RAS-28NK 冷氣", "Hitachi"),
        ("三洋 SW-15DV10 洗衣機", "SANYO"),
        ("IKEA KALLAX 層架", "IKEA"),
        ("宜得利 N-POLARU 電動沙發", "宜得利 NITORI"),
        ("Dyson V12 Detect Slim", "Dyson"),
    ],
)
def test_extracts_common_apparel_and_home_brands(
    text: str,
    expected_brand: str,
) -> None:
    assert PatternIdentifier()._extract_brand(text, None) == expected_brand


@pytest.mark.parametrize(
    ("text", "expected_model"),
    [
        ("New Balance 2002R 復古休閒鞋", "2002R"),
        ("Sony 1000XM5 無線降噪耳機", "1000XM5"),
        ("2017 年 Lexus CT 200H 油電版 CT200H", "CT200H"),
    ],
)
def test_extracts_models_that_may_start_with_digits(
    text: str,
    expected_model: str,
) -> None:
    assert PatternIdentifier()._extract_generic_model(text) == expected_model
