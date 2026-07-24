from backend.services.image_price_service.product.patterm_identifier import (
    PatternIdentifier,
)


def test_multiline_title_is_preserved_before_price_and_ui_text() -> None:
    result = PatternIdentifier().identify_product(
        "Apple iPhone 15 256GB\nNT$25,000\n加入購物車"
    )

    assert result.product_name == "Apple iPhone 15 256GB"
