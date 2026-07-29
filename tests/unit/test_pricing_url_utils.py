"""市場價格來源 URL 正規化工具測試。"""

import pytest

from backend.services.image_price_service.pricing.url_utils import (
    normalize_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "HTTPS://Shop.Example/Items/123/#details",
            "https://shop.example/Items/123",
        ),
        (
            "http://SHOP.EXAMPLE?sort=price#top",
            "http://shop.example/?sort=price",
        ),
        ("ftp://shop.example/item", ""),
        ("not-a-url", ""),
        ("https://[invalid", ""),
    ],
)
def test_normalize_source_url(url: str, expected: str) -> None:
    assert normalize_url(url) == expected
