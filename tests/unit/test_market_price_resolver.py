from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.pricing.market_price_resolver import (
    resolve_market_price,
)


class _FakeOnlinePriceService:
    def __init__(self, value: int, search_tool: str = "serp_api"):
        self.value = value
        self.search_tool = search_tool
        self.calls: list[dict] = []

    def estimate_price(
        self,
        product_query: str,
        max_results: int = 8,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[int, str]:
        self.calls.append(
            {
                "product_query": product_query,
                "max_results": max_results,
                "condition": condition,
                "condition_text": condition_text,
            }
        )
        return self.value, self.search_tool


def test_market_price_prefers_online_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    service = _FakeOnlinePriceService(31_234)

    result = resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
        "iPhone 15 256GB",
    )

    assert result == (31_234, "online", "serp_api")
    assert service.calls[0]["product_query"] == "iPhone 15 256GB"
    assert service.calls[0]["condition"] is MarketplaceCondition.NEW


def test_market_price_falls_back_to_local_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    service = _FakeOnlinePriceService(0)

    assert resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
    ) == (27_900, "fallback_local", "unused")


def test_market_price_uses_product_name_for_unknown_model(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    service = _FakeOnlinePriceService(31_234)

    resolve_market_price(
        service,
        "Apple iPhone 15",
        "未知型號",
        27_900,
    )

    assert service.calls[0]["product_query"] == "Apple iPhone 15"


def test_market_price_forwards_used_condition_and_title(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    service = _FakeOnlinePriceService(18_000)

    resolve_market_price(
        service,
        "PS5 2手 9成新",
        "Sony PlayStation 5",
        0,
        condition=MarketplaceCondition.USED,
        condition_text="PS5 2手 9成新",
    )

    assert service.calls[0]["condition"] is MarketplaceCondition.USED
    assert service.calls[0]["condition_text"] == "PS5 2手 9成新"


def test_disabled_online_price_does_not_call_service(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        False,
    )
    service = _FakeOnlinePriceService(31_234)

    assert resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
    ) == (27_900, "fallback_local", "unused")
    assert service.calls == []
