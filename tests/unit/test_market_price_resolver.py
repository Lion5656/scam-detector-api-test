from backend.services.dto.price_analysis import MarketPriceEstimate
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
)
from backend.services.image_price_service.pricing.market_price_resolver import (
    resolve_market_price,
)


def _estimate(
    condition: MarketplaceCondition,
    median_price: int,
    *,
    status: str = "success",
) -> MarketPriceEstimate:
    return MarketPriceEstimate(
        status=status,
        condition=condition,
        reference_mode="median_low_sample",
        median_price=median_price,
        low_price=round(median_price * 0.75),
        high_price=round(median_price * 1.25),
        sample_count=3 if median_price else 0,
        site_count=3 if median_price else 0,
        source="online",
        confidence=0.6 if median_price else 0.0,
        candidates=(),
    )


class _FakeOnlinePriceService:
    policy = DEFAULT_PRICE_RISK_POLICY

    def __init__(self, estimates: tuple[MarketPriceEstimate, ...]):
        self.estimates = estimates
        self.calls: list[dict] = []

    def estimate_prices(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[MarketPriceEstimate, ...]:
        self.calls.append(
            {
                "product_query": product_query,
                "max_results": max_results,
                "condition": condition,
                "condition_text": condition_text,
            }
        )
        return self.estimates


def test_market_price_resolver_returns_structured_online_result(monkeypatch):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    expected = (_estimate(MarketplaceCondition.NEW, 31_234),)
    service = _FakeOnlinePriceService(expected)

    result = resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
        "iPhone 15 256GB",
    )

    assert result is expected
    assert service.calls == [
        {
            "product_query": "iPhone 15 256GB",
            "max_results": 10,
            "condition": MarketplaceCondition.NEW,
            "condition_text": "",
        }
    ]


def test_market_price_resolver_preserves_unknown_dual_results_without_average(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    expected = (
        _estimate(MarketplaceCondition.NEW, 30_000),
        _estimate(MarketplaceCondition.USED, 20_000),
    )
    service = _FakeOnlinePriceService(expected)

    result = resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
        "iPhone 15 256GB",
        condition=MarketplaceCondition.UNKNOWN,
    )

    assert result is expected
    assert [estimate.median_price for estimate in result] == [30_000, 20_000]
    assert service.calls[0]["condition"] is MarketplaceCondition.UNKNOWN


def test_market_price_resolver_preserves_insufficient_online_status(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    expected = (
        _estimate(
            MarketplaceCondition.USED,
            18_000,
            status="insufficient",
        ),
    )
    service = _FakeOnlinePriceService(expected)

    result = resolve_market_price(
        service,
        "PS5 2手 9成新",
        "Sony PlayStation 5",
        0,
        condition=MarketplaceCondition.USED,
        condition_text="9成新",
    )

    assert result is expected
    assert result[0].status == "insufficient"
    assert service.calls[0]["condition_text"] == "9成新"


def test_market_price_resolver_uses_product_name_for_unknown_model(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        True,
    )
    service = _FakeOnlinePriceService(
        (_estimate(MarketplaceCondition.NEW, 31_234),)
    )

    resolve_market_price(
        service,
        "Apple iPhone 15",
        "未知型號",
        27_900,
    )

    assert service.calls[0]["product_query"] == "Apple iPhone 15"


def test_disabled_online_price_returns_lower_confidence_local_fallback(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.services.image_price_service.pricing.market_price_resolver.settings.ONLINE_PRICE_ENABLED",
        False,
    )
    service = _FakeOnlinePriceService(
        (_estimate(MarketplaceCondition.NEW, 31_234),)
    )

    result = resolve_market_price(
        service,
        "Apple iPhone 15",
        "Apple iPhone 15",
        27_900,
    )

    assert service.calls == []
    assert len(result) == 1
    assert result[0].source == "fallback_local"
    assert result[0].median_price == 27_900
    assert result[0].confidence == 0.0
