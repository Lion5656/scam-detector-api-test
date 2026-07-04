from backend.services.image_service.online_market_price_service import OnlineMarketPriceService


def test_aggregate_from_site_prices_uses_multi_site_median():
    svc = OnlineMarketPriceService()
    site_prices = {
        "momo": [30000, 30500],
        "pchome": [29900, 30200],
        "shopee": [1500],
        "generic": [30100],
    }

    price = svc._aggregate_from_site_prices(site_prices)

    assert 29900 <= price <= 30500


def test_aggregate_from_site_prices_requires_min_sites(monkeypatch):
    svc = OnlineMarketPriceService()
    monkeypatch.setattr("backend.services.image_service.online_market_price_service.settings.ONLINE_PRICE_MIN_SITES", 2)

    site_prices = {
        "momo": [30000, 30500, 31000],
    }

    price = svc._aggregate_from_site_prices(site_prices)

    assert price == 0
