import re
from statistics import median

from backend.config import settings


class OnlineMarketPriceService:
    _PRICE_PATTERN = re.compile(r"(?:NT\$|TWD|台幣|\$)?\s*([1-9]\d{2,6})")
    _DEFAULT_SITES = ("momo.com.tw", "pchome.com.tw", "shopee.tw", "tw.buy.yahoo.com")

    def estimate_taiwan_market_price(self, product_query: str, max_results: int = 8) -> int:
        if not product_query.strip():
            return 0

        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise RuntimeError("缺少 duckduckgo-search 套件，無法進行線上比價") from exc

        sites = self._get_sites()
        site_prices: dict[str, list[int]] = {}

        try:
            with DDGS() as ddgs:
                for site in sites:
                    query = f"{product_query} 台灣 價格 site:{site}"
                    values = self._search_prices(ddgs, query, max_results=max_results)
                    if values:
                        site_prices[site] = values

                # 額外補一輪非 site 限制查詢，避免特定站結果過少。
                generic_query = f"{product_query} 台灣 價格"
                generic_values = self._search_prices(ddgs, generic_query, max_results=max_results)
                if generic_values:
                    site_prices["generic"] = generic_values
        except Exception:
            return 0

        return self._aggregate_from_site_prices(site_prices)

    def _get_sites(self) -> list[str]:
        custom_sites = [s.strip() for s in settings.ONLINE_PRICE_SITES.split(",") if s.strip()]
        if custom_sites:
            return custom_sites
        return list(self._DEFAULT_SITES)

    def _search_prices(self, ddgs, query: str, max_results: int) -> list[int]:
        values: list[int] = []
        results = ddgs.text(query, region="wt-wt", safesearch="off", max_results=max_results)
        for item in results:
            title = str(item.get("title", ""))
            body = str(item.get("body", ""))
            href = str(item.get("href", ""))
            values.extend(self._extract_prices(f"{title} {body} {href}"))
        return values

    def _aggregate_from_site_prices(self, site_prices: dict[str, list[int]]) -> int:
        if not site_prices:
            return 0

        contributing_sites = 0
        flat_prices: list[int] = []
        for values in site_prices.values():
            cleaned = [v for v in values if 300 <= v <= 2_000_000]
            if cleaned:
                contributing_sites += 1
                flat_prices.extend(cleaned)

        if contributing_sites < settings.ONLINE_PRICE_MIN_SITES:
            return 0

        normalized = self._normalize_prices(flat_prices)
        if len(normalized) < settings.ONLINE_PRICE_MIN_PRICE_POINTS:
            return 0

        return int(median(normalized))

    def _extract_prices(self, text: str) -> list[int]:
        values: list[int] = []
        for m in self._PRICE_PATTERN.finditer(text.replace(",", "")):
            value = int(m.group(1))
            if 300 <= value <= 2_000_000:
                values.append(value)
        return values

    def _normalize_prices(self, values: list[int]) -> list[int]:
        if not values:
            return []

        values.sort()
        q1_idx = int(len(values) * 0.25)
        q3_idx = int(len(values) * 0.75)
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1

        if iqr <= 0:
            return values

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [v for v in values if lower <= v <= upper]
