"""搜尋台灣電商價格，並以四分位距與中位數估算市場參考價。"""

import re
from statistics import median

from backend.config import settings


class OnlineMarketPriceService:
    """彙整多個台灣電商搜尋結果並計算市場參考價。

    服務會搜尋設定的電商網站及不限定網域的結果，從標題、摘要與網址擷取
    300 至 2,000,000 元的價格。來源數和價格點數達到設定門檻後，使用四分位距
    排除離群值，再回傳剩餘價格的整數中位數。
    """
    _PRICE_PATTERN = re.compile(r"(?:NT\$|TWD|台幣|\$)?\s*([1-9]\d{2,6})")
    _DEFAULT_SITES = ("momo.com.tw", "pchome.com.tw", "shopee.tw", "tw.buy.yahoo.com")

    def estimate_taiwan_market_price(self, product_query: str, max_results: int = 8) -> int:
        """搜尋商品價格並回傳新臺幣中位數；查詢空白或資料不足時回傳 0。

        搜尋套件載入或查詢過程失敗時，會改以 ImportError 向上層回報。
        """
        if not product_query.strip():
            return 0

        try:
            from ddgs import DDGS

            sites = self._get_sites()
            site_prices: dict[str, list[int]] = {}
            
            with DDGS() as ddgs:
                for site in sites:
                    query = f"{product_query} 台灣 價格 site:{site}"
                    values = self._search_prices(ddgs, query, max_results=max_results)
                    if values:
                        site_prices[site] = values

                # 補充不限定網域的搜尋結果，降低指定網站資料不足的影響。
                generic_query = f"{product_query} 台灣 價格"
                generic_values = self._search_prices(ddgs, generic_query, max_results=max_results)
                if generic_values:
                    site_prices["generic"] = generic_values
        except Exception as e:
            raise ImportError(f"DuckDuckgo Search Failed: {e}")

        return self._aggregate_from_site_prices(site_prices)

    def _get_sites(self) -> list[str]:
        """取得設定的搜尋網站；未設定時使用預設台灣電商清單。"""
        custom_sites = [s.strip() for s in settings.ONLINE_PRICE_SITES.split(",") if s.strip()]
        if custom_sites:
            return custom_sites
        return list(self._DEFAULT_SITES)

    def _search_prices(self, ddgs, query: str, max_results: int) -> list[int]:
        """從搜尋結果的標題、摘要與網址擷取所有有效價格。"""
        values: list[int] = []
        results = ddgs.text(query, region="wt-wt", safesearch="off", max_results=max_results)
        for item in results:
            title = str(item.get("title", ""))
            body = str(item.get("body", ""))
            href = str(item.get("href", ""))
            values.extend(self._extract_prices(f"{title} {body} {href}"))
        return values

    def _aggregate_from_site_prices(self, site_prices: dict[str, list[int]]) -> int:
        """驗證來源與價格點數門檻，排除離群值後計算中位數。"""
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
        """從文字中擷取介於 300 至 2,000,000 的整數價格。"""
        values: list[int] = []
        for m in self._PRICE_PATTERN.finditer(text.replace(",", "")):
            value = int(m.group(1))
            if 300 <= value <= 2_000_000:
                values.append(value)
        return values

    def _normalize_prices(self, values: list[int]) -> list[int]:
        """保留落在 Q1 − 1.5 × IQR 至 Q3 + 1.5 × IQR 內的價格。"""
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

online_marketprice_service = OnlineMarketPriceService()
