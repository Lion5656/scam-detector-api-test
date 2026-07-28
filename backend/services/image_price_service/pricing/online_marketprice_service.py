"""定義價格查詢 prompt、fallback 規則與台灣市場價格統計。"""

import json
import logging
import re
import time
from statistics import median
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from backend.config import settings
from backend.services.dto.price_analysis import SearchTool
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY,
    PriceRiskPolicy,
)
from backend.services.image_price_service.product.product_research_agent import (
    GroqRateLimitError,
    ProductAgentResult,
    ProductResearchAgent,
    create_product_research_agent,
)

logger = logging.getLogger(__name__)

PRICE_EXTRACTION_SYSTEM_PROMPT = """
你是台灣商品價格證據審核助手。程式會先執行搜尋工具並提供標題、摘要與 URL；
請根據這些工具結果，擷取與目標商品、型號及必要規格相符的完整商品售價。

必須排除：
- 運費、訂金、折價金、優惠券、回饋金與折扣差額。
- 分期每期金額、月租費、搭配門號價格與租賃價格。
- 配件、零件、不同型號、不同容量或明顯不同商品的價格。
- 搜尋摘要沒有明確顯示的價格，以及非新台幣價格。

每個價格必須引用搜尋工具結果中原有的 URL 與原文證據，不得自行建立網址或價格。
只回傳以下 JSON，不要加入 Markdown 或額外說明：
{
  "prices": [
    {
      "price": 30500,
      "currency": "TWD",
      "url": "搜尋結果中的 URL",
      "evidence": "包含價格的原文片段",
      "condition": "new、used 或 unknown",
      "product_match": true
    }
  ]
}
""".strip()


class OnlineMarketPriceService:
    """搜尋並彙整台灣市場價格。"""

    _PRIMARY_SEARCH_TOOL = "search_market_prices_serpapi"
    _FALLBACK_SEARCH_TOOLS = (
        "search_market_prices_tavily",
        "search_market_prices_ddgs"
    )
    _SEARCH_TOOL_LABELS: dict[str, SearchTool] = {
        "search_market_prices_serpapi": "serp_api",
        "search_market_prices_tavily": "tavily",
        "search_market_prices_ddgs": "ddgs"
    }
    _PRICE_PATTERN = re.compile(
        r"(?:NT\$|NTD|TWD|台幣|\$)?\s*([1-9]\d{2,6})"
    )
    _TWD_CURRENCIES = {"TWD", "NTD", "NT$", "台幣", "新台幣"}
    _USED_TERM_PATTERN = re.compile(
        r"(?:二\s*手|2\s*手|"
        r"(?:[1-9](?:\.\d)?|10|[一二三四五六七八九十])\s*成新)",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        research_agent: ProductResearchAgent | None = None,
        policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
    ) -> None:
        """建立線上市價服務。"""
        self._research_agent = research_agent
        self.policy = policy

    def estimate_price(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[int, SearchTool]:
        """回傳商品市價與成功使用的搜尋工具。"""
        query = product_query.strip()
        if not query or max_results <= 0:
            return 0, "unused"
        condition_term = self._condition_search_term(
            condition,
            condition_text,
        )
        conditioned_query = self._append_condition_term(
            query,
            condition_term,
        )
        condition_prompt = self._condition_prompt(
            condition,
            condition_term,
        )

        try:
            candidates = self._search_and_extract_prices(
                query=conditioned_query,
                max_results=max_results,
                tool_name=self._PRIMARY_SEARCH_TOOL,
                condition=condition,
                condition_prompt=condition_prompt,
            )
        except GroqRateLimitError as error:
            self._log_rate_limit_stop(
                self._PRIMARY_SEARCH_TOOL,
                error,
            )
            return 0, "unused"
        price = self._aggregate_candidates(candidates)
        if price > 0:
            return price, self._SEARCH_TOOL_LABELS[self._PRIMARY_SEARCH_TOOL]

        for fallback_tool in self._FALLBACK_SEARCH_TOOLS:
            if (
                fallback_tool == "search_market_prices_tavily"
                and not settings.TAVILY_SEARCH_API_KEY.get_secret_value()
            ):
                logger.warning(
                    "price search skipped tool=tavily reason=missing_api_key"
                )
                continue
            self._wait_before_fallback(fallback_tool)
            try:
                fallback_candidates = self._search_and_extract_prices(
                    query=conditioned_query,
                    max_results=max_results,
                    tool_name=fallback_tool,
                    condition=condition,
                    condition_prompt=condition_prompt,
                )
            except GroqRateLimitError as error:
                self._log_rate_limit_stop(fallback_tool, error)
                return 0, "unused"
            candidates.extend(fallback_candidates)
            candidates = self._deduplicate_candidates(candidates)
            price = self._aggregate_candidates(candidates)
            if price > 0:
                return price, self._SEARCH_TOOL_LABELS[fallback_tool]

        return 0, "unused"

    def _search_and_extract_prices(
        self,
        *,
        query: str,
        max_results: int,
        tool_name: str,
        condition: MarketplaceCondition,
        condition_prompt: str,
    ) -> list[dict[str, Any]]:
        """執行單一搜尋工具並驗證價格結果。"""
        tool_label = self._SEARCH_TOOL_LABELS[tool_name]
        logger.info("price search started tool=%s", tool_label)
        try:
            agent = self._get_research_agent()
            agent_result = agent.online_price_search(
                system_prompt=(
                    f"{PRICE_EXTRACTION_SYSTEM_PROMPT}\n\n{condition_prompt}"
                ),
                user_prompt=json.dumps(
                    {
                        "product_query": query,
                        "max_results": max_results,
                        "instruction": (
                            "搜尋工具結果會由程式提供；只接受與 product_query "
                            "及目標品況相符，且摘要中有明確價格的結果。"
                        ),
                        "target_condition": condition.value,
                        "condition_instruction": condition_prompt,
                    },
                    ensure_ascii=False,
                ),
                allowed_tool_names=[tool_name],
            )
        except GroqRateLimitError:
            raise
        except Exception as e:
            logger.error(
                "price search failed tool=%s error=%s",
                tool_label,
                str(e),
            )
            return []
        candidates = self._validate_agent_prices(
            agent_result,
            condition=condition,
        )
        logger.info(
            "price search completed tool=%s candidates=%d",
            tool_label,
            len(candidates),
        )
        return candidates

    def _wait_before_fallback(self, tool_name: str) -> None:
        """切換搜尋工具前依設定短暫等待。"""
        delay = max(
            0.0,
            settings.ONLINE_PRICE_FALLBACK_DELAY_SECONDS,
        )
        if delay <= 0:
            return
        logger.info(
            "price search fallback waiting tool=%s delay=%.1fs",
            self._SEARCH_TOOL_LABELS[tool_name],
            delay,
        )
        time.sleep(delay)

    def _log_rate_limit_stop(
        self,
        tool_name: str,
        error: GroqRateLimitError,
    ) -> None:
        """記錄 Groq 限流與停止原因。"""
        logger.warning(
            "price search stopped tool=%s reason=groq_rate_limit "
            "retry_after=%.1fs",
            self._SEARCH_TOOL_LABELS[tool_name],
            error.retry_after_seconds,
        )

    def _validate_agent_prices(
        self,
        agent_result: ProductAgentResult,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
    ) -> list[dict[str, Any]]:
        """保留來源可驗證的新台幣價格。"""
        known_results = {
            self._normalize_url(str(result.get("link", "")).strip()): result
            for result in agent_result.tool_results
            if self._normalize_url(str(result.get("link", "")).strip())
        }
        raw_prices = agent_result.output.get("prices", [])
        if not isinstance(raw_prices, list):
            return []

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for item in raw_prices:
            if not isinstance(item, dict) or item.get("product_match") is not True:
                continue

            normalized_url = self._normalize_url(
                str(item.get("url", "")).strip()
            )
            source_result = known_results.get(normalized_url)
            if source_result is None:
                continue

            currency = str(item.get("currency", "")).strip().upper()
            if currency not in self._TWD_CURRENCIES:
                continue
            item_condition = str(item.get("condition", "")).strip().lower()
            if (
                condition is not MarketplaceCondition.UNKNOWN
                and item_condition != condition.value
            ):
                continue

            price = self._parse_price(item.get("price"))
            if price is None:
                continue

            searchable_text = (
                f"{source_result.get('title', '')} "
                f"{source_result.get('snippet', '')}"
            )
            if price not in self._extract_prices(searchable_text):
                continue

            deduplication_key = (normalized_url, price)
            if deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            candidates.append(
                {
                    "price": price,
                    "currency": "TWD",
                    "url": str(source_result["link"]),
                    "source": urlsplit(
                        str(source_result["link"])
                    ).netloc.casefold(),
                    "evidence": str(item.get("evidence", "")).strip(),
                    "condition": str(
                        item.get("condition", "unknown")
                    ).strip(),
                }
            )
        return candidates

    @classmethod
    def _condition_search_term(
        cls,
        condition: MarketplaceCondition,
        condition_text: str,
    ) -> str:
        """建立新品或二手的搜尋關鍵字。"""
        if condition is MarketplaceCondition.NEW:
            return "全新"
        if condition is MarketplaceCondition.USED:
            terms = [
                re.sub(r"\s+", "", match.group(0))
                for match in cls._USED_TERM_PATTERN.finditer(condition_text)
            ]
            return " ".join(dict.fromkeys(terms)) or "二手"
        return ""

    @staticmethod
    def _append_condition_term(query: str, condition_term: str) -> str:
        """將品況加入搜尋詞並避免重複。"""
        if not condition_term or condition_term.casefold() in query.casefold():
            return query
        return f"{query} {condition_term}"

    @staticmethod
    def _condition_prompt(
        condition: MarketplaceCondition,
        condition_term: str,
    ) -> str:
        """建立查價使用的品況提示。"""
        if condition is MarketplaceCondition.USED:
            return (
                f"本次目標品況為「{condition_term or '二手'}」。"
                "只擷取相同二手／成新程度的商品價格，不可混入全新品價格；"
                '回傳項目的 condition 必須填 "used"。'
            )
        if condition is MarketplaceCondition.NEW:
            return (
                "本次目標品況為「全新」。只擷取全新商品價格，不可混入二手、"
                '中古或展示品價格；回傳項目的 condition 必須填 "new"。'
            )
        return (
            "本次商品品況未知；只擷取能由原文確認的價格，"
            '回傳項目的 condition 填 "unknown"。'
        )

    def _get_research_agent(self) -> ProductResearchAgent:
        """取得或建立商品研究代理。"""
        if self._research_agent is None:
            self._research_agent = create_product_research_agent()
        return self._research_agent

    def _aggregate_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> int:
        """依網站分組候選價格並計算市價。"""
        site_prices: dict[str, list[int]] = {}
        for candidate in candidates:
            site = urlsplit(str(candidate["url"])).netloc.casefold()
            if not site:
                continue
            site_prices.setdefault(site, []).append(int(candidate["price"]))
        return self._aggregate_from_site_prices(site_prices)

    def _aggregate_from_site_prices(
        self,
        site_prices: dict[str, list[int]],
    ) -> int:
        """驗證資料量並以中位數計算市價。"""
        if not site_prices:
            return 0

        contributing_sites = 0
        flat_prices: list[int] = []
        for values in site_prices.values():
            cleaned = [
                value
                for value in values
                if 0 < value <= self.policy.maximum_supported_price
            ]
            if cleaned:
                contributing_sites += 1
                flat_prices.extend(cleaned)

        if contributing_sites < self.policy.minimum_market_sites:
            return 0

        normalized = self._normalize_prices(flat_prices)
        if len(normalized) < self.policy.minimum_market_samples:
            return 0
        return int(median(normalized))

    def _extract_prices(self, text: str) -> list[int]:
        """從文字擷取合理範圍內的價格。"""
        values: list[int] = []
        for match in self._PRICE_PATTERN.finditer(text.replace(",", "")):
            value = int(match.group(1))
            if 0 < value <= self.policy.maximum_supported_price:
                values.append(value)
        return values

    def _parse_price(self, value: Any) -> int | None:
        """將輸入值轉成合理的整數價格。"""
        try:
            parsed = int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return None
        if not 0 < parsed <= self.policy.maximum_supported_price:
            return None
        return parsed

    def _normalize_prices(self, values: list[int]) -> list[int]:
        """以四分位距排除離群價格。"""
        if not values:
            return []

        values.sort()
        q1 = values[int(len(values) * 0.25)]
        q3 = values[int(len(values) * 0.75)]
        iqr = q3 - q1
        if iqr <= 0:
            return values

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [value for value in values if lower <= value <= upper]

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """依網址與價格移除重複候選。"""
        unique_candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for candidate in candidates:
            key = (
                cls._normalize_url(str(candidate["url"])),
                int(candidate["price"]),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _normalize_url(url: str) -> str:
        """正規化網址供比對與去重。"""
        try:
            parsed = urlsplit(url)
        except ValueError:
            return ""
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/") or "/",
                parsed.query,
                "",
            )
        )


online_marketprice_service = OnlineMarketPriceService()
