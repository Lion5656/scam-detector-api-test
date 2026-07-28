"""定義價格查詢 prompt、候選驗證與台灣市場價格統計。"""

import json
import logging
import re
import time
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from backend.config import settings
from backend.services.dto.price_analysis import (
    MarketPriceCandidateEvidence,
    MarketPriceEstimate,
)
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
      "condition": "new 或 used",
      "product_match": true
    }
  ]
}
""".strip()


class OnlineMarketPriceService:
    """搜尋、驗證並彙整同商品且同品況的台灣市場價格。"""

    _PRIMARY_SEARCH_TOOL = "search_market_prices_serpapi"
    _FALLBACK_SEARCH_TOOLS = (
        "search_market_prices_tavily",
        "search_market_prices_ddgs",
    )
    _SEARCH_TOOL_LABELS = {
        "search_market_prices_serpapi": "serp_api",
        "search_market_prices_tavily": "tavily",
        "search_market_prices_ddgs": "ddgs",
    }
    _TWD_CURRENCIES = {"TWD", "NTD", "NT$", "台幣", "新台幣"}
    _USED_TERM_PATTERN = re.compile(
        r"(?:二\s*手|2\s*手|中古|"
        r"(?:[1-9](?:\.\d)?|10|[一二三四五六七八九十])\s*成新)",
        flags=re.IGNORECASE,
    )
    _USED_GRADE_PATTERN = re.compile(
        r"(?:近\s*全新|良品|良好|尚可|使用痕跡|"
        r"(?:[1-9](?:\.\d)?|10|[一二三四五六七八九十])\s*成新)",
        flags=re.IGNORECASE,
    )
    _NEW_EVIDENCE_PATTERN = re.compile(
        r"(?:全\s*新|未\s*拆(?:封)?|新品)",
        flags=re.IGNORECASE,
    )
    _USED_EVIDENCE_PATTERN = re.compile(
        r"(?:二\s*手|2\s*手|中古|近\s*全新|良品|良好|尚可|"
        r"使用痕跡|(?:[1-9](?:\.\d)?|10|[一二三四五六七八九十])\s*成新)",
        flags=re.IGNORECASE,
    )
    _NON_PRODUCT_PRICE_PATTERN = re.compile(
        r"(?:分期|每期|月付|月租|租賃|訂金|押金|運費|"
        r"折價|優惠券|回饋金|折扣差額|搭配門號)",
        flags=re.IGNORECASE,
    )
    _ACCESSORY_TERMS = (
        "保護殼",
        "手機殼",
        "保護貼",
        "玻璃貼",
        "充電線",
        "充電器",
        "轉接器",
        "支架",
        "收納包",
        "替換零件",
        "維修零件",
        "配件",
        "零件",
        "case",
        "cover",
        "cable",
        "charger",
        "adapter",
    )
    _VERSION_TERMS = {
        "pro",
        "max",
        "plus",
        "mini",
        "ultra",
        "air",
        "oled",
        "slim",
    }
    _QUERY_NOISE_TERMS = {
        "taiwan",
        "twd",
        "price",
        "new",
        "used",
        "台灣",
        "價格",
        "售價",
        "市價",
        "商品",
        "全新",
        "二手",
        "中古",
    }

    def __init__(
        self,
        *,
        research_agent: ProductResearchAgent | None = None,
        policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
    ) -> None:
        """建立線上市價服務。"""
        self._research_agent = research_agent
        self.policy = policy

    def estimate_prices(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[MarketPriceEstimate, ...]:
        """依品況查價；UNKNOWN 固定分成 NEW 與 USED 兩條獨立路徑。"""
        if condition is MarketplaceCondition.UNKNOWN:
            return (
                self.estimate_price(
                    product_query,
                    max_results,
                    condition=MarketplaceCondition.NEW,
                ),
                self.estimate_price(
                    product_query,
                    max_results,
                    condition=MarketplaceCondition.USED,
                ),
            )
        return (
            self.estimate_price(
                product_query,
                max_results,
                condition=condition,
                condition_text=condition_text,
            ),
        )

    def estimate_price(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> MarketPriceEstimate:
        """回傳單一已知品況的結構化市場價格估計。"""
        if condition is MarketplaceCondition.UNKNOWN:
            raise ValueError("UNKNOWN 必須使用 estimate_prices() 執行雙路徑查價")

        query = product_query.strip()
        if not query or max_results <= 0:
            return self._empty_estimate(condition)

        condition_term = self._condition_search_term(condition, condition_text)
        conditioned_query = self._append_condition_term(query, condition_term)
        condition_prompt = self._condition_prompt(condition, condition_term)
        candidates: list[MarketPriceCandidateEvidence] = []

        tools = (self._PRIMARY_SEARCH_TOOL, *self._FALLBACK_SEARCH_TOOLS)
        for index, tool_name in enumerate(tools):
            if (
                tool_name == "search_market_prices_tavily"
                and not settings.TAVILY_SEARCH_API_KEY.get_secret_value()
            ):
                logger.warning(
                    "price search skipped tool=tavily reason=missing_api_key"
                )
                continue
            if index > 0:
                self._wait_before_fallback(tool_name)

            try:
                new_candidates = self._search_and_extract_prices(
                    query=conditioned_query,
                    product_query=query,
                    max_results=max_results,
                    tool_name=tool_name,
                    condition=condition,
                    condition_prompt=condition_prompt,
                    condition_text=condition_text,
                )
            except GroqRateLimitError as error:
                self._log_rate_limit_stop(tool_name, error)
                break

            candidates = self._deduplicate_candidates(
                [*candidates, *new_candidates]
            )
            estimate = self._aggregate_candidates(candidates, condition)
            if estimate.status == "success":
                return estimate

        return self._aggregate_candidates(candidates, condition)

    def _search_and_extract_prices(
        self,
        *,
        query: str,
        product_query: str,
        max_results: int,
        tool_name: str,
        condition: MarketplaceCondition,
        condition_prompt: str,
        condition_text: str,
    ) -> list[MarketPriceCandidateEvidence]:
        """執行單一搜尋工具並由服務端重新驗證模型價格結果。"""
        tool_label = self._SEARCH_TOOL_LABELS[tool_name]
        logger.info("price search started tool=%s", tool_label)
        try:
            agent_result = self._get_research_agent().online_price_search(
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
        except Exception as error:
            logger.error(
                "price search failed tool=%s error=%s",
                tool_label,
                str(error),
            )
            return []

        candidates = self._validate_agent_prices(
            agent_result,
            product_query=product_query,
            condition=condition,
            condition_text=condition_text,
        )
        logger.info(
            "price search completed tool=%s candidates=%d",
            tool_label,
            len(candidates),
        )
        return candidates

    def _validate_agent_prices(
        self,
        agent_result: ProductAgentResult,
        *,
        product_query: str = "",
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> list[MarketPriceCandidateEvidence]:
        """只保留可由搜尋原文驗證的同商品、同規格、同品況價格。"""
        known_results = {
            self._normalize_url(str(result.get("link", "")).strip()): result
            for result in agent_result.tool_results
            if self._normalize_url(str(result.get("link", "")).strip())
        }
        raw_prices = agent_result.output.get("prices", [])
        if not isinstance(raw_prices, list):
            return []

        candidates: list[MarketPriceCandidateEvidence] = []
        for item in raw_prices:
            if not isinstance(item, dict) or item.get("product_match") is not True:
                continue

            normalized_url = self._normalize_url(str(item.get("url", "")).strip())
            source_result = known_results.get(normalized_url)
            if source_result is None:
                continue

            currency = str(item.get("currency", "")).strip().upper()
            if currency not in self._TWD_CURRENCIES:
                continue

            item_condition = str(item.get("condition", "")).strip().lower()
            if item_condition != condition.value:
                continue

            price = self._parse_price(item.get("price"))
            if price is None or 1900 <= price <= 2100:
                continue

            title = str(source_result.get("title", "")).strip()
            snippet = str(source_result.get("snippet", "")).strip()
            source_text = " ".join(part for part in (title, snippet) if part)
            if not self._price_is_supported_by_text(price, source_text):
                continue
            if self._is_non_product_price(product_query, source_text):
                continue
            if product_query and not self._matches_product_and_specs(
                product_query,
                source_text,
            ):
                continue
            if not self._matches_condition(
                condition,
                condition_text,
                source_text,
            ):
                continue

            candidates.append(
                MarketPriceCandidateEvidence(
                    candidate_id=(
                        f"{normalized_url}#{price}#{condition.value}"
                    ),
                    title=title,
                    price=price,
                    condition=condition,
                    url=str(source_result["link"]),
                    evidence=snippet or title,
                )
            )
        return self._deduplicate_candidates(candidates)

    def _aggregate_candidates(
        self,
        candidates: list[MarketPriceCandidateEvidence],
        condition: MarketplaceCondition,
    ) -> MarketPriceEstimate:
        """依 policy 門檻選擇低樣本中位數模式或 IQR 模式。"""
        if not candidates:
            return self._empty_estimate(condition)

        sample_count = len(candidates)
        site_count = self._site_count(candidates)
        confidence = self._market_confidence(sample_count, site_count)
        if (
            sample_count < self.policy.minimum_market_samples
            or site_count < self.policy.minimum_market_sites
            or confidence < self.policy.minimum_market_confidence
        ):
            return self._small_sample_estimate(
                candidates,
                condition,
                status="insufficient",
            )

        if sample_count < self.policy.minimum_iqr_samples:
            return self._small_sample_estimate(
                candidates,
                condition,
                status="success",
            )

        prices = sorted(candidate.price for candidate in candidates)
        initial_p25 = self._percentile(prices, 0.25)
        initial_p75 = self._percentile(prices, 0.75)
        iqr = initial_p75 - initial_p25
        lower_fence = initial_p25 - 1.5 * iqr
        upper_fence = initial_p75 + 1.5 * iqr
        retained = [
            candidate
            for candidate in candidates
            if lower_fence <= candidate.price <= upper_fence
        ]
        retained_sample_count = len(retained)
        retained_site_count = self._site_count(retained)
        retained_confidence = self._market_confidence(
            retained_sample_count,
            retained_site_count,
        )
        if (
            retained_sample_count < self.policy.minimum_market_samples
            or retained_site_count < self.policy.minimum_market_sites
            or retained_confidence < self.policy.minimum_market_confidence
        ):
            return self._small_sample_estimate(
                retained,
                condition,
                status="insufficient",
            )
        if retained_sample_count < self.policy.minimum_iqr_samples:
            return self._small_sample_estimate(
                retained,
                condition,
                status="success",
            )

        retained_prices = sorted(candidate.price for candidate in retained)
        return MarketPriceEstimate(
            status="success",
            condition=condition,
            reference_mode="iqr",
            median_price=round(self._percentile(retained_prices, 0.50)),
            low_price=round(self._percentile(retained_prices, 0.25)),
            high_price=round(self._percentile(retained_prices, 0.75)),
            sample_count=retained_sample_count,
            site_count=retained_site_count,
            source="online",
            confidence=retained_confidence,
            candidates=tuple(retained),
        )

    def _small_sample_estimate(
        self,
        candidates: list[MarketPriceCandidateEvidence],
        condition: MarketplaceCondition,
        *,
        status: Literal["success", "insufficient"],
    ) -> MarketPriceEstimate:
        """以中位數及 policy 的固定相對容許範圍建立估計。"""
        if not candidates:
            return self._empty_estimate(
                condition,
                status="insufficient",
            )

        prices = sorted(candidate.price for candidate in candidates)
        median_price = round(self._percentile(prices, 0.50))
        tolerance = self.policy.small_sample_relative_tolerance
        low_price = round(median_price * (1 - tolerance))
        high_price = round(median_price * (1 + tolerance))
        resolved_status = status
        if (
            low_price <= 0
            or high_price > self.policy.maximum_supported_price
        ):
            resolved_status = "insufficient"
        return MarketPriceEstimate(
            status=resolved_status,
            condition=condition,
            reference_mode="median_low_sample",
            median_price=median_price,
            low_price=low_price,
            high_price=high_price,
            sample_count=len(candidates),
            site_count=self._site_count(candidates),
            source="online",
            confidence=self._market_confidence(
                len(candidates),
                self._site_count(candidates),
            ),
            candidates=tuple(candidates),
        )

    def _empty_estimate(
        self,
        condition: MarketplaceCondition,
        *,
        status: Literal["not_found", "insufficient"] = "not_found",
    ) -> MarketPriceEstimate:
        """建立沒有可用市場候選的估計結果。"""
        return MarketPriceEstimate(
            status=status,
            condition=condition,
            reference_mode="median_low_sample",
            median_price=0,
            low_price=0,
            high_price=0,
            sample_count=0,
            site_count=0,
            source="online",
            confidence=0.0,
            candidates=(),
        )

    def _market_confidence(self, sample_count: int, site_count: int) -> float:
        """依 IQR 樣本完整度與獨立來源完整度計算資料可信度。"""
        sample_confidence = min(
            1.0,
            sample_count / self.policy.minimum_iqr_samples,
        )
        site_confidence = min(
            1.0,
            site_count / self.policy.minimum_market_sites,
        )
        return round(min(sample_confidence, site_confidence), 4)

    @staticmethod
    def _site_count(
        candidates: list[MarketPriceCandidateEvidence],
    ) -> int:
        return len(
            {
                urlsplit(candidate.url).netloc.casefold()
                for candidate in candidates
                if urlsplit(candidate.url).netloc
            }
        )

    @staticmethod
    def _percentile(values: list[int], quantile: float) -> float:
        """使用固定線性插值計算百分位數。"""
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = position - lower_index
        return (
            ordered[lower_index]
            + (ordered[upper_index] - ordered[lower_index]) * fraction
        )

    @classmethod
    def _matches_condition(
        cls,
        condition: MarketplaceCondition,
        condition_text: str,
        source_text: str,
    ) -> bool:
        if condition is MarketplaceCondition.NEW:
            return bool(cls._NEW_EVIDENCE_PATTERN.search(source_text))
        if condition is not MarketplaceCondition.USED:
            return False
        if not cls._USED_EVIDENCE_PATTERN.search(source_text):
            return False

        required_grades = {
            re.sub(r"\s+", "", match.group(0)).casefold()
            for match in cls._USED_GRADE_PATTERN.finditer(condition_text)
        }
        normalized_source = re.sub(r"\s+", "", source_text).casefold()
        return not required_grades or required_grades.issubset(
            {
                grade
                for grade in required_grades
                if grade in normalized_source
            }
        )

    @classmethod
    def _is_non_product_price(
        cls,
        product_query: str,
        source_text: str,
    ) -> bool:
        if cls._NON_PRODUCT_PRICE_PATTERN.search(source_text):
            return True

        normalized_query = product_query.casefold()
        normalized_source = source_text.casefold()
        return any(
            term in normalized_source and term not in normalized_query
            for term in cls._ACCESSORY_TERMS
        )

    @classmethod
    def _matches_product_and_specs(
        cls,
        product_query: str,
        source_text: str,
    ) -> bool:
        query = cls._normalize_matching_text(product_query)
        source = cls._normalize_matching_text(source_text)

        query_capacities = set(
            re.findall(r"\b\d+\s*(?:gb|tb)\b", query)
        )
        source_capacities = set(
            re.findall(r"\b\d+\s*(?:gb|tb)\b", source)
        )
        if query_capacities and not query_capacities.issubset(source_capacities):
            return False
        if query_capacities and source_capacities - query_capacities:
            return False

        query_versions = {
            term for term in cls._VERSION_TERMS if re.search(rf"\b{term}\b", query)
        }
        source_versions = {
            term for term in cls._VERSION_TERMS if re.search(rf"\b{term}\b", source)
        }
        if query_versions != source_versions:
            return False

        query_model_tokens = cls._model_tokens(query)
        source_model_tokens = cls._model_tokens(source)
        if not query_model_tokens.issubset(source_model_tokens):
            return False

        query_words = {
            token
            for token in re.findall(r"\b[a-z][a-z0-9]{1,}\b", query)
            if token not in cls._QUERY_NOISE_TERMS
            and token not in cls._VERSION_TERMS
        }
        source_words = set(
            re.findall(r"\b[a-z][a-z0-9]{1,}\b", source)
        )
        if not query_words.issubset(source_words):
            return False

        query_cjk_terms = {
            token
            for token in re.findall(r"[\u4e00-\u9fff]{2,}", query)
            if token not in cls._QUERY_NOISE_TERMS
        }
        if query_cjk_terms and not any(
            term in source for term in query_cjk_terms
        ):
            return False
        return True

    @staticmethod
    def _normalize_matching_text(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.casefold()).strip()

    @staticmethod
    def _model_tokens(text: str) -> set[str]:
        capacities = {
            re.sub(r"\s+", "", value)
            for value in re.findall(r"\b\d+\s*(?:gb|tb)\b", text)
        }
        mixed_tokens = set(
            re.findall(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+\b", text)
        )
        numbers = {
            value
            for value in re.findall(r"\b\d{1,4}\b", text)
            if not 1900 <= int(value) <= 2100
        }
        return capacities | mixed_tokens | numbers

    @staticmethod
    def _price_is_supported_by_text(price: int, text: str) -> bool:
        compact = text.replace(",", "")
        escaped_price = re.escape(str(price))
        return bool(
            re.search(
                rf"(?:NT\$|NTD|TWD|新台幣|台幣)\s*{escaped_price}(?!\d)",
                compact,
                flags=re.IGNORECASE,
            )
            or re.search(
                rf"(?:售價|價格|特價)\s*[:：]?\s*{escaped_price}\s*元",
                compact,
                flags=re.IGNORECASE,
            )
        )

    def _parse_price(self, value: Any) -> int | None:
        """將輸入值轉成 policy 支援範圍內的整數價格。"""
        try:
            parsed = int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return None
        if not 0 < parsed <= self.policy.maximum_supported_price:
            return None
        return parsed

    @classmethod
    def _condition_search_term(
        cls,
        condition: MarketplaceCondition,
        condition_text: str,
    ) -> str:
        """建立新品或保留二手成色細節的搜尋關鍵字。"""
        if condition is MarketplaceCondition.NEW:
            return "全新"
        if condition is MarketplaceCondition.USED:
            terms = [
                re.sub(r"\s+", "", match.group(0))
                for pattern in (cls._USED_TERM_PATTERN, cls._USED_GRADE_PATTERN)
                for match in pattern.finditer(condition_text)
            ]
            unique_terms = list(dict.fromkeys(terms))
            if not any(
                term in {"二手", "2手", "中古"}
                for term in unique_terms
            ):
                unique_terms.insert(0, "二手")
            return " ".join(unique_terms)
        return ""

    @staticmethod
    def _append_condition_term(query: str, condition_term: str) -> str:
        if not condition_term or condition_term.casefold() in query.casefold():
            return query
        return f"{query} {condition_term}"

    @staticmethod
    def _condition_prompt(
        condition: MarketplaceCondition,
        condition_term: str,
    ) -> str:
        if condition is MarketplaceCondition.USED:
            return (
                f"本次目標品況為「{condition_term or '二手'}」。"
                "只擷取相同二手／成新程度的商品價格，不可混入全新品價格；"
                '回傳項目的 condition 必須填 "used"。'
            )
        return (
            "本次目標品況為「全新」。只擷取全新商品價格，不可混入二手、"
            '中古或展示品價格；回傳項目的 condition 必須填 "new"。'
        )

    def _wait_before_fallback(self, tool_name: str) -> None:
        delay = max(0.0, settings.ONLINE_PRICE_FALLBACK_DELAY_SECONDS)
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
        logger.warning(
            "price search stopped tool=%s reason=groq_rate_limit "
            "retry_after=%.1fs",
            self._SEARCH_TOOL_LABELS[tool_name],
            error.retry_after_seconds,
        )

    def _get_research_agent(self) -> ProductResearchAgent:
        if self._research_agent is None:
            self._research_agent = create_product_research_agent()
        return self._research_agent

    @classmethod
    def _deduplicate_candidates(
        cls,
        candidates: list[MarketPriceCandidateEvidence],
    ) -> list[MarketPriceCandidateEvidence]:
        unique_candidates: list[MarketPriceCandidateEvidence] = []
        seen: set[tuple[str, int, MarketplaceCondition]] = set()
        for candidate in candidates:
            key = (
                cls._normalize_url(candidate.url),
                candidate.price,
                candidate.condition,
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _normalize_url(url: str) -> str:
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
