"""定義價格查詢、候選驗證與台灣市場價格統計。"""

import json
import logging
import time
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from backend.config import settings
from backend.services.dto.price_analysis import (MarketPriceCandidateEvidence,
                                                 MarketPriceEstimate,
                                                 SearchTool)
from backend.services.image_price_service.domain.models import \
    MarketplaceCondition
from backend.services.image_price_service.domain.policy import (
    DEFAULT_PRICE_RISK_POLICY, PriceRiskPolicy)
from backend.services.image_price_service.pricing.search_result_price_extractor import (
    SearchResultPriceExtractor, default_search_result_price_extractor,
    extract_prices_from_search_results)
from backend.services.image_price_service.pricing.search_tools import (
    FALLBACK_SEARCH_TOOLS, PRIMARY_SEARCH_TOOL, SEARCH_FUNCTIONS)

logger = logging.getLogger(__name__)


class OnlineMarketPriceService:
    """搜尋、驗證並彙整同商品且同品況的台灣市場價格。"""

    _SEARCH_TOOL_LABELS = {
        "serpapi": "serp_api",
        "tavily": "tavily",
        "ddgs": "ddgs",
    }

    def __init__(
        self,
        *,
        policy: PriceRiskPolicy = DEFAULT_PRICE_RISK_POLICY,
        search_functions: dict[str, Any] | None = None,
        price_extractor: SearchResultPriceExtractor | None = None,
    ) -> None:
        """建立線上市價服務。"""
        self.policy = policy
        self._search_functions = search_functions or SEARCH_FUNCTIONS
        self._price_extractor = (
            price_extractor or default_search_result_price_extractor
        )

    def estimate_prices(
        self,
        product_query: str,
        max_results: int = 10,
        *,
        condition: MarketplaceCondition = MarketplaceCondition.NEW,
        condition_text: str = "",
    ) -> tuple[MarketPriceEstimate, ...]:
        """一次搜尋後，依品況建立一個或兩個獨立市場價格區間。"""
        if condition is MarketplaceCondition.UNKNOWN:
            candidates, search_tools = self._collect_candidates(
                product_query,
                max_results,
                condition=condition,
            )
            return self._aggregate_target_candidates(
                candidates,
                condition,
                search_tools=search_tools,
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
            raise ValueError("UNKNOWN 必須使用 estimate_prices() 執行雙區間估價")

        candidates, search_tools = self._collect_candidates(
            product_query,
            max_results,
            condition=condition,
        )
        return self._aggregate_candidates(
            candidates,
            condition,
            search_tools=search_tools,
        )

    def _collect_candidates(
        self,
        product_query: str,
        max_results: int,
        *,
        condition: MarketplaceCondition,
    ) -> tuple[list[MarketPriceCandidateEvidence], list[SearchTool]]:
        """以單一簡化 query 搜尋，並累積所有通過驗證的候選。"""
        query = product_query.strip()
        if not query or max_results <= 0:
            return [], []

        condition_term = self._condition_search_term(condition)
        conditioned_query = self._append_condition_term(query, condition_term)
        candidates: list[MarketPriceCandidateEvidence] = []
        search_tools: list[SearchTool] = []

        tools = (PRIMARY_SEARCH_TOOL, *FALLBACK_SEARCH_TOOLS)
        for index, tool_name in enumerate(tools):
            if (
                tool_name == "tavily"
                and not settings.TAVILY_SEARCH_API_KEY.get_secret_value()
            ):
                logger.warning(
                    "價格搜尋已略過 工具=tavily 原因=缺少 API 金鑰"
                )
                continue
            if self._search_functions.get(tool_name) is None:
                logger.warning("找不到搜尋工具：%s", tool_name)
                continue
            if index > 0:
                self._wait_before_fallback(tool_name)

            tool_label = cast(
                SearchTool,
                self._SEARCH_TOOL_LABELS.get(tool_name, tool_name),
            )
            search_tools.append(tool_label)
            try:
                new_candidates = self._search_and_extract_prices(
                    query=conditioned_query,
                    product_query=query,
                    max_results=max_results,
                    tool_name=tool_name,
                    condition=condition,
                )
            except Exception as error:
                tool_label = self._SEARCH_TOOL_LABELS.get(tool_name, tool_name)
                logger.error(
                    "價格搜尋失敗 工具=%s 錯誤=%s",
                    tool_label,
                    str(error),
                )
                continue

            candidates.extend(new_candidates)
            estimates = self._aggregate_target_candidates(
                candidates,
                condition,
                search_tools=search_tools,
            )
            if all(
                estimate.status == "success"
                for estimate in estimates
            ):
                return candidates, search_tools

        return candidates, search_tools

    def _aggregate_target_candidates(
        self,
        candidates: list[MarketPriceCandidateEvidence],
        condition: MarketplaceCondition,
        *,
        search_tools: list[SearchTool] | None = None,
    ) -> tuple[MarketPriceEstimate, ...]:
        """UNKNOWN 固定拆成 NEW、USED 兩個區間，但不重複搜尋。"""
        target_conditions = (
            (MarketplaceCondition.NEW, MarketplaceCondition.USED)
            if condition is MarketplaceCondition.UNKNOWN
            else (condition,)
        )
        return tuple(
            self._aggregate_candidates(
                [
                    candidate
                    for candidate in candidates
                    if candidate.condition is target_condition
                ],
                target_condition,
                search_tools=search_tools,
            )
            for target_condition in target_conditions
        )

    def _search_and_extract_prices(
        self,
        *,
        query: str,
        product_query: str,
        max_results: int,
        tool_name: str,
        condition: MarketplaceCondition,
    ) -> list[MarketPriceCandidateEvidence]:
        """執行單一搜尋工具並以一次 LLM 呼叫整理候選價格。"""
        tool_label = self._SEARCH_TOOL_LABELS.get(tool_name, tool_name)
        logger.info("價格搜尋開始 工具=%s", tool_label)

        search_fn = self._search_functions.get(tool_name)
        if search_fn is None:
            logger.warning("找不到搜尋工具：%s", tool_name)
            return []

        try:
            search_results = search_fn(query, max_results)
        except Exception as error:
            logger.error(
                "搜尋工具執行失敗 工具=%s 錯誤=%s",
                tool_label,
                str(error),
            )
            return []

        if not isinstance(search_results, list):
            return []

        candidates = extract_prices_from_search_results(
            search_results,
            condition,
            product_query=product_query,
            policy=self.policy,
            extractor=self._price_extractor,
        )

        logger.info("價格搜尋完成 工具=%s 候選數=%d", tool_label, len(candidates))
        return candidates

    def _aggregate_candidates(
        self,
        candidates: list[MarketPriceCandidateEvidence],
        condition: MarketplaceCondition,
        *,
        search_tools: list[SearchTool] | None = None,
    ) -> MarketPriceEstimate:
        """依 policy 門檻選擇低樣本中位數模式或 IQR 模式。"""
        if not candidates:
            return self._empty_estimate(
                condition,
                search_tools=search_tools,
            )

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
                search_tools=search_tools,
            )

        if sample_count < self.policy.minimum_iqr_samples:
            return self._small_sample_estimate(
                candidates,
                condition,
                status="success",
                search_tools=search_tools,
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
                search_tools=search_tools,
            )
        if retained_sample_count < self.policy.minimum_iqr_samples:
            return self._small_sample_estimate(
                retained,
                condition,
                status="success",
                search_tools=search_tools,
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
            search_tools=list(search_tools or []),
            candidates=tuple(retained),
        )

    def _small_sample_estimate(
        self,
        candidates: list[MarketPriceCandidateEvidence],
        condition: MarketplaceCondition,
        *,
        status: Literal["success", "insufficient"],
        search_tools: list[SearchTool] | None = None,
    ) -> MarketPriceEstimate:
        """以中位數及 policy 的固定相對容許範圍建立估計。"""
        if not candidates:
            return self._empty_estimate(
                condition,
                status="insufficient",
                search_tools=search_tools,
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
            search_tools=list(search_tools or []),
            candidates=tuple(candidates),
        )

    def _empty_estimate(
        self,
        condition: MarketplaceCondition,
        *,
        status: Literal["not_found", "insufficient"] = "not_found",
        search_tools: list[SearchTool] | None = None,
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
            search_tools=list(search_tools or []),
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

    @staticmethod
    def _condition_search_term(
        condition: MarketplaceCondition,
    ) -> str:
        """只加入必要的全新／二手搜尋關鍵字。"""
        if condition is MarketplaceCondition.NEW:
            return "全新"
        if condition is MarketplaceCondition.USED:
            return "二手"
        return "全新 二手"

    @staticmethod
    def _append_condition_term(query: str, condition_term: str) -> str:
        missing_terms = [
            term
            for term in condition_term.split()
            if term.casefold() not in query.casefold()
        ]
        if not missing_terms:
            return query
        return f"{query} {' '.join(missing_terms)}"

    def _wait_before_fallback(self, tool_name: str) -> None:
        delay = max(0.0, settings.ONLINE_PRICE_FALLBACK_DELAY_SECONDS)
        if delay <= 0:
            return
        logger.info("價格搜尋等待備援工具 工具=%s 延遲=%.1f 秒", self._SEARCH_TOOL_LABELS.get(tool_name, tool_name), delay)
        time.sleep(delay)

online_marketprice_service = OnlineMarketPriceService()
