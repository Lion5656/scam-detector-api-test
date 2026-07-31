"""獨立的搜尋工具函式：SerpApi、DuckDuckGo 與 Tavily。"""

import json
import logging
import os
from typing import Any

import serpapi
from tavily import TavilyClient

from backend.config import settings

logger = logging.getLogger(__name__)


def search_serpapi(
    query: str,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """透過 SerpApi 搜尋並回傳標準化結果。"""
    api_key = (
        settings.SERP_API_KEY.get_secret_value()
        or os.getenv("SERP_API_KEY", "")
    )
    if not api_key:
        raise RuntimeError("尚未設定 SERP_API_KEY")

    logger.info("SerpApi 搜尋 query=%r max_results=%d", query, max_results)
    client = serpapi.Client(api_key=api_key, timeout=60)
    organic_results: list[dict[str, Any]] = []
    for start in range(0, max_results, 10):
        search_params: dict[str, Any] = {
            "engine": "google_light",
            "q": query,
            "google_domain": "google.com.tw",
            "hl": "zh-tw",
            "gl": "tw",
        }
        if start:
            search_params["start"] = start
        response = client.search(search_params)
        if isinstance(response, str):
            response_data = json.loads(response)
        elif isinstance(response, dict):
            response_data = response
        elif hasattr(response, "as_dict"):
            response_data = response.as_dict()
        else:
            response_data = dict(response)
        if not isinstance(response_data, dict):
            raise TypeError("SerpApi 回傳格式不是 JSON 物件")
        if response_data.get("error"):
            raise RuntimeError(str(response_data["error"]))

        page_results = response_data.get("organic_results", [])
        if not isinstance(page_results, list):
            return []
        organic_results.extend(
            item for item in page_results if isinstance(item, dict)
        )
        if not page_results:
            break

    results = [
        {
            "title": str(item.get("title", "")).strip(),
            "snippet": str(item.get("snippet", "")).strip(),
        }
        for item in organic_results[:max_results]
        if isinstance(item, dict)
        if (
            str(item.get("title", "")).strip()
            or str(item.get("snippet", "")).strip()
        )
    ]
    return results


def search_ddgs(
    query: str,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """透過 DuckDuckGo 搜尋並回傳標準化結果。"""
    from ddgs import DDGS

    with DDGS() as ddgs:
        logger.info("DuckDuckGo 搜尋 query=%r max_results=%d", query, max_results)
        raw_results = list(
            ddgs.text(
                query,
                region="tw-tzh",
                safesearch="off",
                backend="duckduckgo, brave, google, bing, yahoo",
                max_results=max_results,
            )
        )

    results = [
        {
            "title": str(item.get("title", "")).strip(),
            "snippet": str(item.get("body", "")).strip(),
        }
        for item in raw_results
        if (
            str(item.get("title", "")).strip()
            or str(item.get("body", "")).strip()
        )
    ]
    return results


def search_tavily(
    query: str,
    max_results: int = 10
) -> list[dict[str, str]]:
    """透過 Tavily 搜尋並回傳標準化結果。"""
    api_key = settings.TAVILY_SEARCH_API_KEY.get_secret_value() or os.getenv(
        "TAVILY_SEARCH_API_KEY",
        "",
    )
    if not api_key:
        raise RuntimeError("尚未設定 TAVILY_SEARCH_API_KEY")

    logger.info("Tavily 搜尋 query=%r max_results=%d", query, max_results)
    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
        country=settings.SEARCH_COUNTRY,
        include_domains=settings.SEARCH_DOMAIN,
        exclude_domains=settings.EXCLUDE_DOMAIN,
    )

    results = [
        {
            "title": str(item.get("title", "")).strip(),
            "snippet": str(item.get("content", "")).strip(),
        }
        for item in response.get("results", [])
        if isinstance(item, dict)
        if (
            str(item.get("title", "")).strip()
            or str(item.get("content", "")).strip()
        )
    ]
    return results


# 搜尋工具名稱到函式的映射
SEARCH_FUNCTIONS: dict[str, Any] = {
    "serpapi": search_serpapi,
    "tavily": search_tavily,
    "ddgs": search_ddgs,
}

# 搜尋工具的順序（主要 → 備援）
PRIMARY_SEARCH_TOOL = "serpapi"
FALLBACK_SEARCH_TOOLS = ("tavily", "ddgs")
SEARCH_TOOL_ORDER = (PRIMARY_SEARCH_TOOL, *FALLBACK_SEARCH_TOOLS)
