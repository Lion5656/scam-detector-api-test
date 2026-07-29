"""逐一呼叫市場搜尋工具，診斷 API、日誌與回傳內容大小。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("market_search_diagnostic")

TOOL_LABELS = {
    "serpapi": "SerpApi",
    "tavily": "Tavily",
    "ddgs": "DuckDuckGo",
}


class SearchTool(Protocol):
    """診斷程式需要的最小搜尋工具介面。"""

    def invoke(self, input: dict[str, object]) -> object:
        ...


@dataclass(frozen=True)
class SearchDiagnostic:
    """單一搜尋工具的診斷摘要。"""

    tool_name: str
    success: bool
    elapsed_seconds: float
    result_count: int
    snippet_characters: int
    largest_snippet_characters: int
    serialized_characters: int
    error: str | None = None


def configure_console() -> None:
    """設定 UTF-8 console 與可看見 INFO 的 root logger。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    root_logger = logging.getLogger()
    logger.info(
        "日誌已初始化 root_level=%s info_enabled=%s handlers=%d",
        logging.getLevelName(root_logger.getEffectiveLevel()),
        root_logger.isEnabledFor(logging.INFO),
        len(root_logger.handlers),
    )


def load_project_tools() -> tuple[dict[str, SearchTool], dict[str, str]]:
    """延遲載入正式工具，確保 logging 已先完成初始化。"""
    from backend.config import settings
    from backend.services.image_price_service.pricing.search_tools import (
        search_ddgs,
        search_serpapi,
        search_tavily,
    )

    class _FnAdapter:
        """將純函式包裝成 SearchTool 介面。"""

        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def invoke(self, input: dict[str, object]) -> object:
            return self._fn(
                str(input.get("query", "")),
                int(input.get("max_results", 10)),
            )

    tools: dict[str, SearchTool] = {
        "serpapi": _FnAdapter(search_serpapi),
        "tavily": _FnAdapter(search_tavily),
        "ddgs": _FnAdapter(search_ddgs),
    }
    credential_status = {
        "serpapi": (
            "已設定"
            if settings.SERP_API_KEY.get_secret_value().strip()
            else "未設定"
        ),
        "tavily": (
            "已設定"
            if settings.TAVILY_SEARCH_API_KEY.get_secret_value().strip()
            else "未設定"
        ),
        "ddgs": "不需 API 金鑰",
    }
    return tools, credential_status


def _normalized_results(raw_results: object) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        raise TypeError(
            f"工具回傳型別應為 list，實際為 {type(raw_results).__name__}"
        )
    return [item for item in raw_results if isinstance(item, dict)]


def diagnose_tool(
    tool_name: str,
    tool: SearchTool,
    *,
    query: str,
    max_results: int,
    preview_characters: int,
    clock: Callable[[], float] = time.perf_counter,
) -> SearchDiagnostic:
    """呼叫一次工具，並記錄成功、耗時、結果數與內容大小。"""
    label = TOOL_LABELS.get(tool_name, tool_name)
    started_at = clock()
    logger.info(
        "搜尋工具呼叫開始 工具=%s query=%r max_results=%d",
        label,
        query,
        max_results,
    )
    try:
        raw_results = tool.invoke(
            {
                "query": query,
                "max_results": max_results,
            }
        )
        results = _normalized_results(raw_results)
    except Exception as error:
        elapsed = max(0.0, clock() - started_at)
        logger.error(
            "搜尋工具呼叫失敗 工具=%s 耗時=%.3f 秒 "
            "錯誤類型=%s 錯誤=%s",
            label,
            elapsed,
            type(error).__name__,
            str(error),
        )
        return SearchDiagnostic(
            tool_name=tool_name,
            success=False,
            elapsed_seconds=elapsed,
            result_count=0,
            snippet_characters=0,
            largest_snippet_characters=0,
            serialized_characters=0,
            error=f"{type(error).__name__}: {error}",
        )

    elapsed = max(0.0, clock() - started_at)
    snippets = [
        str(result.get("snippet", "")).strip()
        for result in results
    ]
    serialized_characters = len(
        json.dumps(results, ensure_ascii=False)
    )
    diagnostic = SearchDiagnostic(
        tool_name=tool_name,
        success=True,
        elapsed_seconds=elapsed,
        result_count=len(results),
        snippet_characters=sum(len(snippet) for snippet in snippets),
        largest_snippet_characters=max(
            (len(snippet) for snippet in snippets),
            default=0,
        ),
        serialized_characters=serialized_characters,
    )
    logger.info(
        "搜尋工具呼叫完成 工具=%s 耗時=%.3f 秒 結果數=%d "
        "摘要總字元=%d 最大摘要字元=%d 序列化總字元=%d",
        label,
        diagnostic.elapsed_seconds,
        diagnostic.result_count,
        diagnostic.snippet_characters,
        diagnostic.largest_snippet_characters,
        diagnostic.serialized_characters,
    )

    for index, result in enumerate(results, start=1):
        title = str(result.get("title", "")).strip() or "(無標題)"
        link = str(result.get("link", "")).strip() or "(無連結)"
        snippet = str(result.get("snippet", "")).strip().replace("\n", " ")
        preview = snippet
        if len(preview) > preview_characters:
            preview = f"{preview[:preview_characters]}..."
        logger.info(
            "搜尋結果 工具=%s 序號=%d 標題=%r 連結=%s "
            "摘要字元=%d 摘要預覽=%r",
            label,
            index,
            title,
            link,
            len(snippet),
            preview,
        )

    return diagnostic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "逐一測試 SerpApi、Tavily、DDGS 搜尋工具；"
            "本程式不會呼叫 Groq。"
        ),
    )
    parser.add_argument(
        "--tool",
        choices=("all", *TOOL_LABELS),
        default="all",
        help="要測試的工具，預設為 all。",
    )
    parser.add_argument(
        "--query",
        default="Apple iPhone 15 128GB 台灣 二手 價格",
        help="測試搜尋關鍵字。",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="每個工具最多回傳幾筆結果，預設為 3。",
    )
    parser.add_argument(
        "--preview-characters",
        type=int,
        default=160,
        help="每筆摘要最多顯示幾個字元，預設為 160。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    args = parse_args(argv)
    if args.max_results <= 0:
        logger.error("--max-results 必須大於 0")
        return 2
    if args.preview_characters <= 0:
        logger.error("--preview-characters 必須大於 0")
        return 2
    if not args.query.strip():
        logger.error("--query 不可為空白")
        return 2

    tools, credential_status = load_project_tools()
    selected_names = (
        tuple(TOOL_LABELS)
        if args.tool == "all"
        else (args.tool,)
    )
    logger.info("本程式只測試搜尋 API，不會呼叫 Groq")
    for name in selected_names:
        logger.info(
            "搜尋工具設定 工具=%s API金鑰=%s",
            TOOL_LABELS[name],
            credential_status[name],
        )

    diagnostics = [
        diagnose_tool(
            name,
            tools[name],
            query=args.query.strip(),
            max_results=args.max_results,
            preview_characters=args.preview_characters,
        )
        for name in selected_names
    ]
    succeeded = sum(item.success for item in diagnostics)
    failed = len(diagnostics) - succeeded
    logger.info(
        "診斷完成 成功工具數=%d 失敗工具數=%d",
        succeeded,
        failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
