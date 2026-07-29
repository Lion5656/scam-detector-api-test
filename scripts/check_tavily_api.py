"""實際呼叫專案的 Tavily 搜尋工具，確認 API 設定與連線是否正常。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.image_price_service.pricing.search_tools import (
    search_tavily,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="送出一次真實 Tavily 搜尋，並顯示標準化搜尋結果。",
    )
    parser.add_argument(
        "--query",
        default="Apple iPhone 15 台灣 全新 價格",
        help="測試搜尋關鍵字",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="最多回傳幾筆結果，預設為 3",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    """避免 Windows 預設 CP950 因搜尋摘要含特殊符號而輸出失敗。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    if args.max_results <= 0:
        print("失敗：--max-results 必須大於 0。", file=sys.stderr)
        return 2

    api_key = settings.TAVILY_SEARCH_API_KEY.get_secret_value().strip()
    if not api_key:
        print(
            "失敗：尚未設定 TAVILY_SEARCH_API_KEY。"
            "請在專案根目錄或 backend/.env 中設定。",
            file=sys.stderr,
        )
        return 2

    print(f"正在呼叫 Tavily：query={args.query!r}, max_results={args.max_results}")
    try:
        results = search_tavily(
            args.query,
            args.max_results,
        )
    except Exception as error:
        print(
            f"失敗：Tavily API 呼叫發生錯誤：{error}",
            file=sys.stderr,
        )
        return 1

    print(f"成功：Tavily API 已回應，共取得 {len(results)} 筆結果。")
    for index, result in enumerate(results, start=1):
        title = str(result.get("title", "")).strip() or "(無標題)"
        link = str(result.get("link", "")).strip() or "(無連結)"
        snippet = str(result.get("snippet", "")).strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = f"{snippet[:157]}..."
        print(f"{index}. {title}")
        print(f"   {link}")
        if snippet:
            print(f"   {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
