"""以真實 Tavily API 驗證確定性價格擷取流程。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.image_price_service.domain.models import MarketplaceCondition
from backend.services.image_price_service.pricing.search_result_price_extractor import (
    extract_prices_from_search_results,
)
from backend.services.image_price_service.pricing.search_tools import (
    search_tavily,
)


def configure_console_encoding() -> None:
    """避免 Windows 預設 CP950 無法輸出搜尋結果中的特殊符號。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="實際執行 Tavily 搜尋，再以程式規則擷取價格。",
    )
    parser.add_argument(
        "--query",
        default="Apple iPhone 15 128GB 台灣 全新 價格",
        help="測試搜尋關鍵字",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Tavily 最多回傳幾筆結果，預設為 5",
    )
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    if args.max_results <= 0:
        print("失敗：--max-results 必須大於 0。", file=sys.stderr)
        return 2

    if not settings.TAVILY_SEARCH_API_KEY.get_secret_value().strip():
        print(
            "失敗：尚未設定 TAVILY_SEARCH_API_KEY。",
            file=sys.stderr,
        )
        return 2

    print("步驟 1/2：呼叫 Tavily 搜尋工具。")
    print("步驟 2/2：以程式規則擷取價格。")
    try:
        results = search_tavily(args.query, args.max_results)
        candidates = extract_prices_from_search_results(
            results,
            MarketplaceCondition.NEW,
            product_query=args.query,
        )
    except Exception as error:
        print(f"失敗：workflow 發生錯誤：{error}", file=sys.stderr)
        return 1

    print(
        "成功：Tavily 與價格擷取流程均已完成。"
        f"Tavily 結果 {len(results)} 筆，"
        f"價格證據 {len(candidates)} 筆。"
    )
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"{index}. price={candidate.price} "
            f"condition={candidate.condition.value}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
