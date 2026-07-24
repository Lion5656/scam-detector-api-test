"""以真實 Tavily 與 Groq API 驗證價格搜尋 workflow。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.services.image_price_service.pricing.online_marketprice_service import (
    PRICE_EXTRACTION_SYSTEM_PROMPT,
)
from backend.services.image_price_service.product.product_research_agent import (
    create_product_research_agent,
)


def configure_console_encoding() -> None:
    """避免 Windows 預設 CP950 無法輸出搜尋結果中的特殊符號。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="實際執行 Tavily 搜尋，再由 Groq 整理價格 JSON。",
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

    missing_keys: list[str] = []
    if not settings.TAVILY_SEARCH_API_KEY.get_secret_value().strip():
        missing_keys.append("TAVILY_SEARCH_API_KEY")
    if not settings.GROQ_API_KEY.get_secret_value().strip():
        missing_keys.append("GROQ_API_KEY")
    if missing_keys:
        print(
            f"失敗：尚未設定 {', '.join(missing_keys)}。",
            file=sys.stderr,
        )
        return 2

    print("步驟 1/2：呼叫 Tavily 搜尋工具。")
    print("步驟 2/2：將 Tavily 結果交給 Groq 產生價格 JSON。")
    try:
        result = create_product_research_agent().online_price_search(
            system_prompt=(
                f"{PRICE_EXTRACTION_SYSTEM_PROMPT}\n\n"
                "本次目標品況為「全新」。只擷取全新商品價格；"
                '回傳項目的 condition 必須填 "new"。'
            ),
            user_prompt=json.dumps(
                {
                    "product_query": args.query,
                    "max_results": args.max_results,
                    "instruction": (
                        "只接受與 product_query 及全新品況相符，"
                        "且摘要中有明確價格的結果。"
                    ),
                    "target_condition": "new",
                },
                ensure_ascii=False,
            ),
            allowed_tool_names=["search_market_prices_tavily"],
        )
    except Exception as error:
        print(f"失敗：workflow 發生錯誤：{error}", file=sys.stderr)
        return 1

    if result.tool_errors:
        print(
            f"失敗：Tavily 工具錯誤：{'; '.join(result.tool_errors)}",
            file=sys.stderr,
        )
        return 1

    prices = result.output.get("prices", [])
    print(
        "成功：Tavily 與 Groq workflow 均已完成。"
        f"Tavily 結果 {len(result.tool_results)} 筆，"
        f"Groq 價格證據 {len(prices) if isinstance(prices, list) else 0} 筆。"
    )
    if isinstance(prices, list):
        for index, price in enumerate(prices, start=1):
            print(
                f"{index}. price={price.get('price')} "
                f"condition={price.get('condition')} "
                f"url={price.get('url')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
