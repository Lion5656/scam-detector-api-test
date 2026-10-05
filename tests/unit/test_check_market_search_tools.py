from scripts.check_market_search_tools import (
    SearchDiagnostic,
    diagnose_tool,
    parse_args,
)


class _FakeTool:
    def __init__(self, result=None, *, error: Exception | None = None):
        self.result = result
        self.error = error
        self.received = None

    def invoke(self, input):
        self.received = input
        if self.error is not None:
            raise self.error
        return self.result


def _clock(*values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_diagnose_tool_reports_result_size_without_real_api():
    tool = _FakeTool(
        [
            {
                "title": "商品 A",
                "snippet": "售價 NT$10,000",
            },
            {
                "title": "商品 B",
                "snippet": "二手近全新",
            },
        ]
    )

    result = diagnose_tool(
        "tavily",
        tool,
        query="測試商品",
        max_results=2,
        preview_characters=20,
        clock=_clock(10.0, 10.25),
    )

    assert result == SearchDiagnostic(
        tool_name="tavily",
        success=True,
        elapsed_seconds=0.25,
        result_count=2,
        snippet_characters=len("售價 NT$10,000二手近全新"),
        largest_snippet_characters=len("售價 NT$10,000"),
        serialized_characters=result.serialized_characters,
        error=None,
    )
    assert result.serialized_characters > result.snippet_characters
    assert tool.received == {
        "query": "測試商品",
        "max_results": 2,
    }


def test_diagnose_tool_records_api_failure_and_continues():
    tool = _FakeTool(error=TimeoutError("連線逾時"))

    result = diagnose_tool(
        "serpapi",
        tool,
        query="測試商品",
        max_results=3,
        preview_characters=20,
        clock=_clock(5.0, 6.5),
    )

    assert result.success is False
    assert result.elapsed_seconds == 1.5
    assert result.error == "TimeoutError: 連線逾時"
    assert result.result_count == 0


def test_parse_args_supports_individual_tool():
    args = parse_args(
        [
            "--tool",
            "ddgs",
            "--query",
            "Nike 球衣 台灣 二手 價格",
            "--max-results",
            "4",
            "--preview-characters",
            "80",
        ]
    )

    assert args.tool == "ddgs"
    assert args.query == "Nike 球衣 台灣 二手 價格"
    assert args.max_results == 4
    assert args.preview_characters == 80
