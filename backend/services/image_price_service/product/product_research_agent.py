"""提供商品市價搜尋工具，以及協調 Groq 工具呼叫與結構化輸出的代理。"""

import json
import logging
import os
import time
from typing import Any, Literal

import serpapi
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage,
                                     ToolCall, ToolMessage)
from langchain_core.tools import BaseTool, tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from tavily import TavilyClient

from backend.config import settings
from backend.services.image_price_service.models import ProductAgentResult

logger = logging.getLogger(__name__)


class GroqRateLimitError(RuntimeError):
    """Groq 請求速率限制。"""

    def __init__(self, retry_after_seconds: float) -> None:
        """保存建議等待秒數。"""
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            "Groq rate limit reached"
            f" (retry_after={self.retry_after_seconds:.1f}s)"
        )


class MarketPriceEvidence(BaseModel):
    """單筆可由搜尋結果驗證的商品價格。"""

    model_config = ConfigDict(extra="forbid")

    # 價格合理性由 OnlineMarketPriceService 逐筆判斷。這裡只驗證型別，
    # 避免單筆過低或過高的搜尋結果讓整批 Groq JSON 驗證失敗。
    price: int
    currency: Literal["TWD", "NTD", "NT$", "台幣", "新台幣"]
    url: str
    evidence: str
    condition: Literal["new", "used", "unknown"]
    product_match: bool


class MarketPriceSearchOutput(BaseModel):
    """Groq 整理搜尋結果後必須回傳的結構。"""

    model_config = ConfigDict(extra="forbid")

    prices: list[MarketPriceEvidence] = Field(default_factory=list)


@tool
def search_market_prices_serpapi(
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

    logging.info("SerpApi 搜尋 query=%r max_results=%d", query, max_results)
    client = serpapi.Client(api_key=api_key, timeout=60)
    response = client.search(
        {
            "engine": "google_light",
            "q": query,
            "google_domain": "google.com.tw",
            "hl": "zh-tw",
            "gl": "tw",
        }
    )
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

    organic_results = response_data.get("organic_results", [])
    if not isinstance(organic_results, list):
        return []

    results = [
        {
            "title": str(item.get("title", "")).strip(),
            "link": str(item.get("link", "")).strip(),
            "snippet": str(item.get("snippet", "")).strip(),
        }
        for item in organic_results[:max_results]
        if isinstance(item, dict)
        if str(item.get("link", "")).strip()
    ]
    return results


@tool
def search_market_prices_ddgs(
    query: str,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """透過 DuckDuckGo 搜尋並回傳標準化結果。"""
    from ddgs import DDGS

    with DDGS() as ddgs:
        logging.info("DuckDuckGo 搜尋 query=%r max_results=%d", query, max_results)
        raw_results = list(
            ddgs.text(
                query,
                region="tw-tzh",
                safesearch="off",
                max_results=max_results,
            )
        )

    results = [
        {
            "title": str(item.get("title", "")).strip(),
            "link": str(item.get("href", "")).strip(),
            "snippet": str(item.get("body", "")).strip(),
        }
        for item in raw_results
        if str(item.get("href", "")).strip()
    ]
    return results


@tool
def search_market_prices_tavily(
    query: str,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """透過 Tavily 搜尋並回傳標準化結果。"""
    api_key = settings.TAVILY_SEARCH_API_KEY.get_secret_value() or os.getenv(
        "TAVILY_SEARCH_API_KEY",
        "",
    )
    if not api_key:
        raise RuntimeError("尚未設定 TAVILY_SEARCH_API_KEY")
    
    logging.info("Tavily 搜尋 query=%r max_results=%d", query, max_results)
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
            "link": str(item.get("url", "")).strip(),
            "snippet": str(item.get("content", "")).strip(),
        }
        for item in response.get("results", [])
        if str(item.get("url", "")).strip()
    ]
    return results


class ProductResearchAgent:
    """協調搜尋工具與 Groq 結構化輸出。"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
    ) -> None:
        """建立代理並註冊可用搜尋工具。"""
        self._llm = llm
        self._tools_by_name = {
            search_tool.name: search_tool
            for search_tool in (tools or [])
        }

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """呼叫模型並解析 JSON 回覆。"""
        response = self._llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return self._parse_json_object(self._message_text(response))

    def online_price_search(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_tool_names: list[str],
    ) -> ProductAgentResult:
        """執行指定搜尋工具並由 Groq 整理價格。"""
        if len(allowed_tool_names) != 1:
            raise ValueError("每次價格搜尋必須指定一個搜尋工具")

        tool_name = allowed_tool_names[0]
        if tool_name not in self._tools_by_name:
            raise ValueError(f"找不到指定的搜尋工具：{tool_name}")

        request = self._parse_json_object(user_prompt)
        query = str(request.get("product_query", "")).strip()
        try:
            max_results = int(request.get("max_results", 10))
        except (TypeError, ValueError) as error:
            raise ValueError("max_results 必須是整數") from error
        if not query or max_results <= 0:
            raise ValueError("價格搜尋需要有效的 product_query 與 max_results")

        direct_tool_call: ToolCall = {
            "name": tool_name,
            "args": {
                "query": query,
                "max_results": max_results,
            },
            "id": f"direct-{tool_name}",
            "type": "tool_call",
        }
        _, tool_results, tool_error = self._execute_tool(
            direct_tool_call,
            allowed_tool_names,
        )

        if tool_error:
            logger.warning(
                "price search tool failed: %s",
                tool_error,
            )
            return ProductAgentResult(
                output={"prices": []},
                tool_results=[],
                tool_errors=[tool_error],
            )

        messages: list[BaseMessage] = [
            HumanMessage(
                content=json.dumps(
                    {
                        "instruction": system_prompt,
                        "request": request,
                        "search_results": tool_results,
                        "required_output_schema": (
                            MarketPriceSearchOutput.model_json_schema()
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        ]
        parsed = self._generate_price_output(messages, tool_name)

        return ProductAgentResult(
            output=parsed.model_dump(),
            tool_results=tool_results,
            tool_errors=[],
        )

    def _generate_price_output(
        self,
        messages: list[BaseMessage],
        tool_name: str,
    ) -> MarketPriceSearchOutput:
        """產生價格 JSON，必要時執行一次降級重試。"""
        structured_llm = self._llm.with_structured_output(
            MarketPriceSearchOutput,
            method="json_mode",
            include_raw=True,
            reasoning_format="hidden",
            reasoning_effort="none",
        )
        json_mode_error: BaseException | None = None
        try:
            structured_result = structured_llm.invoke(messages)
            if isinstance(structured_result, dict):
                parsed = structured_result.get("parsed")
                if isinstance(parsed, MarketPriceSearchOutput):
                    return parsed
                parsing_error = structured_result.get("parsing_error")
                if isinstance(parsing_error, BaseException):
                    json_mode_error = parsing_error
            if json_mode_error is None:
                json_mode_error = ValueError("missing structured output")
        except Exception as error:
            json_mode_error = error

        retry_delay = settings.GROQ_FALLBACK_RETRY_DELAY_SECONDS
        logger.warning(
            "Groq JSON mode failed tool=%s error=%s; "
            "retrying text mode in %.1fs",
            tool_name,
            str(json_mode_error),
            retry_delay,
        )
        if retry_delay > 0:
            time.sleep(retry_delay)

        fallback_llm = self._llm.bind(
            reasoning_format="hidden",
            reasoning_effort="none",
        )
        fallback_response = fallback_llm.invoke(messages)
        fallback_text = self._message_text(fallback_response)
        try:
            fallback_output = self._parse_json_object(fallback_text)
            return MarketPriceSearchOutput.model_validate(fallback_output)
        except Exception as error:
            logger.warning(
                "Groq price output failed tool=%s error=%s response=%r",
                tool_name,
                str(error),
                fallback_text[:500],
            )
            raise ValueError("Groq 未回傳有效的價格結構") from error

    def _execute_tool(
        self,
        tool_call: ToolCall,
        allowed_tool_names: list[str],
    ) -> tuple[ToolMessage, list[dict[str, Any]], str | None]:
        """執行白名單工具並整理結果或錯誤。"""
        tool_name = str(tool_call.get("name", ""))
        tool_call_id = str(tool_call.get("id", ""))
        if tool_name not in allowed_tool_names:
            error = f"Groq 嘗試使用未允許的工具：{tool_name}"
            return (
                ToolMessage(
                    content=error,
                    tool_call_id=tool_call_id,
                    status="error",
                ),
                [],
                error,
            )

        selected_tool = self._tools_by_name[tool_name]
        try:
            raw_result = selected_tool.invoke(tool_call.get("args", {}))
            parsed_results = (
                [item for item in raw_result if isinstance(item, dict)]
                if isinstance(raw_result, list)
                else []
            )
            return (
                ToolMessage(
                    content=json.dumps(
                        parsed_results,
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call_id,
                    status="success",
                ),
                parsed_results,
                None,
            )
        except Exception as error:
            message = f"工具 {tool_name} 執行失敗：{error}"
            return (
                ToolMessage(
                    content=message,
                    tool_call_id=tool_call_id,
                    status="error",
                ),
                [],
                message,
            )

    @staticmethod
    def _message_text(message: Any) -> str:
        """從模型訊息取出文字內容。"""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            if content.strip():
                return content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_text = block.get("text") or block.get("content")
                    if isinstance(block_text, str):
                        text_parts.append(block_text)
            if text_parts:
                return "\n".join(text_parts)

        additional_kwargs = getattr(message, "additional_kwargs", {})
        if isinstance(additional_kwargs, dict):
            reasoning_content = additional_kwargs.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                return reasoning_content

        return str(content)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        """從模型文字中解析第一個 JSON 物件。"""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Groq 未回傳有效 JSON 物件（回覆為空）")

        decoder = json.JSONDecoder()
        last_error: json.JSONDecodeError | None = None
        for start, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError as error:
                last_error = error
                continue
            if isinstance(parsed, dict):
                return parsed

        if last_error is not None:
            raise ValueError(
                f"Groq 回傳 JSON 解析失敗：{last_error.msg}"
            ) from last_error
        else:
            raise ValueError("Groq 未回傳有效 JSON 物件")


def create_product_research_agent() -> ProductResearchAgent:
    """建立已註冊搜尋工具的 Groq 代理。"""
    api_key_value = settings.GROQ_API_KEY.get_secret_value() or os.getenv(
        "GROQ_API_KEY",
        "",
    )
    if not api_key_value:
        raise RuntimeError("尚未設定 GROQ_API_KEY")

    llm = ChatGroq(
        model=settings.PRODUCT_MODEL_NAME,
        temperature=0.1,
        api_key=SecretStr(api_key_value),
    )
    return ProductResearchAgent(
        llm=llm,
        tools=[
            search_market_prices_serpapi,
            search_market_prices_tavily,
            search_market_prices_ddgs
        ],
    )
