"""使用語言模型與網路搜尋工具補足商品識別資訊。"""

import os
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_groq import ChatGroq
from pydantic import SecretStr

from backend.config import settings


PRODUCT_RESEARCH_SYSTEM_PROMPT = """
    你是一個商品識別助手，負責從 OCR 文字判斷商品名稱、品牌與型號。

    只有在下列情況才能使用搜尋工具：
    1. OCR 文字沒有足夠資訊可以識別具體商品。
    2. 需要外部資料確認型號與商品之間的關係。

    如果輸入已經包含明確的商品名稱、品牌與型號，請直接回答，不要使用搜尋工具。
    無法確認的資訊不可自行推測或捏造。

    最終答案必須嚴格使用以下格式，不要加入額外說明：
    product_name: <商品名稱；無法識別時填寫「未知商品」>
    brand_model: <品牌與型號；無法識別時填寫「未知品牌型號」>
""".strip()


class ProductResearchAgent:
    """讓語言模型辨識商品，並依模型要求執行一輪搜尋工具呼叫。"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_prompt: str = PRODUCT_RESEARCH_SYSTEM_PROMPT,
    ) -> None:
        """綁定語言模型、可用搜尋工具與商品識別系統提示。"""
        tool_list = list(tools)
        if not tool_list:
            raise ValueError("ProductResearchAgent 至少需要一個搜尋工具")

        self._llm = llm
        self._tools_by_name = {
            search_tool.name: search_tool for search_tool in tool_list
        }
        self._llm_with_tools = llm.bind_tools(tool_list, tool_choice="auto")
        self._system_prompt = system_prompt

    def run(self, prompt: str) -> str:
        """
            執行商品識別；有工具請求時加入搜尋結果後再取得最終回答\n
            SystemMessage：商品識別規則\n
            HumanMessage：原始 OCR 內容\n
            AIMessage：第一次 Groq 的搜尋工具請求\n
            ToolMessage：Google/DuckDuckGo 搜尋結果
        """
        messages: list[BaseMessage] = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]

        response = self._llm_with_tools.invoke(messages) # 第一次呼叫groq，判斷是否使用搜索工具

        if not response.tool_calls:
            return str(response.text)

        messages.append(response)
        for tool_call in response.tool_calls:
            messages.append(self._execute_tool(tool_call))

        final_response = self._llm.invoke(messages) # 第二次呼叫groq，使用搜索工具
        return str(final_response.text)

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """執行單一工具請求，並回傳帶有請求識別碼與狀態的訊息。"""
        tool_name = tool_call["name"]
        selected_tool = self._tools_by_name.get(tool_name)

        if selected_tool is None:
            return ToolMessage(
                content=f"找不到指定的工具：{tool_name}",
                tool_call_id=tool_call["id"],
                status="error",
            )

        try:
            result = selected_tool.invoke(tool_call["args"])
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                status="success",
            )
        except Exception as error:
            return ToolMessage(
                content=f"工具 {tool_name} 執行失敗：{error}",
                tool_call_id=tool_call["id"],
                status="error",
            )


@tool
def alt_search_product_info(query: str) -> str:
    """透過 Google 自訂搜尋取得最多三筆商品標題與摘要。"""
    from googleapiclient.discovery import build

    api_key = getattr(settings, "GOOGLE_API_KEY", "")
    search_engine_id = getattr(settings, "GOOGLE_CSE_ID", "")
    if not api_key or not search_engine_id:
        raise RuntimeError("尚未設定 Google Custom Search")

    service = build("customsearch", "v1", developerKey=api_key)
    response = service.cse().list(
        q=query,
        cx=search_engine_id,
        num=3,
    ).execute()

    items = response.get("items", [])
    if not items:
        return "找不到符合條件的商品資訊。"

    summary: list[str] = []
    for item in items:
        summary.append(f"標題：{item.get('title', '')}")
        summary.append(f"摘要：{item.get('snippet', '')}")
    return "\n".join(summary)


@tool
def search_product_info(query: str) -> str:
    """透過 DuckDuckGo 取得最多三筆商品標題與摘要。"""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return "找不到符合條件的商品資訊。"

    summary: list[str] = []
    for result in results:
        summary.append(f"標題：{result.get('title', '')}")
        summary.append(f"摘要：{result.get('body', '')}")
        summary.append("---")
    return "\n".join(summary)


def create_product_identifier_agent() -> ProductResearchAgent:
    """依專案設定建立 Groq 語言模型及兩種搜尋工具的商品研究代理。"""
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
        tools=[search_product_info, alt_search_product_info],
    )
