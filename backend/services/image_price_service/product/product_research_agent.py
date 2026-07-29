"""提供商品正規化的 Groq 代理"""

import json
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from backend.config import settings

class ProductResearchAgent:
    """協調 LLM 結構化輸出，用於商品正規化。"""

    def __init__(
        self,
        llm: BaseChatModel,
    ) -> None:
        """建立代理。"""
        self._llm = llm

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
    """建立用於商品正規化的 Groq 代理。"""
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
    return ProductResearchAgent(llm=llm)
