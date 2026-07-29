"""以受限狀態證據執行結構化 LLM 商品狀態複核。"""

import json
from typing import Any

from pydantic import SecretStr

from backend.services.dto.price_analysis import DeepAnalysisReview

_REVIEW_CONTEXT_FIELDS = (
    "product_name",
    "text",
    "condition",
    "condition_detail",
    "condition_source_text",
    "condition_extraction_confidence",
)

_SYSTEM_PROMPT = """
你是商品狀態複核器。請同時查看輸入中的完整刊登文字 text、condition_detail 與
condition_source_text，確認或修正商品狀態，並將狀態正規化為 new、used 或 unknown。
condition_evidence 必須逐字取自這三個欄位其中之一；證據不足時回傳 unknown。
不得推測價格、風險、商品匹配或市場候選，也不得使用輸入以外的資訊。
""".strip()


class GroqConditionReviewer:
    """延遲建立 Groq client"""

    def __init__(self, *, api_key: str, model_name: str) -> None:
        self._api_key = api_key.strip()
        self._model_name = model_name.strip()

    def __call__(
        self,
        context: dict[str, object],
    ) -> DeepAnalysisReview | dict[str, object] | None:
        if not self._api_key or not self._model_name:
            return None

        from langchain_groq import ChatGroq

        limited_context = {
            field: context.get(field)
            for field in _REVIEW_CONTEXT_FIELDS
        }
        llm = ChatGroq(
            model=self._model_name,
            temperature=0,
            api_key=SecretStr(self._api_key),
        )
        structured_llm: Any = llm.with_structured_output(
            DeepAnalysisReview,
        )
        return structured_llm.invoke(
            (
                f"{_SYSTEM_PROMPT}\n\n"
                f"輸入：{json.dumps(limited_context, ensure_ascii=False)}"
            )
        )


__all__ = ["GroqConditionReviewer"]
