import json
import re
from typing import Any, cast

from backend.config import settings
from backend.services.dto.analysis import RagEvidence
from backend.services.ingestion.rag_retriever import format_context, get_retriever, is_rag_ready, normalize_query_text


def _require_langchain() -> tuple[Any, Any, Any]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import OllamaLLM

    return ChatPromptTemplate, RunnablePassthrough, OllamaLLM


def get_llm() -> Any:
    _, _, OllamaLLM = _require_langchain()
    return OllamaLLM(model=settings.OLLAMA_LLM_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0)


def build_rag_chain() -> Any:
    ChatPromptTemplate, RunnablePassthrough, _ = _require_langchain()
    template = """
相似案例：
{context}

待分析訊息：
{question}

注意交易命令相關操作，是否包含詐騙中的「資金流向合理化」話術。
若訊息同時出現第三方帳戶、隱蔽交易、借名操作與機構化包裝等特徵，請提高詐騙風險。
請先列出訊息中的 3 個可疑點，接著對照檢索到的案例，最後才給出詐騙風險評分，理由要符合評分等級
嚴格只輸出單一 JSON 物件，不要加入前言、markdown、程式碼區塊或其他欄位。

輸出格式：
{{
  "label": "(風險等級: 低風險、中等風險、高風險)",
  "score": (分數: 0-100 的數字，低風險40分以下，中風險80分以下，高風險80分以上) ,
  "reason": "原因: 2-3 句簡短理由"
}}
"""
    prompt = ChatPromptTemplate.from_template(template)
    retriever = get_retriever()
    llm = get_llm()

    return (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough() | normalize_query_text,
        }
        | prompt
        | llm
    )


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def extract_reason_from_response(text: str) -> str:
    match = re.search(r"原因[：:]\s*(.+)$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_structured_response(text: str) -> tuple[str, float | None, str]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            label = str(payload.get("label") or "未知風險")
            raw_score = payload.get("score")
            score = float(raw_score) if raw_score is not None else None
            reason = str(payload.get("reason") or "").strip()
            return label, score, reason or text.strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    label = infer_label_from_response(text)
    score = infer_score_from_response(text)
    reason = extract_reason_from_response(text)
    return label, score, reason


def analyze_with_rag(message: str) -> RagEvidence:
    if not is_rag_ready():
        return RagEvidence(used=False)

    rag_chain = build_rag_chain()
    raw_response = clean_response(cast(str, rag_chain.invoke(message)))
    label, score, reason = parse_structured_response(raw_response)

    return RagEvidence(
        used=True,
        label=label,
        score=score,
        reason=reason,
        raw_response=raw_response,
    )


def infer_label_from_response(text: str) -> str:
    if "高風險" in text:
        return "高風險"
    if "中等風險" in text:
        return "中等風險"
    if "低風險" in text:
        return "低風險"
    return "未知風險"


def infer_score_from_response(text: str) -> float | None:
    match = re.search(r"(?:風險分數|分數)[：: ]*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    match = re.search(r"\b(100|[1-9]?\d(?:\.\d+)?)\b", text)
    if match:
        value = float(match.group(1))
        if 0 <= value <= 100:
            return value
    return None
