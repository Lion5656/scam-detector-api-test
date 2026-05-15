import json
import re
import os
from typing import Any

from backend.config import settings
from backend.rag.dto.analysis import RagEvidence, RagResponse
from backend.rag.rag_retriever import format_context, get_retriever, is_rag_ready, normalize_query_text
from backend.utils.text_cleaner import normalize_escape_sequences



def _require_langchain() -> tuple[Any, Any, Any, Any]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_groq import ChatGroq

    return ChatPromptTemplate, RunnablePassthrough, JsonOutputParser, ChatGroq


def _get_prompt() -> Any:
    ChatPromptTemplate, _, _, _ = _require_langchain()

    template = """
    <think>
    你是一個防詐專家，擅長從文字上下文中找出詐騙風險，
    分析待分析訊息，考慮是否為詐騙，
    請先在內部完成風險分析，
    不要直接下結論輸出。
    </think>

    最終請 JSON 結果。

    注意：
    1. reason 輸出禁止直接引用 context 原句，需要說明如何分析
    2. 風險越高分數請給越高，分數要跟風險成正比
    以下是風險類型定義：
    {context}\n

    待分析訊息(禁止引用)：
    {question}\n

    必須輸出繁體中文，且不要輸出引號：
    \n
    輸出訊息： 請嚴格按照以下 JSON schema 輸出：
    {format_instructions}
    """

    parser = _get_parser()
    prompt = ChatPromptTemplate.from_template(
        template, 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt

def _get_llm() -> Any:
    _, _, _, ChatGroq = _require_langchain()
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model_name=settings.RAG_MODEL_NAME, 
        temperature = 0.1,
        api_key=api_key,
        
    )


def _get_parser() -> Any:
    _, _, JsonOutputParser, _  = _require_langchain()
    parser = JsonOutputParser(pydantic_object=RagResponse)
    return parser

def build_rag_chain() -> Any:
    _, RunnablePassthrough, _, _ = _require_langchain()

    prompt = _get_prompt()
    retriever = get_retriever()
    llm = _get_llm()
    parser = _get_parser()

    return (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough() | normalize_query_text,
        }
        | prompt
        | llm
        | parser
    )


def _to_float(v: Any) -> float:
    """安全轉換任意值為浮點數，預設為 0.0"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0




def parse_structured_response(text: str) -> tuple[str, float, str]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        return "回傳錯誤", -1, ""
    money = _to_float(payload.get("money_related"))
    urgency = _to_float(payload.get("urgency"))
    bating = _to_float(payload.get("bating"))
    personal_info = _to_float(payload.get("asks_for_personal_info"))
    reputation = _to_float(payload.get("reputation"))
    
    score = min((money * 0.25 + 
                urgency * 0.30 + 
                bating * 0.25 + 
                personal_info * 0.25 +
                reputation * 0.30) * 100, 100)
    
    score = max(score, 5)
    reason = normalize_escape_sequences(str(payload.get("reason") or ""))
    
    label = "低風險"
    if score >= 80:
        label = "高風險"
    elif score >= 40:
        label = "中等風險"
    
    return label, score, reason


def analyze_with_rag(message: str) -> RagEvidence:
    if not is_rag_ready():
        return RagEvidence(used=False)

    rag_chain = build_rag_chain()
    chain_output = rag_chain.invoke(message)
    if isinstance(chain_output, dict):
        raw_response = json.dumps(chain_output, ensure_ascii=False)
    else:
        raw_response = chain_output
    label, score, reason = parse_structured_response(raw_response)

    return RagEvidence(
        used=True,
        label=label,
        score=score,
        reason=reason,
        raw_response=raw_response,
    )
