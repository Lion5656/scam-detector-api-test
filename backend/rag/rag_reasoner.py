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
        你是一個防詐風險分析專家，請根據訊息內容分析不同風險特徵的符合程度。

        【參考案例 - 僅供理解詐騙模式，禁止複製其中文字到 reason】
        {context}
        【參考案例結束】
        
        【待分析訊息 - 僅供理解內容，禁止複製其中文字到 reason】
        {question}
        【待分析訊息結束】

        請依序完成以下判斷：
        第一步：判斷這則訊息的主要目的
        - A：提醒他人避免詐騙、警示風險、安全建議
        - B：可疑訊息（詐騙、誘導、索資、施壓）
        - C：一般安全訊息（正常日常對話）
        - D：對話類互動異常（對話中出現一方威脅、一方示弱、且具有情緒操控）
        
        第二步：
        - 若第一步為 A，直接輸出下方 JSON，reason：用繁體中文說明，禁止複製原文，其他欄位全部填 0.0：
        {format_instructions}
        
        - 若第一步為 B 或 C，再分析以下維度（0~1）：
        - urgency：催促、限時、要求立即操作
        - money_related：誘導金錢交易、投資、匯款
        - baiting：不合理優惠、保證獲利、隱藏交易
        - asks_for_personal_info：要求個資、帳號、驗證碼
        - reputation：來源可疑、假冒權威、誇大承諾
        - reason：禁止複製原文，用繁體中文說明訊息中出現哪些疑點
        - suggestion：針對此風險給出一句具體的下一步建議，以「建議」開頭

        - 若第一步為 D，分析以下維度（0~1）：
        - urgency：是否出現威脅、限時、製造恐慌
        - money_related：是否涉及借錢、投資、轉帳請求
        - baiting：是否以感情、同情、利益逐步誘導
        - asks_for_personal_info：是否套問個資、帳號、行蹤
        - reputation：身份是否可疑、前後矛盾、來路不明
        - social_engineering：是否存在情緒操控、威脅（0~1）
        - reason：禁止複製原文，用繁體中文具體說明對話中出現哪些可疑行為或話術
        - suggestion：針對此風險給出一句具體的下一步建議，以「建議」開頭
        輸出格式：
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
    baiting = _to_float(payload.get("baiting"))
    personal_info = _to_float(payload.get("asks_for_personal_info"))
    is_anti_fraud = _to_float(payload.get("is_anti_fraud"))
    social_engineering = _to_float(payload.get("social_engineering"))
    reputation = _to_float(payload.get("reputation"))
    
    print(money, urgency, baiting, personal_info, is_anti_fraud, social_engineering, reputation, end='\n')

    score = min((money * 0.25 + 
                urgency * 0.30 + 
                baiting * 0.25 + 
                personal_info * 0.25 +
                social_engineering * 0.30 -
                is_anti_fraud * 1 +
                reputation * 0.30) * 100, 100)
    
    score = max(score, 5)
    suggestion = normalize_escape_sequences(str(payload.get("suggestion") or ""))
    reason = normalize_escape_sequences(str(payload.get("reason") or ""))
    
    label = "低風險"
    if score >= 80:
        label = "高風險"
    elif score >= 40:
        label = "中等風險"
    
    return label, score, f"{reason}{suggestion}"


async def analyze_with_rag(message: str) -> RagEvidence:
    if not is_rag_ready():
        return RagEvidence(used=False)

    rag_chain = build_rag_chain()
    # 避免被同步請求卡住
    chain_output = await rag_chain.ainvoke(message)
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
