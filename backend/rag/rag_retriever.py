from pathlib import Path
from typing import Any

from backend.config import settings
from backend.rag.rag_context import RAGContext
from backend.utils.text_cleaner import normalize_text


def _get_vectorstore() -> Any:
    try:
        return RAGContext.get_vectorstore()
    except RuntimeError as e:
        print(f"取得 vectorstore 失敗: {e}")
        raise


def get_retriever() -> Any:
    vectorstore = _get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": settings.RAG_TOP_K})


def format_context(docs: list[Any]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        blocks.append(f"[相似案例 {idx}]\n{doc.page_content[:190]}") # 限制每句參考在190以內
    return "\n\n".join(blocks)


def normalize_query_text(message: str) -> str:
    cleaned = normalize_text(message)
    return f"待分析訊息: {cleaned}"


def retrieve_similar_cases(message: str) -> list[Any]:
    if not is_rag_ready():
        return []
    return get_retriever().invoke(normalize_query_text(message))


def is_rag_ready() -> bool:
    if not settings.RAG_ENABLED:
        return False
    return Path(settings.RAG_PERSIST_DIR).exists()
