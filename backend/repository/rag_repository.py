from pathlib import Path
from typing import Any

from backend.config import settings
from backend.rag.rag_context import RAGContext


def _require_langchain() -> tuple[Any]:
    from langchain_chroma import Chroma

    return (Chroma,)


def get_vectorstore() -> Any:
    """取得向量資料庫實例"""
    (Chroma,) = _require_langchain()
    return Chroma(
        persist_directory=settings.RAG_PERSIST_DIR,
        embedding_function=RAGContext.get_embeddings(),
    )


def query_documents(query_text: str, k: int | None = None) -> list[Any]:
    """查詢向量庫中相似的文檔
    
    Args:
        query_text: 查詢文本
        k: 返回結果數量，默认使用配置中的 RAG_TOP_K
    
    Returns:
        相似文檔列表
    """
    if k is None:
        k = settings.RAG_TOP_K
    
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query_text)


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[str]:
    """新增文檔到向量庫
    
    Args:
        texts: 文本列表
        metadatas: 元數據列表
    
    Returns:
        新增文檔的 ID 列表
    """
    vectorstore = get_vectorstore()
    return vectorstore.add_texts(texts=texts, metadatas=metadatas)


def delete_documents(ids: list[str]) -> bool:
    """從向量庫刪除文檔
    
    Args:
        ids: 文檔 ID 列表
    
    Returns:
        是否刪除成功
    """
    vectorstore = get_vectorstore()
    try:
        vectorstore.delete(ids=ids)
        return True
    except Exception as e:
        print(f"Error deleting documents: {e}")
        return False
