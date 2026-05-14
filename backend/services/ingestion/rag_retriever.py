from pathlib import Path
from typing import Any

from backend.config import settings
from backend.utils.text_cleaner import normalize_text


def _require_langchain() -> tuple[Any, Any]:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings

    return Chroma, OllamaEmbeddings


def get_embeddings() -> Any:
    _, OllamaEmbeddings = _require_langchain()
    return OllamaEmbeddings(model=settings.OLLAMA_EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)


def get_vectorstore() -> Any:
    Chroma, _ = _require_langchain()
    return Chroma(
        persist_directory=settings.RAG_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever() -> Any:
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": settings.RAG_TOP_K})


def format_context(docs: list[Any]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        blocks.append(f"[相似案例 {idx}]\n{doc.page_content}")
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
