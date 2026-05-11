import json
import shutil
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.services.ingestion.rag_retriever import get_embeddings


def _require_langchain() -> tuple[Any, Any, Any]:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return Chroma, Document, RecursiveCharacterTextSplitter


def load_dataset(dataset_path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    if len(data) < limit:
        raise ValueError(f"Dataset only contains {len(data)} records, less than {limit}.")
    return data[:limit]


def build_case_text(record: dict[str, Any], index: int) -> str:
    return (
        f"案例編號: {index}"
        f"指令: {str(record.get('instruction', '')).strip()}"
        f"輸入: {str(record.get('input', '')).strip()}"
        f"輸出: {str(record.get('output', '')).strip()}"
    )


def build_documents(records: list[dict[str, Any]]) -> list[Any]:
    _, Document, _ = _require_langchain()
    documents: list[Any] = []
    for idx, record in enumerate(records, start=1):
        documents.append(
            Document(
                page_content=build_case_text(record, idx),
                metadata={
                    "case_id": idx,
                    "instruction": str(record.get("instruction", "")),
                    "input": str(record.get("input", "")),
                    "output": str(record.get("output", "")),
                },
            )
        )
    return documents


def split_documents(documents: list[Any]) -> list[Any]:
    _, _, RecursiveCharacterTextSplitter = _require_langchain()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNCK_SIZE,
        chunk_overlap=settings.CHUNCK_OVERLAP
    )
    return splitter.split_documents(documents)


def rebuild_vectorstore(documents: list[Any]) -> Any:
    Chroma, _, _ = _require_langchain()
    persist_dir = Path(settings.RAG_PERSIST_DIR)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    return Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
        persist_directory=str(persist_dir),
    )


def rebuild_index() -> None:
    """重建 RAG 向量庫索引"""
    dataset_path = Path(settings.RAG_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = load_dataset(dataset_path, settings.RAG_RECORD_LIMIT)
    documents = build_documents(records)
    chunks = split_documents(documents)
    rebuild_vectorstore(chunks)
