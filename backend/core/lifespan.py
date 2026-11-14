import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.core.config import settings
from backend.rag.rag_context import RAGContext
from backend.rag.rag_reasoner import get_rag_chain
from backend.rag.rag_retriever import is_rag_ready
from backend.services.text_service.base_classifier import base_classifier
from backend.services.url_service.url_analyzer import url_detector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("應用環境啟動")

    base_classifier.load_model()
    url_detector.load_model()

    # 預熱：ONNX session 首次推論與 tldextract 後綴表載入（可能需連網）都很慢，
    # 在啟動時先跑一次，避免由第一個使用者請求承擔
    base_classifier.predict_text("預熱")
    url_detector.analyze("https://www.example.com/")

    print("初始化 RAG")
    RAGContext.set_app(app)
    if not is_rag_ready():
        RAGContext.load_vectorstore()

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    embeddings.embed_query("warmup")
    app.state.embeddings = embeddings
    app.state.vectorstore = Chroma(
        persist_directory=settings.RAG_PERSIST_DIR, embedding_function=embeddings
    )

    if is_rag_ready():
        try:
            get_rag_chain()
        except Exception:
            # 例如尚未設定 GROQ_API_KEY；保留延遲建立，第一次 RAG 請求時再嘗試
            logger.warning("RAG chain 預先建立失敗，改為首次請求時建立", exc_info=True)
    print("RAG 已初始化")

    yield
    print("應用關閉")
