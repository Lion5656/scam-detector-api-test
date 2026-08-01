from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.core.config import settings
from backend.rag.rag_context import RAGContext
from backend.rag.rag_retriever import is_rag_ready
from backend.services.text_service.base_classifier import base_classifier
from backend.services.url_service.url_analyzer import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("應用環境啟動")
    
    
    base_classifier.load_model()  # 載入 Base 模型
    detector.load_model()

    print("初始化 RAG")
    # 設置 RAG Context
    RAGContext.set_app(app)

    if not is_rag_ready():
        RAGContext.load_vectorstore()

    
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBED_MODEL
    )

    # warmup
    embeddings.embed_query("warmup")

    vectorstore = Chroma(
        persist_directory=settings.RAG_PERSIST_DIR,
        embedding_function=embeddings,
    )
    
    # 儲存到 app.state
    app.state.vectorstore = vectorstore
    app.state.embeddings = embeddings
    print("RAG 已初始化")
    
    yield
    print("清理資源...")
    print("應用關閉")
