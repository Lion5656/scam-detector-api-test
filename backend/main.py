from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routers import text_inference, url_detection
from backend.routers import phone_detection
from backend.services.text_service.transformer_classifier import transformer_classifier
from backend.services.url_service.url_analyzer import detector
from backend.rag.rag_retriever import is_rag_ready
from backend.rag.rag_context import RAGContext
from backend.config import settings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("應用環境啟動")
    
    
    transformer_classifier.load_model()  # 載入 Transformer 模型
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

# 主入口
app = FastAPI(
    lifespan=lifespan,
    title="詐騙風險偵測模型API",
    description="測試用API",
    version="2.0.0"    
)

# 掛載router
app.include_router(text_inference.router)
app.include_router(url_detection.router)
app.include_router(phone_detection.router)

@app.get("/")
async def root():
    return {"message": "AI API is running..."}
