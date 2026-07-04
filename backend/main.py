from contextlib import asynccontextmanager

from fastapi import FastAPI
<<<<<<< HEAD

from backend.routers import image_inference, text_inference, url_detection
from backend.services.text_service.transformer_classifier import transformer_classifier
from backend.services.url_service.url_analyzer import detector
from backend.rag.rag_retriever import is_rag_ready
from backend.rag.rag_context import RAGContext
from backend.config import settings

=======
from fastapi.middleware.cors import CORSMiddleware
>>>>>>> caceac604d1b92bf5150ef7fa765e0dd225bf8c5
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings
from backend.rag.rag_context import RAGContext
from backend.rag.rag_retriever import is_rag_ready
from backend.routers import phone_detection, text_inference, url_detection
from backend.services.text_service.transformer_classifier import \
    transformer_classifier
from backend.services.url_service.url_analyzer import detector


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載router
app.include_router(text_inference.router)
app.include_router(url_detection.router)
<<<<<<< HEAD
app.include_router(image_inference.router)
=======
app.include_router(phone_detection.router)
>>>>>>> caceac604d1b92bf5150ef7fa765e0dd225bf8c5

@app.get("/")
async def root():
    return {"message": "AI API is running..."}
