from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routers import text_inference, url_detection
from backend.services.text_service.transformer_classifier import transformer_classifier
from backend.services.url_service.url_analyzer import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    transformer_classifier.load_model()  # 載入 Transformer 模型
    detector.load_model()
    yield
    print("清理資源...")

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

@app.get("/")
async def root():
    return {"message": "AI API is running..."}
