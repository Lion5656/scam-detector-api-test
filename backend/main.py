from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routers import inference
from backend.services.ai_engine import inference_engine


# 主入口
@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_engine.load_model() # 載入模型
    yield
    print("清理資源...")

app = FastAPI(
    lifespan=lifespan,
    title="詐騙風險偵測模型API",
    description="測試用API",
    version="1.5.0"    
)

# 掛載router
app.include_router(inference.router)

@app.get("/")
async def root():
    return {"message": "AI API is running..."}