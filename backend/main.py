from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.exception_handler import register_exception_handlers
from backend.api.routers import (image_price_validation, phone_detection,
                                 text_inference, url_detection)
from backend.core.lifespan import lifespan
from backend.core.logging import configure_logging

configure_logging()

# 主入口
app = FastAPI(
    lifespan=lifespan,
    title="詐騙風險偵測API",
    description="用於檢測詐騙風險的API",
    version="1.0.0"    
)

# 註冊全域例外處理
register_exception_handlers(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載router
app.include_router(text_inference.router, prefix="/api")
app.include_router(url_detection.router, prefix="/api")
app.include_router(image_price_validation.router, prefix="/api")
app.include_router(phone_detection.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "AI API is running..."}
