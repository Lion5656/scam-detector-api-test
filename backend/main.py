import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, cast

from fastapi import FastAPI, HTTPException
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel
from transformers import AutoTokenizer, Pipeline, pipeline

HF_REPO_ID = "kko12/spam-detector-chinese"
HF_TOKEN = os.getenv("HF_TOKEN")

spam_cls: Pipeline | None = None

# 管理fastapi的生命周期，確保在啟動時載入模型，關閉時清理資源
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("載入量化Bert ONNX推理模型...")
    global spam_cls
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, use_auth_token=HF_TOKEN)
    model = ORTModelForSequenceClassification.from_pretrained(HF_REPO_ID, file_name="model_quantized.onnx", use_auth_token=HF_TOKEN, provider="CPUExecutionProvider")
    spam_cls = pipeline("text-classification", model=cast(Any, model), tokenizer=tokenizer, truncation=True, device="cpu")
    yield
    print("清理資源...")

# 啟動server
app = FastAPI(
    lifespan=lifespan,
    title="詐騙風險偵測模型API",
    description="測試用API",
    version="1.0.0"
)


#請求的datamodel
class Request(BaseModel):
    text: str

# 標準化文字
def normalize_text(text: str):
    text = text.strip()
    # 清掉多餘的空白，只留一個
    text = re.sub(r"\s+", " ", text)
    return text

tochinese: Dict[str, str] = {
    "LOW": "低風險",
    "MEDIUM": "中等風險",
    "HIGH": "高風險", 
    "UNKNOWN": "未知"
}

# 模型預測 + 決策分類
def predict(text: str):
    if spam_cls is None:
        raise RuntimeError("模型未載入")

    assert spam_cls is not None

    results =  cast(List[Dict[str, Any]], spam_cls(text, top_k=None))
    # 如果模型輸出是多層列表，取第一層 [[{"label": "HIGH", "score": 0.95}, ...]]
    if isinstance(results[0], list):
        results = results[0]
    dist: Dict[str, float] = {str(item['label']): float(item['score']) for item in results} # type: ignore

    # 排序標籤分數由大到小
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

    top_label, top_score = sorted_dist[0]
    second_score = sorted_dist[1][1]
    margin = top_score - second_score

    if dist.get("UNKNOWN", 0.0) >= 0.7:
        return "未知", dist.get("UNKNOWN")
    if top_score >= 0.7:
        return tochinese.get(top_label), top_score
    if dist.get("HIGH", 0.0) >= 0.5:
        return "高風險", dist.get("HIGH")
    if margin >= 0.1:
        return tochinese.get(top_label), top_score
    return "中等風險", dist.get("MEDIUM")

@app.post("/predict")
def predict_api(req: Request):
    text = normalize_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="text 不能為空白")
    label, score =  predict(text)
    return {
        "label": label,
        "confidence_score": f"{score:.2f}"
    }

@app.get("/")
def home():
    return {"message": "AI API is running..."}
