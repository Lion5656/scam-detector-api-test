## ✨ 基於 FastAPI 的多模型詐騙偵測 API

整合文字、網址、電話號碼與 Marketplace 商品價格驗證的 API。

### 🧠 使用核心模型

- `scam-project/spam-detector-chinese`：ONNX 量化文字詐騙分類模型
- `scam-project/url-detector`：XGBoost 惡意網址分類模型
- `BAAI/bge-small-zh-v1.5`：RAG 相似案例文本向量化模型
- `llama-3.1-8b-instant`：Groq RAG 語意推理與風險分析模型
- `qwen/qwen3.6-27b`：Groq 商品資訊正規化與搜尋價格整理模型
- `Google Cloud Vision`：商品圖片 OCR、段落座標與版面文字擷取

### ⚡ 核心功能

- 文字詐騙偵測：結合本地模型、關鍵字規則與低信心 RAG 分析。
- URL 風險偵測：分析網址格式與特徵，判斷連結是否具有釣魚或惡意風險。
- 電話號碼查詢：查詢黑白名單與回報紀錄，並支援回報新的可疑電話號碼。
- 商品圖片分析：驗證商品頁面，抽取標題、主價格、品況與賣家。
- 商品品況：優先讀取詳細內容的狀況欄，再依商品標題與說明文字判斷。
- 線上查價：依序使用搜尋工具，再將搜尋結果整理為價格證據。
- 價格風險：驗證價格來源、排除離群值，判斷售價是否明顯偏離市場行情。
- 除錯資訊：回傳實際使用的搜尋工具、價格來源、商品品況與 OCR 警告。

## 🚀 快速開始

### 環境需求
- Python 3.14+
- pip 或 uv 套件管理工具

### 1. 安裝專案依賴

於專案根目錄執行：

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 環境變數設定

本地執行請在專案根目錄建立 `.env` 檔案，依使用功能填入所需設定：

```env
# 商品圖片分析
GROQ_API_KEY=your-groq-key
SERP_API_KEY=your-serpapi-key
TAVILY_SEARCH_API_KEY=your-tavily-key
GCP_OCR_SERVICE_ACCOUNT_JSON={"type":"service_account"}

# Aiven 資料庫
DB_HOST=mysql-xxxxx-yourproject.aivencloud.com
DB_USERNAME=yourusername
DB_NAME=defaultdb
DB_PASSWORD=yourpassword
```
### 3. 啟動服務

```bash
# 開發模式
uv run fastapi dev

# 生產模式
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

服務將於 `http://localhost:8000` 啟動，API 文件請瀏覽 `http://localhost:8000/docs`

---

## 🏗️ 系統架構

**模型與技術棧**

| 組件 | 用途 | 技術 |
|------|------|------|
| 文字分類 | 詐騙文本檢測 | BERT (ONNX 量化) |
| URL 分類 | 惡意 URL 檢測 | XGBoost |
| 向量化 | 語義編碼 | BAAI/bge-small-zh-v1.5 |
| LLM 推理 | 深度語義分析 | Groq (Llama 3.1:8b) |
| 商品辨識 | OCR 商品名稱與型號識別 | Groq (`qwen/qwen3.6-27b` + 搜尋工具) |
| 線上查價 | 市場價格搜尋 | SerpApi、Tavily、DDGS |
| 向量存儲 | 知識庫管理 | ChromaDB |

---

## ✅ 查價工具測試

```powershell
# 測試真實 Tavily API
uv run python scripts/check_tavily_api.py

# 測試 Tavily 搜尋後交由 Groq 整理的完整流程
uv run python scripts/check_groq_tavily_workflow.py
```

若環境已初始化完成，後續啟動僅需：

### 啟動 FastAPI 開發伺服器

```powershell
uv run python -m uvicorn backend.main:app --reload
```

---

## 📦 部署

本專案使用 Hugging Face Spaces Docker 模式部署。

### 環境資訊

- SDK：docker
- Base Image：python:3.10-slim
- Local Dependency Management：uv + pyproject.toml
- Docker Dependency Installation：requirements.txt
- Exposed Port：7860
