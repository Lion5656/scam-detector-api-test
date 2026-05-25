### 基於 FastAPI 的多模型詐騙偵測 API 測試環境

整合文字詐騙分類、URL 詐騙偵測與 RAG 推理流程，提供模組化的詐騙分析能力。

#### 使用核心模型

- bert：負責文字詐騙分類推理
- XGBoost：負責 URL 詐騙網址偵測
- BAAI/bge-small-zh-v1.5：負責 RAG 文本向量化
- llama 3.1:8b-instant：負責 RAG 語意推理與分析

### 📂 專案結構說明
```text
scam-detection-project/
│
├─ backend/                             # 後端主程式
│  ├─ __init__.py
│  ├─ main.py                           # FastAPI 應用入口 & 生命週期管理
│  ├─ config.py                         # 環境變數與全局設定
│  │ 
│  ├─ routers/                          # 🛣️ API 路由層
│  │  ├─ __init__.py
│  │  ├─ text_inference.py             
│  │  └─ url_detection.py              
│  │ 
│  ├─ schemas/                          # 📋 API 資料模型
│  │  ├─ __init__.py
│  │  ├─ text.py                        # 簡訊請求/回應定義
│  │  └─ url.py                         # 網址請求/回應定義
│  │
│  ├─ services/                         # ⚙️ 業務邏輯層
│  │  ├─ __init__.py
│  │  │
│  │  ├─ dto/                           # 服務間通信模型
│  │  │  ├─ __init__.py
│  │  │  └─ analysis.py
│  │  │
│  │  ├─ text_service/                  # 📝 文字詐騙分析流程
│  │  │  ├─ __init__.py
│  │  │  ├─ text_analyzer.py            # [協調層] 功能編排 & 決策路由
│  │  │  ├─ transformer_classifier.py   # [推理層] BERT 模型加載 & 預測
│  │  │  ├─ confidence_router.py        # [決策層] 信心度評估 & RAG 判定
│  │  │  ├─ rag_reasoner.py             # [推理層] LLM 深度分析 & 結果解析
│  │  │  └─ fusion_service.py           # [融合層] 多源結果加權融合
│  │  │
│  │  └─ url_service/                   # 🌐 URL 詐騙分析流程
│  │     ├─ __init__.py
│  │     └─ url_analyzer.py             # 特徵工程 & XGBoost 推理
│  │       
│  ├─ repository/                       # 資料存取層
│  │  ├─ __init__.py
│  │  └─ rag_repository.py
│  │    
│  ├─ rag/                              # 🧠 RAG 模組
│  │  ├─ __init__.py
│  │  ├─ rag_context.py                 # RAG 全局上下文 & 狀態管理
│  │  ├─ rag_retriever.py               # 向量檢索 & Context 組裝
│  │  ├─ rag_reasoner.py                # LLM 推理與結果解析
│  │  ├─ dto/
│  │  │  ├─ __init__.py
│  │  │  └─ analysis.py
│  │
│  └─ utils/                            # 🔧 工具函式庫
│     ├─ __init__.py
│     ├─ features.py                    # URL 特徵工程 (30+ 維度)
│     ├─ pattern.py                     # 關鍵詞匹配 & 正則規則
│     └─ text_cleaner.py                # 文字前處理 & 標準化
```

**📋 分層說明：**
- **API 層**：FastAPI 應用管理
- **路由層**：HTTP 端點定義
- **業務邏輯層**：核心分析引擎 (協調/推理/決策/融合)
- **資料存取層**：資料存取操作
- **RAG 模組**：知識庫檢索與 LLM 推理
- **工具庫**：特徵工程與文字處理

## 快速開始

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

在專案根目錄建立 `.env` 檔案：

```env
# LLM API 設定
GROQ_API_KEY=your_groq_api_key  # 可選，若使用 Groq LLM
```

### 3. 初始化向量資料庫

執行以下指令建立 ChromaDB：

```bash
python backend/db/rebuild.py
```

### 4. 啟動服務

```bash
# 開發模式
uv run fastapi dev

# 生產模式
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

服務將於 `http://localhost:8000` 啟動，API 文件請瀏覽 `http://localhost:8000/docs`

---

## 系統架構

### 模型與技術棧

| 組件 | 用途 | 技術 |
|------|------|------|
| 文字分類 | 詐騙文本檢測 | BERT (ONNX 量化) |
| URL 分類 | 惡意 URL 檢測 | XGBoost |
| 向量化 | 語義編碼 | BAAI/bge-small-zh-v1.5 |
| LLM 推理 | 深度語義分析 | Groq (Llama 3.1:8b) |
| 向量存儲 | 知識庫管理 | ChromaDB |

### 推理流程

```
輸入 (文字/URL)
    ↓
基礎規則檢測 (關鍵詞匹配)
    ↓
信心度評估
    ├─ 高信心 → 直接返回結果
    └─ 低信心 → 進入 RAG 流程
        ├─ 向量檢索 (相似案例)
        ├─ LLM 推理 (語義分析)
        └─ 結果融合 → 返回結果
```

---

## 開發指南

### 專案結構詳解

#### Backend 層級

- **routers/**: API 端點定義
  - `text_inference.py`: 文字檢測路由
  - `url_detection.py`: URL 檢測路由

- **services/**: 業務邏輯
  - `text_service/`: 文字分析流程
    - `text_analyzer.py`: 協調層（功能編排）
    - `transformer_classifier.py`: 模型推理
    - `confidence_router.py`: 決策路由（是否使用 LLM）
    - `rag_reasoner.py`: LLM 推理
    - `fusion_service.py`: 結果融合
  - `url_service/`: URL 分析流程

- **rag/**: RAG 相關模組
  - `rag_retriever.py`: 向量檢索
  - `rag_reasoner.py`: LLM 推理
  - `rag_context.py`: RAG 全局上下文

- **db/**: 資料庫管理
  - `chroma.py`: ChromaDB 初始化
  - `rebuild.py`: 向量庫重建

- **utils/**: 工具函式
  - `features.py`: URL 特徵工程
  - `pattern.py`: 關鍵詞規則比對
  - `text_cleaner.py`: 文字前處理


### 建立 ChromaDB 向量資料庫

執行前請確認：

- 已匯入 `dataset`
- 已下載所需 embedding model

完成後會於：

```text
data/chroma/
```

生成 ChromaDB 向量資料庫。

---

## 後續啟動專案

若環境已初始化完成，後續啟動僅需：

### 啟動 FastAPI 開發伺服器

```powershell
uv run python -m uvicorn backend.main:app --reload
```

---

## Deployment

本專案使用 Hugging Face Spaces Docker 模式部署。

### 環境資訊

- SDK：docker
- Base Image：python:3.10-slim
- Local Dependency Management：uv + pyproject.toml
- Docker Dependency Installation：requirements.txt
- Exposed Port：7860