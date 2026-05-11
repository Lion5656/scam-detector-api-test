---
title: Spam Detector API
emoji: 🚫
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

### 基於 FastAPI 的多模型詐騙偵測 API 測試環境

整合文字詐騙分類、URL 詐騙偵測與 RAG 推理流程，提供模組化的詐騙分析能力。

#### 使用核心模型

- bert：負責文字詐騙分類推理
- XGBoost：負責 URL 詐騙網址偵測
- nomic-embed-text：負責 RAG 文本向量化
- llama 3.2:3b Q5：負責 RAG 語意推理與分析

### 📂 專案結構說明
```text
scam-detection-project/
│
├─ backend/
│  ├─ main.py                           # FastAPI 入口
│  ├─ config.py                         # 專案設定與環境變數
│  │
│  ├─ db/
│  │  ├─ chroma.py                      # ChromaDB 初始化入口
│  │  └─ rebuild.py                     # 向量庫建立與索引重建
│  │
│  ├─ repository/                       # 資料存取層 (CRUD)
│  │  └─ rag_repository.py              # 向量庫查詢、新增、刪除等操作(保留擴充web search)
│  │ 
│  ├─ routers/                          # API 路由層
│  │  ├─ text_inference.py              # 文字檢測 API
│  │  └─ url_detection.py               # URL 檢測 API
│  │ 
│  ├─ schemas/                          # 對外請求 / 回應模型
│  │  └─ analysis.py
│  │
│  ├─ services/                         # 業務邏輯層
│  │  ├─ dto/                           # 服務間共享資料模型
│  │  │  └─ analysis.py
│  │  │
│  │  ├─ ingestion/                     # 資料攝取與向量化層
│  │  │  └─ rag_retriever.py            # 向量庫檢索、context 組裝與 embedding
│  │  │
│  │  ├─ text_service/                  # 文字詐騙分析流程
│  │  │  ├─ text_analyzer.py            # 協調層 - 功能編排與決策路由 (Orchestration)
│  │  │  ├─ transformer_classifier.py   # Transformer 模型加載與推理服務
│  │  │  ├─ confidence_router.py        # decision-logic -> LLM or RAG
│  │  │  ├─ rag_reasoner.py             # 呼叫 LLM 並解析 RAG 結果
│  │  │  └─ fusion_service.py           # 基礎規則、模型與 RAG 結果融合
│  │  │
│  │  └─ url_service/                   # URL 詐騙分析流程
│  │     └─ url_analyzer.py             # URL 特徵與分類推論
│  └─ utils/                            # 共用工具函式
│     ├─ features.py                    # URL 特徵工程
│     ├─ pattern.py                     # 關鍵詞 / 規則比對
│     └─ text_cleaner.py                # 文字前處理
├─ data/
│  ├─ chroma/                           # 向量資料庫輸出目錄
│  └─ raw/
│     └─ scam-dataset.json              # RAG 預設資料集位置
├─ scripts/
│  └─ ingest_data.py                    # 建立 ChromaDB 的執行入口
├─ backend/models/                      # 模型檔案（文字分類與 URL 分類）
```

目前專案架構依據功能邊界設計，採分層式的協調模式：

- **db** 層：ChromaDB 初始化與索引重建
- **repository** 層：向量庫的 CRUD 操作（查詢、新增、刪除）
- **ingestion** 層：資料攝取、embedding 與向量庫檢索
- **text_service** 層：
  - **text_analyzer** (Orchestration)：功能編排與決策路由
  - **transformer_classifier**：Transformer 模型加載與推理
  - **confidence_router**：決策邏輯（是否使用 LLM/RAG）
  - **rag_reasoner**：調用 LLM 並解析 RAG 結果
  - **fusion_service**：融合多種方案的分析結果
- **url_service** 層：URL 特徵與分類推論
- **routers** 層：API 路由與請求處理


## 開發環境設定

### 首次啟動專案

若您首次下載本專案或更換開發環境，請依照以下步驟初始化：

### 1. 安裝專案依賴

於專案根目錄執行：

```powershell
uv sync
```

此指令會根據 `pyproject.toml` 安裝所需依賴並建立虛擬環境。

---

### 2. 啟動 Ollama 服務

本專案使用 Ollama 提供 embedding 模型服務。

```powershell
ollama serve
```

預設 embedding model 為 `nomic-embed-text`，可於 `backend/config.py` 修改：

```python
OLLAMA_EMBED_MODEL
```

---

### 3. 建立 ChromaDB 向量資料庫

執行前請確認：

- 已匯入 `dataset`
- Ollama 服務已啟動
- 已下載所需 embedding model

接著執行向量建庫腳本：

```powershell
python scripts/build_vector_db.py
```

完成後會於：

```text
data/chroma/
```

生成 ChromaDB 向量資料庫。

---

## 後續啟動專案

若環境已初始化完成，後續啟動僅需：

### 1. 啟動 Ollama

```powershell
ollama serve
```

### 2. 啟動 FastAPI 開發伺服器

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

### FastAPI 啟動 Port

```text
0.0.0.0:7860
```