---
title: Spam Detector API
emoji: 🚫
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

### 這是一個基於FastAPI和 XgBoost、Bert + llama3.2量化模型 + 決策型RAG 的詐騙偵測API的測試環境

### 📂 專案結構說明
```text
scam-detection-project/
│
├─ backend/
│  ├─ main.py                           # FastAPI 入口與 lifespan 設定
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

若您首次下載本專案或更換電腦，請按照以下步驟初始化環境：

#### 1. 同步依賴

在專案根目錄執行：

```powershell
uv sync
```

這會根據 `pyproject.toml` 安裝所有必要的 Python 依賴，並設置虛擬環境。

#### 2. 啟動 Ollama 服務

確保 Ollama 服務已啟動（用於 Embedding 生成）：

```powershell
ollama serve
```

預設使用的 embedding model 為 `nomic-embed-text`，可在 `backend/config.py` 中修改 `OLLAMA_EMBED_MODEL` 設定。

#### 3. 構建 ChromaDB 向量資料庫

在新的終端執行（確保虛擬環境已啟動）：

```powershell
uv run python scripts/ingest_data.py
```

**執行前請確認：**
- `data/raw/scam-dataset.json` 已存在
- Ollama 服務已啟動（步驟 2）
- 所需的 embedding model 已下載

執行後會在 `data/chroma/` 生成向量資料庫。

### 後續啟動專案

若環境已設置完成，後續啟動只需：

#### 1. 啟動 Ollama（若未運行）

```powershell
ollama serve
```

#### 2. 啟動 FastAPI 開發伺服器

在新的終端執行：

```powershell
uv run python -m uvicorn backend.main:app --reload
```

應用會在 `http://localhost:8000` 運行。
- API 文檔：`http://localhost:8000/docs`
- 測試資訊：查看 `test.py` 中的測試案例
