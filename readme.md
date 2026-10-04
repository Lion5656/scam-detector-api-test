## ✨ 基於 FastAPI 的多模型詐騙偵測 API

整合**文字訊息、網址、電話號碼、商品圖片價格**四種偵測管道的詐騙風險分析 API，結合本地模型、規則引擎、RAG 相似案例檢索與 LLM 推理，提供可解釋的風險判斷結果。

---

## 🌟 專案特色

- **混合式判斷**：規則比對與分類模型加權融合，模型信心不足或與規則判斷衝突時，才交由 RAG + LLM 深度分析，兼顧速度、成本與準確度。
- **可解釋的結果**：每次分析都會回傳風險等級、分數與判斷原因，而不只是單一標籤。
- **在地化規則**：針對台灣常見詐騙情境設計，包含 `.com.tw` 等受管制網域後綴、台灣常見短網址與中文詐騙關鍵字。
- **詐騙家族關聯分析**：以圖論分析回報資料，找出電話號碼之間的轉介、共用導流端點與號段相近等關聯。
- **多來源查價與容錯**：線上查價有主要與備援搜尋工具，單一服務失效時會自動切換。
- **模型與資料雲端載入**：模型與向量資料庫在啟動時從 Hugging Face 下載，不需要放在版本控制裡。

---

## ⚡ 功能說明

### 📝 文字詐騙偵測
- 中文 BERT 分類模型（ONNX 量化），長文會切段推論並檢查各段結果是否一致。
- 關鍵字與正規表示式規則比對，和模型結果加權融合。
- 信心路由：模型信心過低、分數差距過小、規則與模型判斷衝突時，自動交給 RAG 分析。
- RAG 從向量資料庫檢索相似的詐騙案例，再由 LLM 做語意推理並說明原因。

### 🔗 URL 風險偵測
- 網域後綴策略：辨識台灣受管制後綴（`com.tw`、`gov.tw`、`edu.tw` 等）、公用短網址，以及偽裝成台灣網域的可疑字樣。
- 擷取網址特徵後交由 XGBoost 模型評分，輸出 `HIGH` / `MEDIUM` / `LOW` 風險等級。

### 📞 電話號碼查詢與回報
- 查詢號碼是否在黑名單或白名單中，並回傳通報次數、首次與最近通報時間、詐騙類型。
- 黑名單號碼會附上詐騙家族的直接關聯證據（關聯號碼、關聯強度、原因）。
- 查無資料時，可回報可疑號碼，並記錄轉介方式（電話、LINE、URL、Telegram 等）。

### 🛒 商品圖片價格驗證
- 使用 Google Cloud Vision OCR 讀取商品頁截圖，支援 Facebook Marketplace 版面解析。
- 擷取商品名稱、售價、品況與賣家，再由 LLM 正規化商品名稱與型號。
- 線上查價：以 SerpApi 為主，Tavily、DDGS 為備援，並整理搜尋結果中的價格證據。
- 排除離群值後估算市場行情區間，判斷售價是否明顯低於行情。
- 加上 `?debug=true` 可取得 OCR、搜尋工具、價格來源與判斷層級等除錯資訊。

---

## 🧠 使用模型

| 用途 | 模型 / 服務 |
|------|------|
| 文字詐騙分類 | `scam-project/spam-detector-chinese`（BERT，ONNX 量化） |
| URL 風險分類 | `scam-project/url-detector`（XGBoost） |
| RAG 文本向量化 | `BAAI/bge-small-zh-v1.5` |
| RAG 向量資料庫 | ChromaDB（資料集：`scam-project/scam-rag-db`） |
| RAG 語意推理 | Groq `openai/gpt-oss-20b` |
| 商品名稱正規化、價格整理 | Groq `openai/gpt-oss-120b` |
| 商品品況審查 | Groq `qwen/qwen3.8-27b` |
| 圖片 OCR | Google Cloud Vision |
| 線上查價 | SerpApi、Tavily、DDGS |

> 以上 LLM 模型名稱皆可透過環境變數（`RAG_MODEL_NAME`、`NORMALIZER_MODEL`、`PRICE_MODEL`、`REVIEW_MODEL`）調整。

---

## 🔌 API 端點

所有端點都掛在 `/api/v1` 底下，完整的參數與範例請見 Swagger 文件 `/docs`。

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/api/v1/text/analyze` | 文字詐騙風險分析（規則 + 模型 + RAG 混合判斷） |
| `POST` | `/api/v1/text/analyze/base` | 文字風險分析（僅使用分類模型） |
| `POST` | `/api/v1/url/analyze` | URL 風險分析 |
| `POST` | `/api/v1/phones/search` | 電話號碼黑白名單查詢 |
| `POST` | `/api/v1/phones/report` | 回報可疑電話號碼 |
| `POST` | `/api/v1/price/analyze` | 商品圖片價格風險分析（`multipart/form-data`） |
| `POST` | `/api/v1/image/upload` | 圖片上傳測試（只驗證，不分析） |

---

## 🖥️ 使用環境

| 項目 | 需求 |
|------|------|
| Python | 3.14 以上 |
| 套件管理 | [uv](https://docs.astral.sh/uv/)（推薦）或 pip |
| 資料庫 | MySQL（目前使用 Aiven 雲端資料庫，需 SSL 憑證） |
| 外部服務 | Groq、Google Cloud Vision、SerpApi、Tavily、Hugging Face |
| 執行裝置 | CPU 即可（`DEVICE=cpu`） |

---

## 🚀 快速開始

### 1. 安裝依賴

於專案根目錄執行：

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 設定環境變數

在專案根目錄建立 `.env` 檔案（也可以放在 `backend/.env`）：

```env
# 執行模式：production 時錯誤訊息不會回傳細節
DEBUG=development

# Hugging Face（下載模型與向量資料庫）
HF_TOKEN=your-hf-token

# LLM 與查價
GROQ_API_KEY=your-groq-key
SERP_API_KEY=your-serpapi-key
TAVILY_SEARCH_API_KEY=your-tavily-key

# Google Cloud Vision OCR（整份 service account JSON）
GCP_OCR_SERVICE_ACCOUNT_JSON={"type":"service_account", ...}

# MySQL 資料庫
DB_HOST=mysql-xxxxx-yourproject.aivencloud.com
DB_PORT=12345
DB_USERNAME=yourusername
DB_PASSWORD=yourpassword
DB_NAME=defaultdb
DB_SSL_CA=ca.pem
```

| 變數 | 用於 |
|------|------|
| `HF_TOKEN` | 所有功能（啟動時下載模型與 RAG 資料） |
| `GROQ_API_KEY` | 文字 RAG 分析、商品圖片分析 |
| `SERP_API_KEY`、`TAVILY_SEARCH_API_KEY` | 商品線上查價 |
| `GCP_OCR_SERVICE_ACCOUNT_JSON` | 商品圖片 OCR |
| `DB_*` | 電話號碼查詢與回報 |

其餘可調整的參數（風險門檻、權重、模型名稱等）請參考 `backend/core/config.py`。

### 3. 啟動服務

```bash
# 開發模式（程式碼變更時自動重載）
uv run fastapi dev

# 生產模式
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

第一次啟動時會從 Hugging Face 下載模型與向量資料庫，需要稍等一段時間。看到 `RAG 已初始化` 就代表啟動完成。

- 服務位址：`http://localhost:8000`
- API 文件：`http://localhost:8000/docs`

### 4. 呼叫範例

```bash
# 文字分析
curl -X POST http://localhost:8000/api/v1/text/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "恭喜您中獎，請加 LINE 領取獎金"}'

# URL 分析
curl -X POST http://localhost:8000/api/v1/url/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/login"}'

# 電話查詢
curl -X POST http://localhost:8000/api/v1/phones/search \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "0422255141"}'

# 商品圖片價格分析
curl -X POST "http://localhost:8000/api/v1/price/analyze?debug=true" \
  -F "file=@screenshot.png"
```

---

## ✅ 測試

```bash
# 執行單元與 API 測試
uv run pytest

# 檢查外部服務連線
uv run python scripts/test_db_connection.py
uv run python scripts/check_tavily_api.py
uv run python scripts/check_market_search_tools.py
uv run python scripts/check_groq_tavily_workflow.py
```

---

## 📦 部署

本專案以 Hugging Face Spaces 的 Docker 模式部署：

1. 將 `main` 合併到 `space-deploy` 分支（建議開 PR，會自動檢查是否有超過 10MB 的檔案）。
2. push `space-deploy` 後，GitHub Actions 會自動把程式碼同步到 Hugging Face Space。
3. 環境變數請在 Space 的 **Settings → Variables and secrets** 設定，正式環境記得設 `DEBUG=production`。

| 項目 | 設定 |
|------|------|
| SDK | Docker |
| 依賴安裝 | `requirements.txt` |
| 對外 Port | 7860 |
| GitHub Secret | `HF_TOKEN`（需有 Space 的寫入權限） |
