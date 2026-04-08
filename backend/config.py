import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 模型路徑設定 (使用Path處理跨平台路徑問題)
# Path(__file__).resolve() 取得當前檔案的絕對路徑，parent取得檔案所在的目錄(上一層) 
BASE_DIR = Path(__file__).resolve().parent 
# BASE_MODEL_PATH: str = str(BASE_DIR/"models"/"bert-onnx-v1")

class Settings(BaseSettings):
    # 基本專案資訊
    APP_NAME: str = "Scam detetcor"
    DEBUG: bool = True

    # huggingface repo id
    HF_REPO_ID: str = "kko12/spam-detector-chinese"

    # 推理權重分配
    REGEX_WEIGHT: float = 0.65
    MODEL_WEIGHT: float = 0.35

    # 模型預測標籤門檻
    HIGH_THRESHOLD: float = 0.5
    MEDIUM_THRESHOLD: float = 0.6
    UNKNOWN_THRESHOLD: float = 0.7

    # 硬體配置
    DEVICE: str = "cpu"

    # 載入專案環境變數
    model_config = SettingsConfigDict(env_file = os.path.join(BASE_DIR, ".env"), extra="ignore")

# 實例化全域使用
settings = Settings()