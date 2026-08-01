import os
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CORE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CORE_DIR.parent
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    APP_NAME: str = "Scam detetcor"
    DEBUG: bool = True

    HF_TEXT_REPO_ID: str = "scam-project/spam-detector-chinese"
    HF_URL_REPO_ID: str = "scam-project/url-detector"
    HF_PERSIST_REPO_ID: str = "scam-project/scam-rag-db"
    HF_TOKEN: str = ""

    REGEX_WEIGHT: float = 0.55
    MODEL_WEIGHT: float = 0.45

    HIGH_THRESHOLD: float = 0.5
    MEDIUM_THRESHOLD: float = 0.6
    UNKNOWN_THRESHOLD: float = 0.7

    URL_THRESHOLD: float = 0.46

    DEVICE: str = "cpu"
    BASE_MODEL_PATH: str = str(BACKEND_DIR / "models")

    CHUNCK_SIZE: int = 500
    CHUNCK_OVERLAP: int = 50

    RAG_MODEL_NAME: str = "llama-3.1-8b-instant"
    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 3
    RAG_RECORD_LIMIT: int = 2100
    RAG_DATASET_PATH: str = str(ROOT_DIR / "data" / "raw" / "scam-dataset.json")
    RAG_PERSIST_DIR: str = str(ROOT_DIR / "data" / "chroma")
    EMBED_MODEL: str = "BAAI/bge-small-zh-v1.5"
    
    NORMALIZER_MODEL: str = "openai/gpt-oss-120b"
    PRICE_MODEL: str = "openai/gpt-oss-120b"
    REVIEW_MODEL: str = "qwen/qwen3.6-27b"
    TAVILY_SEARCH_API_KEY: SecretStr = SecretStr("")
    SERP_API_KEY: SecretStr = SecretStr("")
    SEARCH_COUNTRY: str = "taiwan"

    ALLOWED_IMAGE_TYPES : set[str] = {"image/jpeg", "image/png", "image/webp"}
    MAX_IMAGE_BYTES : int = 20 * 1024 * 1024  # 20MB
    MARKET_PRICE_DB_PATH: str = str(ROOT_DIR / "data" / "market_prices_tw.json")
    ONLINE_PRICE_ENABLED: bool = True
    ONLINE_PRICE_MAX_RESULTS: int = 20
    SEARCH_DOMAIN: list[str] = ["biggo.com.tw", "momo.com.tw", "pchome.com.tw", "shopee.tw", "ruten.com.tw", "tw.carousell.com"]
    EXCLUDE_DOMAIN: list[str] = ["zh.wikipedia.org"]
    ONLINE_PRICE_FALLBACK_DELAY_SECONDS: float = 2.0
    GROQ_FALLBACK_RETRY_DELAY_SECONDS: float = 2.0
    GROQ_RATE_LIMIT_MAX_WAIT_SECONDS: float = 30.0
    PRODUCT_MATCH_FUZZY_MIN_SCORE: int = 86
    ENABLE_INTELLIGENT_LAYER: bool = True
    CASE_LOG_PATH: str = str(ROOT_DIR / "data" / "cases" / "image_analysis_cases.jsonl")
    CASE_MEMORY_ENABLED: bool = True
    OCR_PROVIDER: str = "google_vision"
    GCV_LANGUAGE_HINTS: str = "zh-TW,en"
    GCP_OCR_SERVICE_ACCOUNT_JSON : str = ""

    DB_USERNAME: str = ""
    DB_PASSWORD: str = ""
    DB_HOST: str = ""
    DB_PORT: int = 10126
    DB_NAME: str = ""
    DB_TIMEOUT: int = 10
    DB_SSL_CA: str = "ca.pem"

    GROQ_API_KEY: SecretStr = SecretStr("")
    model_config = SettingsConfigDict(
        env_file=[os.path.join(BACKEND_DIR, ".env"), 
                  os.path.join(ROOT_DIR, ".env")],
        extra="ignore",
    )
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value


settings = Settings()
