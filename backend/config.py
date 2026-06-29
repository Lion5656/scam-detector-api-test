import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    APP_NAME: str = "Scam detetcor"
    DEBUG: bool = True

    HF_TEXT_REPO_ID: str = "kko12/spam-detector-chinese"
    HF_URL_REPO_ID: str = "kko12/url-detector"
    HF_PERSIST_REPO_ID: str = "kko12/scam-rag-db"

    REGEX_WEIGHT: float = 0.55
    MODEL_WEIGHT: float = 0.45

    HIGH_THRESHOLD: float = 0.5
    MEDIUM_THRESHOLD: float = 0.6
    UNKNOWN_THRESHOLD: float = 0.7

    DEVICE: str = "cpu"
    BASE_MODEL_PATH: str = str(BASE_DIR / "models")

    CHUNCK_SIZE: int = 500
    CHUNCK_OVERLAP: int = 50

    RAG_MODEL_NAME: str = "llama-3.1-8b-instant"
    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 3
    RAG_RECORD_LIMIT: int = 2100
    RAG_DATASET_PATH: str = str(ROOT_DIR / "data" / "raw" / "scam-dataset.json")
    RAG_PERSIST_DIR: str = str(ROOT_DIR / "data" / "chroma")
    EMBED_MODEL: str = "BAAI/bge-small-zh-v1.5"

    DB_USERNAME: str = "root"
    DB_PASSWORD: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "phone_db"

    GROQ_API_KEY: str = ""
    model_config = SettingsConfigDict(
        env_file=[os.path.join(BASE_DIR, ".env"), os.path.join(ROOT_DIR, ".env")],
        extra="ignore",
    )


settings = Settings()
