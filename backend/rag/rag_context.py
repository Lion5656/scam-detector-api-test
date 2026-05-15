from typing import Any
import os
from huggingface_hub import snapshot_download

from backend.config import settings


class RAGContext:
    """全局 RAG 上下文，存儲 app 實例"""
    
    _app: Any = None

    @staticmethod
    def load_vectorstore() -> None:
        """從 HuggingFace 加載向量資料庫"""
        print("加載向量資料庫...")
        snapshot_download(
            repo_id=settings.HF_PERSIST_REPO_ID,
            repo_type="dataset",
            local_dir=settings.RAG_PERSIST_DIR,
            token=os.getenv("HF_TOKEN")
        )
        print("向量資料庫加載完畢")

    @classmethod
    def set_app(cls, app: Any) -> None:
        """設置 FastAPI app 實例"""
        cls._app = app
    
    @classmethod
    def get_app(cls) -> Any:
        """取得 FastAPI app 實例"""
        if cls._app is None:
            raise RuntimeError("RAG Context 未初始化，請先呼叫 set_app()")
        return cls._app
    
    @classmethod
    def get_vectorstore(cls) -> Any:
        """取得 vectorstore"""
        app = cls.get_app()
        if not hasattr(app.state, 'vectorstore'):
            raise RuntimeError("vectorstore 尚未初始化")
        return app.state.vectorstore
    
    @classmethod
    def get_embeddings(cls) -> Any:
        """取得 embeddings"""
        app = cls.get_app()
        if not hasattr(app.state, 'embeddings'):
            raise RuntimeError("embeddings 尚未初始化")
        return app.state.embeddings
