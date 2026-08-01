"""集中建立 Groq 聊天模型。"""

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from backend.core.config import settings


class GroqProvider:
    """建立具有一致 API key 處理方式的 Groq 聊天模型。"""

    @staticmethod
    def create(
        *,
        model: str,
        temperature: float = 0,
        api_key: str | SecretStr | None = None,
        reasoning_effort: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> BaseChatModel:
        """依用途建立 Groq 模型，僅傳入有設定的選用參數。"""
        from langchain_groq import ChatGroq

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
        }
        secret = GroqProvider.resolve_api_key(api_key)
        if secret is not None:
            kwargs["api_key"] = secret
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if model_kwargs:
            kwargs["model_kwargs"] = dict(model_kwargs)

        return ChatGroq(**kwargs)

    @classmethod
    def resolve_api_key(
        cls,
        value: str | SecretStr | None = None,
    ) -> SecretStr | None:
        """依序解析指定值、應用設定與環境變數中的 Groq API key。"""
        secret = cls._secret(value)
        if secret is not None:
            return secret

        secret = cls._secret(settings.GROQ_API_KEY)
        if secret is not None:
            return secret

        return cls._secret(os.getenv("GROQ_API_KEY"))

    @classmethod
    def is_configured(
        cls,
        value: str | SecretStr | None = None,
    ) -> bool:
        """確認是否有可用的 Groq API key。"""
        return cls.resolve_api_key(value) is not None

    @staticmethod
    def _secret(value: str | SecretStr | None) -> SecretStr | None:
        """將單一非空 API key 正規化為 SecretStr。"""
        if isinstance(value, SecretStr):
            return value if value.get_secret_value() else None
        if isinstance(value, str) and value.strip():
            return SecretStr(value.strip())
        return None


groq_provider = GroqProvider()

__all__ = ["GroqProvider", "groq_provider"]
