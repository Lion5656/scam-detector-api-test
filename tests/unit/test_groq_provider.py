"""Groq 模型 provider 的單元測試。"""

from typing import Any

from pydantic import SecretStr

from backend.providers.groq import GroqProvider


def test_provider_builds_chat_model_with_optional_settings(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_groq(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_groq.ChatGroq", fake_chat_groq)

    GroqProvider.create(
        model="test-model",
        temperature=0.2,
        api_key=" test-key ",
        reasoning_effort="medium",
        model_kwargs={"top_p": 0.7},
    )

    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2
    assert captured["reasoning_effort"] == "medium"
    assert captured["model_kwargs"] == {"top_p": 0.7}
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["api_key"].get_secret_value() == "test-key"


def test_provider_omits_empty_optional_settings(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_groq(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_groq.ChatGroq", fake_chat_groq)
    monkeypatch.setattr(
        "backend.providers.groq.settings.GROQ_API_KEY",
        SecretStr(""),
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    GroqProvider.create(model="test-model", api_key="")

    assert captured == {
        "model": "test-model",
        "temperature": 0,
    }


def test_provider_resolves_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.providers.groq.settings.GROQ_API_KEY",
        SecretStr(""),
    )
    monkeypatch.setenv("GROQ_API_KEY", "environment-key")

    api_key = GroqProvider.resolve_api_key()

    assert api_key is not None
    assert api_key.get_secret_value() == "environment-key"
