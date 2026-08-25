import sys
from types import ModuleType

import pytest

from langgraph_agent_lab.llm import get_llm


class FakeGeminiClient:
    def __init__(self, **config: object) -> None:
        self.config = config


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_get_llm_builds_configured_gemini_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_module = ModuleType("langchain_google_genai")
    fake_module.__dict__["ChatGoogleGenerativeAI"] = FakeGeminiClient
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    client = get_llm(model="test-model", temperature=0.25)

    assert isinstance(client, FakeGeminiClient)
    assert client.config == {
        "model": "test-model",
        "google_api_key": "test-key",
        "temperature": 0.25,
    }


def test_get_llm_requires_a_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)

    with pytest.raises(RuntimeError, match="No LLM API key found"):
        get_llm()


def test_get_llm_uses_current_default_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_module = ModuleType("langchain_google_genai")
    fake_module.__dict__["ChatGoogleGenerativeAI"] = FakeGeminiClient
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    client = get_llm()

    assert isinstance(client, FakeGeminiClient)
    assert client.config["model"] == "gemini-3.6-flash"
