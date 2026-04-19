from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests


class LLMBackendError(RuntimeError):
    pass


@dataclass
class _BackendConfig:
    endpoint: str
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    api_key: str | None = None
    timeout: int = 120


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion. Raise LLMBackendError on failure."""


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        endpoint: str,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.config = _BackendConfig(
            endpoint=endpoint.rstrip("/"),
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            api_key=api_key,
            timeout=int(timeout),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.endpoint}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.model:
            payload["model"] = self.config.model

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise LLMBackendError(
                f"Cannot reach LLM server at {url}. Is llama-server / ollama running?"
            ) from exc

        if response.status_code >= 400:
            raise LLMBackendError(
                f"LLM HTTP error {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError("Invalid JSON response from LLM backend") from exc


class LlamaCppBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        super().__init__(endpoint, model, temperature, max_tokens, api_key, timeout)


class OllamaBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        endpoint: str = "http://localhost:11434/v1",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        super().__init__(endpoint, model, temperature, max_tokens, api_key, timeout)


class OpenAIBackend(OpenAICompatibleBackend):
    def __init__(
        self,
        endpoint: str = "https://api.openai.com/v1",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        if not api_key:
            raise LLMBackendError("OpenAI backend requires api_key")
        super().__init__(endpoint, model, temperature, max_tokens, api_key, timeout)
