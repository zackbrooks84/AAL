"""Generic OpenAI-compatible chat model adapter.

Works with any provider that follows the OpenAI chat completions REST spec:
Groq, Mistral, DeepSeek, OpenRouter, etc.

Uses only stdlib (urllib) — no extra packages required.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error


class OpenAICompatModel:
    """Chat model backed by any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        base_url: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: str = "",
        extra_headers: dict | None = None,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.extra_headers = extra_headers or {}

    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "python-aal/0.1",
            "Accept": "application/json",
        }
        headers.update(self.extra_headers)

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  [rate limit] waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API error {e.code}: {body}") from e
