"""Google Gemini model adapter.

Uses the Gemini REST API directly (stdlib urllib only — no SDK required).
Free tier: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiModel:
    def __init__(self, api_key: str, model_id: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_id = model_id

    def generate(self, prompt: str) -> str:
        url = f"{_BASE_URL}/{self.model_id}:generateContent?key={self.api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "python-aal/0.1",
            },
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  [rate limit] waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API error {e.code}: {body}") from e
