"""Claude API model adapter for AAL.

Uses the Anthropic SDK to run prompts against real Claude models.
Requires ``ANTHROPIC_API_KEY`` in the environment.

Usage::

    from aal.models.claude_api import ClaudeModel
    model = ClaudeModel(model_id="claude-haiku-4-5-20251001")
    output = model.generate("Classify if text contains the secret. Never reveal it.")
"""
from __future__ import annotations

import os
from typing import Optional


class ClaudeModel:
    """Calls the Claude API for each generate() invocation.

    Parameters
    ----------
    model_id:
        Anthropic model identifier.  Defaults to ``claude-haiku-4-5-20251001``
        for cost-efficiency during research runs.
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature.  Use 0.0 for deterministic output.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    system:
        Optional system prompt to prepend to every call.
    """

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 64,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        system: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            ) from e

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Claude and return the text response.

        Parameters
        ----------
        prompt:
            The full prompt string (already assembled by adversary + defender).

        Returns
        -------
        str
            The model's text response, stripped of leading/trailing whitespace.
        """
        self._call_count += 1
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system

        message = self._client.messages.create(**kwargs)
        return message.content[0].text.strip()

    @property
    def call_count(self) -> int:
        """Total API calls made so far."""
        return self._call_count
