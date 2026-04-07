"""
Model client utilities for the Defense LLM Evaluation Framework.

Provides a unified interface for interacting with multiple LLM providers
(OpenAI, Anthropic) with rate limiting and token counting.
"""

import functools
import time
from typing import Any

import anthropic
import openai


def rate_limit(max_calls_per_minute: int = 60):
    """Decorator that enforces rate limiting on API calls.

    Args:
        max_calls_per_minute: Maximum number of calls allowed per minute.

    Returns:
        Decorated function with rate limiting applied.
    """
    min_interval = 60.0 / max_calls_per_minute

    def decorator(func):
        last_called = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ModelClient:
    """Unified client for LLM inference across providers.

    Supports OpenAI and Anthropic APIs with consistent interface
    for generation, token counting, and error handling.

    Args:
        provider: API provider name ('openai' or 'anthropic').
        model: Model identifier string.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in the generated response.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_token_count: int = 0

        if provider == "openai":
            self._client = openai.OpenAI()
        elif provider == "anthropic":
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'.")

    @rate_limit(max_calls_per_minute=50)
    def generate(self, prompt: str) -> dict[str, Any]:
        """Generate a response from the model given a prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            Parsed JSON response from the model, or raw text in a dict
            if JSON parsing fails.

        Raises:
            RuntimeError: If the API call fails after retries.
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt)
        except Exception as e:
            raise RuntimeError(f"Generation failed for {self.provider}/{self.model}: {e}") from e

    def _generate_openai(self, prompt: str) -> dict[str, Any]:
        """Call OpenAI API and return parsed response.

        Args:
            prompt: The input prompt string.

        Returns:
            Parsed response dictionary.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.last_token_count = response.usage.total_tokens if response.usage else 0
        content = response.choices[0].message.content or ""
        return self._parse_response(content)

    def _generate_anthropic(self, prompt: str) -> dict[str, Any]:
        """Call Anthropic API and return parsed response.

        Args:
            prompt: The input prompt string.

        Returns:
            Parsed response dictionary.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        self.last_token_count = (response.usage.input_tokens + response.usage.output_tokens)
        content = response.content[0].text if response.content else ""
        return self._parse_response(content)

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        """Attempt to parse model output as JSON.

        Args:
            content: Raw text output from the model.

        Returns:
            Parsed JSON dictionary, or {"raw_text": content} if parsing fails.
        """
        import json

        content = content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_text": content}


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text string.

    Uses a simple heuristic of ~4 characters per token.
    For exact counts, use the provider's tokenizer.

    Args:
        text: Input text string.

    Returns:
        Estimated number of tokens.
    """
    return max(1, len(text) // 4)
