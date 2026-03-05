"""Synthetic data generation pipeline using LLMs.

Generates diverse test inputs for evaluation by prompting an LLM with
strategy-specific templates. Supports batch generation with automatic
deduplication and validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from evalkit.core.config import LLMProviderConfig
from evalkit.generators.base import BaseGenerator
from evalkit.generators.templates import GenerationStrategy, render_template

logger = structlog.get_logger(__name__)


class SyntheticGenerator(BaseGenerator):
    """Generates synthetic test inputs using an LLM backend.

    Uses configurable prompt templates to produce test cases across
    different strategies (standard, adversarial, edge-case, distribution).

    Attributes:
        llm_config: Configuration for the LLM provider.
        strategy: The generation strategy to use.
    """

    def __init__(
        self,
        generator_id: str = "synthetic-default",
        llm_config: LLMProviderConfig | None = None,
        strategy: GenerationStrategy = GenerationStrategy.STANDARD,
    ) -> None:
        """Initialize the synthetic generator.

        Args:
            generator_id: Unique identifier for this generator.
            llm_config: LLM provider configuration.
            strategy: Generation strategy to use.
        """
        super().__init__(generator_id=generator_id)
        self.llm_config = llm_config or LLMProviderConfig()
        self.strategy = strategy

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        """Send a generation prompt to the LLM provider.

        Args:
            prompt: The generation prompt.

        Returns:
            Raw text response.
        """
        api_key = self.llm_config.api_key

        if self.llm_config.provider == "openai":
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_config.model,
                    "temperature": max(self.llm_config.temperature, 0.7),
                    "max_tokens": self.llm_config.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.llm_config.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        elif self.llm_config.provider == "anthropic":
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_config.model,
                    "temperature": max(self.llm_config.temperature, 0.7),
                    "max_tokens": self.llm_config.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.llm_config.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]
        else:
            raise ValueError(f"Unsupported provider: {self.llm_config.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _acall_llm(self, prompt: str) -> str:
        """Async version of _call_llm.

        Args:
            prompt: The generation prompt.

        Returns:
            Raw text response.
        """
        api_key = self.llm_config.api_key

        async with httpx.AsyncClient(timeout=self.llm_config.timeout_seconds) as client:
            if self.llm_config.provider == "openai":
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_config.model,
                        "temperature": max(self.llm_config.temperature, 0.7),
                        "max_tokens": self.llm_config.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            elif self.llm_config.provider == "anthropic":
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.llm_config.model,
                        "temperature": max(self.llm_config.temperature, 0.7),
                        "max_tokens": self.llm_config.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            else:
                raise ValueError(f"Unsupported provider: {self.llm_config.provider}")

    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, str]]:
        """Parse the LLM's JSON response into test case dicts.

        Args:
            raw: Raw text response from the LLM.

        Returns:
            List of test case dictionaries.

        Raises:
            ValueError: If parsing fails.
        """
        text = raw.strip()

        # Strip markdown code fences
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse generator response: {e}") from e

        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

        # Validate all items have an "input" key
        validated: list[dict[str, str]] = []
        for item in parsed:
            if isinstance(item, dict) and "input" in item:
                validated.append({k: str(v) for k, v in item.items()})

        return validated

    @staticmethod
    def _deduplicate(items: list[dict[str, str]]) -> list[dict[str, str]]:
        """Remove duplicate inputs.

        Args:
            items: List of test case dicts.

        Returns:
            Deduplicated list.
        """
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for item in items:
            key = item.get("input", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def generate(
        self,
        topic: str,
        count: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Generate synthetic test inputs synchronously.

        Args:
            topic: Subject domain for generation.
            count: Number of test cases to generate.
            **kwargs: Additional parameters passed to the template.

        Returns:
            List of test case dictionaries.
        """
        extra = kwargs.get("extra_instructions", "")
        prompt = render_template(self.strategy, topic, count, extra)

        raw = self._call_llm(prompt)
        items = self._parse_response(raw)
        items = self._deduplicate(items)

        logger.info(
            "generator.complete",
            generator_id=self.generator_id,
            strategy=self.strategy.value,
            requested=count,
            generated=len(items),
        )
        return items

    async def agenerate(
        self,
        topic: str,
        count: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Generate synthetic test inputs asynchronously.

        Args:
            topic: Subject domain for generation.
            count: Number of test cases to generate.
            **kwargs: Additional parameters passed to the template.

        Returns:
            List of test case dictionaries.
        """
        extra = kwargs.get("extra_instructions", "")
        prompt = render_template(self.strategy, topic, count, extra)

        raw = await self._acall_llm(prompt)
        items = self._parse_response(raw)
        items = self._deduplicate(items)

        logger.info(
            "generator.complete",
            generator_id=self.generator_id,
            strategy=self.strategy.value,
            requested=count,
            generated=len(items),
        )
        return items
