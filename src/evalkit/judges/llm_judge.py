"""LLM-as-Judge implementation with configurable rubrics.

Uses an LLM (OpenAI or Anthropic) to evaluate model outputs against
structured rubrics. The judge sends a structured prompt containing the
rubric criteria and expects a JSON response with scores and reasoning.

See ADR-001 for the rationale behind LLM-as-judge architecture.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from evalkit.core.config import LLMProviderConfig
from evalkit.core.models import JudgeScore, Rubric
from evalkit.judges.base import BaseJudge

logger = structlog.get_logger(__name__)


def _build_evaluation_prompt(
    rubric: Rubric,
    input_text: str,
    output_text: str,
    reference_text: str | None = None,
) -> str:
    """Construct the evaluation prompt sent to the judge LLM.

    Args:
        rubric: The rubric defining evaluation criteria.
        input_text: The original prompt.
        output_text: The model response to evaluate.
        reference_text: Optional reference answer.

    Returns:
        Formatted prompt string.
    """
    criteria_block = "\n".join(
        f"  - {c.name} (weight: {c.weight}, scale: {c.scale.value}): {c.description}"
        for c in rubric.criteria
    )

    reference_block = ""
    if reference_text:
        reference_block = f"""
## Reference Answer
{reference_text}
"""

    return f"""You are an expert evaluation judge. Score the following model output
against each criterion in the rubric below. Be rigorous and consistent.

## Rubric: {rubric.name}
{rubric.description}

### Criteria
{criteria_block}

## Input
{input_text}

## Model Output
{output_text}
{reference_block}
## Instructions
For each criterion, provide:
1. A numeric score within the criterion's scale
2. A brief reasoning (1-2 sentences) justifying the score

Respond with a JSON array of objects, each with keys: "criterion", "score", "reasoning".
Example:
[
  {{"criterion": "Faithfulness", "score": 4, "reasoning": "The summary is mostly accurate..."}},
  {{"criterion": "Coverage", "score": 3, "reasoning": "Misses one key point..."}}
]

Return ONLY the JSON array, no other text.
"""


def _parse_judge_response(raw_response: str, judge_id: str) -> list[dict[str, Any]]:
    """Parse the LLM judge's JSON response.

    Attempts to extract a JSON array from the response, handling cases
    where the LLM wraps the JSON in markdown code blocks.

    Args:
        raw_response: Raw text response from the LLM.
        judge_id: Judge identifier for logging.

    Returns:
        List of score dictionaries.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("judge.parse_failed", judge_id=judge_id, error=str(e))
        raise ValueError(f"Failed to parse judge response as JSON: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    return parsed  # type: ignore[return-value]


class LLMJudge(BaseJudge):
    """Evaluation judge backed by an LLM provider.

    Sends structured evaluation prompts to an LLM and parses the
    returned scores. Supports OpenAI and Anthropic providers with
    automatic retry on transient failures.

    Attributes:
        llm_config: Configuration for the LLM provider.
    """

    def __init__(
        self,
        judge_id: str,
        rubric: Rubric,
        llm_config: LLMProviderConfig | None = None,
    ) -> None:
        """Initialize the LLM judge.

        Args:
            judge_id: Unique identifier for this judge.
            rubric: Evaluation rubric.
            llm_config: LLM provider configuration. Defaults to OpenAI GPT-4o.
        """
        super().__init__(judge_id=judge_id, rubric=rubric)
        self.llm_config = llm_config or LLMProviderConfig()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the configured LLM provider.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Raw text response from the LLM.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
            EnvironmentError: If the API key is not configured.
        """
        api_key = self.llm_config.api_key

        if self.llm_config.provider == "openai":
            return self._call_openai(prompt, api_key)
        elif self.llm_config.provider == "anthropic":
            return self._call_anthropic(prompt, api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.llm_config.provider}")

    def _call_openai(self, prompt: str, api_key: str) -> str:
        """Call the OpenAI chat completions API.

        Args:
            prompt: Evaluation prompt.
            api_key: OpenAI API key.

        Returns:
            Response content string.
        """
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_config.model,
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.llm_config.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def _call_anthropic(self, prompt: str, api_key: str) -> str:
        """Call the Anthropic messages API.

        Args:
            prompt: Evaluation prompt.
            api_key: Anthropic API key.

        Returns:
            Response content string.
        """
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_config.model,
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.llm_config.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json()["content"][0]["text"])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _acall_llm(self, prompt: str) -> str:
        """Async version of _call_llm.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Raw text response from the LLM.
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
                        "temperature": self.llm_config.temperature,
                        "max_tokens": self.llm_config.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"])
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
                        "temperature": self.llm_config.temperature,
                        "max_tokens": self.llm_config.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return str(response.json()["content"][0]["text"])
            else:
                raise ValueError(f"Unsupported provider: {self.llm_config.provider}")

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate model output synchronously using the configured LLM.

        Args:
            input_text: The original prompt.
            output_text: The model response to evaluate.
            reference_text: Optional reference answer.

        Returns:
            List of JudgeScore instances, one per rubric criterion.
        """
        prompt = _build_evaluation_prompt(self.rubric, input_text, output_text, reference_text)
        raw_response = self._call_llm(prompt)
        parsed = _parse_judge_response(raw_response, self.judge_id)

        return self._scores_from_parsed(parsed, raw_response)

    async def aevaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate model output asynchronously using the configured LLM.

        Args:
            input_text: The original prompt.
            output_text: The model response to evaluate.
            reference_text: Optional reference answer.

        Returns:
            List of JudgeScore instances, one per rubric criterion.
        """
        prompt = _build_evaluation_prompt(self.rubric, input_text, output_text, reference_text)
        raw_response = await self._acall_llm(prompt)
        parsed = _parse_judge_response(raw_response, self.judge_id)

        return self._scores_from_parsed(parsed, raw_response)

    def _scores_from_parsed(
        self, parsed: list[dict[str, Any]], raw_response: str
    ) -> list[JudgeScore]:
        """Convert parsed JSON score dicts to JudgeScore model instances.

        Args:
            parsed: List of score dictionaries from the LLM response.
            raw_response: Original raw response for debugging.

        Returns:
            List of JudgeScore instances.
        """
        scores: list[JudgeScore] = []
        for item in parsed:
            scores.append(
                JudgeScore(
                    judge_id=self.judge_id,
                    criterion=item.get("criterion", "unknown"),
                    score=float(item.get("score", 0)),
                    reasoning=item.get("reasoning", ""),
                    raw_response=raw_response,
                )
            )

        logger.info(
            "judge.evaluation_complete",
            judge_id=self.judge_id,
            criteria_count=len(scores),
        )
        return scores
