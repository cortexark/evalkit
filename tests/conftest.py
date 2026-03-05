"""Shared fixtures for evalkit test suite."""

from __future__ import annotations

import pytest

from evalkit.core.config import EvalConfig, LLMProviderConfig, StorageConfig
from evalkit.core.models import (
    EvalResult,
    JudgeScore,
    Rubric,
    RubricCriteria,
    ScoreScale,
)
from evalkit.core.storage import DuckDBStorage


@pytest.fixture
def sample_criteria() -> list[RubricCriteria]:
    """Provide sample rubric criteria for testing."""
    return [
        RubricCriteria(
            name="Accuracy",
            description="How factually accurate is the response?",
            weight=2.0,
            scale=ScoreScale.LIKERT_5,
        ),
        RubricCriteria(
            name="Clarity",
            description="How clear and understandable is the response?",
            weight=1.0,
            scale=ScoreScale.LIKERT_5,
        ),
        RubricCriteria(
            name="Completeness",
            description="Does the response fully address the question?",
            weight=1.5,
            scale=ScoreScale.LIKERT_5,
        ),
    ]


@pytest.fixture
def sample_rubric(sample_criteria: list[RubricCriteria]) -> Rubric:
    """Provide a sample rubric for testing."""
    return Rubric(
        name="Test Rubric",
        description="A rubric for unit testing.",
        criteria=sample_criteria,
        version="1.0.0",
    )


@pytest.fixture
def sample_scores() -> list[JudgeScore]:
    """Provide sample judge scores for testing."""
    return [
        JudgeScore(
            judge_id="test-judge-1",
            criterion="Accuracy",
            score=4.0,
            reasoning="Mostly accurate with minor issues.",
        ),
        JudgeScore(
            judge_id="test-judge-1",
            criterion="Clarity",
            score=5.0,
            reasoning="Very clear and well-written.",
        ),
        JudgeScore(
            judge_id="test-judge-1",
            criterion="Completeness",
            score=3.0,
            reasoning="Misses a couple of important points.",
        ),
    ]


@pytest.fixture
def sample_eval_result(sample_scores: list[JudgeScore]) -> EvalResult:
    """Provide a sample evaluation result for testing."""
    return EvalResult(
        id="test-result-001",
        model_id="gpt-4o",
        model_version="v2024.01",
        input_text="Explain the theory of relativity.",
        output_text="The theory of relativity, proposed by Einstein...",
        reference_text="Einstein's theory of relativity consists of...",
        scores=sample_scores,
        aggregate_score=4.0,
        rubric_name="Test Rubric",
    )


@pytest.fixture
def in_memory_storage() -> DuckDBStorage:
    """Provide an in-memory DuckDB storage instance."""
    storage = DuckDBStorage(db_path=":memory:")
    yield storage
    storage.close()


@pytest.fixture
def storage_with_data(in_memory_storage: DuckDBStorage) -> DuckDBStorage:
    """Provide storage pre-populated with test data across two versions."""
    results_v1 = [
        EvalResult(
            id=f"v1-{i}",
            model_id="test-model",
            model_version="v1.0",
            input_text=f"Question {i}",
            output_text=f"Answer {i} from v1",
            aggregate_score=3.5 + (i * 0.1),
            rubric_name="Test Rubric",
        )
        for i in range(5)
    ]
    results_v2 = [
        EvalResult(
            id=f"v2-{i}",
            model_id="test-model",
            model_version="v2.0",
            input_text=f"Question {i}",
            output_text=f"Answer {i} from v2",
            aggregate_score=3.8 + (i * 0.1),
            rubric_name="Test Rubric",
        )
        for i in range(5)
    ]

    in_memory_storage.store_results(results_v1)
    in_memory_storage.store_results(results_v2)

    return in_memory_storage


@pytest.fixture
def sample_config() -> EvalConfig:
    """Provide a sample configuration for testing."""
    return EvalConfig(
        project_name="test-project",
        storage=StorageConfig(database_path=":memory:"),
    )


@pytest.fixture
def sample_llm_config() -> LLMProviderConfig:
    """Provide a sample LLM config (without a real API key)."""
    return LLMProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key_env_var="TEST_OPENAI_API_KEY",
        temperature=0.0,
        max_tokens=512,
    )
