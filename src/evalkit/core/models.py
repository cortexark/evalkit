"""Pydantic v2 domain models for evaluation results, scoring, and reporting.

All models use strict validation and are serializable to JSON and DuckDB-compatible
dictionaries for persistence in the analytics store.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a new unique identifier."""
    return uuid.uuid4().hex


class ScoreScale(str, Enum):
    """Supported scoring scales for rubric criteria."""

    BINARY = "binary"  # 0 or 1
    LIKERT_5 = "likert_5"  # 1-5
    LIKERT_10 = "likert_10"  # 1-10
    PERCENTAGE = "percentage"  # 0.0-1.0


class VotingStrategy(str, Enum):
    """Ensemble voting strategies."""

    MAJORITY = "majority"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"


class RubricCriteria(BaseModel):
    """A single evaluation criterion within a rubric.

    Attributes:
        name: Human-readable criterion name (e.g. "Factual Accuracy").
        description: Detailed description guiding the judge on how to score.
        weight: Relative weight of this criterion in the overall score.
        scale: The scoring scale to apply.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    scale: ScoreScale = Field(default=ScoreScale.LIKERT_5)

    @property
    def max_score(self) -> float:
        """Return the maximum possible score for this criterion's scale."""
        return {
            ScoreScale.BINARY: 1.0,
            ScoreScale.LIKERT_5: 5.0,
            ScoreScale.LIKERT_10: 10.0,
            ScoreScale.PERCENTAGE: 1.0,
        }[self.scale]


class Rubric(BaseModel):
    """A complete evaluation rubric composed of weighted criteria.

    Attributes:
        name: Name of the rubric (e.g. "Summarization Quality").
        description: Purpose and scope of the rubric.
        criteria: List of evaluation criteria.
        version: Rubric version for tracking changes over time.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    criteria: list[RubricCriteria] = Field(..., min_length=1)
    version: str = Field(default="1.0.0")

    @property
    def total_weight(self) -> float:
        """Sum of all criteria weights."""
        return sum(c.weight for c in self.criteria)


class JudgeScore(BaseModel):
    """Score produced by a single judge for one evaluation.

    Attributes:
        judge_id: Identifier of the judge that produced this score.
        criterion: The criterion being scored.
        score: Numeric score within the criterion's scale.
        reasoning: Free-text explanation of why this score was given.
        raw_response: The unprocessed response from the judge (for debugging).
        metadata: Additional key-value pairs for context.
    """

    model_config = ConfigDict(frozen=True)

    judge_id: str
    criterion: str
    score: float = Field(..., ge=0.0)
    reasoning: str = Field(default="")
    raw_response: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Complete evaluation result for a single input-output pair.

    Attributes:
        id: Unique evaluation identifier.
        model_id: Identifier of the model being evaluated.
        model_version: Version string of the model.
        input_text: The prompt or input given to the model.
        output_text: The model's response being evaluated.
        reference_text: Optional gold-standard reference for comparison.
        scores: List of individual judge scores.
        aggregate_score: Weighted aggregate across all scores.
        rubric_name: Name of the rubric used for evaluation.
        created_at: Timestamp of when the evaluation was performed.
        metadata: Additional context (e.g. dataset name, split).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    model_id: str
    model_version: str = Field(default="unknown")
    input_text: str
    output_text: str
    reference_text: str | None = Field(default=None)
    scores: list[JudgeScore] = Field(default_factory=list)
    aggregate_score: float | None = Field(default=None, ge=0.0)
    rubric_name: str = Field(default="default")
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_aggregate(self, weights: dict[str, float] | None = None) -> float:
        """Compute a weighted average across all scores.

        Args:
            weights: Optional mapping of criterion name to weight override.
                     If not provided, all scores are weighted equally.

        Returns:
            Weighted average score.

        Raises:
            ValueError: If there are no scores to aggregate.
        """
        if not self.scores:
            raise ValueError("Cannot compute aggregate with no scores.")

        if weights is None:
            total = sum(s.score for s in self.scores)
            return total / len(self.scores)

        weighted_sum = 0.0
        weight_total = 0.0
        for s in self.scores:
            w = weights.get(s.criterion, 1.0)
            weighted_sum += s.score * w
            weight_total += w

        if weight_total == 0.0:
            raise ValueError("Total weight is zero; cannot compute aggregate.")

        return weighted_sum / weight_total

    def to_storage_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary suitable for DuckDB insertion."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "reference_text": self.reference_text or "",
            "aggregate_score": self.aggregate_score,
            "rubric_name": self.rubric_name,
            "created_at": self.created_at.isoformat(),
            "scores_json": self.model_dump_json(include={"scores"}),
            "metadata_json": self.model_dump_json(include={"metadata"}),
        }


class RegressionDelta(BaseModel):
    """Score change for a single criterion between two model versions.

    Attributes:
        criterion: Name of the criterion.
        baseline_score: Average score in the baseline version.
        candidate_score: Average score in the candidate version.
        delta: Absolute change (candidate - baseline).
        relative_delta_pct: Percentage change.
        is_regression: Whether this represents a meaningful regression.
    """

    model_config = ConfigDict(frozen=True)

    criterion: str
    baseline_score: float
    candidate_score: float
    delta: float
    relative_delta_pct: float
    is_regression: bool = Field(default=False)


class RegressionReport(BaseModel):
    """Summary report comparing two model versions.

    Attributes:
        id: Unique report identifier.
        baseline_version: The reference model version.
        candidate_version: The model version being compared.
        model_id: Identifier of the model.
        rubric_name: Name of the rubric used.
        deltas: Per-criterion score deltas.
        overall_delta: Aggregate score change.
        sample_count_baseline: Number of evaluations in baseline.
        sample_count_candidate: Number of evaluations in candidate.
        has_regression: Whether any criterion shows meaningful regression.
        created_at: When the report was generated.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    baseline_version: str
    candidate_version: str
    model_id: str
    rubric_name: str = Field(default="default")
    deltas: list[RegressionDelta] = Field(default_factory=list)
    overall_delta: float = Field(default=0.0)
    sample_count_baseline: int = Field(default=0, ge=0)
    sample_count_candidate: int = Field(default=0, ge=0)
    has_regression: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
