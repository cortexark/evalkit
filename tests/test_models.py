"""Tests for core Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalkit.core.models import (
    EvalResult,
    JudgeScore,
    RegressionDelta,
    RegressionReport,
    Rubric,
    RubricCriteria,
    ScoreScale,
    VotingStrategy,
)


class TestRubricCriteria:
    """Tests for the RubricCriteria model."""

    def test_create_with_defaults(self) -> None:
        criteria = RubricCriteria(name="Test", description="A test criterion")
        assert criteria.name == "Test"
        assert criteria.weight == 1.0
        assert criteria.scale == ScoreScale.LIKERT_5

    def test_max_score_likert_5(self) -> None:
        criteria = RubricCriteria(name="Test", description="desc", scale=ScoreScale.LIKERT_5)
        assert criteria.max_score == 5.0

    def test_max_score_binary(self) -> None:
        criteria = RubricCriteria(name="Test", description="desc", scale=ScoreScale.BINARY)
        assert criteria.max_score == 1.0

    def test_max_score_percentage(self) -> None:
        criteria = RubricCriteria(name="Test", description="desc", scale=ScoreScale.PERCENTAGE)
        assert criteria.max_score == 1.0

    def test_max_score_likert_10(self) -> None:
        criteria = RubricCriteria(name="Test", description="desc", scale=ScoreScale.LIKERT_10)
        assert criteria.max_score == 10.0

    def test_frozen_model(self) -> None:
        criteria = RubricCriteria(name="Test", description="desc")
        with pytest.raises(ValidationError):
            criteria.name = "Changed"  # type: ignore[misc]

    def test_invalid_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            RubricCriteria(name="", description="desc")

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RubricCriteria(name="Test", description="desc", weight=-1.0)

        with pytest.raises(ValidationError):
            RubricCriteria(name="Test", description="desc", weight=101.0)


class TestRubric:
    """Tests for the Rubric model."""

    def test_create_rubric(self, sample_criteria: list[RubricCriteria]) -> None:
        rubric = Rubric(name="Test Rubric", criteria=sample_criteria)
        assert rubric.name == "Test Rubric"
        assert len(rubric.criteria) == 3

    def test_total_weight(self, sample_criteria: list[RubricCriteria]) -> None:
        rubric = Rubric(name="Test Rubric", criteria=sample_criteria)
        assert rubric.total_weight == pytest.approx(4.5)

    def test_empty_criteria_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Rubric(name="Empty", criteria=[])

    def test_rubric_version(self, sample_criteria: list[RubricCriteria]) -> None:
        rubric = Rubric(name="Versioned", criteria=sample_criteria, version="2.0.0")
        assert rubric.version == "2.0.0"


class TestJudgeScore:
    """Tests for the JudgeScore model."""

    def test_create_score(self) -> None:
        score = JudgeScore(
            judge_id="judge-1",
            criterion="Accuracy",
            score=4.5,
            reasoning="Good accuracy.",
        )
        assert score.judge_id == "judge-1"
        assert score.score == 4.5

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="j", criterion="c", score=-1.0)

    def test_default_metadata(self) -> None:
        score = JudgeScore(judge_id="j", criterion="c", score=3.0)
        assert score.metadata == {}
        assert score.reasoning == ""

    def test_with_metadata(self) -> None:
        score = JudgeScore(
            judge_id="j",
            criterion="c",
            score=3.0,
            metadata={"latency_ms": 120},
        )
        assert score.metadata["latency_ms"] == 120


class TestEvalResult:
    """Tests for the EvalResult model."""

    def test_create_result(self, sample_eval_result: EvalResult) -> None:
        assert sample_eval_result.model_id == "gpt-4o"
        assert len(sample_eval_result.scores) == 3

    def test_auto_generated_id(self) -> None:
        result = EvalResult(
            model_id="m",
            input_text="in",
            output_text="out",
        )
        assert len(result.id) == 32  # hex uuid4

    def test_compute_aggregate_equal_weights(self, sample_eval_result: EvalResult) -> None:
        avg = sample_eval_result.compute_aggregate()
        assert avg == pytest.approx(4.0)  # (4 + 5 + 3) / 3

    def test_compute_aggregate_custom_weights(self, sample_eval_result: EvalResult) -> None:
        weights = {"Accuracy": 2.0, "Clarity": 1.0, "Completeness": 1.0}
        avg = sample_eval_result.compute_aggregate(weights)
        # (4*2 + 5*1 + 3*1) / (2+1+1) = 16/4 = 4.0
        assert avg == pytest.approx(4.0)

    def test_compute_aggregate_no_scores(self) -> None:
        result = EvalResult(model_id="m", input_text="in", output_text="out", scores=[])
        with pytest.raises(ValueError, match="no scores"):
            result.compute_aggregate()

    def test_to_storage_dict(self, sample_eval_result: EvalResult) -> None:
        d = sample_eval_result.to_storage_dict()
        assert d["model_id"] == "gpt-4o"
        assert d["id"] == "test-result-001"
        assert "scores_json" in d
        assert "metadata_json" in d

    def test_default_version(self) -> None:
        result = EvalResult(model_id="m", input_text="in", output_text="out")
        assert result.model_version == "unknown"


class TestRegressionDelta:
    """Tests for the RegressionDelta model."""

    def test_create_delta(self) -> None:
        delta = RegressionDelta(
            criterion="Accuracy",
            baseline_score=4.0,
            candidate_score=3.5,
            delta=-0.5,
            relative_delta_pct=-12.5,
            is_regression=True,
        )
        assert delta.is_regression is True
        assert delta.delta == -0.5

    def test_positive_delta_not_regression(self) -> None:
        delta = RegressionDelta(
            criterion="Clarity",
            baseline_score=3.0,
            candidate_score=4.0,
            delta=1.0,
            relative_delta_pct=33.33,
            is_regression=False,
        )
        assert delta.is_regression is False


class TestRegressionReport:
    """Tests for the RegressionReport model."""

    def test_create_report(self) -> None:
        report = RegressionReport(
            baseline_version="v1.0",
            candidate_version="v2.0",
            model_id="test-model",
        )
        assert report.baseline_version == "v1.0"
        assert report.has_regression is False

    def test_report_with_regression(self) -> None:
        delta = RegressionDelta(
            criterion="Accuracy",
            baseline_score=4.0,
            candidate_score=3.0,
            delta=-1.0,
            relative_delta_pct=-25.0,
            is_regression=True,
        )
        report = RegressionReport(
            baseline_version="v1.0",
            candidate_version="v2.0",
            model_id="test-model",
            deltas=[delta],
            overall_delta=-1.0,
            has_regression=True,
        )
        assert report.has_regression is True
        assert len(report.deltas) == 1


class TestEnums:
    """Tests for enum types."""

    def test_score_scale_values(self) -> None:
        assert ScoreScale.BINARY.value == "binary"
        assert ScoreScale.LIKERT_5.value == "likert_5"

    def test_voting_strategy_values(self) -> None:
        assert VotingStrategy.MAJORITY.value == "majority"
        assert VotingStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert VotingStrategy.UNANIMOUS.value == "unanimous"
