"""Tests for judge engine: base, LLM judge, ensemble, and rubrics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from evalkit.core.models import (
    JudgeScore,
    Rubric,
    RubricCriteria,
    ScoreScale,
    VotingStrategy,
)
from evalkit.judges.base import BaseJudge
from evalkit.judges.ensemble import EnsembleJudge
from evalkit.judges.llm_judge import (
    LLMJudge,
    _build_evaluation_prompt,
    _parse_judge_response,
)
from evalkit.judges.rubrics import (
    FACTUAL_ACCURACY_RUBRIC,
    HELPFULNESS_RUBRIC,
    SAFETY_RUBRIC,
    SUMMARIZATION_RUBRIC,
    build_rubric,
)


# ---------------------------------------------------------------------------
# Concrete stub for testing BaseJudge
# ---------------------------------------------------------------------------


class StubJudge(BaseJudge):
    """Concrete judge returning fixed scores for testing."""

    def __init__(
        self,
        judge_id: str,
        rubric: Rubric,
        fixed_scores: list[JudgeScore] | None = None,
    ) -> None:
        super().__init__(judge_id=judge_id, rubric=rubric)
        self._fixed_scores = fixed_scores or []

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        return self._fixed_scores

    async def aevaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        return self._fixed_scores


class TestBaseJudge:
    """Tests for the BaseJudge abstract class."""

    def test_cannot_instantiate_abstract(self, sample_rubric: Rubric) -> None:
        with pytest.raises(TypeError):
            BaseJudge(judge_id="x", rubric=sample_rubric)  # type: ignore[abstract]

    def test_stub_evaluate(self, sample_rubric: Rubric, sample_scores: list[JudgeScore]) -> None:
        judge = StubJudge("stub-1", sample_rubric, sample_scores)
        scores = judge.evaluate("in", "out")
        assert len(scores) == 3
        assert scores[0].criterion == "Accuracy"

    def test_repr(self, sample_rubric: Rubric) -> None:
        judge = StubJudge("stub-1", sample_rubric)
        assert "StubJudge" in repr(judge)
        assert "stub-1" in repr(judge)

    def test_evaluate_result(self, sample_rubric: Rubric, sample_scores: list[JudgeScore]) -> None:
        from evalkit.core.models import EvalResult

        judge = StubJudge("stub-1", sample_rubric, sample_scores)
        result = EvalResult(model_id="m", input_text="in", output_text="out")
        scores = judge.evaluate_result(result)
        assert len(scores) == 3


class TestBuildRubric:
    """Tests for the build_rubric utility."""

    def test_build_simple_rubric(self) -> None:
        rubric = build_rubric(
            "Simple",
            [
                {"name": "A", "description": "Criterion A"},
                {"name": "B", "description": "Criterion B", "weight": 2.0},
            ],
        )
        assert rubric.name == "Simple"
        assert len(rubric.criteria) == 2
        assert rubric.criteria[1].weight == 2.0

    def test_build_rubric_custom_scale(self) -> None:
        rubric = build_rubric(
            "Binary",
            [{"name": "Pass", "description": "Does it pass?"}],
            scale=ScoreScale.BINARY,
        )
        assert rubric.criteria[0].scale == ScoreScale.BINARY


class TestPrebuiltRubrics:
    """Tests for pre-built rubric templates."""

    @pytest.mark.parametrize(
        "rubric",
        [SUMMARIZATION_RUBRIC, FACTUAL_ACCURACY_RUBRIC, HELPFULNESS_RUBRIC, SAFETY_RUBRIC],
        ids=["summarization", "factual_accuracy", "helpfulness", "safety"],
    )
    def test_rubric_has_criteria(self, rubric: Rubric) -> None:
        assert len(rubric.criteria) >= 2
        assert rubric.name
        assert rubric.total_weight > 0


class TestParseJudgeResponse:
    """Tests for LLM response parsing."""

    def test_parse_clean_json(self) -> None:
        raw = json.dumps(
            [
                {"criterion": "A", "score": 4, "reasoning": "Good"},
                {"criterion": "B", "score": 3, "reasoning": "OK"},
            ]
        )
        result = _parse_judge_response(raw, "test")
        assert len(result) == 2

    def test_parse_json_in_code_block(self) -> None:
        raw = '```json\n[{"criterion": "A", "score": 5, "reasoning": "Great"}]\n```'
        result = _parse_judge_response(raw, "test")
        assert len(result) == 1
        assert result[0]["score"] == 5

    def test_parse_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_judge_response("not json at all", "test")

    def test_parse_non_array(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_judge_response('{"key": "value"}', "test")


class TestBuildEvaluationPrompt:
    """Tests for evaluation prompt construction."""

    def test_prompt_contains_rubric_name(self, sample_rubric: Rubric) -> None:
        prompt = _build_evaluation_prompt(sample_rubric, "input text", "output text")
        assert "Test Rubric" in prompt
        assert "input text" in prompt
        assert "output text" in prompt

    def test_prompt_contains_criteria(self, sample_rubric: Rubric) -> None:
        prompt = _build_evaluation_prompt(sample_rubric, "in", "out")
        assert "Accuracy" in prompt
        assert "Clarity" in prompt

    def test_prompt_with_reference(self, sample_rubric: Rubric) -> None:
        prompt = _build_evaluation_prompt(sample_rubric, "in", "out", "reference answer")
        assert "Reference Answer" in prompt
        assert "reference answer" in prompt

    def test_prompt_without_reference(self, sample_rubric: Rubric) -> None:
        prompt = _build_evaluation_prompt(sample_rubric, "in", "out")
        assert "Reference Answer" not in prompt


class TestLLMJudge:
    """Tests for the LLMJudge class (mocked LLM calls)."""

    def test_init_default_config(self, sample_rubric: Rubric) -> None:
        judge = LLMJudge(judge_id="llm-1", rubric=sample_rubric)
        assert judge.judge_id == "llm-1"
        assert judge.llm_config.provider == "openai"

    @patch.object(LLMJudge, "_call_llm")
    def test_evaluate_calls_llm(self, mock_call: MagicMock, sample_rubric: Rubric) -> None:
        mock_call.return_value = json.dumps(
            [
                {"criterion": "Accuracy", "score": 4, "reasoning": "Good"},
                {"criterion": "Clarity", "score": 5, "reasoning": "Clear"},
                {"criterion": "Completeness", "score": 3, "reasoning": "OK"},
            ]
        )
        judge = LLMJudge(judge_id="llm-1", rubric=sample_rubric)
        scores = judge.evaluate("What is Python?", "Python is a language...")
        assert len(scores) == 3
        assert scores[0].judge_id == "llm-1"
        assert scores[0].score == 4.0
        mock_call.assert_called_once()

    @patch.object(LLMJudge, "_call_llm")
    def test_evaluate_preserves_raw_response(
        self, mock_call: MagicMock, sample_rubric: Rubric
    ) -> None:
        raw = json.dumps([{"criterion": "A", "score": 3, "reasoning": "r"}])
        mock_call.return_value = raw
        judge = LLMJudge(judge_id="llm-1", rubric=sample_rubric)
        scores = judge.evaluate("in", "out")
        assert scores[0].raw_response == raw


class TestEnsembleJudge:
    """Tests for the EnsembleJudge class."""

    def _make_stub(self, judge_id: str, rubric: Rubric, scores: dict[str, float]) -> StubJudge:
        """Create a StubJudge with fixed scores per criterion."""
        judge_scores = [
            JudgeScore(judge_id=judge_id, criterion=c, score=s, reasoning=f"r-{c}")
            for c, s in scores.items()
        ]
        return StubJudge(judge_id, rubric, judge_scores)

    def test_weighted_average(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0, "B": 3.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 2.0, "B": 5.0})

        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
            judges=[(j1, 1.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )
        scores = ensemble.evaluate("in", "out")
        score_map = {s.criterion: s.score for s in scores}
        assert score_map["A"] == pytest.approx(3.0)  # (4+2)/2
        assert score_map["B"] == pytest.approx(4.0)  # (3+5)/2

    def test_weighted_average_with_weights(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 2.0})

        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
            judges=[(j1, 3.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )
        scores = ensemble.evaluate("in", "out")
        # (4*3 + 2*1) / (3+1) = 14/4 = 3.5
        assert scores[0].score == pytest.approx(3.5)

    def test_majority_vote(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 4.0})
        j3 = self._make_stub("j3", sample_rubric, {"A": 2.0})

        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
            judges=[(j1, 1.0), (j2, 1.0), (j3, 1.0)],
            voting_strategy=VotingStrategy.MAJORITY,
        )
        scores = ensemble.evaluate("in", "out")
        assert scores[0].score == 4.0

    def test_majority_vote_tie_breaks_high(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 3.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 5.0})

        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
            judges=[(j1, 1.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.MAJORITY,
        )
        scores = ensemble.evaluate("in", "out")
        assert scores[0].score == 5.0  # tie -> higher score

    def test_unanimous_returns_minimum(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 2.0})
        j3 = self._make_stub("j3", sample_rubric, {"A": 5.0})

        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
            judges=[(j1, 1.0), (j2, 1.0), (j3, 1.0)],
            voting_strategy=VotingStrategy.UNANIMOUS,
        )
        scores = ensemble.evaluate("in", "out")
        assert scores[0].score == 2.0

    def test_empty_ensemble_raises(self, sample_rubric: Rubric) -> None:
        ensemble = EnsembleJudge(
            judge_id="ens-empty",
            rubric=sample_rubric,
            judges=[],
        )
        with pytest.raises(ValueError, match="no judges"):
            ensemble.evaluate("in", "out")

    def test_add_judge(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0})
        ensemble = EnsembleJudge(
            judge_id="ens-1",
            rubric=sample_rubric,
        )
        ensemble.add_judge(j1, weight=2.0)
        assert len(ensemble.judges) == 1
        assert ensemble.judges[0][1] == 2.0

    @pytest.mark.asyncio
    async def test_async_weighted_average(self, sample_rubric: Rubric) -> None:
        j1 = self._make_stub("j1", sample_rubric, {"A": 4.0, "B": 3.0})
        j2 = self._make_stub("j2", sample_rubric, {"A": 2.0, "B": 5.0})

        ensemble = EnsembleJudge(
            judge_id="ens-async",
            rubric=sample_rubric,
            judges=[(j1, 1.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )
        scores = await ensemble.aevaluate("in", "out")
        score_map = {s.criterion: s.score for s in scores}
        assert score_map["A"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_async_empty_ensemble_raises(self, sample_rubric: Rubric) -> None:
        ensemble = EnsembleJudge(
            judge_id="ens-empty",
            rubric=sample_rubric,
            judges=[],
        )
        with pytest.raises(ValueError, match="no judges"):
            await ensemble.aevaluate("in", "out")
