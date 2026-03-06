"""API Contract Tests -- validates public interfaces, signatures, types, and Pydantic validation.

Tests every public class and function to ensure correct parameter handling,
return types, input validation, and edge-case rejection.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from evalkit.core.config import (
    EnsembleConfig,
    EvalConfig,
    JudgeConfig,
    LLMProviderConfig,
    StorageConfig,
)
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
from evalkit.core.storage import DuckDBStorage
from evalkit.generators.base import BaseGenerator
from evalkit.generators.synthetic import SyntheticGenerator
from evalkit.generators.templates import GenerationStrategy, render_template
from evalkit.judges.base import BaseJudge
from evalkit.judges.ensemble import EnsembleJudge
from evalkit.judges.llm_judge import LLMJudge, _build_evaluation_prompt, _parse_judge_response
from evalkit.judges.rubrics import (
    FACTUAL_ACCURACY_RUBRIC,
    HELPFULNESS_RUBRIC,
    SAFETY_RUBRIC,
    SUMMARIZATION_RUBRIC,
    build_rubric,
)
from evalkit.regression.comparator import ComparisonMethod, ComparisonResult, OutputComparator
from evalkit.regression.reporter import RegressionReporter
from evalkit.regression.tracker import RegressionTracker


# ===========================================================================
# Section 1: Core Models -- Parameter validation and type contracts
# ===========================================================================


class TestRubricCriteriaContract:
    """Validate RubricCriteria Pydantic contracts."""

    def test_required_fields_name_and_description(self) -> None:
        """name and description are required; omitting either raises ValidationError."""
        with pytest.raises(ValidationError):
            RubricCriteria(description="desc")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            RubricCriteria(name="Name")  # type: ignore[call-arg]

    def test_name_max_length_boundary(self) -> None:
        """Name at max 200 chars succeeds; 201 chars fails."""
        ok = RubricCriteria(name="A" * 200, description="d")
        assert len(ok.name) == 200
        with pytest.raises(ValidationError):
            RubricCriteria(name="A" * 201, description="d")

    def test_name_min_length_boundary(self) -> None:
        """Name of 1 char succeeds; empty string fails."""
        ok = RubricCriteria(name="X", description="d")
        assert ok.name == "X"
        with pytest.raises(ValidationError):
            RubricCriteria(name="", description="d")

    def test_description_min_length(self) -> None:
        """Empty description is rejected."""
        with pytest.raises(ValidationError):
            RubricCriteria(name="N", description="")

    def test_weight_exact_boundaries(self) -> None:
        """Weight at 0.0 and 100.0 are valid; -0.001 and 100.001 fail."""
        c0 = RubricCriteria(name="N", description="d", weight=0.0)
        assert c0.weight == 0.0
        c100 = RubricCriteria(name="N", description="d", weight=100.0)
        assert c100.weight == 100.0
        with pytest.raises(ValidationError):
            RubricCriteria(name="N", description="d", weight=-0.001)
        with pytest.raises(ValidationError):
            RubricCriteria(name="N", description="d", weight=100.001)

    def test_scale_default_is_likert_5(self) -> None:
        c = RubricCriteria(name="N", description="d")
        assert c.scale == ScoreScale.LIKERT_5

    def test_all_score_scales_accepted(self) -> None:
        for scale in ScoreScale:
            c = RubricCriteria(name="N", description="d", scale=scale)
            assert c.scale == scale

    def test_invalid_scale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RubricCriteria(name="N", description="d", scale="invalid_scale")  # type: ignore[arg-type]

    def test_max_score_property_all_scales(self) -> None:
        expected = {
            ScoreScale.BINARY: 1.0,
            ScoreScale.LIKERT_5: 5.0,
            ScoreScale.LIKERT_10: 10.0,
            ScoreScale.PERCENTAGE: 1.0,
        }
        for scale, expected_max in expected.items():
            c = RubricCriteria(name="N", description="d", scale=scale)
            assert c.max_score == expected_max

    def test_frozen_prevents_mutation(self) -> None:
        c = RubricCriteria(name="N", description="d")
        with pytest.raises(ValidationError):
            c.weight = 5.0  # type: ignore[misc]


class TestRubricContract:
    """Validate Rubric model contracts."""

    def test_criteria_minimum_one(self) -> None:
        with pytest.raises(ValidationError):
            Rubric(name="R", criteria=[])

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            Rubric(criteria=[RubricCriteria(name="C", description="d")])  # type: ignore[call-arg]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Rubric(name="", criteria=[RubricCriteria(name="C", description="d")])

    def test_default_version(self) -> None:
        r = Rubric(name="R", criteria=[RubricCriteria(name="C", description="d")])
        assert r.version == "1.0.0"

    def test_total_weight_single_criterion(self) -> None:
        r = Rubric(name="R", criteria=[RubricCriteria(name="C", description="d", weight=3.5)])
        assert r.total_weight == pytest.approx(3.5)

    def test_total_weight_multiple_criteria(self) -> None:
        r = Rubric(
            name="R",
            criteria=[
                RubricCriteria(name="A", description="d", weight=1.0),
                RubricCriteria(name="B", description="d", weight=2.5),
                RubricCriteria(name="C", description="d", weight=0.5),
            ],
        )
        assert r.total_weight == pytest.approx(4.0)


class TestJudgeScoreContract:
    """Validate JudgeScore model contracts."""

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="j", criterion="c")  # type: ignore[call-arg]  # missing score

    def test_score_zero_valid(self) -> None:
        s = JudgeScore(judge_id="j", criterion="c", score=0.0)
        assert s.score == 0.0

    def test_score_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(judge_id="j", criterion="c", score=-0.01)

    def test_very_large_score_accepted(self) -> None:
        s = JudgeScore(judge_id="j", criterion="c", score=99999.0)
        assert s.score == 99999.0

    def test_default_fields(self) -> None:
        s = JudgeScore(judge_id="j", criterion="c", score=1.0)
        assert s.reasoning == ""
        assert s.raw_response == ""
        assert s.metadata == {}

    def test_metadata_accepts_nested(self) -> None:
        s = JudgeScore(
            judge_id="j",
            criterion="c",
            score=1.0,
            metadata={"nested": {"key": "value"}, "list": [1, 2, 3]},
        )
        assert s.metadata["nested"]["key"] == "value"


class TestEvalResultContract:
    """Validate EvalResult model contracts."""

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            EvalResult(model_id="m", input_text="i")  # type: ignore[call-arg]  # missing output_text

    def test_auto_id_generation(self) -> None:
        r1 = EvalResult(model_id="m", input_text="i", output_text="o")
        r2 = EvalResult(model_id="m", input_text="i", output_text="o")
        assert r1.id != r2.id
        assert len(r1.id) == 32

    def test_auto_timestamp(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o")
        assert isinstance(r.created_at, datetime)
        assert r.created_at.tzinfo is not None  # should be timezone-aware

    def test_default_values(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o")
        assert r.model_version == "unknown"
        assert r.reference_text is None
        assert r.scores == []
        assert r.aggregate_score is None
        assert r.rubric_name == "default"
        assert r.metadata == {}

    def test_aggregate_score_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvalResult(
                model_id="m",
                input_text="i",
                output_text="o",
                aggregate_score=-0.01,
            )

    def test_aggregate_score_zero_valid(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o", aggregate_score=0.0)
        assert r.aggregate_score == 0.0

    def test_compute_aggregate_no_weights(self) -> None:
        scores = [
            JudgeScore(judge_id="j", criterion="A", score=2.0),
            JudgeScore(judge_id="j", criterion="B", score=4.0),
        ]
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=scores)
        assert r.compute_aggregate() == pytest.approx(3.0)

    def test_compute_aggregate_with_weights(self) -> None:
        scores = [
            JudgeScore(judge_id="j", criterion="A", score=2.0),
            JudgeScore(judge_id="j", criterion="B", score=4.0),
        ]
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=scores)
        result = r.compute_aggregate(weights={"A": 3.0, "B": 1.0})
        # (2*3 + 4*1) / (3+1) = 10/4 = 2.5
        assert result == pytest.approx(2.5)

    def test_compute_aggregate_empty_raises(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=[])
        with pytest.raises(ValueError, match="no scores"):
            r.compute_aggregate()

    def test_compute_aggregate_zero_weight_raises(self) -> None:
        scores = [JudgeScore(judge_id="j", criterion="A", score=2.0)]
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=scores)
        with pytest.raises(ValueError, match="zero"):
            r.compute_aggregate(weights={"A": 0.0})

    def test_compute_aggregate_missing_criterion_weight(self) -> None:
        """Criteria not in weights dict get weight 1.0."""
        scores = [
            JudgeScore(judge_id="j", criterion="A", score=2.0),
            JudgeScore(judge_id="j", criterion="B", score=4.0),
        ]
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=scores)
        result = r.compute_aggregate(weights={"A": 2.0})
        # A: 2*2=4, B: 4*1=4, total_weight=3 => 8/3 ~ 2.6667
        assert result == pytest.approx(8.0 / 3.0)

    def test_to_storage_dict_keys(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o")
        d = r.to_storage_dict()
        expected_keys = {
            "id",
            "model_id",
            "model_version",
            "input_text",
            "output_text",
            "reference_text",
            "aggregate_score",
            "rubric_name",
            "created_at",
            "scores_json",
            "metadata_json",
        }
        assert set(d.keys()) == expected_keys

    def test_to_storage_dict_reference_text_none_becomes_empty(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o", reference_text=None)
        d = r.to_storage_dict()
        assert d["reference_text"] == ""

    def test_to_storage_dict_created_at_is_isoformat(self) -> None:
        r = EvalResult(model_id="m", input_text="i", output_text="o")
        d = r.to_storage_dict()
        # Should be parseable as isoformat
        datetime.fromisoformat(d["created_at"])

    def test_to_storage_dict_scores_json_is_valid(self) -> None:
        scores = [JudgeScore(judge_id="j", criterion="A", score=3.0)]
        r = EvalResult(model_id="m", input_text="i", output_text="o", scores=scores)
        d = r.to_storage_dict()
        parsed = json.loads(d["scores_json"])
        assert "scores" in parsed
        assert len(parsed["scores"]) == 1


class TestRegressionDeltaContract:
    """Validate RegressionDelta model contracts."""

    def test_all_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RegressionDelta(criterion="A")  # type: ignore[call-arg]

    def test_default_is_regression_false(self) -> None:
        d = RegressionDelta(
            criterion="A",
            baseline_score=3.0,
            candidate_score=3.5,
            delta=0.5,
            relative_delta_pct=16.67,
        )
        assert d.is_regression is False


class TestRegressionReportContract:
    """Validate RegressionReport model contracts."""

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RegressionReport(baseline_version="v1")  # type: ignore[call-arg]

    def test_default_values(self) -> None:
        r = RegressionReport(
            baseline_version="v1",
            candidate_version="v2",
            model_id="m",
        )
        assert r.deltas == []
        assert r.overall_delta == 0.0
        assert r.sample_count_baseline == 0
        assert r.sample_count_candidate == 0
        assert r.has_regression is False

    def test_sample_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            RegressionReport(
                baseline_version="v1",
                candidate_version="v2",
                model_id="m",
                sample_count_baseline=-1,
            )


# ===========================================================================
# Section 2: Config Module Contracts
# ===========================================================================


class TestLLMProviderConfigContract:
    """Validate LLMProviderConfig contracts."""

    def test_default_values(self) -> None:
        c = LLMProviderConfig()
        assert c.provider == "openai"
        assert c.model == "gpt-4o"
        assert c.api_key_env_var == "OPENAI_API_KEY"
        assert c.temperature == 0.0
        assert c.max_tokens == 1024
        assert c.timeout_seconds == 60

    def test_temperature_bounds(self) -> None:
        LLMProviderConfig(temperature=0.0)
        LLMProviderConfig(temperature=2.0)
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMProviderConfig(temperature=2.1)

    def test_max_tokens_minimum(self) -> None:
        LLMProviderConfig(max_tokens=1)
        with pytest.raises(ValidationError):
            LLMProviderConfig(max_tokens=0)

    def test_timeout_minimum(self) -> None:
        LLMProviderConfig(timeout_seconds=1)
        with pytest.raises(ValidationError):
            LLMProviderConfig(timeout_seconds=0)

    def test_api_key_property_reads_env(self) -> None:
        c = LLMProviderConfig(api_key_env_var="TEST_EVALKIT_KEY_12345")
        os.environ["TEST_EVALKIT_KEY_12345"] = "sk-test-key"
        try:
            assert c.api_key == "sk-test-key"
        finally:
            del os.environ["TEST_EVALKIT_KEY_12345"]

    def test_api_key_missing_raises(self) -> None:
        c = LLMProviderConfig(api_key_env_var="DEFINITELY_NOT_SET_EVALKIT_XYZ")
        # Make sure it's not set
        os.environ.pop("DEFINITELY_NOT_SET_EVALKIT_XYZ", None)
        with pytest.raises(EnvironmentError, match="API key not found"):
            _ = c.api_key


class TestEvalConfigContract:
    """Validate EvalConfig contracts."""

    def test_default_values(self) -> None:
        c = EvalConfig()
        assert c.project_name == "evalkit"
        assert c.default_model_id == "default-model"
        assert isinstance(c.ensemble, EnsembleConfig)
        assert isinstance(c.storage, StorageConfig)

    def test_from_yaml_loads_valid_config(self) -> None:
        data = {
            "project_name": "test-proj",
            "default_model_id": "my-model",
            "storage": {"database_path": ":memory:"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(data, f)
            path = f.name

        config = EvalConfig.from_yaml(path)
        assert config.project_name == "test-proj"
        assert config.default_model_id == "my-model"
        Path(path).unlink()

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            EvalConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_invalid_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("just a string, not a mapping")
            path = f.name

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            EvalConfig.from_yaml(path)
        Path(path).unlink()

    def test_to_yaml_roundtrip(self) -> None:
        config = EvalConfig(project_name="roundtrip-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            loaded = EvalConfig.from_yaml(path)
            assert loaded.project_name == "roundtrip-test"

    def test_to_yaml_creates_parent_dirs(self) -> None:
        config = EvalConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "config.yaml"
            config.to_yaml(path)
            assert path.exists()

    def test_from_yaml_with_ensemble_config(self) -> None:
        data = {
            "project_name": "test",
            "ensemble": {
                "voting_strategy": "majority",
                "regression_threshold": -0.2,
                "judges": [
                    {"judge_id": "j1", "judge_type": "llm", "weight": 2.0},
                ],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(data, f)
            path = f.name

        config = EvalConfig.from_yaml(path)
        assert config.ensemble.voting_strategy == VotingStrategy.MAJORITY
        assert config.ensemble.regression_threshold == -0.2
        assert len(config.ensemble.judges) == 1
        assert config.ensemble.judges[0].weight == 2.0
        Path(path).unlink()


class TestStorageConfigContract:
    """Validate StorageConfig contracts."""

    def test_default_is_memory(self) -> None:
        c = StorageConfig()
        assert c.database_path == ":memory:"


class TestJudgeConfigContract:
    """Validate JudgeConfig contracts."""

    def test_required_judge_id(self) -> None:
        with pytest.raises(ValidationError):
            JudgeConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        c = JudgeConfig(judge_id="j1")
        assert c.judge_type == "llm"
        assert c.rubric_name == "default"
        assert c.weight == 1.0


class TestEnsembleConfigContract:
    """Validate EnsembleConfig contracts."""

    def test_defaults(self) -> None:
        c = EnsembleConfig()
        assert c.judges == []
        assert c.voting_strategy == VotingStrategy.WEIGHTED_AVERAGE
        assert c.regression_threshold == -0.1


# ===========================================================================
# Section 3: Storage Contracts
# ===========================================================================


class TestDuckDBStorageContract:
    """Validate DuckDBStorage public interface."""

    def test_init_signature(self) -> None:
        sig = inspect.signature(DuckDBStorage.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "db_path" in params

    def test_store_result_return_type(self) -> None:
        with DuckDBStorage() as s:
            r = EvalResult(model_id="m", input_text="i", output_text="o")
            result = s.store_result(r)
            assert result is None  # returns None

    def test_store_results_return_type(self) -> None:
        with DuckDBStorage() as s:
            result = s.store_results([])
            assert result is None

    def test_get_results_return_type(self) -> None:
        with DuckDBStorage() as s:
            results = s.get_results()
            assert isinstance(results, list)

    def test_count_results_return_type(self) -> None:
        with DuckDBStorage() as s:
            count = s.count_results()
            assert isinstance(count, int)

    def test_get_distinct_versions_return_type(self) -> None:
        with DuckDBStorage() as s:
            versions = s.get_distinct_versions("m")
            assert isinstance(versions, list)

    def test_context_manager_protocol(self) -> None:
        s = DuckDBStorage()
        assert hasattr(s, "__enter__")
        assert hasattr(s, "__exit__")
        s.close()


# ===========================================================================
# Section 4: Judge Contracts
# ===========================================================================


class TestBaseJudgeContract:
    """Validate BaseJudge abstract interface."""

    def test_is_abstract(self) -> None:
        assert inspect.isabstract(BaseJudge)

    def test_abstract_methods(self) -> None:
        abstract_methods = set()
        for name, method in inspect.getmembers(BaseJudge):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)
        assert "evaluate" in abstract_methods
        assert "aevaluate" in abstract_methods

    def test_evaluate_result_is_concrete(self) -> None:
        assert not getattr(BaseJudge.evaluate_result, "__isabstractmethod__", False)


class TestLLMJudgeContract:
    """Validate LLMJudge public interface."""

    def test_init_signature(self) -> None:
        sig = inspect.signature(LLMJudge.__init__)
        params = list(sig.parameters.keys())
        assert "judge_id" in params
        assert "rubric" in params
        assert "llm_config" in params

    def test_default_llm_config(self) -> None:
        rubric = Rubric(name="R", criteria=[RubricCriteria(name="C", description="d")])
        judge = LLMJudge(judge_id="j1", rubric=rubric)
        assert judge.llm_config.provider == "openai"

    def test_evaluate_return_type(self) -> None:
        rubric = Rubric(name="R", criteria=[RubricCriteria(name="C", description="d")])
        judge = LLMJudge(judge_id="j1", rubric=rubric)
        with patch.object(LLMJudge, "_call_llm") as mock:
            mock.return_value = json.dumps([{"criterion": "C", "score": 4, "reasoning": "ok"}])
            result = judge.evaluate("in", "out")
            assert isinstance(result, list)
            assert all(isinstance(s, JudgeScore) for s in result)


class TestEnsembleJudgeContract:
    """Validate EnsembleJudge public interface."""

    def test_init_signature(self) -> None:
        sig = inspect.signature(EnsembleJudge.__init__)
        params = list(sig.parameters.keys())
        assert "judge_id" in params
        assert "rubric" in params
        assert "judges" in params
        assert "voting_strategy" in params

    def test_add_judge_signature(self) -> None:
        sig = inspect.signature(EnsembleJudge.add_judge)
        params = list(sig.parameters.keys())
        assert "judge" in params
        assert "weight" in params


# ===========================================================================
# Section 5: Rubrics Contracts
# ===========================================================================


class TestBuildRubricContract:
    """Validate build_rubric factory function."""

    def test_signature(self) -> None:
        sig = inspect.signature(build_rubric)
        params = list(sig.parameters.keys())
        assert "name" in params
        assert "criteria_defs" in params
        assert "description" in params
        assert "scale" in params
        assert "version" in params

    def test_return_type(self) -> None:
        r = build_rubric("Test", [{"name": "C", "description": "d"}])
        assert isinstance(r, Rubric)

    def test_custom_version(self) -> None:
        r = build_rubric("Test", [{"name": "C", "description": "d"}], version="2.0.0")
        assert r.version == "2.0.0"

    def test_custom_description(self) -> None:
        r = build_rubric("Test", [{"name": "C", "description": "d"}], description="My desc")
        assert r.description == "My desc"

    def test_weight_from_criteria_defs(self) -> None:
        r = build_rubric(
            "Test",
            [
                {"name": "A", "description": "d", "weight": 3.0},
                {"name": "B", "description": "d"},
            ],
        )
        assert r.criteria[0].weight == 3.0
        assert r.criteria[1].weight == 1.0  # default


class TestPrebuiltRubricsContract:
    """Validate all 4 pre-built rubrics are correctly defined."""

    @pytest.mark.parametrize(
        "rubric,expected_name,min_criteria",
        [
            (SUMMARIZATION_RUBRIC, "Summarization Quality", 4),
            (FACTUAL_ACCURACY_RUBRIC, "Factual Accuracy", 3),
            (HELPFULNESS_RUBRIC, "Helpfulness", 3),
            (SAFETY_RUBRIC, "Safety", 3),
        ],
    )
    def test_rubric_structure(self, rubric: Rubric, expected_name: str, min_criteria: int) -> None:
        assert rubric.name == expected_name
        assert len(rubric.criteria) >= min_criteria
        assert rubric.version == "1.0.0"
        assert all(isinstance(c, RubricCriteria) for c in rubric.criteria)
        assert rubric.total_weight > 0

    def test_safety_rubric_uses_binary_scale(self) -> None:
        for criterion in SAFETY_RUBRIC.criteria:
            assert criterion.scale == ScoreScale.BINARY

    def test_summarization_rubric_uses_likert_5(self) -> None:
        for criterion in SUMMARIZATION_RUBRIC.criteria:
            assert criterion.scale == ScoreScale.LIKERT_5


# ===========================================================================
# Section 6: Generator Contracts
# ===========================================================================


class TestBaseGeneratorContract:
    """Validate BaseGenerator abstract interface."""

    def test_is_abstract(self) -> None:
        assert inspect.isabstract(BaseGenerator)

    def test_abstract_methods(self) -> None:
        abstract_methods = set()
        for name, method in inspect.getmembers(BaseGenerator):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)
        assert "generate" in abstract_methods
        assert "agenerate" in abstract_methods


class TestSyntheticGeneratorContract:
    """Validate SyntheticGenerator public interface."""

    def test_init_defaults(self) -> None:
        gen = SyntheticGenerator()
        assert gen.generator_id == "synthetic-default"
        assert gen.strategy == GenerationStrategy.STANDARD
        assert isinstance(gen.llm_config, LLMProviderConfig)

    def test_all_strategies_accepted(self) -> None:
        for strategy in GenerationStrategy:
            gen = SyntheticGenerator(strategy=strategy)
            assert gen.strategy == strategy

    def test_parse_response_return_type(self) -> None:
        raw = json.dumps([{"input": "test"}])
        result = SyntheticGenerator._parse_response(raw)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_deduplicate_return_type(self) -> None:
        items = [{"input": "a"}, {"input": "b"}]
        result = SyntheticGenerator._deduplicate(items)
        assert isinstance(result, list)


class TestGenerationStrategyContract:
    """Validate GenerationStrategy enum."""

    def test_all_four_strategies_exist(self) -> None:
        assert len(GenerationStrategy) == 4
        assert GenerationStrategy.STANDARD.value == "standard"
        assert GenerationStrategy.ADVERSARIAL.value == "adversarial"
        assert GenerationStrategy.EDGE_CASE.value == "edge_case"
        assert GenerationStrategy.DISTRIBUTION_MATCHING.value == "distribution_matching"


class TestRenderTemplateContract:
    """Validate render_template function."""

    def test_return_type(self) -> None:
        result = render_template(GenerationStrategy.STANDARD, "topic", 5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_count_and_topic(self) -> None:
        result = render_template(GenerationStrategy.ADVERSARIAL, "security", 20)
        assert "security" in result
        assert "20" in result


# ===========================================================================
# Section 7: Regression Module Contracts
# ===========================================================================


class TestRegressionTrackerContract:
    """Validate RegressionTracker public interface."""

    def test_init_signature(self) -> None:
        sig = inspect.signature(RegressionTracker.__init__)
        params = list(sig.parameters.keys())
        assert "storage" in params
        assert "threshold" in params

    def test_default_threshold(self) -> None:
        with DuckDBStorage() as s:
            tracker = RegressionTracker(storage=s)
            assert tracker.threshold == -0.1

    def test_compare_versions_return_type(self) -> None:
        with DuckDBStorage() as s:
            tracker = RegressionTracker(storage=s)
            # Populate data
            for i in range(3):
                s.store_result(
                    EvalResult(
                        id=f"b-{i}",
                        model_id="m",
                        model_version="v1",
                        input_text=f"i{i}",
                        output_text=f"o{i}",
                        aggregate_score=3.0,
                    )
                )
                s.store_result(
                    EvalResult(
                        id=f"c-{i}",
                        model_id="m",
                        model_version="v2",
                        input_text=f"i{i}",
                        output_text=f"o{i}",
                        aggregate_score=3.5,
                    )
                )
            report = tracker.compare_versions("m", "v1", "v2")
            assert isinstance(report, RegressionReport)


class TestOutputComparatorContract:
    """Validate OutputComparator public interface."""

    def test_init_defaults(self) -> None:
        comp = OutputComparator()
        assert comp.method == ComparisonMethod.FUZZY
        assert comp.similarity_threshold == 0.9

    def test_compare_return_type(self) -> None:
        comp = OutputComparator()
        result = comp.compare("a", "b")
        assert isinstance(result, ComparisonResult)

    def test_compare_batch_return_type(self) -> None:
        comp = OutputComparator()
        results = comp.compare_batch([("a", "b"), ("c", "c")])
        assert isinstance(results, list)
        assert all(isinstance(r, ComparisonResult) for r in results)

    def test_comparison_result_attributes(self) -> None:
        r = ComparisonResult(
            method=ComparisonMethod.EXACT,
            similarity=1.0,
            is_match=True,
        )
        assert hasattr(r, "method")
        assert hasattr(r, "similarity")
        assert hasattr(r, "is_match")
        assert hasattr(r, "diff")
        assert hasattr(r, "details")


class TestRegressionReporterContract:
    """Validate RegressionReporter public interface."""

    def test_to_markdown_return_type(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(baseline_version="v1", candidate_version="v2", model_id="m")
        result = reporter.to_markdown(report)
        assert isinstance(result, str)

    def test_to_json_return_type(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(baseline_version="v1", candidate_version="v2", model_id="m")
        result = reporter.to_json(report)
        assert isinstance(result, str)
        json.loads(result)  # should be valid JSON

    def test_to_dict_return_type(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(baseline_version="v1", candidate_version="v2", model_id="m")
        result = reporter.to_dict(report)
        assert isinstance(result, dict)

    def test_to_console_return_type(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(baseline_version="v1", candidate_version="v2", model_id="m")
        result = reporter.to_console(report)
        assert isinstance(result, str)

    def test_save_report_creates_file(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(baseline_version="v1", candidate_version="v2", model_id="m")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.md")
            reporter.save_report(report, path)
            assert Path(path).exists()


# ===========================================================================
# Section 8: Enum Completeness
# ===========================================================================


class TestEnumCompleteness:
    """Ensure all enum members are properly defined."""

    def test_score_scale_all_members(self) -> None:
        members = {s.value for s in ScoreScale}
        assert members == {"binary", "likert_5", "likert_10", "percentage"}

    def test_voting_strategy_all_members(self) -> None:
        members = {s.value for s in VotingStrategy}
        assert members == {"majority", "weighted_average", "unanimous"}

    def test_comparison_method_all_members(self) -> None:
        members = {m.value for m in ComparisonMethod}
        assert members == {"exact", "fuzzy", "structural"}

    def test_generation_strategy_all_members(self) -> None:
        members = {s.value for s in GenerationStrategy}
        assert members == {"standard", "adversarial", "edge_case", "distribution_matching"}
