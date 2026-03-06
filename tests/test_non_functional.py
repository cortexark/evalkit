"""Non-Functional Tests -- error handling, concurrency, data integrity, config edge cases.

Covers:
- Error handling for invalid config, missing API keys, bad JSON from LLM
- Concurrency: EnsembleJudge.aevaluate() runs judges concurrently
- Data integrity: DuckDB upsert, concurrent writes
- Configuration: YAML with missing fields, env var override
- Boundary values: Score scales (BINARY 0/1, LIKERT_5 1-5, etc.)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import threading
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
from evalkit.generators.synthetic import SyntheticGenerator
from evalkit.generators.templates import GenerationStrategy, render_template
from evalkit.judges.base import BaseJudge
from evalkit.judges.ensemble import EnsembleJudge
from evalkit.judges.llm_judge import LLMJudge, _parse_judge_response
from evalkit.judges.rubrics import build_rubric
from evalkit.regression.comparator import ComparisonMethod, OutputComparator
from evalkit.regression.reporter import RegressionReporter
from evalkit.regression.tracker import RegressionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rubric() -> Rubric:
    return Rubric(
        name="TestRubric",
        criteria=[
            RubricCriteria(name="Quality", description="Overall quality", weight=1.0),
        ],
    )


class StubJudge(BaseJudge):
    """Deterministic judge for testing, with optional delay for concurrency tests."""

    def __init__(
        self,
        judge_id: str,
        rubric: Rubric,
        scores: list[JudgeScore] | None = None,
        delay: float = 0.0,
    ) -> None:
        super().__init__(judge_id=judge_id, rubric=rubric)
        self._scores = scores or []
        self._delay = delay
        self.call_count = 0

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        return self._scores

    async def aevaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._scores


# ===========================================================================
# Section 1: Error Handling
# ===========================================================================


class TestLLMJudgeErrorHandling:
    """Test error paths in LLMJudge."""

    def test_missing_api_key_raises(self) -> None:
        """Calling evaluate without an API key should raise EnvironmentError."""
        rubric = _make_rubric()
        config = LLMProviderConfig(
            provider="openai",
            api_key_env_var="EVALKIT_NONEXISTENT_API_KEY_FOR_TEST",
        )
        os.environ.pop("EVALKIT_NONEXISTENT_API_KEY_FOR_TEST", None)
        judge = LLMJudge(judge_id="j1", rubric=rubric, llm_config=config)

        with pytest.raises(EnvironmentError, match="API key not found"):
            judge.evaluate("in", "out")

    def test_unsupported_provider_raises(self) -> None:
        """Unsupported provider should raise ValueError from _call_llm."""
        rubric = _make_rubric()
        config = LLMProviderConfig(provider="unsupported_provider")
        judge = LLMJudge(judge_id="j1", rubric=rubric, llm_config=config)

        # Mock the api_key property to bypass env var check, then call _call_llm directly
        with patch.object(LLMProviderConfig, "api_key", new_callable=lambda: property(lambda self: "dummy")):
            with pytest.raises(ValueError, match="Unsupported provider"):
                judge._call_llm("test prompt")

    def test_bad_json_from_llm_raises(self) -> None:
        """If LLM returns invalid JSON, evaluate should raise ValueError."""
        rubric = _make_rubric()
        judge = LLMJudge(judge_id="j1", rubric=rubric)
        with patch.object(LLMJudge, "_call_llm") as mock:
            mock.return_value = "This is not JSON at all"
            with pytest.raises(ValueError, match="Failed to parse"):
                judge.evaluate("in", "out")

    def test_llm_returns_object_not_array(self) -> None:
        """If LLM returns a JSON object instead of array, should raise ValueError."""
        rubric = _make_rubric()
        judge = LLMJudge(judge_id="j1", rubric=rubric)
        with patch.object(LLMJudge, "_call_llm") as mock:
            mock.return_value = '{"criterion": "A", "score": 3}'
            with pytest.raises(ValueError, match="Expected JSON array"):
                judge.evaluate("in", "out")

    def test_llm_returns_incomplete_score_objects(self) -> None:
        """If LLM returns scores without criterion or score key, should handle gracefully."""
        rubric = _make_rubric()
        judge = LLMJudge(judge_id="j1", rubric=rubric)
        with patch.object(LLMJudge, "_call_llm") as mock:
            mock.return_value = json.dumps([
                {"reasoning": "no criterion or score key"},
            ])
            scores = judge.evaluate("in", "out")
            assert len(scores) == 1
            assert scores[0].criterion == "unknown"
            assert scores[0].score == 0.0


class TestParseJudgeResponseErrorPaths:
    """Test edge cases in _parse_judge_response."""

    def test_empty_string(self) -> None:
        with pytest.raises(ValueError):
            _parse_judge_response("", "judge-1")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            _parse_judge_response("   \n\t  ", "judge-1")

    def test_json_with_trailing_text(self) -> None:
        # JSON followed by extra text -- json.loads will fail if not array-only
        raw = '[{"criterion":"A","score":3}] some trailing text'
        with pytest.raises(ValueError):
            _parse_judge_response(raw, "judge-1")

    def test_nested_code_blocks(self) -> None:
        raw = '```json\n[{"criterion":"A","score":3,"reasoning":"ok"}]\n```'
        result = _parse_judge_response(raw, "judge-1")
        assert len(result) == 1

    def test_code_block_without_json_tag(self) -> None:
        raw = '```\n[{"criterion":"A","score":3,"reasoning":"ok"}]\n```'
        result = _parse_judge_response(raw, "judge-1")
        assert len(result) == 1


class TestSyntheticGeneratorErrorHandling:
    """Test error paths in SyntheticGenerator."""

    def test_parse_response_empty_string(self) -> None:
        with pytest.raises(ValueError):
            SyntheticGenerator._parse_response("")

    def test_parse_response_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            SyntheticGenerator._parse_response("not json {[")

    def test_parse_response_non_array(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON array"):
            SyntheticGenerator._parse_response('{"key": "value"}')

    def test_parse_response_array_of_non_dicts(self) -> None:
        raw = json.dumps(["string1", "string2"])
        result = SyntheticGenerator._parse_response(raw)
        assert result == []  # non-dict items filtered out

    def test_parse_response_items_missing_input_key(self) -> None:
        raw = json.dumps([
            {"question": "no input key"},
            {"input": "has input key"},
        ])
        result = SyntheticGenerator._parse_response(raw)
        assert len(result) == 1

    def test_missing_api_key_for_generate(self) -> None:
        config = LLMProviderConfig(api_key_env_var="EVALKIT_TEST_MISSING_KEY_XYZ")
        os.environ.pop("EVALKIT_TEST_MISSING_KEY_XYZ", None)
        gen = SyntheticGenerator(llm_config=config)
        with pytest.raises(EnvironmentError):
            gen.generate("topic", count=3)


class TestConfigErrorHandling:
    """Test configuration error paths."""

    def test_from_yaml_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            EvalConfig.from_yaml(path)
        Path(path).unlink()

    def test_from_yaml_list_content(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            path = f.name
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            EvalConfig.from_yaml(path)
        Path(path).unlink()

    def test_from_yaml_invalid_field_type(self) -> None:
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"project_name": 12345, "storage": "not_a_dict"}, f)
            path = f.name
        with pytest.raises(ValidationError):
            EvalConfig.from_yaml(path)
        Path(path).unlink()

    def test_from_yaml_with_extra_fields(self) -> None:
        """Pydantic should accept extra fields without error (by default model is strict=False for extras)."""
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"project_name": "test", "unknown_field": "value"}, f)
            path = f.name
        # Should not raise -- Pydantic ignores extra by default
        try:
            config = EvalConfig.from_yaml(path)
            assert config.project_name == "test"
        except ValidationError:
            # If strict mode rejects extras, that's also valid behavior
            pass
        Path(path).unlink()


class TestStorageErrorHandling:
    """Test storage error paths."""

    def test_query_after_close_raises(self) -> None:
        s = DuckDBStorage()
        s.close()
        with pytest.raises(Exception):  # duckdb raises on closed connection
            s.get_results()

    def test_store_after_close_raises(self) -> None:
        s = DuckDBStorage()
        s.close()
        r = EvalResult(model_id="m", input_text="i", output_text="o")
        with pytest.raises(Exception):
            s.store_result(r)


class TestRegressionTrackerErrorHandling:
    """Test regression tracker error paths."""

    def test_compare_missing_both_versions(self) -> None:
        with DuckDBStorage() as s:
            tracker = RegressionTracker(storage=s)
            with pytest.raises(ValueError, match="No results found"):
                tracker.compare_versions("m", "v1", "v2")

    def test_compare_only_baseline_exists(self) -> None:
        with DuckDBStorage() as s:
            s.store_result(EvalResult(
                model_id="m", model_version="v1",
                input_text="i", output_text="o",
                aggregate_score=3.0,
            ))
            tracker = RegressionTracker(storage=s)
            with pytest.raises(ValueError, match="No results found"):
                tracker.compare_versions("m", "v1", "v2")


# ===========================================================================
# Section 2: Concurrency
# ===========================================================================


class TestEnsembleConcurrency:
    """Test that EnsembleJudge.aevaluate runs judges concurrently."""

    @pytest.mark.asyncio
    async def test_aevaluate_runs_concurrently(self) -> None:
        """If 3 judges each take 0.1s, concurrent execution should take ~0.1s, not ~0.3s."""
        rubric = _make_rubric()
        scores = [JudgeScore(judge_id="j", criterion="Quality", score=4.0)]

        j1 = StubJudge("j1", rubric, scores, delay=0.1)
        j2 = StubJudge("j2", rubric, scores, delay=0.1)
        j3 = StubJudge("j3", rubric, scores, delay=0.1)

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 1.0), (j2, 1.0), (j3, 1.0)],
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )

        start = time.monotonic()
        result = await ensemble.aevaluate("in", "out")
        elapsed = time.monotonic() - start

        # Concurrent: should be ~0.1s. Sequential would be ~0.3s.
        # Allow generous margin but should be well under 0.25s
        assert elapsed < 0.25, f"Expected concurrent execution (<0.25s), took {elapsed:.3f}s"
        assert j1.call_count == 1
        assert j2.call_count == 1
        assert j3.call_count == 1
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_aevaluate_empty_raises(self) -> None:
        rubric = _make_rubric()
        ensemble = EnsembleJudge(judge_id="ens", rubric=rubric, judges=[])
        with pytest.raises(ValueError, match="no judges"):
            await ensemble.aevaluate("in", "out")


# ===========================================================================
# Section 3: Data Integrity
# ===========================================================================


class TestDuckDBDataIntegrity:
    """Test DuckDB storage data integrity guarantees."""

    def test_upsert_replaces_on_duplicate_id(self) -> None:
        """INSERT OR REPLACE should update all fields on duplicate primary key."""
        with DuckDBStorage() as s:
            r1 = EvalResult(
                id="dup-test",
                model_id="m",
                input_text="original",
                output_text="original-out",
                aggregate_score=2.0,
            )
            r2 = EvalResult(
                id="dup-test",
                model_id="m",
                input_text="updated",
                output_text="updated-out",
                aggregate_score=5.0,
            )
            s.store_result(r1)
            s.store_result(r2)

            results = s.get_results()
            assert len(results) == 1
            assert results[0]["input_text"] == "updated"
            assert results[0]["output_text"] == "updated-out"
            assert results[0]["aggregate_score"] == 5.0

    def test_store_results_batch_atomicity(self) -> None:
        """Batch store should store all or fail -- test all stored on success."""
        with DuckDBStorage() as s:
            results = [
                EvalResult(
                    id=f"batch-{i}",
                    model_id="m",
                    input_text=f"in-{i}",
                    output_text=f"out-{i}",
                    aggregate_score=float(i),
                )
                for i in range(10)
            ]
            s.store_results(results)
            assert s.count_results() == 10

    def test_null_aggregate_score_stored_correctly(self) -> None:
        """Results with None aggregate_score should be stored and retrieved."""
        with DuckDBStorage() as s:
            r = EvalResult(
                model_id="m", input_text="i", output_text="o",
                aggregate_score=None,
            )
            s.store_result(r)
            results = s.get_results()
            assert len(results) == 1
            assert results[0]["aggregate_score"] is None

    def test_empty_strings_stored_correctly(self) -> None:
        with DuckDBStorage() as s:
            r = EvalResult(
                model_id="m",
                input_text="",
                output_text="",
                reference_text=None,
            )
            s.store_result(r)
            results = s.get_results()
            assert results[0]["input_text"] == ""
            assert results[0]["output_text"] == ""

    def test_unicode_stored_correctly(self) -> None:
        with DuckDBStorage() as s:
            r = EvalResult(
                model_id="m",
                input_text="Unicode test: \u4f60\u597d\u4e16\u754c \U0001f600 \u00e9\u00e0\u00fc",
                output_text="\u2603 \u2764 \u2728",
            )
            s.store_result(r)
            results = s.get_results()
            assert "\u4f60\u597d" in results[0]["input_text"]
            assert "\u2603" in results[0]["output_text"]

    def test_very_long_text_stored(self) -> None:
        with DuckDBStorage() as s:
            long_text = "x" * 100_000
            r = EvalResult(
                model_id="m",
                input_text=long_text,
                output_text=long_text,
            )
            s.store_result(r)
            results = s.get_results()
            assert len(results[0]["input_text"]) == 100_000

    def test_special_characters_in_model_id(self) -> None:
        with DuckDBStorage() as s:
            r = EvalResult(
                model_id="model/v2.0-beta+build.123",
                input_text="i",
                output_text="o",
            )
            s.store_result(r)
            results = s.get_results(model_id="model/v2.0-beta+build.123")
            assert len(results) == 1

    def test_get_results_filters_combine(self) -> None:
        """Multiple filters should AND together."""
        with DuckDBStorage() as s:
            s.store_result(EvalResult(
                id="1", model_id="m1", model_version="v1",
                input_text="i", output_text="o", rubric_name="r1",
            ))
            s.store_result(EvalResult(
                id="2", model_id="m1", model_version="v2",
                input_text="i", output_text="o", rubric_name="r1",
            ))
            s.store_result(EvalResult(
                id="3", model_id="m2", model_version="v1",
                input_text="i", output_text="o", rubric_name="r2",
            ))

            results = s.get_results(model_id="m1", model_version="v1")
            assert len(results) == 1
            assert results[0]["id"] == "1"

            results = s.get_results(model_id="m1", rubric_name="r1")
            assert len(results) == 2


# ===========================================================================
# Section 4: Configuration Edge Cases
# ===========================================================================


class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_yaml_with_only_project_name(self) -> None:
        """YAML with only project_name should use defaults for everything else."""
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"project_name": "minimal"}, f)
            path = f.name
        config = EvalConfig.from_yaml(path)
        assert config.project_name == "minimal"
        assert config.default_model_id == "default-model"
        assert config.storage.database_path == ":memory:"
        Path(path).unlink()

    def test_yaml_empty_dict(self) -> None:
        """Empty YAML dict should produce default config."""
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = f.name
        config = EvalConfig.from_yaml(path)
        assert config.project_name == "evalkit"
        Path(path).unlink()

    def test_config_roundtrip_preserves_all_fields(self) -> None:
        """to_yaml -> from_yaml should preserve all fields."""
        original = EvalConfig(
            project_name="roundtrip-full",
            default_model_id="my-model",
            storage=StorageConfig(database_path="/tmp/test.duckdb"),
            ensemble=EnsembleConfig(
                voting_strategy=VotingStrategy.MAJORITY,
                regression_threshold=-0.05,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            original.to_yaml(path)
            loaded = EvalConfig.from_yaml(path)

            assert loaded.project_name == original.project_name
            assert loaded.default_model_id == original.default_model_id
            assert loaded.storage.database_path == original.storage.database_path
            assert loaded.ensemble.voting_strategy == original.ensemble.voting_strategy
            assert loaded.ensemble.regression_threshold == original.ensemble.regression_threshold

    def test_api_key_env_var_name_customizable(self) -> None:
        """Custom env var name should be used for API key lookup."""
        config = LLMProviderConfig(api_key_env_var="MY_CUSTOM_KEY_VAR")
        os.environ["MY_CUSTOM_KEY_VAR"] = "test-key-value"
        try:
            assert config.api_key == "test-key-value"
        finally:
            del os.environ["MY_CUSTOM_KEY_VAR"]


# ===========================================================================
# Section 5: Boundary Value Testing
# ===========================================================================


class TestScoreBoundaryValues:
    """Test boundary values for different score scales."""

    def test_binary_valid_scores(self) -> None:
        """BINARY scale: valid scores are 0 and 1."""
        s0 = JudgeScore(judge_id="j", criterion="c", score=0.0)
        s1 = JudgeScore(judge_id="j", criterion="c", score=1.0)
        assert s0.score == 0.0
        assert s1.score == 1.0

    def test_likert_5_boundary_values(self) -> None:
        """LIKERT_5 scale: valid range is 1-5 (but model doesn't enforce upper bound)."""
        for val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            s = JudgeScore(judge_id="j", criterion="c", score=val)
            assert s.score == val

    def test_likert_10_boundary_values(self) -> None:
        """LIKERT_10 scale: valid range is 1-10."""
        for val in [0.0, 1.0, 5.0, 10.0]:
            s = JudgeScore(judge_id="j", criterion="c", score=val)
            assert s.score == val

    def test_percentage_boundary_values(self) -> None:
        """PERCENTAGE scale: valid range is 0.0-1.0."""
        for val in [0.0, 0.5, 1.0]:
            s = JudgeScore(judge_id="j", criterion="c", score=val)
            assert s.score == val

    def test_score_above_scale_max_not_rejected(self) -> None:
        """JudgeScore doesn't enforce scale-specific max -- only ge=0 constraint."""
        # Score of 100 on a BINARY scale is valid at the model level
        s = JudgeScore(judge_id="j", criterion="c", score=100.0)
        assert s.score == 100.0

    def test_weight_boundary_zero(self) -> None:
        c = RubricCriteria(name="N", description="d", weight=0.0)
        assert c.weight == 0.0

    def test_weight_boundary_max(self) -> None:
        c = RubricCriteria(name="N", description="d", weight=100.0)
        assert c.weight == 100.0

    def test_aggregate_score_at_zero(self) -> None:
        r = EvalResult(
            model_id="m", input_text="i", output_text="o",
            aggregate_score=0.0,
        )
        assert r.aggregate_score == 0.0


class TestComparatorBoundaryValues:
    """Test comparator boundary values."""

    def test_exact_compare_empty_strings(self) -> None:
        comp = OutputComparator()
        result = comp.compare("", "", ComparisonMethod.EXACT)
        assert result.is_match is True
        assert result.similarity == 1.0

    def test_fuzzy_compare_empty_strings(self) -> None:
        comp = OutputComparator()
        result = comp.compare("", "", ComparisonMethod.FUZZY)
        assert result.similarity == pytest.approx(1.0)

    def test_structural_compare_empty_strings(self) -> None:
        comp = OutputComparator(similarity_threshold=0.5)
        result = comp.compare("", "", ComparisonMethod.STRUCTURAL)
        # Empty strings should have high structural similarity
        assert result.similarity >= 0.5

    def test_exact_compare_very_long_identical(self) -> None:
        text = "a" * 50_000
        comp = OutputComparator()
        result = comp.compare(text, text, ComparisonMethod.EXACT)
        assert result.is_match is True

    def test_fuzzy_threshold_exactly_met(self) -> None:
        """When similarity equals threshold exactly, should match."""
        comp = OutputComparator(similarity_threshold=1.0)
        result = comp.compare("identical", "identical", ComparisonMethod.FUZZY)
        assert result.is_match is True

    def test_fuzzy_threshold_at_zero(self) -> None:
        """With threshold 0, everything matches."""
        comp = OutputComparator(similarity_threshold=0.0)
        result = comp.compare("abc", "xyz", ComparisonMethod.FUZZY)
        assert result.is_match is True  # any non-negative similarity >= 0

    def test_structural_single_character_vs_paragraph(self) -> None:
        comp = OutputComparator(similarity_threshold=0.9)
        text_a = "a"
        text_b = "This is a long paragraph.\n\nWith multiple paragraphs.\n\nAnd more content."
        result = comp.compare(text_a, text_b, ComparisonMethod.STRUCTURAL)
        assert result.is_match is False


class TestRegressionThresholdBoundaries:
    """Test regression tracker threshold boundary values."""

    def test_zero_threshold_exact_equality_not_regression(self) -> None:
        """With threshold=0, equal scores should not be flagged as regression."""
        with DuckDBStorage() as s:
            for i in range(3):
                s.store_result(EvalResult(
                    id=f"b-{i}", model_id="m", model_version="v1",
                    input_text=f"i{i}", output_text=f"o{i}", aggregate_score=3.0,
                ))
                s.store_result(EvalResult(
                    id=f"c-{i}", model_id="m", model_version="v2",
                    input_text=f"i{i}", output_text=f"o{i}", aggregate_score=3.0,
                ))
            tracker = RegressionTracker(storage=s, threshold=0.0)
            report = tracker.compare_versions("m", "v1", "v2")
            assert report.overall_delta == pytest.approx(0.0)
            assert report.has_regression is False

    def test_very_tight_threshold(self) -> None:
        """With threshold=-0.001, even tiny drops should be flagged."""
        with DuckDBStorage() as s:
            s.store_result(EvalResult(
                id="b1", model_id="m", model_version="v1",
                input_text="i", output_text="o", aggregate_score=3.0,
            ))
            s.store_result(EvalResult(
                id="c1", model_id="m", model_version="v2",
                input_text="i", output_text="o", aggregate_score=2.998,
            ))
            tracker = RegressionTracker(storage=s, threshold=-0.001)
            report = tracker.compare_versions("m", "v1", "v2")
            assert report.has_regression is True

    def test_positive_threshold_marks_regression(self) -> None:
        """BUG: With threshold=999, the comparison 'delta < threshold' (-4 < 999)
        is True, so it flags regression even though a positive threshold should
        mean 'never regress'. The threshold comparison uses raw '<' with no
        guard for positive thresholds. Documenting actual behavior here.

        See bug report: threshold logic does not handle positive thresholds correctly.
        """
        with DuckDBStorage() as s:
            s.store_result(EvalResult(
                id="b1", model_id="m", model_version="v1",
                input_text="i", output_text="o", aggregate_score=5.0,
            ))
            s.store_result(EvalResult(
                id="c1", model_id="m", model_version="v2",
                input_text="i", output_text="o", aggregate_score=1.0,
            ))
            tracker = RegressionTracker(storage=s, threshold=999.0)
            report = tracker.compare_versions("m", "v1", "v2")
            # BUG: This incorrectly flags as regression because -4.0 < 999.0
            assert report.has_regression is True


# ===========================================================================
# Section 6: Ensemble Voting Strategy Edge Cases
# ===========================================================================


class TestEnsembleVotingEdgeCases:
    """Test ensemble voting strategies with edge cases."""

    def test_weighted_average_all_zero_weights(self) -> None:
        """If all judge weights are 0, weighted average should handle gracefully."""
        rubric = _make_rubric()
        scores = [JudgeScore(judge_id="j", criterion="Quality", score=4.0)]
        j1 = StubJudge("j1", rubric, scores)
        j2 = StubJudge("j2", rubric, scores)

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 0.0), (j2, 0.0)],
            voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
        )
        result = ensemble.evaluate("in", "out")
        # With zero weights, code sets avg_score = 0.0
        assert result[0].score == 0.0

    def test_majority_vote_single_judge(self) -> None:
        """Majority with single judge should return that judge's score."""
        rubric = _make_rubric()
        scores = [JudgeScore(judge_id="j1", criterion="Quality", score=3.0)]
        j1 = StubJudge("j1", rubric, scores)

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 1.0)],
            voting_strategy=VotingStrategy.MAJORITY,
        )
        result = ensemble.evaluate("in", "out")
        assert result[0].score == 3.0

    def test_unanimous_all_agree(self) -> None:
        """When all judges agree, unanimous should return that score with 'unanimous' tag."""
        rubric = _make_rubric()
        scores = [JudgeScore(judge_id="j", criterion="Quality", score=5.0)]
        j1 = StubJudge("j1", rubric, scores)
        j2 = StubJudge("j2", rubric, scores)

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 1.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.UNANIMOUS,
        )
        result = ensemble.evaluate("in", "out")
        assert result[0].score == 5.0
        assert "unanimous" in result[0].reasoning

    def test_unanimous_disagreement_takes_minimum(self) -> None:
        rubric = _make_rubric()
        j1 = StubJudge("j1", rubric, [JudgeScore(judge_id="j1", criterion="Q", score=5.0)])
        j2 = StubJudge("j2", rubric, [JudgeScore(judge_id="j2", criterion="Q", score=1.0)])

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 1.0), (j2, 1.0)],
            voting_strategy=VotingStrategy.UNANIMOUS,
        )
        result = ensemble.evaluate("in", "out")
        assert result[0].score == 1.0
        assert "disagreement" in result[0].reasoning

    def test_majority_with_many_criteria(self) -> None:
        """Majority vote should work correctly with multiple criteria."""
        rubric = Rubric(
            name="Multi",
            criteria=[
                RubricCriteria(name="A", description="d"),
                RubricCriteria(name="B", description="d"),
            ],
        )
        j1_scores = [
            JudgeScore(judge_id="j1", criterion="A", score=4.0),
            JudgeScore(judge_id="j1", criterion="B", score=3.0),
        ]
        j2_scores = [
            JudgeScore(judge_id="j2", criterion="A", score=4.0),
            JudgeScore(judge_id="j2", criterion="B", score=5.0),
        ]
        j3_scores = [
            JudgeScore(judge_id="j3", criterion="A", score=2.0),
            JudgeScore(judge_id="j3", criterion="B", score=5.0),
        ]
        j1 = StubJudge("j1", rubric, j1_scores)
        j2 = StubJudge("j2", rubric, j2_scores)
        j3 = StubJudge("j3", rubric, j3_scores)

        ensemble = EnsembleJudge(
            judge_id="ens",
            rubric=rubric,
            judges=[(j1, 1.0), (j2, 1.0), (j3, 1.0)],
            voting_strategy=VotingStrategy.MAJORITY,
        )
        result = ensemble.evaluate("in", "out")
        score_map = {s.criterion: s.score for s in result}
        assert score_map["A"] == 4.0  # 4, 4, 2 -> majority is 4
        assert score_map["B"] == 5.0  # 3, 5, 5 -> majority is 5


# ===========================================================================
# Section 7: Reporter Edge Cases
# ===========================================================================


class TestReporterEdgeCases:
    """Test reporter with edge-case data."""

    def test_report_with_no_deltas(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(
            baseline_version="v1", candidate_version="v2", model_id="m",
        )
        md = reporter.to_markdown(report)
        assert "Regression Report" in md
        console = reporter.to_console(report)
        assert "ALL CLEAR" in console

    def test_report_with_zero_delta(self) -> None:
        reporter = RegressionReporter()
        delta = RegressionDelta(
            criterion="A", baseline_score=3.0, candidate_score=3.0,
            delta=0.0, relative_delta_pct=0.0,
        )
        report = RegressionReport(
            baseline_version="v1", candidate_version="v2", model_id="m",
            deltas=[delta], overall_delta=0.0,
        )
        md = reporter.to_markdown(report)
        assert "+0.0000" in md or "0.0000" in md

    def test_save_report_json_format(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(
            baseline_version="v1", candidate_version="v2", model_id="m",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.json")
            reporter.save_report(report, path, fmt="json")
            content = Path(path).read_text()
            parsed = json.loads(content)
            assert parsed["model_id"] == "m"

    def test_save_report_unknown_format_defaults_to_markdown(self) -> None:
        reporter = RegressionReporter()
        report = RegressionReport(
            baseline_version="v1", candidate_version="v2", model_id="m",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.txt")
            reporter.save_report(report, path, fmt="unknown")
            content = Path(path).read_text()
            assert "Regression Report" in content  # falls through to markdown
