"""Tests for regression tracking, comparison, and reporting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evalkit.core.models import EvalResult, RegressionDelta, RegressionReport
from evalkit.core.storage import DuckDBStorage
from evalkit.regression.comparator import (
    ComparisonMethod,
    ComparisonResult,
    OutputComparator,
)
from evalkit.regression.reporter import RegressionReporter
from evalkit.regression.tracker import RegressionTracker


class TestRegressionTracker:
    """Tests for the RegressionTracker class."""

    def test_record_single(self, in_memory_storage: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=in_memory_storage)
        result = EvalResult(
            model_id="m",
            model_version="v1",
            input_text="in",
            output_text="out",
            aggregate_score=4.0,
        )
        tracker.record(result)
        assert in_memory_storage.count_results() == 1

    def test_record_batch(self, in_memory_storage: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=in_memory_storage)
        results = [
            EvalResult(
                id=f"r-{i}",
                model_id="m",
                model_version="v1",
                input_text=f"in-{i}",
                output_text=f"out-{i}",
                aggregate_score=float(i),
            )
            for i in range(5)
        ]
        tracker.record_batch(results)
        assert in_memory_storage.count_results() == 5

    def test_compare_versions_improvement(self, storage_with_data: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=storage_with_data)
        report = tracker.compare_versions("test-model", "v1.0", "v2.0")

        assert report.model_id == "test-model"
        assert report.baseline_version == "v1.0"
        assert report.candidate_version == "v2.0"
        assert report.overall_delta > 0  # v2 scores higher
        assert report.has_regression is False
        assert report.sample_count_baseline == 5
        assert report.sample_count_candidate == 5

    def test_compare_versions_regression(self, storage_with_data: DuckDBStorage) -> None:
        # Compare in reverse: v2 as baseline, v1 as candidate (regression)
        tracker = RegressionTracker(storage=storage_with_data, threshold=-0.05)
        report = tracker.compare_versions("test-model", "v2.0", "v1.0")

        assert report.overall_delta < 0
        assert report.has_regression is True

    def test_compare_missing_baseline(self, in_memory_storage: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=in_memory_storage)
        with pytest.raises(ValueError, match="No results found"):
            tracker.compare_versions("m", "v1", "v2")

    def test_compare_missing_candidate(self, storage_with_data: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=storage_with_data)
        with pytest.raises(ValueError, match="No results found"):
            tracker.compare_versions("test-model", "v1.0", "v99.0")

    def test_get_version_history(self, storage_with_data: DuckDBStorage) -> None:
        tracker = RegressionTracker(storage=storage_with_data)
        versions = tracker.get_version_history("test-model")
        assert set(versions) == {"v1.0", "v2.0"}


class TestOutputComparator:
    """Tests for the OutputComparator class."""

    def test_exact_match(self) -> None:
        comp = OutputComparator()
        result = comp.compare("hello", "hello", ComparisonMethod.EXACT)
        assert result.is_match is True
        assert result.similarity == 1.0

    def test_exact_no_match(self) -> None:
        comp = OutputComparator()
        result = comp.compare("hello", "world", ComparisonMethod.EXACT)
        assert result.is_match is False
        assert result.similarity == 0.0
        assert result.diff  # should have a diff

    def test_fuzzy_match_identical(self) -> None:
        comp = OutputComparator(similarity_threshold=0.9)
        result = comp.compare("hello world", "hello world", ComparisonMethod.FUZZY)
        assert result.is_match is True
        assert result.similarity == pytest.approx(1.0)

    def test_fuzzy_match_similar(self) -> None:
        comp = OutputComparator(similarity_threshold=0.5)
        result = comp.compare(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox leaps over the lazy dog",
            ComparisonMethod.FUZZY,
        )
        assert result.is_match is True
        assert result.similarity > 0.5

    def test_fuzzy_no_match(self) -> None:
        comp = OutputComparator(similarity_threshold=0.99)
        result = comp.compare("abc", "xyz")
        assert result.is_match is False

    def test_structural_match_similar_structure(self) -> None:
        comp = OutputComparator(similarity_threshold=0.7)
        text_a = "Line one\nLine two\nLine three"
        text_b = "First line\nSecond line\nThird line"
        result = comp.compare(text_a, text_b, ComparisonMethod.STRUCTURAL)
        assert result.similarity > 0.7
        assert "line_similarity" in result.details

    def test_structural_different_structure(self) -> None:
        comp = OutputComparator(similarity_threshold=0.95)
        text_a = "Short"
        text_b = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10"
        result = comp.compare(text_a, text_b, ComparisonMethod.STRUCTURAL)
        assert result.is_match is False

    def test_compare_batch(self) -> None:
        comp = OutputComparator()
        pairs = [
            ("hello", "hello"),
            ("foo", "bar"),
            ("same text", "same text"),
        ]
        results = comp.compare_batch(pairs, ComparisonMethod.EXACT)
        assert len(results) == 3
        assert results[0].is_match is True
        assert results[1].is_match is False
        assert results[2].is_match is True

    def test_default_method_is_fuzzy(self) -> None:
        comp = OutputComparator()
        result = comp.compare("test", "test")
        assert result.method == ComparisonMethod.FUZZY

    def test_comparison_result_repr(self) -> None:
        r = ComparisonResult(
            method=ComparisonMethod.EXACT,
            similarity=1.0,
            is_match=True,
        )
        assert "EXACT" in repr(r) or "exact" in repr(r)


class TestRegressionReporter:
    """Tests for the RegressionReporter class."""

    @pytest.fixture
    def sample_report(self) -> RegressionReport:
        return RegressionReport(
            id="report-001",
            baseline_version="v1.0",
            candidate_version="v2.0",
            model_id="test-model",
            rubric_name="Test Rubric",
            deltas=[
                RegressionDelta(
                    criterion="Accuracy",
                    baseline_score=4.0,
                    candidate_score=4.5,
                    delta=0.5,
                    relative_delta_pct=12.5,
                    is_regression=False,
                ),
                RegressionDelta(
                    criterion="Clarity",
                    baseline_score=3.5,
                    candidate_score=3.0,
                    delta=-0.5,
                    relative_delta_pct=-14.29,
                    is_regression=True,
                ),
            ],
            overall_delta=0.0,
            sample_count_baseline=100,
            sample_count_candidate=100,
            has_regression=True,
        )

    def test_to_markdown(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        md = reporter.to_markdown(sample_report)
        assert "Regression Report" in md
        assert "test-model" in md
        assert "Accuracy" in md
        assert "Clarity" in md
        assert "YES" in md  # Clarity is a regression

    def test_to_markdown_pass(self) -> None:
        report = RegressionReport(
            baseline_version="v1",
            candidate_version="v2",
            model_id="m",
            has_regression=False,
        )
        reporter = RegressionReporter()
        md = reporter.to_markdown(report)
        assert "PASS" in md

    def test_to_json(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        j = reporter.to_json(sample_report)
        parsed = json.loads(j)
        assert parsed["model_id"] == "test-model"
        assert len(parsed["deltas"]) == 2

    def test_to_dict(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        d = reporter.to_dict(sample_report)
        assert d["model_id"] == "test-model"

    def test_to_console(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        console = reporter.to_console(sample_report)
        assert "REGRESSION DETECTED" in console
        assert "test-model" in console

    def test_to_console_all_clear(self) -> None:
        report = RegressionReport(
            baseline_version="v1",
            candidate_version="v2",
            model_id="m",
            has_regression=False,
        )
        reporter = RegressionReporter()
        console = reporter.to_console(report)
        assert "ALL CLEAR" in console

    def test_save_report_markdown(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        reporter.save_report(sample_report, path, fmt="markdown")
        content = Path(path).read_text()
        assert "Regression Report" in content

    def test_save_report_json(self, sample_report: RegressionReport) -> None:
        reporter = RegressionReporter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        reporter.save_report(sample_report, path, fmt="json")
        content = Path(path).read_text()
        parsed = json.loads(content)
        assert parsed["model_id"] == "test-model"
