"""Scaling Tests -- performance and load testing for storage, batch operations, and comparisons.

Covers:
- Storage: Insert 1000+ results and query performance
- Batch operations: store_results with large batches
- Regression comparison with large result sets
- compare_batch with 100+ output pairs
"""

from __future__ import annotations

import time

import pytest

from evalkit.core.models import (
    EvalResult,
    JudgeScore,
)
from evalkit.core.storage import DuckDBStorage
from evalkit.regression.comparator import ComparisonMethod, OutputComparator
from evalkit.regression.reporter import RegressionReporter
from evalkit.regression.tracker import RegressionTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_result(
    idx: int,
    model_id: str = "scale-model",
    model_version: str = "v1.0",
    aggregate_score: float = 3.5,
) -> EvalResult:
    """Create an EvalResult with deterministic fields based on index."""
    return EvalResult(
        id=f"{model_id}-{model_version}-{idx}",
        model_id=model_id,
        model_version=model_version,
        input_text=f"Input question number {idx} about a complex topic",
        output_text=f"Output answer number {idx} with detailed explanation of the topic",
        reference_text=f"Reference answer {idx}",
        aggregate_score=aggregate_score + (idx % 10) * 0.1,
        rubric_name="scale-rubric",
        scores=[
            JudgeScore(
                judge_id="auto-judge",
                criterion="Quality",
                score=aggregate_score + (idx % 10) * 0.1,
                reasoning=f"Score reasoning for item {idx}",
            ),
        ],
        metadata={"batch": idx // 100, "index": idx},
    )


# ===========================================================================
# Section 1: Storage Scaling
# ===========================================================================


class TestStorageScaling:
    """Test DuckDB storage with large data volumes."""

    @pytest.mark.slow
    def test_insert_1000_results(self) -> None:
        """Insert 1000 results and verify count."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(1000)]
            start = time.monotonic()
            s.store_results(results)
            elapsed = time.monotonic() - start

            count = s.count_results()
            assert count == 1000
            # Should complete in reasonable time (< 10 seconds)
            assert elapsed < 10.0, f"Inserting 1000 results took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_insert_2000_results(self) -> None:
        """Insert 2000 results and verify count and query."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(2000)]
            s.store_results(results)

            assert s.count_results() == 2000
            # Query should also be fast
            start = time.monotonic()
            queried = s.get_results(model_id="scale-model", limit=2000)
            query_elapsed = time.monotonic() - start
            assert len(queried) == 2000
            assert query_elapsed < 5.0

    @pytest.mark.slow
    def test_insert_5000_results_batch(self) -> None:
        """Insert 5000 results in one batch."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(5000)]
            start = time.monotonic()
            s.store_results(results)
            elapsed = time.monotonic() - start

            assert s.count_results() == 5000
            assert elapsed < 30.0, f"Inserting 5000 results took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_query_performance_with_filters(self) -> None:
        """Query with filters on a dataset of 1000+ results should be fast."""
        with DuckDBStorage() as s:
            # Insert results across 5 versions
            for v in range(5):
                version = f"v{v}.0"
                for i in range(200):
                    s.store_result(
                        _make_eval_result(
                            i + v * 200,
                            model_version=version,
                            aggregate_score=3.0 + v * 0.2,
                        )
                    )

            assert s.count_results() == 1000

            # Test filtered query performance
            start = time.monotonic()
            results = s.get_results(
                model_id="scale-model",
                model_version="v2.0",
            )
            elapsed = time.monotonic() - start
            assert len(results) == 200
            assert elapsed < 2.0

    @pytest.mark.slow
    def test_get_distinct_versions_many_versions(self) -> None:
        """get_distinct_versions with many versions should be fast."""
        with DuckDBStorage() as s:
            for v in range(50):
                s.store_result(
                    _make_eval_result(
                        v,
                        model_version=f"v{v}.0",
                    )
                )

            start = time.monotonic()
            versions = s.get_distinct_versions("scale-model")
            elapsed = time.monotonic() - start

            assert len(versions) == 50
            assert elapsed < 2.0

    @pytest.mark.slow
    def test_count_results_performance(self) -> None:
        """count_results should be fast even with many records."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(1000)]
            s.store_results(results)

            start = time.monotonic()
            for _ in range(100):
                count = s.count_results()
            elapsed = time.monotonic() - start

            assert count == 1000
            # 100 count queries should take < 2 seconds
            assert elapsed < 2.0

    @pytest.mark.slow
    def test_upsert_at_scale(self) -> None:
        """Upserting 500 results over existing 500 should maintain correct count."""
        with DuckDBStorage() as s:
            # Insert initial batch
            initial = [_make_eval_result(i) for i in range(500)]
            s.store_results(initial)
            assert s.count_results() == 500

            # Upsert with same IDs but different scores
            updated = [_make_eval_result(i, aggregate_score=5.0) for i in range(500)]
            s.store_results(updated)
            assert s.count_results() == 500  # still 500, not 1000

            # Verify scores were updated
            results = s.get_results(limit=500)
            for r in results:
                # All scores should be >= 5.0 (base) since we updated
                assert r["aggregate_score"] >= 5.0

    @pytest.mark.slow
    def test_multiple_model_ids_at_scale(self) -> None:
        """Test with multiple model IDs, 200 results each."""
        with DuckDBStorage() as s:
            for model_idx in range(5):
                model_id = f"model-{model_idx}"
                results = [_make_eval_result(i, model_id=model_id) for i in range(200)]
                s.store_results(results)

            assert s.count_results() == 1000

            # Query each model
            for model_idx in range(5):
                model_id = f"model-{model_idx}"
                count = s.count_results(model_id=model_id)
                assert count == 200


# ===========================================================================
# Section 2: Batch Operations Scaling
# ===========================================================================


class TestBatchOperationsScaling:
    """Test batch operations at scale."""

    @pytest.mark.slow
    def test_store_results_incremental_batches(self) -> None:
        """Insert multiple batches of 100 results each."""
        with DuckDBStorage() as s:
            for batch in range(10):
                results = [
                    _make_eval_result(
                        i + batch * 100,
                        model_version=f"v{batch}.0",
                    )
                    for i in range(100)
                ]
                s.store_results(results)

            assert s.count_results() == 1000
            versions = s.get_distinct_versions("scale-model")
            assert len(versions) == 10

    @pytest.mark.slow
    def test_get_results_with_large_limit(self) -> None:
        """Query with limit=5000 on 1000 results should return all."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(1000)]
            s.store_results(results)

            queried = s.get_results(limit=5000)
            assert len(queried) == 1000

    @pytest.mark.slow
    def test_get_results_pagination_like_pattern(self) -> None:
        """Simulate pagination by querying with small limits."""
        with DuckDBStorage() as s:
            results = [_make_eval_result(i) for i in range(500)]
            s.store_results(results)

            # Query in pages of 50
            pages_total = 0
            for _page in range(10):
                queried = s.get_results(limit=50)
                pages_total += len(queried)
                assert len(queried) == 50  # limit works
            # Note: without offset, each page returns the same results


# ===========================================================================
# Section 3: Regression Comparison at Scale
# ===========================================================================


class TestRegressionComparisonScaling:
    """Test regression comparison with large result sets."""

    @pytest.mark.slow
    def test_compare_versions_1000_results_each(self) -> None:
        """Compare two versions with 1000 results each."""
        with DuckDBStorage() as s:
            # Version 1: scores around 3.5
            v1_results = [
                _make_eval_result(i, model_version="v1.0", aggregate_score=3.5)
                for i in range(1000)
            ]
            # Version 2: scores slightly higher (around 3.7)
            v2_results = [
                _make_eval_result(i, model_version="v2.0", aggregate_score=3.7)
                for i in range(1000)
            ]
            s.store_results(v1_results)
            s.store_results(v2_results)

            tracker = RegressionTracker(storage=s)

            start = time.monotonic()
            report = tracker.compare_versions("scale-model", "v1.0", "v2.0")
            elapsed = time.monotonic() - start

            assert report.sample_count_baseline == 1000
            assert report.sample_count_candidate == 1000
            assert report.overall_delta > 0  # v2 is better
            assert report.has_regression is False
            assert elapsed < 5.0, f"Comparison took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_compare_versions_with_regression_large(self) -> None:
        """Detect regression across large datasets."""
        with DuckDBStorage() as s:
            v1 = [
                _make_eval_result(i, model_version="v1.0", aggregate_score=4.0) for i in range(500)
            ]
            v2 = [
                _make_eval_result(i, model_version="v2.0", aggregate_score=3.0) for i in range(500)
            ]
            s.store_results(v1)
            s.store_results(v2)

            tracker = RegressionTracker(storage=s, threshold=-0.1)
            report = tracker.compare_versions("scale-model", "v1.0", "v2.0")

            assert report.has_regression is True
            assert report.overall_delta < 0

    @pytest.mark.slow
    def test_version_history_with_many_versions(self) -> None:
        """Get version history when there are 20 versions."""
        with DuckDBStorage() as s:
            for v in range(20):
                for i in range(10):
                    s.store_result(
                        _make_eval_result(
                            i + v * 10,
                            model_version=f"v{v}.0",
                        )
                    )

            tracker = RegressionTracker(storage=s)
            versions = tracker.get_version_history("scale-model")
            assert len(versions) == 20

    @pytest.mark.slow
    def test_record_batch_1000(self) -> None:
        """Record a batch of 1000 results via tracker."""
        with DuckDBStorage() as s:
            tracker = RegressionTracker(storage=s)
            results = [_make_eval_result(i) for i in range(1000)]

            start = time.monotonic()
            tracker.record_batch(results)
            elapsed = time.monotonic() - start

            assert s.count_results() == 1000
            assert elapsed < 10.0

    @pytest.mark.slow
    def test_report_generation_at_scale(self) -> None:
        """Generate reports for large comparison."""
        with DuckDBStorage() as s:
            v1 = [
                _make_eval_result(i, model_version="v1.0", aggregate_score=3.5) for i in range(500)
            ]
            v2 = [
                _make_eval_result(i, model_version="v2.0", aggregate_score=3.8) for i in range(500)
            ]
            s.store_results(v1)
            s.store_results(v2)

            tracker = RegressionTracker(storage=s)
            report = tracker.compare_versions("scale-model", "v1.0", "v2.0")

            reporter = RegressionReporter()

            # All format generations should be fast
            start = time.monotonic()
            md = reporter.to_markdown(report)
            json_str = reporter.to_json(report)
            console = reporter.to_console(report)
            d = reporter.to_dict(report)
            elapsed = time.monotonic() - start

            assert len(md) > 0
            assert len(json_str) > 0
            assert len(console) > 0
            assert isinstance(d, dict)
            assert elapsed < 2.0


# ===========================================================================
# Section 4: Comparator Batch Scaling
# ===========================================================================


class TestComparatorBatchScaling:
    """Test compare_batch with large numbers of output pairs."""

    @pytest.mark.slow
    def test_compare_batch_100_pairs_exact(self) -> None:
        """Compare 100 output pairs using EXACT method."""
        comp = OutputComparator()
        pairs = [
            (
                f"Output {i} from model A",
                f"Output {i} from model A" if i % 2 == 0 else f"Output {i} from model B",
            )
            for i in range(100)
        ]

        start = time.monotonic()
        results = comp.compare_batch(pairs, ComparisonMethod.EXACT)
        elapsed = time.monotonic() - start

        assert len(results) == 100
        matches = sum(1 for r in results if r.is_match)
        assert matches == 50  # every even index matches
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_compare_batch_200_pairs_fuzzy(self) -> None:
        """Compare 200 output pairs using FUZZY method."""
        comp = OutputComparator(similarity_threshold=0.8)
        pairs = [
            (
                f"The quick brown fox jumps over the lazy dog number {i}",
                f"The quick brown fox leaps over the lazy dog number {i}",
            )
            for i in range(200)
        ]

        start = time.monotonic()
        results = comp.compare_batch(pairs, ComparisonMethod.FUZZY)
        elapsed = time.monotonic() - start

        assert len(results) == 200
        # All should be similar (high fuzzy match)
        matches = sum(1 for r in results if r.is_match)
        assert matches >= 190  # at least 95% should match
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_compare_batch_500_pairs_exact(self) -> None:
        """Compare 500 output pairs using EXACT method."""
        comp = OutputComparator()
        pairs = [(f"output-{i}", f"output-{i}") for i in range(500)]

        start = time.monotonic()
        results = comp.compare_batch(pairs, ComparisonMethod.EXACT)
        elapsed = time.monotonic() - start

        assert len(results) == 500
        assert all(r.is_match for r in results)
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_compare_batch_100_pairs_structural(self) -> None:
        """Compare 100 output pairs using STRUCTURAL method."""
        comp = OutputComparator(similarity_threshold=0.7)
        pairs = [
            (
                f"Line 1 of output {i}\nLine 2 of output {i}\nLine 3",
                f"First line {i}\nSecond line {i}\nThird line",
            )
            for i in range(100)
        ]

        start = time.monotonic()
        results = comp.compare_batch(pairs, ComparisonMethod.STRUCTURAL)
        elapsed = time.monotonic() - start

        assert len(results) == 100
        # Structural similarity should be high (same structure)
        matches = sum(1 for r in results if r.is_match)
        assert matches >= 80  # most should match structurally
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_compare_batch_mixed_lengths(self) -> None:
        """Compare pairs with varying text lengths."""
        comp = OutputComparator(similarity_threshold=0.5)
        pairs = []
        for i in range(150):
            text_a = "word " * (i + 1)
            text_b = "word " * (i + 1) if i % 3 == 0 else "different " * (i + 1)
            pairs.append((text_a.strip(), text_b.strip()))

        start = time.monotonic()
        results = comp.compare_batch(pairs, ComparisonMethod.FUZZY)
        elapsed = time.monotonic() - start

        assert len(results) == 150
        assert elapsed < 15.0


# ===========================================================================
# Section 5: Stress Tests
# ===========================================================================


class TestStressTests:
    """Stress tests pushing system boundaries."""

    @pytest.mark.slow
    def test_rapid_sequential_inserts(self) -> None:
        """Insert 500 results one at a time (not batched) to test insert throughput."""
        with DuckDBStorage() as s:
            start = time.monotonic()
            for i in range(500):
                s.store_result(_make_eval_result(i))
            elapsed = time.monotonic() - start

            assert s.count_results() == 500
            assert elapsed < 15.0, f"500 sequential inserts took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_alternating_write_read(self) -> None:
        """Alternate between writes and reads to test consistency."""
        with DuckDBStorage() as s:
            for i in range(100):
                s.store_result(_make_eval_result(i))
                count = s.count_results()
                assert count == i + 1

    @pytest.mark.slow
    def test_large_metadata(self) -> None:
        """Store results with large metadata dictionaries."""
        with DuckDBStorage() as s:
            for i in range(50):
                metadata = {f"key_{k}": f"value_{k}_" * 100 for k in range(20)}
                result = EvalResult(
                    id=f"meta-{i}",
                    model_id="m",
                    input_text="i",
                    output_text="o",
                    metadata=metadata,
                )
                s.store_result(result)

            assert s.count_results() == 50
            results = s.get_results(limit=50)
            assert len(results) == 50

    @pytest.mark.slow
    def test_many_scores_per_result(self) -> None:
        """Store results that each have 20 judge scores."""
        with DuckDBStorage() as s:
            for i in range(50):
                scores = [
                    JudgeScore(
                        judge_id=f"judge-{j}",
                        criterion=f"criterion-{j}",
                        score=float(j % 5) + 1,
                        reasoning=f"Reasoning for criterion {j}",
                    )
                    for j in range(20)
                ]
                result = EvalResult(
                    id=f"multi-score-{i}",
                    model_id="m",
                    input_text=f"input-{i}",
                    output_text=f"output-{i}",
                    scores=scores,
                    aggregate_score=3.0,
                )
                s.store_result(result)

            assert s.count_results() == 50

    @pytest.mark.slow
    def test_regression_comparison_speed(self) -> None:
        """Benchmark regression comparison on 2000 results (1000 per version)."""
        with DuckDBStorage() as s:
            v1 = [
                _make_eval_result(i, model_version="v1.0", aggregate_score=3.0)
                for i in range(1000)
            ]
            v2 = [
                _make_eval_result(i, model_version="v2.0", aggregate_score=3.5)
                for i in range(1000)
            ]
            s.store_results(v1)
            s.store_results(v2)

            tracker = RegressionTracker(storage=s)

            # Benchmark 10 comparisons
            start = time.monotonic()
            for _ in range(10):
                report = tracker.compare_versions("scale-model", "v1.0", "v2.0")
            elapsed = time.monotonic() - start

            assert report.sample_count_baseline == 1000
            # 10 comparisons on 2000 results should take < 10 seconds
            assert elapsed < 10.0, f"10 comparisons took {elapsed:.2f}s"
