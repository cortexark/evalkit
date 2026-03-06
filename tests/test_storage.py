"""Tests for DuckDB storage backend."""

from __future__ import annotations

from evalkit.core.models import EvalResult
from evalkit.core.storage import DuckDBStorage


class TestDuckDBStorage:
    """Tests for the DuckDBStorage class."""

    def test_initialize_in_memory(self) -> None:
        storage = DuckDBStorage(db_path=":memory:")
        assert storage.db_path == ":memory:"
        storage.close()

    def test_store_and_retrieve_single_result(
        self, in_memory_storage: DuckDBStorage, sample_eval_result: EvalResult
    ) -> None:
        in_memory_storage.store_result(sample_eval_result)
        results = in_memory_storage.get_results(model_id="gpt-4o")
        assert len(results) == 1
        assert results[0]["model_id"] == "gpt-4o"

    def test_store_batch(self, in_memory_storage: DuckDBStorage) -> None:
        results = [
            EvalResult(
                id=f"batch-{i}",
                model_id="batch-model",
                model_version="v1",
                input_text=f"input {i}",
                output_text=f"output {i}",
                aggregate_score=float(i),
            )
            for i in range(5)
        ]
        in_memory_storage.store_results(results)
        stored = in_memory_storage.get_results(model_id="batch-model")
        assert len(stored) == 5

    def test_filter_by_model_version(self, storage_with_data: DuckDBStorage) -> None:
        v1_results = storage_with_data.get_results(model_id="test-model", model_version="v1.0")
        assert len(v1_results) == 5
        assert all(r["model_version"] == "v1.0" for r in v1_results)

    def test_filter_by_rubric(self, storage_with_data: DuckDBStorage) -> None:
        results = storage_with_data.get_results(rubric_name="Test Rubric")
        assert len(results) == 10

    def test_count_results(self, storage_with_data: DuckDBStorage) -> None:
        total = storage_with_data.count_results()
        assert total == 10

        v1_count = storage_with_data.count_results(model_id="test-model", model_version="v1.0")
        assert v1_count == 5

    def test_get_distinct_versions(self, storage_with_data: DuckDBStorage) -> None:
        versions = storage_with_data.get_distinct_versions("test-model")
        assert set(versions) == {"v1.0", "v2.0"}

    def test_get_version_scores(self, storage_with_data: DuckDBStorage) -> None:
        scores = storage_with_data.get_version_scores("test-model", "v1.0")
        assert len(scores) == 5
        assert all(s["aggregate_score"] is not None for s in scores)

    def test_limit_parameter(self, storage_with_data: DuckDBStorage) -> None:
        results = storage_with_data.get_results(limit=3)
        assert len(results) == 3

    def test_upsert_on_duplicate_id(self, in_memory_storage: DuckDBStorage) -> None:
        result1 = EvalResult(
            id="dup-1",
            model_id="m",
            input_text="in",
            output_text="out1",
            aggregate_score=3.0,
        )
        result2 = EvalResult(
            id="dup-1",
            model_id="m",
            input_text="in",
            output_text="out2",
            aggregate_score=4.0,
        )
        in_memory_storage.store_result(result1)
        in_memory_storage.store_result(result2)
        results = in_memory_storage.get_results()
        assert len(results) == 1
        assert results[0]["output_text"] == "out2"

    def test_context_manager(self) -> None:
        with DuckDBStorage(db_path=":memory:") as storage:
            result = EvalResult(
                model_id="m",
                input_text="in",
                output_text="out",
            )
            storage.store_result(result)
            assert storage.count_results() == 1

    def test_empty_query(self, in_memory_storage: DuckDBStorage) -> None:
        results = in_memory_storage.get_results(model_id="nonexistent")
        assert results == []

    def test_count_empty(self, in_memory_storage: DuckDBStorage) -> None:
        assert in_memory_storage.count_results() == 0
