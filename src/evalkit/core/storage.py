"""DuckDB storage backend for evaluation results and regression data.

DuckDB is chosen for its embedded, zero-config nature and columnar analytics
performance. See ADR-002 for the full rationale.
"""

from __future__ import annotations

from typing import Any

import duckdb
import structlog

from evalkit.core.models import EvalResult

logger = structlog.get_logger(__name__)

_EVAL_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS eval_results (
    id              VARCHAR PRIMARY KEY,
    model_id        VARCHAR NOT NULL,
    model_version   VARCHAR NOT NULL,
    input_text      VARCHAR NOT NULL,
    output_text     VARCHAR NOT NULL,
    reference_text  VARCHAR DEFAULT '',
    aggregate_score DOUBLE,
    rubric_name     VARCHAR DEFAULT 'default',
    created_at      VARCHAR NOT NULL,
    scores_json     VARCHAR DEFAULT '{}',
    metadata_json   VARCHAR DEFAULT '{}'
);
"""

_REGRESSION_REPORTS_DDL = """
CREATE TABLE IF NOT EXISTS regression_reports (
    id                   VARCHAR PRIMARY KEY,
    baseline_version     VARCHAR NOT NULL,
    candidate_version    VARCHAR NOT NULL,
    model_id             VARCHAR NOT NULL,
    rubric_name          VARCHAR DEFAULT 'default',
    overall_delta        DOUBLE DEFAULT 0.0,
    sample_count_baseline  INTEGER DEFAULT 0,
    sample_count_candidate INTEGER DEFAULT 0,
    has_regression       BOOLEAN DEFAULT FALSE,
    created_at           VARCHAR NOT NULL,
    deltas_json          VARCHAR DEFAULT '[]'
);
"""


class DuckDBStorage:
    """Manages persistence of evaluation data in DuckDB.

    Supports both file-backed and in-memory databases. All schema
    migrations are applied automatically on initialization.

    Attributes:
        db_path: Path to the DuckDB file, or ":memory:" for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize storage and apply schema migrations.

        Args:
            db_path: Path to the DuckDB database file, or ":memory:".
        """
        self.db_path = db_path
        self._conn = duckdb.connect(db_path)
        self._migrate()
        logger.info("storage.initialized", db_path=db_path)

    def _migrate(self) -> None:
        """Apply all schema DDL statements."""
        self._conn.execute(_EVAL_RESULTS_DDL)
        self._conn.execute(_REGRESSION_REPORTS_DDL)

    def store_result(self, result: EvalResult) -> None:
        """Persist a single evaluation result.

        Args:
            result: The EvalResult to store.
        """
        data = result.to_storage_dict()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO eval_results
                (id, model_id, model_version, input_text, output_text,
                 reference_text, aggregate_score, rubric_name, created_at,
                 scores_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                data["id"],
                data["model_id"],
                data["model_version"],
                data["input_text"],
                data["output_text"],
                data["reference_text"],
                data["aggregate_score"],
                data["rubric_name"],
                data["created_at"],
                data["scores_json"],
                data["metadata_json"],
            ],
        )
        logger.debug("storage.result_stored", result_id=result.id)

    def store_results(self, results: list[EvalResult]) -> None:
        """Persist multiple evaluation results in a batch.

        Args:
            results: List of EvalResult instances.
        """
        for result in results:
            self.store_result(result)
        logger.info("storage.batch_stored", count=len(results))

    def get_results(
        self,
        model_id: str | None = None,
        model_version: str | None = None,
        rubric_name: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query evaluation results with optional filters.

        Args:
            model_id: Filter by model identifier.
            model_version: Filter by model version.
            rubric_name: Filter by rubric name.
            limit: Maximum number of results to return.

        Returns:
            List of result dictionaries.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if model_version is not None:
            conditions.append("model_version = ?")
            params.append(model_version)
        if rubric_name is not None:
            conditions.append("rubric_name = ?")
            params.append(rubric_name)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM eval_results
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        result = self._conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_version_scores(self, model_id: str, model_version: str) -> list[dict[str, Any]]:
        """Get all results for a specific model version.

        Args:
            model_id: Model identifier.
            model_version: Version string.

        Returns:
            List of result dictionaries.
        """
        return self.get_results(model_id=model_id, model_version=model_version)

    def get_distinct_versions(self, model_id: str) -> list[str]:
        """List all distinct versions recorded for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of version strings, newest first.
        """
        result = self._conn.execute(
            """
            SELECT DISTINCT model_version
            FROM eval_results
            WHERE model_id = ?
            ORDER BY created_at DESC
            """,
            [model_id],
        )
        return [row[0] for row in result.fetchall()]

    def count_results(self, model_id: str | None = None, model_version: str | None = None) -> int:
        """Count evaluation results with optional filters.

        Args:
            model_id: Filter by model identifier.
            model_version: Filter by model version.

        Returns:
            Number of matching results.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if model_version is not None:
            conditions.append("model_version = ?")
            params.append(model_version)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT COUNT(*) FROM eval_results {where_clause}"

        result = self._conn.execute(query, params)
        row = result.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("storage.closed")

    def __enter__(self) -> DuckDBStorage:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
