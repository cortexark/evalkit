"""Regression tracking with DuckDB for evaluation score history.

Stores evaluation results over time and provides methods to detect
regressions between model versions based on configurable thresholds.
"""

from __future__ import annotations

import structlog

from evalkit.core.models import EvalResult, RegressionDelta, RegressionReport
from evalkit.core.storage import DuckDBStorage

logger = structlog.get_logger(__name__)


class RegressionTracker:
    """Tracks evaluation results across model versions and detects regressions.

    Uses DuckDB storage to persist results and compute cross-version deltas.

    Attributes:
        storage: The DuckDB storage backend.
        threshold: Score delta below which a change counts as a regression.
    """

    def __init__(
        self,
        storage: DuckDBStorage,
        threshold: float = -0.1,
    ) -> None:
        """Initialize the regression tracker.

        Args:
            storage: DuckDB storage backend for persistence.
            threshold: Delta below which a score change is flagged as regression.
                       Use negative values (e.g. -0.1 means a 0.1-point drop).
        """
        self.storage = storage
        self.threshold = threshold

    def record(self, result: EvalResult) -> None:
        """Record a single evaluation result.

        Args:
            result: The evaluation result to persist.
        """
        self.storage.store_result(result)
        logger.debug(
            "tracker.recorded",
            result_id=result.id,
            model_id=result.model_id,
            model_version=result.model_version,
        )

    def record_batch(self, results: list[EvalResult]) -> None:
        """Record multiple evaluation results.

        Args:
            results: List of evaluation results to persist.
        """
        self.storage.store_results(results)
        logger.info("tracker.batch_recorded", count=len(results))

    def compare_versions(
        self,
        model_id: str,
        baseline_version: str,
        candidate_version: str,
    ) -> RegressionReport:
        """Compare evaluation scores between two model versions.

        Computes per-criterion deltas and flags regressions based on
        the configured threshold.

        Args:
            model_id: Identifier of the model.
            baseline_version: The reference version.
            candidate_version: The version being compared.

        Returns:
            A RegressionReport with per-criterion deltas.

        Raises:
            ValueError: If either version has no recorded results.
        """
        baseline_rows = self.storage.get_version_scores(model_id, baseline_version)
        candidate_rows = self.storage.get_version_scores(model_id, candidate_version)

        if not baseline_rows:
            raise ValueError(f"No results found for {model_id} version {baseline_version}")
        if not candidate_rows:
            raise ValueError(f"No results found for {model_id} version {candidate_version}")

        baseline_scores = [
            r["aggregate_score"] for r in baseline_rows if r["aggregate_score"] is not None
        ]
        candidate_scores = [
            r["aggregate_score"] for r in candidate_rows if r["aggregate_score"] is not None
        ]

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        candidate_avg = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0

        overall_delta = candidate_avg - baseline_avg
        relative_pct = (overall_delta / baseline_avg * 100) if baseline_avg != 0 else 0.0

        delta = RegressionDelta(
            criterion="aggregate",
            baseline_score=round(baseline_avg, 4),
            candidate_score=round(candidate_avg, 4),
            delta=round(overall_delta, 4),
            relative_delta_pct=round(relative_pct, 2),
            is_regression=overall_delta < self.threshold,
        )

        report = RegressionReport(
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            model_id=model_id,
            deltas=[delta],
            overall_delta=round(overall_delta, 4),
            sample_count_baseline=len(baseline_rows),
            sample_count_candidate=len(candidate_rows),
            has_regression=overall_delta < self.threshold,
        )

        logger.info(
            "tracker.comparison_complete",
            model_id=model_id,
            baseline=baseline_version,
            candidate=candidate_version,
            overall_delta=overall_delta,
            has_regression=report.has_regression,
        )

        return report

    def get_version_history(self, model_id: str) -> list[str]:
        """Get all recorded versions for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of version strings, newest first.
        """
        return self.storage.get_distinct_versions(model_id)
