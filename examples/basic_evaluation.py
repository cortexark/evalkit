"""Basic evaluation example using evalkit.

Demonstrates:
- Creating evaluation results with scores
- Storing results in DuckDB
- Tracking regressions between model versions
- Generating a regression report

This example runs entirely offline using pre-computed scores.
For LLM-backed evaluation, see custom_rubric.py.
"""

from evalkit.core.models import EvalResult, JudgeScore
from evalkit.core.storage import DuckDBStorage
from evalkit.regression.reporter import RegressionReporter
from evalkit.regression.tracker import RegressionTracker


def main() -> None:
    # 1. Create an in-memory storage backend
    storage = DuckDBStorage(db_path=":memory:")
    tracker = RegressionTracker(storage=storage, threshold=-0.2)

    # 2. Record evaluation results for model v1.0
    v1_results = [
        EvalResult(
            id=f"v1-{i}",
            model_id="my-summarizer",
            model_version="v1.0",
            input_text=f"Summarize document {i}",
            output_text=f"Summary of document {i} from v1",
            scores=[
                JudgeScore(
                    judge_id="manual",
                    criterion="Faithfulness",
                    score=3.5 + (i * 0.2),
                    reasoning="Generally faithful with minor omissions.",
                ),
                JudgeScore(
                    judge_id="manual",
                    criterion="Conciseness",
                    score=4.0,
                    reasoning="Appropriately concise.",
                ),
            ],
            aggregate_score=3.75 + (i * 0.1),
        )
        for i in range(5)
    ]
    tracker.record_batch(v1_results)

    # 3. Record evaluation results for model v2.0 (improved)
    v2_results = [
        EvalResult(
            id=f"v2-{i}",
            model_id="my-summarizer",
            model_version="v2.0",
            input_text=f"Summarize document {i}",
            output_text=f"Summary of document {i} from v2 (improved)",
            scores=[
                JudgeScore(
                    judge_id="manual",
                    criterion="Faithfulness",
                    score=4.0 + (i * 0.15),
                    reasoning="Highly faithful to source material.",
                ),
                JudgeScore(
                    judge_id="manual",
                    criterion="Conciseness",
                    score=4.5,
                    reasoning="Concise and well-structured.",
                ),
            ],
            aggregate_score=4.25 + (i * 0.08),
        )
        for i in range(5)
    ]
    tracker.record_batch(v2_results)

    # 4. Compare versions
    report = tracker.compare_versions("my-summarizer", "v1.0", "v2.0")

    # 5. Generate and print the report
    reporter = RegressionReporter()
    print(reporter.to_console(report))
    print()
    print(reporter.to_markdown(report))

    # 6. Check the verdict
    if report.has_regression:
        print("\nRegression detected! Investigate before deploying v2.0.")
    else:
        print(f"\nNo regression. v2.0 improved by {report.overall_delta:+.4f} points.")

    storage.close()


if __name__ == "__main__":
    main()
