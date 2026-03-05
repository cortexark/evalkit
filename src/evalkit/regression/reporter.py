"""Regression report generation in multiple output formats.

Produces human-readable and machine-parsable reports from regression
analysis results. Supports markdown, JSON, and console table formats.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from evalkit.core.models import RegressionReport

logger = structlog.get_logger(__name__)


class RegressionReporter:
    """Generates formatted regression reports from comparison data.

    Supports multiple output formats and configurable detail levels.
    """

    def to_markdown(self, report: RegressionReport) -> str:
        """Render a regression report as a Markdown document.

        Args:
            report: The regression report to render.

        Returns:
            Formatted Markdown string.
        """
        status = "REGRESSION DETECTED" if report.has_regression else "PASS"
        status_icon = "[FAIL]" if report.has_regression else "[PASS]"

        lines = [
            f"# Regression Report {status_icon}",
            "",
            f"**Model:** {report.model_id}",
            f"**Baseline:** {report.baseline_version} "
            f"({report.sample_count_baseline} samples)",
            f"**Candidate:** {report.candidate_version} "
            f"({report.sample_count_candidate} samples)",
            f"**Status:** {status}",
            f"**Overall Delta:** {report.overall_delta:+.4f}",
            f"**Generated:** {report.created_at.isoformat()}",
            "",
            "## Per-Criterion Deltas",
            "",
            "| Criterion | Baseline | Candidate | Delta | Relative | Regression? |",
            "|-----------|----------|-----------|-------|----------|-------------|",
        ]

        for delta in report.deltas:
            reg_flag = "YES" if delta.is_regression else "no"
            lines.append(
                f"| {delta.criterion} "
                f"| {delta.baseline_score:.4f} "
                f"| {delta.candidate_score:.4f} "
                f"| {delta.delta:+.4f} "
                f"| {delta.relative_delta_pct:+.2f}% "
                f"| {reg_flag} |"
            )

        lines.extend(["", "---", f"*Report ID: {report.id}*"])
        return "\n".join(lines)

    def to_json(self, report: RegressionReport) -> str:
        """Render a regression report as a JSON string.

        Args:
            report: The regression report to render.

        Returns:
            JSON string.
        """
        return report.model_dump_json(indent=2)

    def to_dict(self, report: RegressionReport) -> dict[str, Any]:
        """Convert a regression report to a dictionary.

        Args:
            report: The regression report to convert.

        Returns:
            Dictionary representation.
        """
        return report.model_dump(mode="json")

    def to_console(self, report: RegressionReport) -> str:
        """Render a regression report as a console-friendly table.

        Args:
            report: The regression report to render.

        Returns:
            Formatted string for terminal display.
        """
        status = "REGRESSION DETECTED" if report.has_regression else "ALL CLEAR"
        width = 72

        lines = [
            "=" * width,
            f"  REGRESSION REPORT: {status}",
            "=" * width,
            f"  Model:     {report.model_id}",
            f"  Baseline:  {report.baseline_version} ({report.sample_count_baseline} samples)",
            f"  Candidate: {report.candidate_version} ({report.sample_count_candidate} samples)",
            f"  Delta:     {report.overall_delta:+.4f}",
            "-" * width,
        ]

        if report.deltas:
            header = f"  {'Criterion':<20} {'Base':>8} {'Cand':>8} {'Delta':>8} {'Reg?':>6}"
            lines.append(header)
            lines.append("  " + "-" * (width - 4))

            for d in report.deltas:
                reg = " YES" if d.is_regression else "  no"
                lines.append(
                    f"  {d.criterion:<20} "
                    f"{d.baseline_score:>8.4f} "
                    f"{d.candidate_score:>8.4f} "
                    f"{d.delta:>+8.4f} "
                    f"{reg:>6}"
                )

        lines.append("=" * width)
        return "\n".join(lines)

    def save_report(
        self,
        report: RegressionReport,
        path: str,
        fmt: str = "markdown",
    ) -> None:
        """Save a regression report to a file.

        Args:
            report: The regression report to save.
            path: File path to write.
            fmt: Output format ("markdown", "json").
        """
        if fmt == "json":
            content = self.to_json(report)
        else:
            content = self.to_markdown(report)

        with open(path, "w") as f:
            f.write(content)

        logger.info("reporter.saved", path=path, format=fmt)
