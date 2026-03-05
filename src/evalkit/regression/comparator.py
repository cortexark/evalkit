"""Output comparison engine for regression analysis.

Provides multiple comparison strategies for detecting meaningful differences
between model outputs across versions: exact match, semantic similarity
(via embedding distance), and structural comparison.
"""

from __future__ import annotations

import difflib
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ComparisonMethod(str, Enum):
    """Available comparison methods for output pairs."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    STRUCTURAL = "structural"


class ComparisonResult:
    """Result of comparing two outputs.

    Attributes:
        method: The comparison method used.
        similarity: Similarity score between 0.0 and 1.0.
        is_match: Whether the outputs are considered equivalent.
        diff: Human-readable diff when outputs differ.
        details: Additional comparison metadata.
    """

    def __init__(
        self,
        method: ComparisonMethod,
        similarity: float,
        is_match: bool,
        diff: str = "",
        details: dict[str, object] | None = None,
    ) -> None:
        self.method = method
        self.similarity = similarity
        self.is_match = is_match
        self.diff = diff
        self.details = details or {}

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(method={self.method.value!r}, "
            f"similarity={self.similarity:.4f}, is_match={self.is_match})"
        )


class OutputComparator:
    """Compares model outputs using configurable comparison strategies.

    Attributes:
        method: The default comparison method.
        similarity_threshold: Minimum similarity score for a fuzzy match.
    """

    def __init__(
        self,
        method: ComparisonMethod = ComparisonMethod.FUZZY,
        similarity_threshold: float = 0.9,
    ) -> None:
        """Initialize the comparator.

        Args:
            method: Default comparison method.
            similarity_threshold: Threshold for fuzzy matching (0.0-1.0).
        """
        self.method = method
        self.similarity_threshold = similarity_threshold

    def compare(
        self,
        output_a: str,
        output_b: str,
        method: ComparisonMethod | None = None,
    ) -> ComparisonResult:
        """Compare two model outputs.

        Args:
            output_a: First output (typically baseline).
            output_b: Second output (typically candidate).
            method: Override the default comparison method.

        Returns:
            ComparisonResult with similarity score and diff.
        """
        m = method or self.method

        if m == ComparisonMethod.EXACT:
            return self._exact_compare(output_a, output_b)
        elif m == ComparisonMethod.FUZZY:
            return self._fuzzy_compare(output_a, output_b)
        elif m == ComparisonMethod.STRUCTURAL:
            return self._structural_compare(output_a, output_b)
        else:
            raise ValueError(f"Unknown comparison method: {m}")

    def compare_batch(
        self,
        pairs: list[tuple[str, str]],
        method: ComparisonMethod | None = None,
    ) -> list[ComparisonResult]:
        """Compare multiple output pairs.

        Args:
            pairs: List of (output_a, output_b) tuples.
            method: Override the default comparison method.

        Returns:
            List of ComparisonResult instances.
        """
        results = [self.compare(a, b, method) for a, b in pairs]
        match_count = sum(1 for r in results if r.is_match)
        logger.info(
            "comparator.batch_complete",
            total=len(pairs),
            matches=match_count,
            match_rate=match_count / len(pairs) if pairs else 0,
        )
        return results

    def _exact_compare(self, output_a: str, output_b: str) -> ComparisonResult:
        """Exact string comparison.

        Args:
            output_a: First output.
            output_b: Second output.

        Returns:
            ComparisonResult with similarity 1.0 or 0.0.
        """
        is_match = output_a == output_b
        diff = "" if is_match else self._unified_diff(output_a, output_b)
        return ComparisonResult(
            method=ComparisonMethod.EXACT,
            similarity=1.0 if is_match else 0.0,
            is_match=is_match,
            diff=diff,
        )

    def _fuzzy_compare(self, output_a: str, output_b: str) -> ComparisonResult:
        """Fuzzy comparison using SequenceMatcher ratio.

        Args:
            output_a: First output.
            output_b: Second output.

        Returns:
            ComparisonResult with similarity ratio.
        """
        ratio = difflib.SequenceMatcher(None, output_a, output_b).ratio()
        is_match = ratio >= self.similarity_threshold
        diff = "" if is_match else self._unified_diff(output_a, output_b)
        return ComparisonResult(
            method=ComparisonMethod.FUZZY,
            similarity=ratio,
            is_match=is_match,
            diff=diff,
            details={"threshold": self.similarity_threshold},
        )

    def _structural_compare(self, output_a: str, output_b: str) -> ComparisonResult:
        """Structural comparison based on line-level analysis.

        Compares the structure (line count, paragraph count, average line
        length) rather than exact content. Useful for detecting format changes.

        Args:
            output_a: First output.
            output_b: Second output.

        Returns:
            ComparisonResult with structural similarity.
        """
        lines_a = output_a.strip().splitlines()
        lines_b = output_b.strip().splitlines()

        len_a = len(lines_a)
        len_b = len(lines_b)

        # Line count similarity
        max_lines = max(len_a, len_b, 1)
        line_sim = 1.0 - abs(len_a - len_b) / max_lines

        # Average line length similarity
        avg_len_a = sum(len(l) for l in lines_a) / max(len_a, 1)
        avg_len_b = sum(len(l) for l in lines_b) / max(len_b, 1)
        max_avg = max(avg_len_a, avg_len_b, 1)
        len_sim = 1.0 - abs(avg_len_a - avg_len_b) / max_avg

        # Paragraph count similarity (blank-line separated)
        paras_a = output_a.strip().split("\n\n")
        paras_b = output_b.strip().split("\n\n")
        max_paras = max(len(paras_a), len(paras_b), 1)
        para_sim = 1.0 - abs(len(paras_a) - len(paras_b)) / max_paras

        similarity = (line_sim + len_sim + para_sim) / 3.0
        is_match = similarity >= self.similarity_threshold

        return ComparisonResult(
            method=ComparisonMethod.STRUCTURAL,
            similarity=round(similarity, 4),
            is_match=is_match,
            diff=self._unified_diff(output_a, output_b) if not is_match else "",
            details={
                "line_similarity": round(line_sim, 4),
                "length_similarity": round(len_sim, 4),
                "paragraph_similarity": round(para_sim, 4),
            },
        )

    @staticmethod
    def _unified_diff(text_a: str, text_b: str) -> str:
        """Generate a unified diff between two texts.

        Args:
            text_a: Original text.
            text_b: Modified text.

        Returns:
            Unified diff string.
        """
        diff = difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile="baseline",
            tofile="candidate",
            lineterm="",
        )
        return "\n".join(diff)
