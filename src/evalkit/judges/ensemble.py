"""Ensemble judge that aggregates scores from multiple judge instances.

Supports majority voting, weighted average, and unanimous agreement strategies.
See ADR-003 for the rationale behind voting strategy choices.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict

import structlog

from evalkit.core.models import JudgeScore, Rubric, VotingStrategy
from evalkit.judges.base import BaseJudge

logger = structlog.get_logger(__name__)


class EnsembleJudge(BaseJudge):
    """Aggregates evaluations from multiple judges using a voting strategy.

    The ensemble collects scores from each constituent judge, then applies
    the configured voting strategy to produce a single set of consensus scores.

    Attributes:
        judges: List of (judge, weight) tuples.
        voting_strategy: The aggregation strategy to use.
    """

    def __init__(
        self,
        judge_id: str,
        rubric: Rubric,
        judges: list[tuple[BaseJudge, float]] | None = None,
        voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED_AVERAGE,
    ) -> None:
        """Initialize the ensemble judge.

        Args:
            judge_id: Unique identifier for this ensemble.
            rubric: Evaluation rubric (shared across all judges).
            judges: List of (judge_instance, weight) pairs.
            voting_strategy: How to aggregate scores across judges.
        """
        super().__init__(judge_id=judge_id, rubric=rubric)
        self.judges: list[tuple[BaseJudge, float]] = judges or []
        self.voting_strategy = voting_strategy

    def add_judge(self, judge: BaseJudge, weight: float = 1.0) -> None:
        """Add a judge to the ensemble.

        Args:
            judge: Judge instance to add.
            weight: Weight for this judge's scores in aggregation.
        """
        self.judges.append((judge, weight))
        logger.info(
            "ensemble.judge_added",
            ensemble_id=self.judge_id,
            judge_id=judge.judge_id,
            weight=weight,
        )

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate by collecting scores from all judges and aggregating.

        Args:
            input_text: The original prompt.
            output_text: The model response to evaluate.
            reference_text: Optional reference answer.

        Returns:
            Aggregated list of JudgeScore instances, one per criterion.

        Raises:
            ValueError: If no judges are configured.
        """
        if not self.judges:
            raise ValueError("Ensemble has no judges configured.")

        all_scores: list[tuple[list[JudgeScore], float]] = []
        for judge, weight in self.judges:
            scores = judge.evaluate(input_text, output_text, reference_text)
            all_scores.append((scores, weight))

        return self._aggregate(all_scores)

    async def aevaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate asynchronously, running all judges concurrently.

        Args:
            input_text: The original prompt.
            output_text: The model response to evaluate.
            reference_text: Optional reference answer.

        Returns:
            Aggregated list of JudgeScore instances, one per criterion.

        Raises:
            ValueError: If no judges are configured.
        """
        if not self.judges:
            raise ValueError("Ensemble has no judges configured.")

        tasks = [
            judge.aevaluate(input_text, output_text, reference_text) for judge, _ in self.judges
        ]
        results = await asyncio.gather(*tasks)

        all_scores: list[tuple[list[JudgeScore], float]] = [
            (scores, weight) for scores, (_, weight) in zip(results, self.judges)
        ]

        return self._aggregate(all_scores)

    def _aggregate(self, all_scores: list[tuple[list[JudgeScore], float]]) -> list[JudgeScore]:
        """Dispatch to the appropriate voting strategy.

        Args:
            all_scores: List of (judge_scores, weight) pairs.

        Returns:
            Aggregated JudgeScore instances.
        """
        if self.voting_strategy == VotingStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(all_scores)
        elif self.voting_strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(all_scores)
        elif self.voting_strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous(all_scores)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def _weighted_average(
        self, all_scores: list[tuple[list[JudgeScore], float]]
    ) -> list[JudgeScore]:
        """Compute weighted average scores across judges per criterion.

        Args:
            all_scores: List of (judge_scores, weight) pairs.

        Returns:
            Averaged JudgeScore instances.
        """
        criterion_scores: dict[str, list[tuple[float, float, str]]] = defaultdict(list)

        for scores, weight in all_scores:
            for s in scores:
                criterion_scores[s.criterion].append((s.score, weight, s.reasoning))

        result: list[JudgeScore] = []
        for criterion, entries in criterion_scores.items():
            total_weight = sum(w for _, w, _ in entries)
            if total_weight == 0:
                avg_score = 0.0
            else:
                avg_score = sum(s * w for s, w, _ in entries) / total_weight

            reasonings = [r for _, _, r in entries if r]
            combined_reasoning = " | ".join(reasonings) if reasonings else ""

            result.append(
                JudgeScore(
                    judge_id=self.judge_id,
                    criterion=criterion,
                    score=round(avg_score, 4),
                    reasoning=f"[weighted_average] {combined_reasoning}",
                )
            )

        logger.info(
            "ensemble.aggregated",
            strategy="weighted_average",
            criteria_count=len(result),
        )
        return result

    def _majority_vote(self, all_scores: list[tuple[list[JudgeScore], float]]) -> list[JudgeScore]:
        """Use majority voting -- the most common score wins per criterion.

        For continuous scales, scores are rounded to the nearest integer
        before counting votes. Ties are broken by choosing the higher score.

        Args:
            all_scores: List of (judge_scores, weight) pairs.

        Returns:
            Majority-voted JudgeScore instances.
        """
        criterion_votes: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for scores, _ in all_scores:
            for s in scores:
                rounded = round(s.score)
                criterion_votes[s.criterion].append((rounded, s.reasoning))

        result: list[JudgeScore] = []
        for criterion, votes in criterion_votes.items():
            score_counts = Counter(v for v, _ in votes)
            max_count = max(score_counts.values())
            # Break ties by choosing the highest score
            winning_score = max(s for s, c in score_counts.items() if c == max_count)

            reasonings = [r for v, r in votes if v == winning_score and r]
            combined = reasonings[0] if reasonings else ""

            result.append(
                JudgeScore(
                    judge_id=self.judge_id,
                    criterion=criterion,
                    score=float(winning_score),
                    reasoning=f"[majority_vote] {combined}",
                )
            )

        logger.info(
            "ensemble.aggregated",
            strategy="majority",
            criteria_count=len(result),
        )
        return result

    def _unanimous(self, all_scores: list[tuple[list[JudgeScore], float]]) -> list[JudgeScore]:
        """Require unanimous agreement -- return minimum score per criterion.

        If judges disagree, the most conservative (lowest) score is chosen.
        This is appropriate for safety-critical evaluations.

        Args:
            all_scores: List of (judge_scores, weight) pairs.

        Returns:
            Conservative JudgeScore instances.
        """
        criterion_scores: dict[str, list[tuple[float, str]]] = defaultdict(list)

        for scores, _ in all_scores:
            for s in scores:
                criterion_scores[s.criterion].append((s.score, s.reasoning))

        result: list[JudgeScore] = []
        for criterion, entries in criterion_scores.items():
            min_score = min(s for s, _ in entries)
            max_score = max(s for s, _ in entries)
            unanimous = min_score == max_score

            min_reasoning = next((r for s, r in entries if s == min_score and r), "")
            status = "unanimous" if unanimous else "disagreement-conservative"

            result.append(
                JudgeScore(
                    judge_id=self.judge_id,
                    criterion=criterion,
                    score=min_score,
                    reasoning=f"[{status}] {min_reasoning}",
                )
            )

        logger.info(
            "ensemble.aggregated",
            strategy="unanimous",
            criteria_count=len(result),
        )
        return result
