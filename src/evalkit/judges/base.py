"""Abstract base class for all judge implementations.

Every judge in evalkit implements this interface, enabling polymorphic
evaluation pipelines and ensemble composition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from evalkit.core.models import EvalResult, JudgeScore, Rubric


class BaseJudge(ABC):
    """Abstract base for evaluation judges.

    Subclasses must implement both synchronous and asynchronous evaluation
    methods. The base class provides identity and shared utilities.

    Attributes:
        judge_id: Unique identifier for this judge instance.
        rubric: The evaluation rubric this judge applies.
    """

    def __init__(self, judge_id: str, rubric: Rubric) -> None:
        """Initialize the judge.

        Args:
            judge_id: Unique identifier for this judge.
            rubric: Evaluation rubric defining criteria and scales.
        """
        self.judge_id = judge_id
        self.rubric = rubric

    @abstractmethod
    def evaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate a model output synchronously.

        Args:
            input_text: The prompt or input given to the model.
            output_text: The model's response to evaluate.
            reference_text: Optional gold-standard reference.

        Returns:
            List of JudgeScore instances, one per rubric criterion.
        """

    @abstractmethod
    async def aevaluate(
        self,
        input_text: str,
        output_text: str,
        reference_text: str | None = None,
    ) -> list[JudgeScore]:
        """Evaluate a model output asynchronously.

        Args:
            input_text: The prompt or input given to the model.
            output_text: The model's response to evaluate.
            reference_text: Optional gold-standard reference.

        Returns:
            List of JudgeScore instances, one per rubric criterion.
        """

    def evaluate_result(self, result: EvalResult) -> list[JudgeScore]:
        """Convenience method to evaluate from an existing EvalResult.

        Args:
            result: An EvalResult containing input/output text.

        Returns:
            List of JudgeScore instances.
        """
        return self.evaluate(
            input_text=result.input_text,
            output_text=result.output_text,
            reference_text=result.reference_text,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(judge_id={self.judge_id!r}, rubric={self.rubric.name!r})"
        )
