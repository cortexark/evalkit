"""Abstract base class for test data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    """Abstract base for generating evaluation test inputs.

    Generators produce structured test cases that can be fed into
    the evaluation pipeline. Each generator targets a specific
    generation strategy (adversarial, edge-case, etc.).

    Attributes:
        generator_id: Unique identifier for this generator instance.
    """

    def __init__(self, generator_id: str) -> None:
        """Initialize the generator.

        Args:
            generator_id: Unique identifier for this generator.
        """
        self.generator_id = generator_id

    @abstractmethod
    def generate(
        self,
        topic: str,
        count: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Generate test inputs synchronously.

        Args:
            topic: The subject domain for generated inputs.
            count: Number of test cases to generate.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of dicts with at minimum an "input" key.
        """

    @abstractmethod
    async def agenerate(
        self,
        topic: str,
        count: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        """Generate test inputs asynchronously.

        Args:
            topic: The subject domain for generated inputs.
            count: Number of test cases to generate.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of dicts with at minimum an "input" key.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(generator_id={self.generator_id!r})"
