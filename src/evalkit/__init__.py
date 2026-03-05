"""evalkit -- Production-grade LLM evaluation framework.

Provides judge ensembles, synthetic data generation, regression tracking,
and an analytics dashboard for systematic LLM output quality measurement.
"""

from evalkit.core.models import EvalResult, JudgeScore, RubricCriteria

__version__ = "0.1.0"

__all__ = [
    "EvalResult",
    "JudgeScore",
    "RubricCriteria",
    "__version__",
]
