"""Custom rubric example using evalkit.

Demonstrates:
- Building a custom evaluation rubric
- Configuring an LLM judge with a custom rubric
- Setting up an ensemble of judges with different voting strategies
- Using pre-built rubrics as starting points

NOTE: This example requires API keys to run the LLM judge.
      Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment.
      The rubric and ensemble setup works without API keys.
"""

from evalkit.core.config import LLMProviderConfig
from evalkit.core.models import Rubric, RubricCriteria, ScoreScale, VotingStrategy
from evalkit.judges.ensemble import EnsembleJudge
from evalkit.judges.llm_judge import LLMJudge
from evalkit.judges.rubrics import SUMMARIZATION_RUBRIC, build_rubric


def main() -> None:
    # ---------------------------------------------------------------------------
    # 1. Build a custom rubric from scratch
    # ---------------------------------------------------------------------------
    code_review_rubric = Rubric(
        name="Code Review Quality",
        description="Evaluates the quality of LLM-generated code review comments.",
        version="1.0.0",
        criteria=[
            RubricCriteria(
                name="Specificity",
                description=(
                    "Does the review comment identify a specific issue with a clear "
                    "explanation of what is wrong and why it matters?"
                ),
                weight=3.0,
                scale=ScoreScale.LIKERT_5,
            ),
            RubricCriteria(
                name="Actionability",
                description=(
                    "Does the review comment suggest a concrete fix or improvement "
                    "that the author can implement?"
                ),
                weight=2.0,
                scale=ScoreScale.LIKERT_5,
            ),
            RubricCriteria(
                name="Tone",
                description=(
                    "Is the review comment constructive and professional, avoiding "
                    "dismissive or aggressive language?"
                ),
                weight=1.0,
                scale=ScoreScale.LIKERT_5,
            ),
            RubricCriteria(
                name="Correctness",
                description=(
                    "Is the technical assessment in the review comment accurate? "
                    "Does it correctly identify real issues rather than false positives?"
                ),
                weight=3.0,
                scale=ScoreScale.LIKERT_5,
            ),
        ],
    )

    print(f"Custom rubric: {code_review_rubric.name}")
    print(f"  Criteria: {len(code_review_rubric.criteria)}")
    print(f"  Total weight: {code_review_rubric.total_weight}")
    print()

    # ---------------------------------------------------------------------------
    # 2. Build a rubric using the shorthand builder
    # ---------------------------------------------------------------------------
    quick_rubric = build_rubric(
        name="Quick QA Check",
        criteria_defs=[
            {"name": "Correct", "description": "Is the answer correct?", "weight": 3.0},
            {"name": "Complete", "description": "Is the answer complete?", "weight": 2.0},
            {"name": "Clear", "description": "Is the answer clear?", "weight": 1.0},
        ],
        scale=ScoreScale.LIKERT_5,
    )
    print(f"Quick rubric: {quick_rubric.name}")
    print(f"  Criteria: {[c.name for c in quick_rubric.criteria]}")
    print()

    # ---------------------------------------------------------------------------
    # 3. Use a pre-built rubric
    # ---------------------------------------------------------------------------
    print(f"Pre-built rubric: {SUMMARIZATION_RUBRIC.name}")
    print(f"  Criteria: {[c.name for c in SUMMARIZATION_RUBRIC.criteria]}")
    print()

    # ---------------------------------------------------------------------------
    # 4. Configure judges (LLM calls require API keys)
    # ---------------------------------------------------------------------------
    openai_config = LLMProviderConfig(
        provider="openai",
        model="gpt-4o",
        api_key_env_var="OPENAI_API_KEY",
        temperature=0.0,
    )

    anthropic_config = LLMProviderConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        temperature=0.0,
    )

    judge_gpt = LLMJudge(
        judge_id="gpt4o-judge",
        rubric=code_review_rubric,
        llm_config=openai_config,
    )

    judge_claude = LLMJudge(
        judge_id="claude-judge",
        rubric=code_review_rubric,
        llm_config=anthropic_config,
    )

    print(f"Judge 1: {judge_gpt}")
    print(f"Judge 2: {judge_claude}")
    print()

    # ---------------------------------------------------------------------------
    # 5. Create an ensemble with weighted voting
    # ---------------------------------------------------------------------------
    ensemble = EnsembleJudge(
        judge_id="code-review-ensemble",
        rubric=code_review_rubric,
        judges=[
            (judge_gpt, 1.0),      # equal weight
            (judge_claude, 1.0),   # equal weight
        ],
        voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
    )

    print(f"Ensemble: {ensemble.judge_id}")
    print(f"  Strategy: {ensemble.voting_strategy.value}")
    print(f"  Judges: {len(ensemble.judges)}")
    print()

    # ---------------------------------------------------------------------------
    # 6. To actually run evaluation (requires API keys):
    # ---------------------------------------------------------------------------
    print("To run an actual evaluation:")
    print()
    print("  scores = ensemble.evaluate(")
    print('      input_text="Review this code: def add(a, b): return a + b",')
    print('      output_text="This function lacks type hints and a docstring.",')
    print("  )")
    print("  for score in scores:")
    print("      print(f'{score.criterion}: {score.score} -- {score.reasoning}')")


if __name__ == "__main__":
    main()
