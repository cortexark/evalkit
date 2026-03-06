# evalkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/cortexark/evalkit/ci.yml?branch=main&label=CI)](https://github.com/cortexark/evalkit/actions)
[![Coverage](https://img.shields.io/codecov/c/github/cortexark/evalkit)](https://codecov.io/gh/cortexark/evalkit)

**Production-grade LLM evaluation framework** with judge ensembles, synthetic data generation, regression tracking, and an analytics dashboard.

---

evalkit addresses a core problem in LLM-powered product development: **how do you know if your model got better or worse?** Traditional metrics like BLEU and ROUGE correlate poorly with human judgment on open-ended generation tasks. evalkit replaces them with structured, LLM-as-judge evaluation that scales beyond human annotation budgets while producing explainable, per-criterion scores.

Built for engineering teams that ship LLM features and need to catch regressions before they reach production.

## Architecture

```
                         +------------------+
                         |   User / CI/CD   |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
           +--------v--------+        +--------v--------+
           |  Judge Engine   |        | Synthetic Gen.  |
           |  (LLM / Ens.)  |        |  (LLM-backed)   |
           +--------+--------+        +--------+--------+
                    |                           |
                    |     +-----------+         |
                    +---->|  Pydantic |<--------+
                          |  Models   |
                          +-----+-----+
                                |
                    +-----------+-----------+
                    |                       |
           +--------v--------+    +--------v--------+
           | Regression      |    | DuckDB Storage  |
           | Tracker         +--->| Backend         |
           +--------+--------+    +-----------------+
                    |
           +--------v--------+
           | Dashboard       |
           | (Streamlit)     |
           +-----------------+
```

### Data Flow

```
Rubric + Strategy                     Your Model
     |                                     |
     v                                     v
SyntheticGenerator ──> test inputs ──> inference ──> (input, output) pairs
                                                          |
                                                          v
                                             LLMJudge / EnsembleJudge
                                                          |
                                                          v
                                                   list[JudgeScore]
                                                          |
                                                          v
                                           EvalResult ──> DuckDB Storage
                                                          |
                                                          v
                                           RegressionTracker.compare_versions()
                                                          |
                                              +-----------+-----------+
                                              |                       |
                                              v                       v
                                     Console / Markdown        Streamlit Dashboard
```

### Functional Goals

| Goal | How It Works |
|------|-------------|
| **Evaluate LLM outputs** | Structured rubrics scored by LLM judges with per-criterion reasoning |
| **Detect regressions** | Automated version comparison with configurable thresholds |
| **Generate test data** | Synthetic inputs via 4 strategies (standard, adversarial, edge case, distribution) |
| **Visualize trends** | Interactive Streamlit dashboard backed by DuckDB analytics |

### Non-Functional Goals

| Goal | Design Decision |
|------|----------------|
| **Zero-config setup** | DuckDB embedded storage -- no database server required |
| **Reproducibility** | Immutable Pydantic models (`frozen=True`) + schema-versioned rubrics |
| **Provider independence** | Thin LLM abstraction -- swap OpenAI/Anthropic/Gemini without code changes |
| **CI/CD integration** | All operations scriptable, JSON output, non-zero exit on regression |
| **Cost efficiency** | ~$2.50 per 1,000 evaluations at GPT-4o pricing |

## What This System Does Best

- **Catches regressions before production.** Compare model versions on every CI run -- know within minutes if quality degraded.
- **Replaces expensive human annotation.** LLM-as-judge at ~$2.50/1,000 samples vs $50-125 for human annotators.
- **Produces explainable scores.** Per-criterion reasoning ("factual accuracy dropped on medical queries") instead of opaque BLEU numbers.
- **Composable subsystems.** Use just the judge, just the tracker, or just the generator -- no framework lock-in.

## Limitations

- **Judge quality depends on the judge model.** Blind spots in the judge model produce inflated scores. Mitigate with ensemble voting and golden-set calibration.
- **No concurrent writes.** DuckDB single-writer model means parallel CI jobs need separate database files.
- **Cost scales linearly.** 10,000 samples × 3 judges = 30,000 API calls. Start with representative samples (~100-500).
- **No built-in inference.** evalkit evaluates outputs but does not run models -- intentionally framework-agnostic.
- **Single-machine ceiling.** DuckDB handles ~100M rows; beyond that, migrate to a columnar warehouse.

## Quick Start

### Install

```bash
pip install evalkit
# or with all optional dependencies:
pip install evalkit[all]
```

### 5-Line Evaluation

```python
from evalkit.core.models import EvalResult, JudgeScore
from evalkit.core.storage import DuckDBStorage
from evalkit.regression.tracker import RegressionTracker

storage = DuckDBStorage(db_path="./evals.duckdb")
tracker = RegressionTracker(storage=storage)
result = EvalResult(
    model_id="my-model", model_version="v2.1",
    input_text="Summarize this article...",
    output_text="The article discusses...",
    aggregate_score=4.2,
)
tracker.record(result)
```

## Features

### Judge Engine

Evaluate LLM outputs against structured rubrics using LLM-as-judge or multi-judge ensembles.

```python
from evalkit.judges.llm_judge import LLMJudge
from evalkit.judges.rubrics import SUMMARIZATION_RUBRIC

judge = LLMJudge(judge_id="gpt4o", rubric=SUMMARIZATION_RUBRIC)
scores = judge.evaluate(
    input_text="Summarize: The mitochondria is the powerhouse...",
    output_text="Mitochondria generate cellular energy via ATP.",
)
for s in scores:
    print(f"{s.criterion}: {s.score}/5 -- {s.reasoning}")
```

**Pre-built rubrics:** Summarization, Factual Accuracy, Helpfulness, Safety

### Ensemble Judges

Combine multiple judges for robust evaluation with three voting strategies:

```python
from evalkit.judges.ensemble import EnsembleJudge
from evalkit.core.models import VotingStrategy

ensemble = EnsembleJudge(
    judge_id="panel",
    rubric=SUMMARIZATION_RUBRIC,
    judges=[(judge_gpt, 1.0), (judge_claude, 1.0)],
    voting_strategy=VotingStrategy.WEIGHTED_AVERAGE,
)
consensus = ensemble.evaluate(input_text="...", output_text="...")
```

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `WEIGHTED_AVERAGE` | Weighted mean of scores | General evaluation |
| `MAJORITY` | Most common score wins | Binary/categorical |
| `UNANIMOUS` | Minimum score (conservative) | Safety-critical |

### Custom Rubrics

Define evaluation criteria tailored to your use case:

```python
from evalkit.core.models import Rubric, RubricCriteria, ScoreScale

rubric = Rubric(
    name="API Response Quality",
    criteria=[
        RubricCriteria(
            name="Schema Compliance",
            description="Does the response match the expected JSON schema?",
            weight=3.0, scale=ScoreScale.BINARY,
        ),
        RubricCriteria(
            name="Data Accuracy",
            description="Are the returned values correct?",
            weight=2.0, scale=ScoreScale.LIKERT_5,
        ),
    ],
)
```

### Synthetic Data Generation

Generate diverse test inputs using strategy-specific templates:

```python
from evalkit.generators.synthetic import SyntheticGenerator
from evalkit.generators.templates import GenerationStrategy

gen = SyntheticGenerator(strategy=GenerationStrategy.ADVERSARIAL)
test_cases = gen.generate("customer support chatbot", count=20)
```

| Strategy | What It Generates |
|----------|-------------------|
| `STANDARD` | Typical user queries across difficulty levels |
| `ADVERSARIAL` | Inputs designed to expose model weaknesses |
| `EDGE_CASE` | Boundary conditions, unusual formats, empty inputs |
| `DISTRIBUTION_MATCHING` | Inputs matching production traffic distribution |

### Regression Tracking

Track evaluation scores across model versions and detect regressions:

```python
from evalkit.regression.tracker import RegressionTracker
from evalkit.regression.reporter import RegressionReporter

tracker = RegressionTracker(storage=storage, threshold=-0.1)
tracker.record_batch(v1_results)
tracker.record_batch(v2_results)

report = tracker.compare_versions("my-model", "v1.0", "v2.0")
reporter = RegressionReporter()
print(reporter.to_console(report))
```

Output:
```
========================================================================
  REGRESSION REPORT: ALL CLEAR
========================================================================
  Model:     my-model
  Baseline:  v1.0 (100 samples)
  Candidate: v2.0 (100 samples)
  Delta:     +0.3200
------------------------------------------------------------------------
  Criterion            Base     Cand    Delta   Reg?
  --------------------------------------------------------------------
  aggregate            3.8500   4.1700  +0.3200     no
========================================================================
```

### Output Comparison

Compare model outputs across versions using multiple strategies:

```python
from evalkit.regression.comparator import OutputComparator, ComparisonMethod

comparator = OutputComparator(similarity_threshold=0.9)
result = comparator.compare(baseline_output, candidate_output, ComparisonMethod.FUZZY)
print(f"Similarity: {result.similarity:.2%}, Match: {result.is_match}")
```

### Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
evalkit
# or: streamlit run src/evalkit/dashboard/app.py
```

### Configuration

evalkit supports YAML configuration:

```yaml
project_name: my-eval-project
storage:
  database_path: ./evals.duckdb
ensemble:
  voting_strategy: weighted_average
  judges:
    - judge_id: gpt4o
      judge_type: llm
      llm:
        provider: openai
        model: gpt-4o
        api_key_env_var: OPENAI_API_KEY
      weight: 1.0
```

```python
from evalkit.core.config import EvalConfig
config = EvalConfig.from_yaml("evalkit.yml")
```

## Architecture Decisions

- [ADR-001: LLM-as-Judge Architecture](docs/adr/001-llm-as-judge-architecture.md) -- Why LLM judges over heuristic scoring
- [ADR-002: DuckDB Over PostgreSQL](docs/adr/002-duckdb-over-postgres.md) -- Embedded columnar storage for zero-config analytics
- [ADR-003: Ensemble Voting Strategy](docs/adr/003-ensemble-voting-strategy.md) -- Majority vs weighted vs unanimous voting

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/cortexark/evalkit.git
cd evalkit
make dev

# Run tests
make test

# Lint and format
make lint
make format

# Type check
make typecheck

# Full CI pipeline locally
make ci
```

### Project Structure

```
src/evalkit/
  core/        -- Pydantic models, config, DuckDB storage
  judges/      -- BaseJudge, LLMJudge, EnsembleJudge, rubrics
  generators/  -- Synthetic data generation pipeline
  regression/  -- Tracker, comparator, reporter
  dashboard/   -- Streamlit visualization
tests/         -- pytest suite with fixtures
docs/adr/      -- Architecture Decision Records
examples/      -- Runnable usage examples
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests first (TDD -- every feature starts with a failing test)
4. Implement the feature
5. Run `make ci` to verify lint, types, and tests pass
6. Open a pull request with a clear description

### Guidelines

- All public methods must have docstrings with Args/Returns sections
- Type hints on every function signature
- No hardcoded API keys -- use environment variables via `LLMProviderConfig`
- Pydantic models should be frozen (`frozen=True`)
- New features need corresponding tests with >85% coverage

## License

[MIT](LICENSE)
