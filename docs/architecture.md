# evalkit Architecture

## Overview

evalkit is a modular LLM evaluation framework built around four core subsystems: **Judge Engine**, **Synthetic Generator**, **Regression Tracker**, and **Analytics Dashboard**. Each subsystem is independently usable and composes with the others through shared Pydantic models.

## System Architecture

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

## Module Dependency Graph

```
evalkit.core.models      <-- Foundation: all other modules depend on this
evalkit.core.config      <-- Configuration management
evalkit.core.storage     <-- DuckDB persistence (depends on models)

evalkit.judges.base      <-- Abstract judge interface (depends on models)
evalkit.judges.rubrics   <-- Pre-built rubrics (depends on models)
evalkit.judges.llm_judge <-- LLM-backed judge (depends on base, config)
evalkit.judges.ensemble  <-- Multi-judge aggregation (depends on base)

evalkit.generators.base      <-- Abstract generator interface
evalkit.generators.templates <-- Prompt templates
evalkit.generators.synthetic <-- LLM-backed generator (depends on base, templates, config)

evalkit.regression.tracker    <-- Version tracking (depends on storage, models)
evalkit.regression.comparator <-- Output diff engine (standalone)
evalkit.regression.reporter   <-- Report formatting (depends on models)

evalkit.dashboard.app    <-- Streamlit UI (depends on storage)
```

## Data Flow

### Evaluation Pipeline

1. **Input Generation** (optional): `SyntheticGenerator` produces test inputs
2. **Model Inference** (external): User runs their model on the inputs
3. **Judging**: `LLMJudge` or `EnsembleJudge` scores each output against the rubric
4. **Storage**: `EvalResult` objects are persisted via `DuckDBStorage`
5. **Regression Analysis**: `RegressionTracker` compares scores across versions
6. **Reporting**: `RegressionReporter` produces Markdown/JSON reports
7. **Visualization**: `Dashboard` renders interactive charts

### Data Model Hierarchy

```
Rubric
  └── RubricCriteria[]

EvalResult
  ├── JudgeScore[]
  ├── model_id, model_version
  └── aggregate_score

RegressionReport
  ├── RegressionDelta[]
  ├── baseline_version, candidate_version
  └── has_regression
```

## Design Principles

1. **Composition over inheritance**: Judges, generators, and storage are composed through interfaces, not deep class hierarchies.

2. **Immutable models**: All Pydantic models use `frozen=True` to prevent mutation after creation, supporting safe concurrency and reproducibility.

3. **Async-ready**: All judge and generator methods have async counterparts (`aevaluate`, `agenerate`) for concurrent evaluation.

4. **Provider-agnostic**: LLM calls go through a thin provider abstraction. Adding a new provider requires implementing two HTTP calls (sync and async).

5. **Schema-versioned**: Rubrics carry version strings. Database migrations are idempotent. Evaluation results are immutable once stored.

## Key Decisions

- [ADR-001: LLM-as-Judge Architecture](adr/001-llm-as-judge-architecture.md)
- [ADR-002: DuckDB Over PostgreSQL](adr/002-duckdb-over-postgres.md)
- [ADR-003: Ensemble Voting Strategy](adr/003-ensemble-voting-strategy.md)
