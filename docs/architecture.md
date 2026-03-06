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

### End-to-End Evaluation Pipeline

```
User defines Rubric + selects GenerationStrategy
     |
     v
SyntheticGenerator.generate(domain, count)
     |
     v
list[str] (test inputs)
     |
     +---> User runs their model on each input (external)
     |
     v
list[(input, output)] pairs
     |
     v
LLMJudge.evaluate(input, output) -- or EnsembleJudge for multi-judge
     |
     v
list[JudgeScore] per sample
     |
     v
EvalResult(model_id, version, scores, aggregate)
     |
     v
DuckDBStorage.store_results(results)
     |
     v
RegressionTracker.compare_versions("v1", "v2")
     |
     v
RegressionReport { deltas[], has_regression }
     |
     +---> RegressionReporter.to_console() / .to_markdown() / .to_json()
     |
     +---> Dashboard (Streamlit) reads from DuckDB
```

### Ensemble Judging Flow

```
Input + Output
     |
     +---> Judge A (e.g., GPT-4o)  --> list[JudgeScore]
     |
     +---> Judge B (e.g., Claude)  --> list[JudgeScore]
     |
     +---> Judge C (e.g., Gemini)  --> list[JudgeScore]
     |
     v
EnsembleJudge._aggregate(all_scores, strategy)
     |
     +---> WEIGHTED_AVERAGE: sum(score * weight) / sum(weights)
     +---> MAJORITY: most common score wins
     +---> UNANIMOUS: min(scores) -- conservative
     |
     v
list[JudgeScore] (consensus scores with combined reasoning)
```

### Regression Detection Flow

```
DuckDBStorage
     |
     +---> query baseline (model_id, v1) --> list[EvalResult]
     +---> query candidate (model_id, v2) --> list[EvalResult]
     |
     v
RegressionTracker._compute_deltas()
     |
     v
per-criterion delta = candidate_mean - baseline_mean
     |
     +---> if delta < -threshold --> REGRESSION DETECTED
     +---> if delta >= 0         --> IMPROVEMENT or STABLE
     |
     v
RegressionReport
     |
     v
OutputComparator.compare(baseline_output, candidate_output)
     |
     +---> EXACT: strict equality
     +---> FUZZY: normalized similarity ratio
     +---> SEMANTIC: embedding cosine similarity
```

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

## What This System Does Best

1. **Catches regressions before production.** The regression tracker compares model versions on every CI run. Teams know within minutes if a prompt change, fine-tune, or model swap degraded quality -- before any user sees the output.

2. **Replaces expensive human annotation at scale.** LLM-as-judge evaluation costs ~$2.50 per 1,000 samples (GPT-4o at standard pricing). A human annotation team doing the same work costs 10-50x more and takes days instead of minutes. Ensemble voting with 3 judges still costs under $10 per 1,000 samples.

3. **Produces explainable, per-criterion scores.** Unlike BLEU/ROUGE which output a single number, evalkit returns structured rubric scores with reasoning. Engineers can see *why* a score dropped -- "factual accuracy degraded on medical queries" is actionable; "BLEU went from 0.43 to 0.41" is not.

4. **Zero-config local development.** DuckDB means no database server to install, no connection strings to manage, no Docker compose. Clone, install, run. The entire evaluation history lives in a single `.duckdb` file that can be committed, shared, or backed up.

5. **Composable subsystems.** Each piece works independently. Use just the judge engine for one-off evaluations. Use just the regression tracker with your own scoring. Use just the synthetic generator to build test sets. No framework lock-in.

## Limitations

1. **LLM judge quality depends on the judge model.** If the judge model has blind spots (e.g., poor math reasoning), it will give inflated scores on tasks it cannot evaluate well. Mitigation: use ensemble voting with diverse providers and calibrate against a human-labeled golden set.

2. **No concurrent write support.** DuckDB uses a single-writer model. Two CI jobs writing to the same database file simultaneously will fail. For teams with parallel CI pipelines, use separate database files per job and merge results, or migrate to PostgreSQL using the provided adapter interface.

3. **Cost scales linearly with sample count and judge count.** Evaluating 10,000 samples with a 3-judge ensemble requires 30,000 LLM API calls. There is no caching or deduplication of identical inputs across runs. Teams should start with small representative samples (~100-500) and scale up selectively.

4. **No built-in model inference.** evalkit evaluates outputs but does not run models. Users must implement their own inference loop and pass (input, output) pairs to the judge. This is intentional -- evalkit stays framework-agnostic -- but it means more integration code.

5. **Single-machine scale ceiling.** DuckDB handles ~100M rows on a single machine before query performance degrades. For teams generating millions of evaluation records per month, plan a migration to a columnar warehouse (BigQuery, Snowflake) using the storage adapter pattern.

6. **Rubric drift over time.** As products evolve, rubrics need manual updates. There is no automatic detection of criteria becoming stale or irrelevant. Teams should review rubrics quarterly alongside prompt and model changes.

## Key Decisions

- [ADR-001: LLM-as-Judge Architecture](adr/001-llm-as-judge-architecture.md)
- [ADR-002: DuckDB Over PostgreSQL](adr/002-duckdb-over-postgres.md)
- [ADR-003: Ensemble Voting Strategy](adr/003-ensemble-voting-strategy.md)
