# Changelog

All notable changes to evalkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-05

### Added
- Core evaluation models with Pydantic v2
- LLM-as-Judge engine with configurable prompts
- Ensemble judge with weighted average, majority vote, and unanimous strategies
- 4 pre-built rubrics: relevance, coherence, faithfulness, safety
- Synthetic data generation with 4 strategies (paraphrase, adversarial, edge case, augmentation)
- Regression tracking and comparison with DuckDB backend
- Regression reporter with markdown and JSON output
- Streamlit dashboard for evaluation visualization
- YAML-based configuration system
- GitHub Actions CI pipeline (Python 3.11/3.12/3.13)
- Architecture Decision Records (3 ADRs)
- Example scripts for basic evaluation and custom rubrics
