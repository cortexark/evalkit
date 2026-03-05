# ADR-001: LLM-as-Judge Architecture

## Status

Accepted

## Date

2026-03-01

## Context

Evaluating LLM outputs at scale requires automated quality assessment. Traditional approaches rely on lexical metrics (BLEU, ROUGE) or hand-crafted heuristics, both of which correlate poorly with human judgment for open-ended generation tasks. We need an evaluation strategy that:

1. Scales beyond human annotation budgets
2. Handles subjective quality dimensions (helpfulness, coherence, safety)
3. Produces interpretable, explainable scores
4. Adapts to new evaluation criteria without re-engineering

## Decision

We adopt an **LLM-as-Judge** architecture where a capable language model evaluates outputs against structured rubrics. The judge receives the input, the model output, an optional reference, and a rubric defining criteria with scoring scales, then returns per-criterion scores with reasoning.

### Key design choices

- **Structured rubrics over free-form prompting:** Rubrics constrain the evaluation space, improve consistency across runs, and make scores comparable. Each rubric is a first-class data object (`Rubric` model) versioned independently of code.

- **Multi-provider support:** Judges can use OpenAI or Anthropic models. This avoids vendor lock-in and allows provider diversification in ensembles.

- **JSON-structured output:** The judge prompt requests JSON arrays, enabling deterministic parsing. A fallback parser handles markdown code fences that models sometimes emit.

- **Retry with exponential backoff:** Transient API failures are retried up to 3 times using `tenacity`, with exponential backoff to respect rate limits.

## Trade-offs

### Cost
LLM judge calls cost money per evaluation. At GPT-4o pricing (~$5/M input tokens), evaluating 1,000 samples with a 500-token prompt costs approximately $2.50. Mitigation: use cheaper models (GPT-4o-mini) for development, reserve expensive models for final evaluations.

### Latency
Each evaluation requires an LLM API round-trip (200ms-2s). Mitigation: async evaluation with concurrent judge calls, batch processing.

### Self-bias
An LLM may favor its own outputs or outputs similar to its training distribution. Mitigation: use cross-model evaluation (GPT-4o judging Claude outputs and vice versa), ensemble judges from different providers, and periodic calibration against human annotations.

### Calibration drift
Judge behavior may change as underlying models are updated. Mitigation: pin model versions, store raw judge responses for audit, version rubrics, and maintain a calibration test set with known scores.

## Alternatives Considered

1. **Lexical metrics (BLEU/ROUGE):** Rejected. Poor correlation with human judgment for generative tasks. Useful only as supplementary signals.

2. **Embedding similarity:** Rejected as primary metric. Captures semantic proximity but not quality dimensions like safety or helpfulness. Considered for regression comparison (ADR future).

3. **Human annotation:** Not rejected but not primary. Too expensive and slow for CI/CD integration. Used as calibration ground truth.

4. **Fine-tuned reward models:** Rejected for v1. Requires training data and infrastructure. May be adopted in future versions for specific domains.

## Consequences

- Every evaluation produces explainable, per-criterion scores with reasoning
- Rubrics can be versioned and iterated independently
- Cost scales linearly with evaluation volume
- Ensemble judges (ADR-003) can mitigate single-judge bias
- Storage backend (ADR-002) must persist raw responses for auditability
