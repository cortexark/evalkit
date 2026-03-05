# ADR-003: Ensemble Voting Strategy

## Status

Accepted

## Date

2026-03-01

## Context

Single-judge evaluation suffers from systematic biases, inconsistency across runs, and lack of robustness. Research on LLM-as-judge systems shows that:

- Individual LLM judges have measurable self-preference bias (Zheng et al., 2023)
- Judge agreement with human annotations improves with ensemble methods
- Different models have complementary strengths (e.g., GPT-4 on reasoning, Claude on safety)

We need an ensemble mechanism that aggregates scores from multiple judges into a single consensus evaluation.

## Decision

We implement three voting strategies in the `EnsembleJudge`, selectable per evaluation:

### 1. Weighted Average (default)

Each judge's score is weighted by a configurable factor, and the final score is the weighted mean.

```
final_score = sum(score_i * weight_i) / sum(weight_i)
```

**When to use:** General-purpose evaluation where all judges contribute proportionally to the final score. Weights can reflect judge quality (e.g., higher weight for a model known to correlate well with human judgment).

### 2. Majority Vote

Scores are rounded to integers and the most common score wins. Ties are broken by choosing the higher score (optimistic tiebreak).

**When to use:** Categorical or binary evaluations (pass/fail, safe/unsafe) where the question is "what do most judges think?" rather than "what is the average?"

### 3. Unanimous (Conservative)

The minimum score across all judges is selected. If any judge flags a problem, the evaluation inherits that concern.

**When to use:** Safety-critical evaluations where false negatives are more costly than false positives. A single judge identifying harmful content should override others that missed it.

## Implementation Details

- **Async concurrency:** When using `aevaluate()`, all judges run concurrently via `asyncio.gather()`. This reduces latency from `N * per-judge-latency` to `max(per-judge-latency)`.

- **Per-criterion aggregation:** Voting is applied independently per criterion. An ensemble evaluation of a 3-criterion rubric produces 3 aggregated scores, each computed from the judges' scores for that specific criterion.

- **Reasoning aggregation:** For weighted average, all judge reasonings are concatenated with `|` separators. For majority vote, the reasoning from a winning judge is used. For unanimous, the reasoning from the most conservative judge is used.

## Trade-offs

### Cost multiplication
An ensemble of 3 judges costs 3x a single judge. Mitigation: use ensembles selectively for high-stakes evaluations; use single judges for development iteration.

### Latency
Even with async concurrency, total latency is bounded by the slowest judge. Mitigation: set per-judge timeouts, use faster models where appropriate.

### Correlation assumption
Weighted average assumes judges are not perfectly correlated. If two judges always agree, adding both does not improve robustness. Mitigation: use judges from different providers (OpenAI + Anthropic) or different model sizes.

### Tiebreak policy
The optimistic tiebreak in majority voting (choosing the higher score on ties) may not suit all use cases. The conservative tiebreak alternative is available via the unanimous strategy.

## Calibration

We recommend periodic calibration:

1. Maintain a "golden set" of 50-100 evaluations with known human scores
2. Run each judge and the ensemble against the golden set
3. Compute Spearman correlation between judge scores and human scores
4. Adjust weights based on correlation (higher correlation = higher weight)
5. Track calibration drift over time via the regression tracker

## Alternatives Considered

1. **Learned aggregation (meta-judge):** A separate model that takes judge scores as input and produces a calibrated output. Rejected for v1 due to training data requirements. May be explored in v2.

2. **Median score:** Less sensitive to outliers than mean. Rejected because it discards information from extreme judges that may be capturing real quality issues.

3. **Bayesian aggregation:** Model each judge's bias and reliability, then compute posterior scores. Rejected for v1 complexity. Academically interesting but requires calibration data.

## Consequences

- Users can choose the aggregation strategy that matches their risk tolerance
- Ensemble evaluation produces more robust scores than any single judge
- Cost scales linearly with the number of judges in the ensemble
- The framework supports heterogeneous ensembles (mixing providers and models)
- Calibration data can be tracked through the regression module
