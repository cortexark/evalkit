# ADR-002: DuckDB Over PostgreSQL for Analytics Storage

## Status

Accepted

## Date

2026-03-01

## Context

evalkit needs a storage backend for evaluation results that supports:

1. **Analytical queries:** Aggregations across model versions, time-series trends, per-criterion breakdowns
2. **Zero-config setup:** Users should run evalkit without provisioning infrastructure
3. **Embedded operation:** The framework is a library, not a service
4. **Schema evolution:** Adding new columns as the evaluation model grows
5. **Reasonable scale:** Thousands to low millions of evaluation records

## Decision

We use **DuckDB** as the primary storage backend.

### Why DuckDB

- **Embedded, zero-config:** DuckDB runs in-process as a library. No server, no ports, no Docker. `pip install duckdb` is the only requirement.

- **Columnar analytics:** DuckDB uses a columnar storage format optimized for analytical queries (aggregations, group-by, window functions). Evaluation workloads are almost entirely analytical.

- **Single-file database:** The entire database is one file, trivially portable. Users can commit it to version control, copy it between machines, or share it with teammates.

- **SQL interface:** Standard SQL for queries. No ORM overhead, no custom query language.

- **In-memory mode:** For testing and ephemeral workflows, DuckDB runs entirely in memory with `:memory:`.

- **Python-native:** First-class Python API with zero-copy NumPy/Pandas integration when needed for the dashboard.

### Storage schema

```sql
CREATE TABLE eval_results (
    id              VARCHAR PRIMARY KEY,
    model_id        VARCHAR NOT NULL,
    model_version   VARCHAR NOT NULL,
    input_text      VARCHAR NOT NULL,
    output_text     VARCHAR NOT NULL,
    reference_text  VARCHAR DEFAULT '',
    aggregate_score DOUBLE,
    rubric_name     VARCHAR DEFAULT 'default',
    created_at      VARCHAR NOT NULL,
    scores_json     VARCHAR DEFAULT '{}',
    metadata_json   VARCHAR DEFAULT '{}'
);
```

Complex nested data (individual judge scores, metadata) is stored as JSON strings within VARCHAR columns. DuckDB supports JSON extraction functions for querying into these fields.

## Trade-offs

### Scale ceiling
DuckDB handles millions of rows comfortably on a single machine. Beyond ~100M rows or multi-user concurrent write workloads, PostgreSQL would be more appropriate. For evalkit's target use case (team-scale evaluation), this ceiling is acceptable.

### No concurrent writes
DuckDB supports single-writer semantics. Multiple processes cannot write to the same database file simultaneously. Mitigation: evaluation pipelines are typically single-process batch jobs. For multi-process scenarios, use separate database files and merge.

### No network access
DuckDB is embedded-only; there is no client-server mode. Teams needing shared access must share the database file or export to a shared store. Mitigation: the reporter module exports to JSON/Markdown for sharing.

## Alternatives Considered

1. **PostgreSQL:** Rejected for v1. Requires a running server, connection management, and operational overhead. Excellent choice for production services but inappropriate for an evaluation library meant to run locally.

2. **SQLite:** Considered. SQLite is embedded and zero-config, but its row-oriented storage is suboptimal for analytical aggregation queries. DuckDB outperforms SQLite 10-100x on typical evalkit query patterns.

3. **Parquet files:** Considered for cold storage. Parquet is excellent for archival and interchange but lacks the query interface needed for interactive analysis. May be added as an export format.

4. **Redis/in-memory stores:** Rejected. No persistence, no analytical query support.

## Consequences

- Users install evalkit and start evaluating immediately -- no infrastructure setup
- Analytical queries over evaluation history are fast and SQL-native
- The database file is portable and easy to back up
- Multi-user concurrent writes require coordination outside evalkit
- Migration to PostgreSQL is straightforward if needed (same SQL, swap the connection)
