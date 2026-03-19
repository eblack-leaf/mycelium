# mycelium

Natural language → SurrealQL. Runs locally, embedded.

## Goal

Query a personal SurrealDB database in plain English. No large language models, no server — fully embedded, inference only.

## Pipeline

```
NL prompt
  → septa   dep parse → typed span nodes (NP, Comparison, Root, Modifier)
  → hyphae  schema + valid ops as possibility graph
            inject: cross-edges from span nodes to schema candidates
            SageConv: propagate features through combined graph
            bilinear head: resolve each span node to schema target
  → stipe   arrange resolved nodes → QueryIr
            render → deterministic SurrealQL
```

## Crates

- **septa** — dep parser, extracts typed span nodes from NL. No schema knowledge.
- **hyphae** — schema graph, GNN (SageConv + bilinear head), predictions
- **stipe** — Mycelium interface, QueryIr assembly, SurrealQL render

## Key ideas

- Dep parse preserves compositional structure — "older than 30" stays a unit, not isolated tokens
- Possibility graph encodes everything valid for a given schema — ops, fields, comparators, modifiers
- GNN weights are structural and general — trained once on synthetic SurrealQL schemas
- Schema specifics enter through node features and graph topology, not node identity
- render() is fully deterministic — no generation, no hallucination
- Query history used for template selection at arrange step — no model
