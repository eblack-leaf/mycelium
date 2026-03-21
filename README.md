# Mycelium

Natural language to SurrealQL. A neural pipeline that translates conversational English into
structured database queries against a known schema.

```
"show me all tasks where priority is above 3 order by due_date"
                          |
                      Mycelium
                          |
SELECT * FROM task WHERE priority > 3 ORDER BY due_date
```

## Architecture

Mycelium is a two-stage model trained jointly end-to-end:

```
                      raw text
                         |
                  +------+------+
                  |    Septa    |   BiGRU span encoder
                  +------+------+
                         |
              Semantics + SpanHiddens
                         |
             +-----------+-----------+
             |        Hyphae         |   R-GCN + bilinear heads
             |  SchemaGraph.inject() |   with bilinear heads
             +-----------+-----------+
                         |
                      QueryIr
                         |
                    render(values)
                         |
                      SurrealQL
```

**Stage 1 -- Septa** identifies semantic spans in the input: intent verbs, entity references,
projection fields, conditions, assignments, and modifiers. A stacked BiGRU encodes the token
sequence, then mean-pools hidden states over each span's token range to produce fixed-size
span embeddings.

**Stage 2 -- Hyphae** grounds those spans against the database schema. A heterogeneous R-GCN
runs message passing over a graph containing schema nodes (tables, fields), fixed vocabulary
nodes (operations, comparators, modifier types), and the span nodes from Septa. Eight bilinear
resolution heads each score a span embedding against its candidate set to resolve the span to a
concrete schema element.

The two stages share a single autodiff graph during training -- gradients flow from the GNN
resolution loss back through the span projection into the BiGRU encoder.

### Resolution heads

| Head | Span type | Candidates | Output |
|------|-----------|------------|--------|
| intent | intent verb | 4 operations (SELECT/CREATE/UPDATE/DELETE) | `Intent` |
| entity | table reference | all tables in schema | table name |
| projection | field mention | entity-table fields | `table.field` |
| cond_field | condition LHS | entity-table fields | `table.field` |
| cond_cmp | comparator text | 7 comparators (=, !=, >, >=, <, <=, CONTAINS) | `Comparator` |
| assignment | assignment LHS | entity-table fields | `table.field` |
| mod_type | modifier phrase | 3 types (OrderBy, Limit, Fetch) | `ModifierKind` |
| mod_field | modifier argument | entity-table fields | `table.field` |

Field-resolving heads (projection, cond_field, assignment, mod_field) are entity-conditioned:
at resolve time, logits are masked to only consider fields belonging to the table selected by
the entity head. This eliminates cross-table ambiguity (e.g. "name" appearing on multiple tables).

### Graph topology

The grounded graph has three zones of nodes connected by 7 directed edge types:

**Schema structure** (static per schema):
- `HasField` / `FieldOf` -- table-field ownership and its reverse
- `LinksTo` / `LinkedFrom` -- record-type field references between tables

**Span routing** (added per query via `inject()`):
- `EntityToSpan` -- entity span broadcasts table context to field-resolving spans
- `SpanToTable` -- field-resolving spans send to all table nodes (bridge to field subgraph)
- `ProjectionToFetch` -- links projection spans to their fetch modifier when co-referenced

Trivially-solved heads (intent, entity, comparator, modifier type) have no graph edges --
BiGRU text embeddings + bilinear scoring is sufficient. The graph focuses entirely on
field resolution, where table context from EntityToSpan and the SpanToTable relay path
provide the disambiguation signal that text alone cannot.

Message passing uses per-edge-type linear projections with scatter-add aggregation,
ReLU activation, and L2 row normalization at each layer.

## Crates

```
mycelium/
  septa/       Span encoder (BiGRU + future CRF tagger)
  hyphae/      Schema graph, GNN, bilinear heads, QueryIr, SurrealQL renderer
  basidium/    Training data generation, training loop, inference CLI
  stipe/       Public interface -- loads pretrained model, runs full pipeline
  ui/          Tauri + SolidJS frontend (connects to SurrealDB via stipe)
```

**septa** -- Tokenizes input via character n-gram hashing into learned embedding buckets.
Stacked bidirectional GRU layers encode the sequence. Given known span boundaries (from
`Semantics`), mean-pools BiGRU hidden states over each span's token range to produce
`SpanHiddens` -- one vector per span.

**hyphae** -- Parses `.surql` schema files into `Schema`. Builds a `SchemaGraph` with
precomputed n-gram bucket indices for table/field names. `inject()` creates a per-query
`GroundedGraph` by adding span nodes and cross edges. The `Hyphae` model runs R-GCN
message passing then scores each resolution head via bilinear dot products. `resolve()`
argmaxes the logits into a `QueryIr`. `render()` produces SurrealQL.

**basidium** -- Generates synthetic training data from schema via template expansion with
natural language variation (~4000 datums from 6 tables). Handles train/val splitting,
dataset statistics. The training loop uses AdamW with cosine annealing, micro-batch
gradient accumulation, per-head accuracy tracking, and early stopping. Saves/loads model
checkpoints via Burn's binary recorder.

**stipe** -- Thin inference interface. `Mycelium::load()` reads schema + pretrained weights.
`query()` runs the full Septa-to-SurrealQL pipeline.

## Usage

### Generate training data

```sh
cargo run -p basidium -- generate
cargo run -p basidium -- stats
```

Writes `data/train.json` and `data/val.json`. Stats shows per-head label distributions.

### Train

```sh
cargo run --release -p basidium -- train
```

Trains Basidium (Septa + Hyphae jointly) with:
- Batch size 64, micro-batch size 8 (gradient accumulation)
- AdamW optimizer, cosine annealing from 1e-3 to 1e-5
- Early stopping with patience 10
- Best model saved to `weights/basidium/best.bin`

### Inference

```sh
cargo run --release -p basidium -- infer        # first 20 val datums
cargo run --release -p basidium -- infer data 50 # first 50
```

Loads pretrained weights and runs val datums through the full pipeline, printing
NL input alongside predicted SurrealQL and per-datum correctness.

## Schema

Define tables as `.surql` files in `stipe/fixtures/schema/`:

```sql
DEFINE TABLE task;
DEFINE FIELD title    ON task TYPE string;
DEFINE FIELD priority ON task TYPE int;
DEFINE FIELD assignee ON task TYPE record<user>;
DEFINE FIELD due_date ON task TYPE datetime;
```

Supported field types: `string`, `int`, `float`, `decimal`, `number`, `bool`, `datetime`,
`record<table>`, `array<type>`. Record fields create `LinksTo`/`LinkedFrom` edges in the
graph and enable Fetch modifier resolution.

## Stack

- [Burn](https://burn.dev) 0.20 -- deep learning framework (wgpu backend)
- [SurrealDB](https://surrealdb.com) -- target database
- [Tauri](https://tauri.app) + [SolidJS](https://solidjs.com) -- desktop UI
