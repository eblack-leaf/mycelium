# mycelium-gnn

Natural language to SurrealQL query resolution via a 3-stage pipeline:
rule-based NLP parse, cross-encoder candidate matching, GNN graph resolution.

## Setup

```bash
# Download pretrained models + GloVe (one time, ~1GB total)
./fetch_models.sh

# Generate training data (pick one or both)
cargo run --release --example gen_dataset -p gnn-burn       # synthetic (fake scores)
cargo run --release --example gen_dataset_nlp -p gnn-burn   # real (cross-encoder scores)

# Train
cargo run --release --example train -p gnn-burn

# Run end-to-end demo
cargo run --release --example pipeline_demo -p gnn-burn
```

## Models

| Model | What it does | File | Pretrained? |
|-------|-------------|------|-------------|
| **Bi-encoder** (sentence-transformers/all-MiniLM-L6-v2, ONNX) | Takes a text string, returns a 384-dim embedding vector. Used after span extraction to give each span a semantic representation. | `models/model.onnx` | Yes |
| **Cross-encoder** (cross-encoder/ms-marco-MiniLM-L-6-v2, ONNX) | Takes a (phrase, schema_name) pair, returns a 0-1 relevance score. Used to find which schema nodes each NL phrase might refer to. | `models/cross-encoder.onnx` | Yes |
| **GNN** (SAGEConv encoder + OutputHead) | Takes the combined graph (schema + linguistic + candidate edges), resolves each NL phrase to a schema role and target node. | `demo/gnn_model` | No, trained by you |

## Pipeline walkthrough

Concrete example: `"find goods where cost over 100"`

### Stage 1: NLP Parse

**Code:** `nlp.rs:194` `NlpModel::parse()`

The rule-based parser (`nlp.rs:216` `rule_based_parse()`) scans words left to right:

- `"find"` matches the intent word list → **Intent** node
- `"goods"` doesn't match any special pattern → **NounPhrase** node
- `"where"` is a delimiter → flushes the current phrase
- `"cost"` → **NounPhrase** node
- `"over"` matches the comparator word list → starts a **Comparator** node
- `"100"` is a number following a comparator → absorbed into the Comparator

Dependency edges are inferred from adjacency and span types:

- Intent→NounPhrase adjacency → `IntentTarget` edge
- Comparator before NounPhrase → `Comparison` edge

After span extraction, `encode_pooled()` (`nlp.rs:111`) runs each span's text
through the bi-encoder ONNX model to produce a 384-dim embedding. The parser
decided what the spans are; the bi-encoder just gives each one a vector.

```
Output LinguisticGraph:
  nodes:
    [0] Intent("find")                 embedding: [0.12, -0.04, ...]  384-dim
    [1] NounPhrase("goods")            embedding: [0.08, 0.31, ...]   384-dim
    [2] NounPhrase("cost")             embedding: [-0.02, 0.19, ...]  384-dim
    [3] Comparator("over 100")         embedding: [0.05, -0.11, ...]  384-dim
  edges:
    0 --IntentTarget--> 1    ("find" targets "goods")
    3 --Comparison-->   2    ("over 100" modifies "cost")
```

### Stage 2: Cross-encoder candidate matching

**Code:** `candidate_matcher.rs:78` `CandidateMatcher::match_candidates()`

For each linguistic node, the cross-encoder scores it against every schema name
(table names, field names, operation names). This is separate from the
bi-encoder — the cross-encoder sees both texts together as a pair and judges
relevance directly.

```
cross_encode("goods", "products") → 0.82   (high — "goods" likely means products)
cross_encode("goods", "users")    → 0.04   (low)
cross_encode("cost", "goods.cost")→ 0.88   (high — "cost" matches the cost field)
cross_encode("cost", "products.price") → 0.41
cross_encode("over 100", "gt")    → 0.79   (high — "over" implies greater-than)
cross_encode("find", "SELECT")    → 0.71
```

`collect_top_k()` (`candidate_matcher.rs:114`) keeps the top-k matches above a
minimum score threshold per node per schema type. No role assignment happens
here — "goods" might match a table or a field; the cross-encoder doesn't decide.

```
Output CandidateSet: ~20-40 edges, each with a score
```

### Stage 3: GNN Resolution

Three substeps:

#### 3a. Build combined graph

**Code:** `linguistic_graph.rs:144` `LinguisticConv::new()`

Merges three edge sets into one heterogeneous graph:

```
Schema edges (add_schema_edges, line 208):
  table:goods --has_field--> field:goods.cost       (and inverse)
  field:goods.cost --compatible_op--> op:gt          (and inverse)
  table:goods --table_op--> op:SELECT                (and inverse)
  ...

Linguistic edges (add_linguistic_edges, line 287):
  intent:"find" --targets--> np:"goods"              (and inverse)
  comp:"over 100" --comparison--> np:"cost"          (and inverse)

Candidate edges WITH WEIGHTS (add_candidate_edges, line 330):
  np:"goods" --candidate_table[0.82]--> table:products
  np:"goods" --candidate_table[0.12]--> table:goods
  np:"cost"  --candidate_field[0.88]--> field:goods.cost
  comp:"over 100" --candidate_op[0.79]--> op:gt
  intent:"find" --candidate_op[0.71]--> op:SELECT
  ...all bidirectional, scores stored in ConvRelation.weights
```

Candidate edge weights are the cross-encoder scores. They flow through
message passing so a 0.82 match contributes more than a 0.12 match
(`scatter_weighted_mean` in `tensor_ops.rs:79`).

#### 3b. Embed all nodes

**Code:** `embed.rs:146` `Embedder::embed_all()`

```
Schema nodes:  GloVe("goods") + type_embed("table") + [confidence, is_nl] → 68-dim
Ling nodes:    bi-encoder("goods") → 384-dim → ling_proj (Linear 384→68) → 68-dim
```

Schema nodes use GloVe + a learned type embedding. Linguistic nodes use the
384-dim bi-encoder output, projected down to 68-dim by a learned linear layer
(`training.rs:149` `project_linguistic()`). The projection is trained alongside
the GNN — not truncated.

#### 3c. Message passing (SAGEConv, 2 layers)

**Code:** `sage.rs:180` `Encoder::forward()`

Each layer: for every edge type, gather source features, weight by edge score,
scatter to destinations, linear transform + ReLU + L2 normalize.

```
Layer 1:
  np:"goods" receives from:
    table:products (weight 0.82)  ← strong signal from cross-encoder
    table:goods    (weight 0.12)  ← weak signal
    intent:"find"  (uniform)      ← structural signal
  → np:"goods" now carries table-like information

  np:"cost" receives from:
    field:goods.cost     (weight 0.88) ← strong
    field:products.price (weight 0.41) ← weaker
    comp:"over 100"      (uniform)     ← structural
  → np:"cost" now carries field-like information

Layer 2:
  np:"goods" also sees the UPDATED field:goods.cost
  (which received info from np:"cost" and op:gt in layer 1)
  → structural constraints propagate across the graph
```

#### 3d. Output head

**Code:** `head.rs:193` `OutputHead::forward()`

For each linguistic node, predicts:

1. **Role** — which schema role this phrase plays (Linear → 6 classes):
   - `np:"goods"` → Collection
   - `np:"cost"` → FilterField
   - `comp:"over 100"` → Modifier
   - `intent:"find"` → None

2. **Target** — which specific schema node (bilinear score, masked by candidates):
   - `np:"goods"` vs tables: products=0.9, goods=0.4 (others masked to -inf)
   - `np:"cost"` vs fields: goods.cost=0.95, products.price=0.3 (others masked)
   - `comp:"over 100"` vs ops: gt=0.88 (others masked)

The candidate mask (`head.rs:60` `CandidateMask`) adds -1e9 to all
(ling, schema) pairs that the cross-encoder didn't surface, so the head only
picks among pre-screened candidates.

**Result:** goods→Collection→table:products, cost→FilterField→field:goods.cost,
over 100→Modifier→op:gt, find→None

### Stage 4: SQL Emit (TODO)

**Code:** `orchestrator.rs` — not yet implemented.

Walk the resolved assignments and emit SurrealQL:
```sql
SELECT * FROM products WHERE cost > 100
```

## Training

Two dataset generators:

- **Synthetic** (`gen_dataset.rs` → `demo/dataset.json`): constructs
  LinguisticGraph and CandidateSet by hand with fabricated scores (correct
  target 0.65-0.95, distractors 0.05-0.45). Tests whether the GNN architecture
  can learn role+target resolution in isolation, assuming perfect upstream.

- **NLP** (`gen_dataset_nlp.rs` → `demo/dataset_nlp.json`): generates the same
  NL queries but runs them through the real NlpModel + CandidateMatcher. The
  cross-encoder scores are real — messier, sometimes the correct target isn't
  highest. Trains the GNN on what it will actually see at inference.

The GNN trains: `ling_proj` (384→68 projection), `type_embed` (learned type
vectors), `Encoder` (SAGEConv weights), `OutputHead` (role classifier + bilinear
target scoring). GloVe vectors are frozen.

## File map

```
src/
  nlp.rs                Stage 1: rule-based parse + bi-encoder embed + cross-encoder
  candidate_matcher.rs  Stage 2: cross-encoder scoring, top-k filtering
  linguistic_graph.rs   Stage 3 topology: combined graph with weighted candidate edges
  embed.rs              Stage 3 init: GloVe + type_embed (schema), bi-encoder + ling_proj (ling)
  sage.rs               Stage 3 GNN: SAGEConv with weighted scatter, HeteroConv, Encoder
  head.rs               Stage 3 output: role classifier + masked bilinear target scoring
  training.rs           Training loop, GnnModel, ling_proj, loss, accuracy, save/load
  tensor_ops.rs         Primitives: gather, scatter_mean, scatter_weighted_mean, l2_normalize
  schema.rs             SurrealQL schema parsing (.surql files)
  graph.rs              Schema → heterogeneous graph (tables, fields, record links)
  operations.rs         34 fixed SurrealQL operations (SELECT, gt, LIMIT, count, ...)
  orchestrator.rs       Stage 4 (TODO): ResolvedGraph → SurrealQL string
  lib.rs                Pipeline struct wiring stages 1-2

examples/
  gen_dataset.rs        Synthetic dataset generator
  gen_dataset_nlp.rs    Real NLP dataset generator
  train.rs              Training entrypoint
  pipeline_demo.rs      End-to-end inference demo

demo/
  schema.surql          Test schema (goods, users, posts, messages, products)
  dataset.json          Synthetic training data (gitignored)
  dataset_nlp.json      NLP training data (gitignored)
  glove.6B.50d.txt      GloVe vectors (gitignored, fetched by fetch_models.sh)
  gnn_model             Trained model weights (gitignored)

models/
  model.onnx            Bi-encoder ONNX (gitignored)
  tokenizer.json        Bi-encoder tokenizer (gitignored)
  cross-encoder.onnx    Cross-encoder ONNX (gitignored)
  cross-tokenizer.json  Cross-encoder tokenizer (gitignored)
```
