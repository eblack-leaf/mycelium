# TODO

## hyphae — GNN

- [ ] `GroundedGraph::forward` — resolve all bilinear heads → `QueryIr`:
      for each Resolution, score span embedding against candidate embeddings,
      argmax to pick winning QueryNode, assemble QueryIr fields
- [ ] `Hyphae::forward` — wire SageConv message passing + bilinear heads:
      init node features → SageConv layers → per-head bilinear scoring → QueryIr
- [ ] `QueryIr::render` — deterministic SurrealQL string from resolved IR:
      SELECT [fields|*] FROM table[:id] [WHERE conds] [ORDER BY field [DESC]] [LIMIT n] [FETCH field]
      CREATE table[:id] SET field = val, ...
      UPDATE table[:id] SET field = val, ... [WHERE conds]
      DELETE table[:id] [WHERE conds]
      Substitute Slot(n) → values[n], normalise TemporalExpr at render time
- [ ] Node feature initialisation — embed QueryNode variants into fixed-dim vectors:
      schema nodes (table/field name embeddings), vocab nodes (learned embeddings for
      Operation/Comparator/Modifier), span nodes (from Septa hidden states or text embeddings)
- [ ] Bilinear heads — one per resolution type (intent, entity, projection, condition_field,
      condition_cmp, assignment, modifier_type, modifier_field)

## septa — BiLSTM-CRF

- [ ] `Septa::forward` — BiLSTM-CRF inference:
      word embeddings → BiLSTM → emission scores → CRF Viterbi decode → BIO tags
- [ ] `Semantics::parse` — call Septa::forward, convert BIO tags to typed spans
- [ ] Word vocabulary + pretrained embeddings

## basidium — training

- [ ] `Datum::generate(schema)` — synthetic data generation:
      enumerate query patterns (SELECT/CREATE/UPDATE/DELETE × tables × fields × modifiers)
      generate NL surface forms per pattern, derive Semantics + SpanLabels + QueryIr ground truth
- [ ] `Trainer::train` — epoch loop, train/val split, early stopping, scheduler step
- [ ] `Septa::step` / `Septa::evaluate` / `Septa::save` — forward + CRF loss + backward
- [ ] `Hyphae::step` / `Hyphae::evaluate` / `Hyphae::save` — forward + cross-entropy loss + backward

## stipe — pipeline

- [ ] Query history — persistent store of past (Semantics, QueryIr, SurrealQL) triples
- [ ] History retrieval — structural match for template reuse and ambiguity resolution
