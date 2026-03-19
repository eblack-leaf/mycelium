# TODO

## hyphae

- [ ] Register `sage.rs` in `lib.rs` (`pub mod sage`)
- [ ] `SchemaGraph::new` — build full possibility graph:
      OperationToTable (all ops × all tables)
      OperationToModifier (SELECT→[Where,OrderBy,Limit,Fetch,GroupBy], UPDATE/DELETE→[Where])
      FieldHasComparator (int→[Gt,Lt,Eq,Gte,Lte], string→[Eq,Contains], bool→[Eq], datetime→[Gt,Lt,Eq])
      ModifierToField (Where/OrderBy→all fields, Fetch→record link fields only)
- [ ] `SchemaGraph::inject(semantics)` — add span nodes + typed cross edges + inter-span edges from entity_index
- [ ] Bilinear head — stub `BilinearHead` struct in `hyphae/src/head.rs`:
      `forward(span_embs, schema_embs) -> Predictions`
      one projection per span type (entity, projection, condition, assignment, modifier)
- [ ] `GnnModel` — wire `SageConv` from `sage.rs` + `BilinearHead`
- [ ] `SageConvLayer::forward` — R-GCN aggregation:
      for each edge type: gather src features, apply W_r, mean-pool per dst node
      sum across edge types + self projection, apply ReLU

## septa

- [ ] `Model::forward` — BiLSTM-CRF inference:
      tokenize at word level → BiLSTM → emission scores → CRF Viterbi decode → spans
- [ ] `Semantics::parse` — call `Model::forward`, extract typed spans into `Slots`
- [ ] Word vocabulary + embeddings

## basidium

- [ ] `Datum::generate(schema)` — for each schema:
      enumerate query patterns (SELECT/INSERT/UPDATE/DELETE × tables × fields)
      generate NL surface forms per pattern
      derive Slots from SurrealQL automatically
- [ ] `Trainer::train` — implement epoch loop, early stopping on val_loss, scheduler step
- [ ] `SeptaModel::step` / `GnnModel::step` — forward + loss + backward

## stipe — query history

- [ ] `QueryLog` — persistent store of past (Slots, SurrealQL) pairs, saved to disk between sessions
- [ ] Retrieval — structural match over Slots: intent exact, entity/field text fuzzy, top-k by slot overlap score
- [ ] Template selection — if overlap above threshold, borrow SurrealQL structure from past entry, substitute new resolved values
- [ ] Re-ranking — use history co-occurrence to resolve ambiguous Predictions (which table/field appeared together before)

## stipe

- [ ] `QueryIr::new(predictions, semantics)` — thread septa values (comparator, value, modifier kind)
      back through Predictions resolutions into ResolvedCondition/Assignment/Modifier
- [ ] `QueryIr::render()` — deterministic SurrealQL:
      SELECT [fields|*] FROM [tables] WHERE [conditions] ORDER BY/LIMIT/FETCH [modifiers]
      INSERT INTO [table] SET [assignments]
      UPDATE [table] SET [assignments] WHERE [conditions]
      DELETE FROM [table] WHERE [conditions]
