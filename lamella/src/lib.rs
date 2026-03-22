pub mod schema;
pub mod query;
pub mod embed;
pub mod catalog;
pub mod model;
pub mod train;

use catalog::SchemaCatalog;
use model::SlotCounts;
use query::{Comparator, Intent, ModifierKind, QueryIr, ValueRef};
use serde::{Deserialize, Serialize};

// =============================================================================
// LamellaDatum — training datum with pre-resolved integer indices
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LamellaDatum {
    pub nl: String,
    pub intent: usize,
    pub entity: usize,
    pub proj_fields: Vec<usize>,
    pub cond_fields: Vec<usize>,
    pub cond_cmps: Vec<usize>,
    pub asgn_fields: Vec<usize>,
    pub mod_types: Vec<usize>,
    pub mod_fields: Vec<usize>,
    pub values: DatumValues,
    pub ir: Option<QueryIr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatumValues {
    pub record_id: Option<ValueRef>,
    pub cond_values: Vec<ValueRef>,
    pub asgn_values: Vec<ValueRef>,
    pub asgn_fields_text: Vec<Option<String>>,
    pub mod_values: Vec<ValueRef>,
    pub mod_descending: Vec<bool>,
}

impl DatumValues {
    pub fn empty() -> Self {
        Self {
            record_id: None,
            cond_values: vec![],
            asgn_values: vec![],
            asgn_fields_text: vec![],
            mod_values: vec![],
            mod_descending: vec![],
        }
    }
}

impl LamellaDatum {
    pub fn slot_counts(&self) -> SlotCounts {
        SlotCounts {
            projections: self.proj_fields.len(),
            conditions: self.cond_fields.len(),
            assignments: self.asgn_fields.len(),
            mod_types: self.mod_types.len(),
            mod_fields: self.mod_fields.len(),
        }
    }
}

// =============================================================================
// Hand-written datum format — human-readable, resolved at load time
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDatum {
    pub nl: String,
    pub intent: String,
    pub table: String,
    #[serde(default)]
    pub projections: Vec<String>,
    #[serde(default)]
    pub conditions: Vec<RawCondition>,
    #[serde(default)]
    pub assignments: Vec<RawAssignment>,
    #[serde(default)]
    pub modifiers: Vec<RawModifier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCondition {
    pub field: String,
    pub cmp: String,
    #[serde(default = "default_value")]
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawAssignment {
    pub field: String,
    #[serde(default = "default_value")]
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RawModifier {
    #[serde(rename = "order_by")]
    OrderBy {
        field: String,
        #[serde(default)]
        descending: bool,
    },
    #[serde(rename = "limit")]
    Limit {
        #[serde(default = "default_limit")]
        value: String,
    },
    #[serde(rename = "fetch")]
    Fetch { field: String },
}

fn default_value() -> String { "?".into() }
fn default_limit() -> String { "10".into() }

// =============================================================================
// Resolve RawDatum → LamellaDatum using SchemaCatalog
// =============================================================================

pub fn resolve_raw(raw: &RawDatum, catalog: &SchemaCatalog) -> Result<LamellaDatum, String> {
    let intent = parse_intent(&raw.intent)
        .ok_or_else(|| format!("unknown intent '{}' in: {}", raw.intent, raw.nl))?;
    let intent_idx = catalog.op_index(&intent);

    let entity = catalog.table_index(&raw.table)
        .ok_or_else(|| format!("unknown table '{}' in: {}", raw.table, raw.nl))?;

    let proj_fields: Vec<usize> = raw.projections.iter()
        .map(|f| catalog.field_index(&raw.table, f)
            .ok_or_else(|| format!("unknown field '{}.{}' in: {}", raw.table, f, raw.nl)))
        .collect::<Result<_, _>>()?;

    let mut cond_fields = Vec::new();
    let mut cond_cmps = Vec::new();
    let mut cond_values = Vec::new();
    for c in &raw.conditions {
        let fid = catalog.field_index(&raw.table, &c.field)
            .ok_or_else(|| format!("unknown field '{}.{}' in: {}", raw.table, c.field, raw.nl))?;
        let cmp = parse_cmp(&c.cmp)
            .ok_or_else(|| format!("unknown comparator '{}' in: {}", c.cmp, raw.nl))?;
        cond_fields.push(fid);
        cond_cmps.push(catalog.cmp_index(&cmp));
        cond_values.push(ValueRef::Literal(c.value.clone()));
    }

    let mut asgn_fields = Vec::new();
    let mut asgn_values = Vec::new();
    let mut asgn_fields_text = Vec::new();
    for a in &raw.assignments {
        let fid = catalog.field_index(&raw.table, &a.field)
            .ok_or_else(|| format!("unknown field '{}.{}' in: {}", raw.table, a.field, raw.nl))?;
        asgn_fields.push(fid);
        asgn_values.push(ValueRef::Literal(a.value.clone()));
        asgn_fields_text.push(Some(a.field.clone()));
    }

    let mut mod_types = Vec::new();
    let mut mod_fields = Vec::new();
    let mut mod_values = Vec::new();
    let mut mod_descending = Vec::new();
    for m in &raw.modifiers {
        match m {
            RawModifier::OrderBy { field, descending } => {
                mod_types.push(catalog.mod_index(&ModifierKind::OrderBy));
                let fid = catalog.field_index(&raw.table, field)
                    .ok_or_else(|| format!("unknown field '{}.{}' in: {}", raw.table, field, raw.nl))?;
                mod_fields.push(fid);
                mod_descending.push(*descending);
            }
            RawModifier::Limit { value } => {
                mod_types.push(catalog.mod_index(&ModifierKind::Limit));
                mod_values.push(ValueRef::Literal(value.clone()));
            }
            RawModifier::Fetch { field } => {
                mod_types.push(catalog.mod_index(&ModifierKind::Fetch));
                let fid = catalog.field_index(&raw.table, field)
                    .ok_or_else(|| format!("unknown field '{}.{}' in: {}", raw.table, field, raw.nl))?;
                mod_fields.push(fid);
            }
        }
    }

    Ok(LamellaDatum {
        nl: raw.nl.clone(),
        intent: intent_idx,
        entity,
        proj_fields,
        cond_fields,
        cond_cmps,
        asgn_fields,
        mod_types,
        mod_fields,
        values: DatumValues {
            record_id: None,
            cond_values,
            asgn_values,
            asgn_fields_text,
            mod_values,
            mod_descending,
        },
        ir: None,
    })
}

/// Load all .jsonl files from a directory, resolve against catalog,
/// stratified split by table.
pub fn load_dataset(
    dir: &str,
    catalog: &SchemaCatalog,
    val_ratio: f64,
) -> (Vec<LamellaDatum>, Vec<LamellaDatum>) {
    let mut by_table: std::collections::HashMap<usize, Vec<LamellaDatum>> =
        std::collections::HashMap::new();
    let mut errors = 0;
    let mut total_lines = 0;

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("Failed to read directory {dir}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();

    if files.is_empty() {
        panic!("No .jsonl files found in {dir}");
    }

    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|_| panic!("Failed to read {}", file.display()));
        let fname = file.file_name().unwrap().to_string_lossy();

        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") { continue; }
            total_lines += 1;

            let raw: RawDatum = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{fname}:{}: parse error: {e}", i + 1);
                    errors += 1;
                    continue;
                }
            };

            match resolve_raw(&raw, catalog) {
                Ok(datum) => {
                    by_table.entry(datum.entity).or_default().push(datum);
                }
                Err(e) => {
                    eprintln!("{fname}:{}: {e}", i + 1);
                    errors += 1;
                }
            }
        }
    }

    println!("Loaded {} datums from {} file(s) in {dir}", total_lines - errors, files.len());
    if errors > 0 {
        eprintln!("{errors} datum(s) skipped due to errors");
    }

    // Stratified split: per table, take last val_ratio fraction as val
    let mut train = Vec::new();
    let mut val = Vec::new();

    let mut tables: Vec<usize> = by_table.keys().copied().collect();
    tables.sort();

    for table_idx in tables {
        let mut datums = by_table.remove(&table_idx).unwrap();
        // Deterministic shuffle per table
        let mut rng: u64 = table_idx as u64 * 6364136223846793005 + 1;
        for i in (1..datums.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            datums.swap(i, j);
        }

        let split = ((datums.len() as f64) * (1.0 - val_ratio)).ceil() as usize;
        let (t, v) = datums.split_at(split);
        train.extend_from_slice(t);
        val.extend_from_slice(v);
    }

    (train, val)
}

// =============================================================================
// Parsing helpers
// =============================================================================

fn parse_intent(s: &str) -> Option<Intent> {
    match s.to_lowercase().as_str() {
        "select" => Some(Intent::Select),
        "create" => Some(Intent::Create),
        "update" => Some(Intent::Update),
        "delete" => Some(Intent::Delete),
        _ => None,
    }
}

fn parse_cmp(s: &str) -> Option<Comparator> {
    match s.to_lowercase().as_str() {
        "eq" | "=" | "==" => Some(Comparator::Eq),
        "neq" | "!=" | "<>" => Some(Comparator::Neq),
        "gt" | ">" => Some(Comparator::Gt),
        "gte" | ">=" => Some(Comparator::Gte),
        "lt" | "<" => Some(Comparator::Lt),
        "lte" | "<=" => Some(Comparator::Lte),
        "contains" | "~" => Some(Comparator::Contains),
        _ => None,
    }
}
