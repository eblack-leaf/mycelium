use crate::embed::char_ngram_buckets;
use crate::query::{Comparator, Intent, ModifierKind};
use crate::schema::Schema;

/// Flat index of all resolution targets. Built once from Schema at startup.
pub struct SchemaCatalog {
    // Fixed vocab
    pub ops: Vec<Intent>,
    pub cmps: Vec<Comparator>,
    pub mods: Vec<ModifierKind>,

    // Schema nodes
    pub tables: Vec<String>,
    pub fields: Vec<(String, String)>, // (table_name, field_name)

    // Precomputed n-gram bucket indices
    pub table_ngrams: Vec<Vec<usize>>,
    pub field_ngrams: Vec<Vec<usize>>,

    // Per-table field mask: table index → vec of field indices in self.fields
    pub table_field_indices: Vec<Vec<usize>>,
}

impl SchemaCatalog {
    pub fn from_schema(schema: &Schema, ngram_buckets: usize) -> Self {
        let ops = vec![Intent::Select, Intent::Create, Intent::Update, Intent::Delete];
        let cmps = vec![
            Comparator::Eq, Comparator::Neq, Comparator::Gt,
            Comparator::Gte, Comparator::Lt, Comparator::Lte, Comparator::Contains,
        ];
        let mods = vec![ModifierKind::OrderBy, ModifierKind::Limit, ModifierKind::Fetch];

        let tables: Vec<String> = schema.tables.iter().map(|t| t.name.clone()).collect();
        let table_ngrams: Vec<Vec<usize>> = tables.iter()
            .map(|name| char_ngram_buckets(name, ngram_buckets))
            .collect();

        let mut fields: Vec<(String, String)> = Vec::new();
        let mut field_ngrams: Vec<Vec<usize>> = Vec::new();
        let mut table_field_indices: Vec<Vec<usize>> = Vec::new();

        for table in &schema.tables {
            let mut indices = Vec::new();
            for field in &table.fields {
                let idx = fields.len();
                fields.push((table.name.clone(), field.name.clone()));
                field_ngrams.push(char_ngram_buckets(&field.name, ngram_buckets));
                indices.push(idx);
            }
            table_field_indices.push(indices);
        }

        Self { ops, cmps, mods, tables, fields, table_ngrams, field_ngrams, table_field_indices }
    }

    pub fn table_index(&self, name: &str) -> Option<usize> {
        self.tables.iter().position(|t| t == name)
    }

    pub fn field_index(&self, table: &str, field: &str) -> Option<usize> {
        self.fields.iter().position(|(t, f)| t == table && f == field)
    }

    pub fn op_index(&self, intent: &Intent) -> usize {
        self.ops.iter().position(|o| o == intent).unwrap()
    }

    pub fn cmp_index(&self, cmp: &Comparator) -> usize {
        self.cmps.iter().position(|c| c == cmp).unwrap()
    }

    pub fn mod_index(&self, kind: &ModifierKind) -> usize {
        self.mods.iter().position(|m| m == kind).unwrap()
    }
}
