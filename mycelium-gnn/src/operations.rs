// =============================================================================
// operations.rs — Fixed vocabulary of SurrealQL operations as graph nodes
//
// These are NOT extracted from NL — they're always present in the graph.
// The GNN scores which operations are active for a given query by combining
// structural edges (compatible_op, table_op) with NL cross-edges (matches_op).
//
// Each operation specifies what node type it structurally connects to:
//   "field" — via compatible_op, filtered by field type
//   "table" — via table_op, connects to all tables
//   "none"  — no structural edges, only reachable via Grounding model cross-edges
// =============================================================================

use crate::schema::FieldType;

/// One operation node in the graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    pub id: usize,
    pub name: String,
    pub category: OpCategory,
    /// Which field types this operation is compatible with.
    /// Only used when connects_to == "field".
    pub compatible_types: Vec<OpType>,
    /// What node type this operation structurally connects to.
    pub connects_to: ConnectsTo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Statement,
    Clause,
    Comparison,
    StringOp,
    MathOp,
    Aggregate,
    Traversal,
}

/// What node type an operation structurally connects to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectsTo {
    /// Connects to field nodes via compatible_op (type-filtered)
    Field,
    /// Connects to table nodes via table_op
    Table,
    /// No structural edges — only reachable via Grounding model cross-edges
    None,
}

/// Simplified type classes for compatibility matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Numeric,   // int, float, decimal, number
    String,    // string
    Bool,      // bool
    Temporal,  // datetime, duration
    Record,    // record<T>
    Array,     // array
    Set,       // set
    Geometry,  // geometry
}

impl OpType {
    pub fn from_field_type(ft: &FieldType) -> Option<Self> {
        match ft {
            FieldType::Int | FieldType::Float | FieldType::Decimal | FieldType::Number => Some(OpType::Numeric),
            FieldType::String | FieldType::Regex | FieldType::Literal { .. } => Some(OpType::String),
            FieldType::Bool => Some(OpType::Bool),
            FieldType::Datetime | FieldType::Duration => Some(OpType::Temporal),
            FieldType::Record { .. } => Some(OpType::Record),
            FieldType::Array { .. } => Some(OpType::Array),
            FieldType::Set { .. } => Some(OpType::Set),
            FieldType::Geometry { .. } => Some(OpType::Geometry),
            FieldType::Option { inner } => OpType::from_field_type(inner),
            // Any/Object/Bytes/Range — no specific type class
            _ => None,
        }
    }
}

/// The full fixed vocabulary. Called once at graph construction time.
pub fn all_operations() -> Vec<OpNode> {
    use OpCategory::*;
    use OpType::*;
    use ConnectsTo::*;

    let defs: Vec<(&str, OpCategory, Vec<OpType>, ConnectsTo)> = vec![
        // Statements — connect to tables
        ("SELECT",  Statement,  vec![], Table),
        ("CREATE",  Statement,  vec![], Table),
        ("UPDATE",  Statement,  vec![], Table),
        ("DELETE",  Statement,  vec![], Table),
        ("RELATE",  Statement,  vec![], Table),
        ("INSERT",  Statement,  vec![], Table),

        // Clauses — field-dependent ones connect to fields, others have no structural edges
        ("ORDER_BY", Clause, vec![Numeric, String, Temporal],         Field),
        ("GROUP_BY", Clause, vec![Numeric, String, Bool, Temporal],   Field),
        ("FETCH",    Clause, vec![Record],                            Field),
        ("SPLIT",    Clause, vec![Array, Set],                        Field),
        ("LIMIT",    Clause, vec![],                                  None),

        // Comparisons — connect to fields by type
        ("eq",  Comparison, vec![Numeric, String, Bool, Temporal, Record], Field),
        ("neq", Comparison, vec![Numeric, String, Bool, Temporal, Record], Field),
        ("gt",  Comparison, vec![Numeric, String, Temporal],               Field),
        ("lt",  Comparison, vec![Numeric, String, Temporal],               Field),
        ("gte", Comparison, vec![Numeric, String, Temporal],               Field),
        ("lte", Comparison, vec![Numeric, String, Temporal],               Field),

        // String operations — connect to string fields
        ("LIKE",        StringOp, vec![String],              Field),
        ("CONTAINS",    StringOp, vec![String, Array, Set],  Field),
        ("STARTS_WITH", StringOp, vec![String],              Field),
        ("ENDS_WITH",   StringOp, vec![String],              Field),

        // Math — connect to numeric fields
        ("add", MathOp, vec![Numeric],           Field),
        ("sub", MathOp, vec![Numeric, Temporal],  Field),
        ("mul", MathOp, vec![Numeric],           Field),
        ("div", MathOp, vec![Numeric],           Field),

        // Aggregates — type-dependent ones connect to fields, universal ones connect to tables
        ("count",        Aggregate, vec![],                    Table),
        ("sum",          Aggregate, vec![Numeric],             Field),
        ("avg",          Aggregate, vec![Numeric],             Field),
        ("min",          Aggregate, vec![Numeric, Temporal],   Field),
        ("max",          Aggregate, vec![Numeric, Temporal],   Field),
        ("array_group",  Aggregate, vec![],                    Table),

        // Traversals — connect to record fields
        ("arrow_right", Traversal, vec![Record], Field),
        ("arrow_left",  Traversal, vec![Record], Field),
        ("arrow_both",  Traversal, vec![Record], Field),
    ];

    defs.into_iter()
        .enumerate()
        .map(|(id, (name, category, compatible_types, connects_to))| OpNode {
            id,
            name: name.to_string(),
            category,
            compatible_types,
            connects_to,
        })
        .collect()
}

/// Check if an operation is compatible with a given field type.
/// Only meaningful when op.connects_to == ConnectsTo::Field.
pub fn is_compatible(op: &OpNode, field_type: &FieldType) -> bool {
    if op.compatible_types.is_empty() {
        return false; // no compatible types = doesn't connect to fields
    }
    match OpType::from_field_type(field_type) {
        Some(ft) => op.compatible_types.contains(&ft),
        None => false,
    }
}
