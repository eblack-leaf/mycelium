use serde::{Deserialize, Serialize};

// =============================================================================
// Enums (from septa, inlined here for independence)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    Select,
    Create,
    Update,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Comparator {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueRef {
    Literal(String),
    Slot(usize),
    Temporal(TemporalExpr),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalExpr {
    Today,
    Yesterday,
    DaysAgo(u32),
    WeeksAgo(u32),
    MonthsAgo(u32),
    Iso(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModifierKind {
    OrderBy,
    Limit,
    Fetch,
}

// =============================================================================
// Query IR — resolved structure, renderable to SurrealQL
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryIr {
    pub intent: Intent,
    pub table: String,
    pub record_id: Option<ValueRef>,
    pub projections: Vec<ResolvedField>,
    pub conditions: Vec<ResolvedCondition>,
    pub assignments: Vec<ResolvedAssignment>,
    pub modifiers: Vec<ResolvedModifier>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedField {
    pub table: String,
    pub field: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCondition {
    pub table: String,
    pub field: String,
    pub comparator: Comparator,
    pub value: ValueRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedAssignment {
    pub table: String,
    pub field: Option<String>,
    pub value: ValueRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolvedModifier {
    OrderBy {
        table: String,
        field: String,
        descending: bool,
    },
    Limit {
        value: ValueRef,
    },
    Fetch {
        field: String,
    },
}

pub struct Query {
    pub surql: String,
    /// Runtime parameters in render order: (param_name, field_name).
    /// e.g. [("param1", "salary"), ("param2", "name")]
    /// Populated for every "?" value slot — caller binds these before executing.
    pub params: Vec<(String, String)>,
}

impl QueryIr {
    pub fn render(&self, values: &[String]) -> Query {
        // Track runtime params: each "?" literal emits $param1, $param2, …
        // and records which field it belongs to.
        let param_n = std::cell::Cell::new(1usize);
        let params: std::cell::RefCell<Vec<(String, String)>> = std::cell::RefCell::new(Vec::new());

        // Resolve a value, recording a param entry when field context is known.
        let resolve_value_for = |v: &ValueRef, field: &str| -> String {
            match v {
                ValueRef::Literal(s) if s == "?" => {
                    let n = param_n.get();
                    param_n.set(n + 1);
                    let name = format!("param{n}");
                    params.borrow_mut().push((name.clone(), field.to_string()));
                    format!("${name}")
                }
                ValueRef::Literal(s) => {
                    if s.parse::<f64>().is_ok() || s == "true" || s == "false" {
                        s.clone()
                    } else {
                        format!("'{}'", s.replace('\'', "\\'"))
                    }
                }
                ValueRef::Slot(n) => {
                    if *n < values.len() { values[*n].clone() }
                    else { format!("${n}") }
                }
                ValueRef::Temporal(t) => render_temporal(t),
            }
        };

        let table = &self.table;
        let surql = match &self.intent {
            Intent::Select => {
                let proj = if self.projections.is_empty() {
                    "*".to_string()
                } else {
                    self.projections.iter().map(|p| p.field.clone()).collect::<Vec<_>>().join(", ")
                };
                let mut q = format!("SELECT {proj} FROM {table}");
                if let Some(ref rid) = self.record_id {
                    q = format!("SELECT {proj} FROM {table}:{}", resolve_value_for(rid, "id"));
                }
                if !self.conditions.is_empty() {
                    q.push_str(" WHERE ");
                    let conds: Vec<String> = self.conditions.iter().map(|c| {
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value_for(&c.value, &c.field))
                    }).collect();
                    q.push_str(&conds.join(" AND "));
                }
                for m in &self.modifiers {
                    match m {
                        ResolvedModifier::OrderBy { field, descending, .. } => {
                            q.push_str(&format!(" ORDER BY {field}"));
                            if *descending { q.push_str(" DESC"); }
                        }
                        ResolvedModifier::Limit { value } => {
                            q.push_str(&format!(" LIMIT {}", resolve_value_for(value, "limit")));
                        }
                        ResolvedModifier::Fetch { field } => {
                            q.push_str(&format!(" FETCH {field}"));
                        }
                    }
                }
                q
            }
            // CREATE and UPDATE always use SET — no CONTENT branch.
            Intent::Create => {
                let mut q = format!("CREATE {table}");
                if !self.assignments.is_empty() {
                    q.push_str(" SET ");
                    let sets: Vec<String> = self.assignments.iter()
                        .filter_map(|a| a.field.as_ref().map(|f| {
                            format!("{f} = {}", resolve_value_for(&a.value, f))
                        }))
                        .collect();
                    q.push_str(&sets.join(", "));
                }
                q
            }
            Intent::Update => {
                let mut q = format!("UPDATE {table}");
                if !self.assignments.is_empty() {
                    q.push_str(" SET ");
                    let sets: Vec<String> = self.assignments.iter()
                        .filter_map(|a| a.field.as_ref().map(|f| {
                            format!("{f} = {}", resolve_value_for(&a.value, f))
                        }))
                        .collect();
                    q.push_str(&sets.join(", "));
                }
                if !self.conditions.is_empty() {
                    q.push_str(" WHERE ");
                    let conds: Vec<String> = self.conditions.iter().map(|c| {
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value_for(&c.value, &c.field))
                    }).collect();
                    q.push_str(&conds.join(" AND "));
                }
                q
            }
            Intent::Delete => {
                let mut q = format!("DELETE {table}");
                if !self.conditions.is_empty() {
                    q.push_str(" WHERE ");
                    let conds: Vec<String> = self.conditions.iter().map(|c| {
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value_for(&c.value, &c.field))
                    }).collect();
                    q.push_str(&conds.join(" AND "));
                }
                q
            }
        };

        Query { surql, params: params.into_inner() }
    }
}

fn render_comparator(c: &Comparator) -> &'static str {
    match c {
        Comparator::Eq => "=",
        Comparator::Neq => "!=",
        Comparator::Gt => ">",
        Comparator::Gte => ">=",
        Comparator::Lt => "<",
        Comparator::Lte => "<=",
        Comparator::Contains => "CONTAINS",
    }
}

fn render_temporal(t: &TemporalExpr) -> String {
    match t {
        TemporalExpr::Today => "time::floor(time::now(), 1d)".into(),
        TemporalExpr::Yesterday => "(time::floor(time::now(), 1d) - 1d)".into(),
        TemporalExpr::DaysAgo(n) => format!("(time::now() - {n}d)"),
        TemporalExpr::WeeksAgo(n) => format!("(time::now() - {n}w)"),
        TemporalExpr::MonthsAgo(n) => format!("(time::now() - {n}mo)"),
        TemporalExpr::Iso(s) => format!("'{s}'"),
    }
}
