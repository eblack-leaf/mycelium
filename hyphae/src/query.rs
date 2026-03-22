use serde::{Deserialize, Serialize};
use septa::{Comparator, Intent, TemporalExpr, ValueRef};

/// All node types in the grounded graph — each is a bilinear resolution target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryNode {
    Table(String),
    Field { table: String, name: String },
    Operation(Intent),
    Comparator(Comparator),
    Modifier(ModifierKind),
    /// Typed span nodes added by inject(). Features come from SpanHiddens (BiLSTM
    /// output) + a learned role embedding. Graph edges are role-specific.
    IntentSpan,
    EntitySpan,
    ProjSpan,
    CondFieldSpan,
    CondCmpSpan,
    AsgnSpan,
    ModTypeSpan,
    ModFieldSpan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModifierKind {
    OrderBy,
    Limit,
    Fetch,
}

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
    pub field: Option<String>, // None = expand slot object via schema types at render
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
}

impl QueryIr {
    /// Render to SurrealQL. values[n] is substituted for Slot(n) references.
    pub fn render(&self, values: &[String]) -> Query {
        let resolve_value = |v: &ValueRef| -> String {
            match v {
                ValueRef::Literal(s) => {
                    // Numeric / bool literals pass through, strings get quoted
                    if s.parse::<f64>().is_ok() || s == "true" || s == "false" {
                        s.clone()
                    } else {
                        format!("'{}'", s.replace('\'', "\\'"))
                    }
                }
                ValueRef::Slot(n) => {
                    if *n < values.len() { values[*n].clone() }
                    else { format!("${}", n) }
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
                    q = format!("SELECT {proj} FROM {table}:{}", resolve_value(rid));
                }
                if !self.conditions.is_empty() {
                    q.push_str(" WHERE ");
                    let conds: Vec<String> = self.conditions.iter().map(|c| {
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value(&c.value))
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
                            q.push_str(&format!(" LIMIT {}", resolve_value(value)));
                        }
                        ResolvedModifier::Fetch { field } => {
                            q.push_str(&format!(" FETCH {field}"));
                        }
                    }
                }
                q
            }
            Intent::Create => {
                let mut q = format!("CREATE {table}");
                if self.assignments.len() == 1 && self.assignments[0].field.is_none() {
                    q.push_str(&format!(" CONTENT {}", resolve_value(&self.assignments[0].value)));
                } else if !self.assignments.is_empty() {
                    q.push_str(" SET ");
                    let sets: Vec<String> = self.assignments.iter()
                        .filter_map(|a| a.field.as_ref().map(|f| format!("{f} = {}", resolve_value(&a.value))))
                        .collect();
                    q.push_str(&sets.join(", "));
                }
                q
            }
            Intent::Update => {
                let mut q = format!("UPDATE {table}");
                if self.assignments.len() == 1 && self.assignments[0].field.is_none() {
                    q.push_str(&format!(" CONTENT {}", resolve_value(&self.assignments[0].value)));
                } else if !self.assignments.is_empty() {
                    q.push_str(" SET ");
                    let sets: Vec<String> = self.assignments.iter()
                        .filter_map(|a| a.field.as_ref().map(|f| format!("{f} = {}", resolve_value(&a.value))))
                        .collect();
                    q.push_str(&sets.join(", "));
                }
                if !self.conditions.is_empty() {
                    q.push_str(" WHERE ");
                    let conds: Vec<String> = self.conditions.iter().map(|c| {
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value(&c.value))
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
                        format!("{} {} {}", c.field, render_comparator(&c.comparator), resolve_value(&c.value))
                    }).collect();
                    q.push_str(&conds.join(" AND "));
                }
                q
            }
        };

        Query { surql }
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
        TemporalExpr::WeeksAgo(n) => format!("(time::now() - {w}w)", w = n),
        TemporalExpr::MonthsAgo(n) => format!("(time::now() - {n}mo)"),
        TemporalExpr::Iso(s) => format!("'{s}'"),
    }
}
