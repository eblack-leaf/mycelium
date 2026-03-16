use crate::condition::{Condition, Value};
use crate::intent::{Intent, Modifiers, OrderDir};
use crate::schema::Schema;

/// Assembled query IR — everything needed to emit SurrealQL.
/// Produced by combining intent + table + conditions + modifiers.
pub struct QueryIR {
    pub intent: Intent,
    pub table: String,
    pub fields: Vec<String>,
    pub conditions: Vec<Condition>,
    pub modifiers: Modifiers,
}

/// Emits a SurrealQL string from the query IR.
/// Deterministic — no models, just formatting.
pub fn build(ir: &QueryIR, schema: &Schema) -> String {
    let mut parts = Vec::new();

    // SELECT / SELECT count() / CREATE / UPDATE / DELETE
    match ir.intent {
        Intent::Select | Intent::Aggregate => {
            let fields = if ir.fields.is_empty() {
                "*".to_string()
            } else {
                ir.fields.join(", ")
            };
            parts.push(format!("SELECT {fields}"));
        }
        Intent::Count => parts.push("SELECT count()".to_string()),
        Intent::Create => parts.push("CREATE".to_string()),
        Intent::Update => parts.push("UPDATE".to_string()),
        Intent::Delete => parts.push("DELETE".to_string()),
    }

    // FROM table
    parts.push(format!("FROM {}", ir.table));

    // WHERE conditions
    if !ir.conditions.is_empty() {
        let clauses: Vec<String> = ir
            .conditions
            .iter()
            .map(|c| {
                let val = resolve_value(&c.value, &c.field, schema);
                format!("{} {} {}", c.field, c.op.to_surreal(), val)
            })
            .collect();
        parts.push(format!("WHERE {}", clauses.join(" AND ")));
    }

    // GROUP BY
    if let Some(ref group) = ir.modifiers.group_by {
        parts.push(format!("GROUP BY {group}"));
    }

    // ORDER BY
    if let Some(ref order) = ir.modifiers.order_by {
        let dir = match ir.modifiers.order_dir {
            Some(OrderDir::Desc) => " DESC",
            _ => " ASC",
        };
        parts.push(format!("ORDER BY {order}{dir}"));
    }

    // LIMIT
    if let Some(limit) = ir.modifiers.limit {
        parts.push(format!("LIMIT {limit}"));
    }

    parts.push(";".to_string());
    parts.join(" ")
}

/// Resolve a Value into a SurrealQL literal string.
/// Handles temporal resolution and type-aware formatting.
fn resolve_value(value: &Value, _field_name: &str, _schema: &Schema) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::String(v) => format!("'{}'", v.replace('\'', "\\'")),
        Value::Temporal(marker) => resolve_temporal(marker),
    }
}

/// Resolve a temporal marker into a SurrealQL time expression.
fn resolve_temporal(marker: &str) -> String {
    let lower = marker.to_lowercase();
    let lower = lower.trim();

    // "N units ago" pattern
    if let Some(expr) = parse_ago(lower) {
        return expr;
    }

    // Named shorthands
    match lower {
        "today" => "time::now() - 1d".to_string(),
        "yesterday" => "time::now() - 2d".to_string(),
        "this week" | "last week" => "time::now() - 1w".to_string(),
        "this month" | "last month" => "time::now() - 4w".to_string(),
        "this year" | "last year" => "time::now() - 1y".to_string(),
        "now" => "time::now()".to_string(),
        // Fallback — pass through as-is, user can correct at confirmation step
        other => format!("'{other}'"),
    }
}

fn parse_ago(s: &str) -> Option<String> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() == 3 && parts[2] == "ago" {
        let n: u64 = parts[0].parse().ok()?;
        let unit = match parts[1].trim_end_matches('s') {
            "second" | "sec" => "s",
            "minute" | "min" => "m",
            "hour" | "hr" => "h",
            "day" => "d",
            "week" => "w",
            "year" => "y",
            _ => return None,
        };
        Some(format!("time::now() - {n}{unit}"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condition::Op;
    use crate::intent::Intent;

    fn empty_schema() -> Schema {
        Schema {
            fields: vec![],
            phrases: vec![],
            temporal_markers: vec![],
            op_phrases: vec![],
        }
    }

    #[test]
    fn simple_select() {
        let ir = QueryIR {
            intent: Intent::Select,
            table: "products".into(),
            fields: vec![],
            conditions: vec![Condition {
                field: "stock".into(),
                op: Op::Lt,
                value: Value::Int(10),
            }],
            modifiers: Modifiers::default(),
        };
        let sql = build(&ir, &empty_schema());
        assert_eq!(sql, "SELECT * FROM products WHERE stock < 10 ;");
    }

    #[test]
    fn select_with_temporal() {
        let ir = QueryIR {
            intent: Intent::Select,
            table: "users".into(),
            fields: vec![],
            conditions: vec![Condition {
                field: "created".into(),
                op: Op::Gt,
                value: Value::Temporal("this week".into()),
            }],
            modifiers: Modifiers::default(),
        };
        let sql = build(&ir, &empty_schema());
        assert_eq!(
            sql,
            "SELECT * FROM users WHERE created > time::now() - 1w ;"
        );
    }

    #[test]
    fn select_with_modifiers() {
        let ir = QueryIR {
            intent: Intent::Select,
            table: "products".into(),
            fields: vec!["name".into(), "price".into()],
            conditions: vec![],
            modifiers: Modifiers {
                order_by: Some("price".into()),
                order_dir: Some(OrderDir::Desc),
                limit: Some(5),
                group_by: None,
            },
        };
        let sql = build(&ir, &empty_schema());
        assert_eq!(
            sql,
            "SELECT name, price FROM products ORDER BY price DESC LIMIT 5 ;"
        );
    }

    #[test]
    fn count_with_group() {
        let ir = QueryIR {
            intent: Intent::Count,
            table: "sessions".into(),
            fields: vec![],
            conditions: vec![],
            modifiers: Modifiers {
                group_by: Some("day".into()),
                ..Default::default()
            },
        };
        let sql = build(&ir, &empty_schema());
        assert_eq!(sql, "SELECT count() FROM sessions GROUP BY day ;");
    }

    #[test]
    fn temporal_ago() {
        assert_eq!(resolve_temporal("3 days ago"), "time::now() - 3d");
        assert_eq!(resolve_temporal("1 week ago"), "time::now() - 1w");
        assert_eq!(resolve_temporal("this week"), "time::now() - 1w");
    }

    #[test]
    fn delete_with_condition() {
        let ir = QueryIR {
            intent: Intent::Delete,
            table: "logs".into(),
            fields: vec![],
            conditions: vec![Condition {
                field: "created".into(),
                op: Op::Lt,
                value: Value::Temporal("30 days ago".into()),
            }],
            modifiers: Modifiers::default(),
        };
        let sql = build(&ir, &empty_schema());
        assert_eq!(
            sql,
            "DELETE FROM logs WHERE created < time::now() - 30d ;"
        );
    }
}
