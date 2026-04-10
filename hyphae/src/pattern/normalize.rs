use regex::Regex;
use std::sync::OnceLock;

/// Normalize a raw SurrealQL query string into a structural template.
/// Replaces literals, IDs, and placeholders with type tokens so that
/// semantically equivalent queries collapse to the same key.
///
/// Examples:
///   "SELECT * FROM user WHERE id = user:abc123"  →  "SELECT * FROM <table> WHERE id = <id>"
///   "UPDATE user:abc SET name = 'Alice'"          →  "UPDATE <id> SET name = <str>"
///   "SELECT * FROM @target-user"                  →  "SELECT * FROM <placeholder>"
pub fn normalize(query: &str) -> String {
    static RECORD_ID: OnceLock<Regex> = OnceLock::new();
    static STRING_LIT: OnceLock<Regex> = OnceLock::new();
    static NUMBER_LIT: OnceLock<Regex> = OnceLock::new();
    static PLACEHOLDER: OnceLock<Regex> = OnceLock::new();
    static TABLE_ID: OnceLock<Regex> = OnceLock::new();

    let q = PLACEHOLDER
        .get_or_init(|| Regex::new(r"[@$]\w[\w-]*").unwrap())
        .replace_all(query, "<placeholder>");

    let q = RECORD_ID
        .get_or_init(|| Regex::new(r"\b[a-zA-Z_]\w*:[a-zA-Z0-9_-]+\b").unwrap())
        .replace_all(&q, "<id>");

    let q = STRING_LIT
        .get_or_init(|| Regex::new(r#"'[^']*'|"[^"]*""#).unwrap())
        .replace_all(&q, "<str>");

    let q = NUMBER_LIT
        .get_or_init(|| Regex::new(r"\b\d+(\.\d+)?\b").unwrap())
        .replace_all(&q, "<num>");

    // Collapse remaining bare identifiers that look like table names
    // (appear after FROM / UPDATE / CREATE / DELETE / RELATE)
    let q = TABLE_ID
        .get_or_init(|| {
            Regex::new(r"(?i)(FROM|UPDATE|CREATE|DELETE|RELATE)\s+([a-zA-Z_]\w*)").unwrap()
        })
        .replace_all(&q, |caps: &regex::Captures| {
            format!("{} <table>", &caps[1].to_uppercase())
        });

    // Normalize whitespace
    q.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_select() {
        let q = "SELECT * FROM user WHERE id = user:abc123";
        let n = normalize(q);
        assert!(n.contains("<table>"), "got: {n}");
        assert!(n.contains("<id>"), "got: {n}");
    }

    #[test]
    fn normalizes_placeholder() {
        let q = "SELECT * FROM @target-user";
        let n = normalize(q);
        assert!(n.contains("<placeholder>"), "got: {n}");
    }
}
