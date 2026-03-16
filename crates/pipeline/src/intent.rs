use serde::{Deserialize, Serialize};

/// Top-level query intent — what operation the user wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Intent {
    Select,
    Count,
    Create,
    Update,
    Delete,
    Aggregate,
}

impl Intent {
    pub fn to_surreal_prefix(&self) -> &'static str {
        match self {
            Self::Select => "SELECT",
            Self::Count => "SELECT count()",
            Self::Create => "CREATE",
            Self::Update => "UPDATE",
            Self::Delete => "DELETE",
            Self::Aggregate => "SELECT",
        }
    }
}

/// Modifiers extracted alongside or after conditions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Modifiers {
    pub order_by: Option<String>,
    pub order_dir: Option<OrderDir>,
    pub limit: Option<u64>,
    pub group_by: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderDir {
    Asc,
    Desc,
}

pub trait IntentClassifier {
    fn classify(&self, input: &str) -> Intent;
}

pub trait ModifierExtractor {
    fn extract(&self, input: &str) -> Modifiers;
}
