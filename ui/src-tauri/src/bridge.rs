use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct TaskParam {
    pub(crate) name: String,
    pub(crate) description: String,
}

/// Bridge view of a task — `path` is internal and omitted.
#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct TaskMeta {
    pub(crate) name: String,
    pub(crate) params: Vec<TaskParam>,
}

impl From<hyphae::task::TaskMeta> for TaskMeta {
    fn from(t: hyphae::task::TaskMeta) -> Self {
        Self {
            name: t.name,
            params: t
                .params
                .into_iter()
                .map(|p| TaskParam {
                    name: p.name,
                    description: p.description,
                })
                .collect(),
        }
    }
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) enum BlockState {
    Composing,
    Executing,
    Done,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct Block {
    pub(crate) id: String,
    pub(crate) query: String,
    pub(crate) state: BlockState,
    pub(crate) result: Option<String>,
}

#[derive(Serialize, Deserialize, TS, Default, Clone)]
#[ts(export)]
pub(crate) struct Suggestion {
    pub(crate) text: String,
    pub(crate) metadata: String,
}

#[derive(Serialize, Deserialize, TS, Default, Clone)]
#[ts(export)]
pub(crate) struct Suggestions {
    pub(crate) placeholders: Vec<Suggestion>,
    pub(crate) schema: Vec<Suggestion>,
    pub(crate) other: Vec<Suggestion>,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct PlaceholderValue {
    pub(crate) name: String,
    pub(crate) value: String,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct Settings {
    pub(crate) surreal_endpoint: String,
    pub(crate) surreal_namespace: String,
    pub(crate) surreal_database: String,
    pub(crate) surreal_username: String,
    pub(crate) surreal_password: String,
    pub(crate) placeholder_prefix: String,
    #[serde(default)]
    pub(crate) task_dir: String,
}

#[derive(Serialize, Deserialize, TS, Clone)]
#[ts(export)]
pub(crate) struct PasteResult {
    pub(crate) name: String,
    pub(crate) values: Vec<PlaceholderValue>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            surreal_endpoint: "ws://localhost:8000".to_string(),
            surreal_namespace: "test".to_string(),
            surreal_database: "test".to_string(),
            surreal_username: "root".to_string(),
            surreal_password: "root".to_string(),
            placeholder_prefix: "@".to_string(),
            task_dir: String::new(),
        }
    }
}
