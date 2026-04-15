use crate::db::ConnConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMeta {
    pub name: String,
    pub params: Vec<TaskParam>,
}

/// Context passed into every task execution.
pub struct TaskRunContext<'a> {
    pub conn_cfg: &'a ConnConfig,
    pub registry: &'a TaskRegistry,
    pub depth: u8,
}

impl TaskRunContext<'_> {
    /// Invoke another registered task from within a task. Depth-capped at 8.
    pub fn run_child(&self, name: &str, params: &HashMap<String, String>) -> String {
        if self.depth >= 8 {
            return serde_json::json!([{"error": "run_task: max recursion depth (8) exceeded"}])
                .to_string();
        }
        let child = TaskRunContext {
            conn_cfg: self.conn_cfg,
            registry: self.registry,
            depth: self.depth + 1,
        };
        self.registry.run(&child, name, params)
    }
}

/// Implement this trait in `src-tauri/src/tasks.rs` for each task.
pub trait Task: Send + Sync {
    fn name(&self) -> &str;
    fn params(&self) -> Vec<TaskParam>;
    fn run(&self, ctx: &TaskRunContext<'_>, params: &HashMap<String, String>) -> String;
}

/// Built once at startup; holds all registered task implementations.
pub struct TaskRegistry {
    tasks: Vec<Box<dyn Task>>,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self { tasks: vec![] }
    }

    /// Builder-style — chain `.register(MyTask)` calls in `tasks::build_registry`.
    pub fn register(mut self, task: impl Task + 'static) -> Self {
        self.tasks.push(Box::new(task));
        self
    }

    pub fn list(&self) -> Vec<TaskMeta> {
        self.tasks
            .iter()
            .map(|t| TaskMeta {
                name: t.name().to_string(),
                params: t.params(),
            })
            .collect()
    }

    pub fn run(
        &self,
        ctx: &TaskRunContext<'_>,
        name: &str,
        params: &HashMap<String, String>,
    ) -> String {
        match self.tasks.iter().find(|t| t.name() == name) {
            Some(task) => task.run(ctx, params),
            None => {
                serde_json::json!([{"error": format!("task '{}' not found", name)}]).to_string()
            }
        }
    }
}

pub struct TaskInvocation {
    pub task_name: String,
    pub params: HashMap<String, String>,
}

/// Parse `/taskname k=v k="quoted value"` into a `TaskInvocation`.
/// Values may be unquoted (no spaces) or wrapped in `"..."` / `'...'`
/// (which are stripped and support `\"` / `\'` escapes inside).
pub fn parse_invocation(input: &str) -> Option<TaskInvocation> {
    let without_slash = input.trim_start().strip_prefix('/')?;
    let mut it = without_slash.chars().peekable();

    // task name — up to first whitespace
    while it.peek().map_or(false, |c| c.is_whitespace()) { it.next(); }
    let mut task_name = String::new();
    while let Some(&c) = it.peek() {
        if c.is_whitespace() { break; }
        it.next();
        task_name.push(c);
    }
    if task_name.is_empty() { return None; }

    let mut params = HashMap::new();
    loop {
        // skip whitespace
        while it.peek().map_or(false, |c| c.is_whitespace()) { it.next(); }
        if it.peek().is_none() { break; }

        // key — up to '=' or whitespace
        let mut key = String::new();
        while let Some(&c) = it.peek() {
            if c == '=' || c.is_whitespace() { break; }
            it.next();
            key.push(c);
        }

        if it.peek().map_or(true, |c| *c != '=') {
            continue; // bare token with no '=', skip
        }
        it.next(); // consume '='

        // value — quoted or bare
        let value = if let Some(&q @ ('"' | '\'')) = it.peek() {
            it.next(); // consume opening quote
            let mut v = String::new();
            loop {
                match it.next() {
                    None => break,
                    Some(c) if c == q => break,
                    Some('\\') => { if let Some(esc) = it.next() { v.push(esc); } }
                    Some(c) => v.push(c),
                }
            }
            v
        } else {
            let mut v = String::new();
            while let Some(&c) = it.peek() {
                if c.is_whitespace() { break; }
                it.next();
                v.push(c);
            }
            v
        };

        if !key.is_empty() {
            params.insert(key, value);
        }
    }

    Some(TaskInvocation { task_name, params })
}
