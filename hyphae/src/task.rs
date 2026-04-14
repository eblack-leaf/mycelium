use crate::db::ConnConfig;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskParam {
    pub name: String,
    pub description: String,
}

/// Full task metadata — `path` is internal and skipped in serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMeta {
    pub name: String,
    #[serde(skip)]
    pub path: PathBuf,
    pub params: Vec<TaskParam>,
}

/// Scan a directory for `.rhai` files and return sorted `TaskMeta` list.
/// Returns an empty vec if the directory does not exist or is unreadable.
pub fn scan_tasks(dir: &Path) -> Vec<TaskMeta> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut tasks: Vec<TaskMeta> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rhai"))
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_stem()?.to_string_lossy().into_owned();
            let content = std::fs::read_to_string(&path).ok()?;
            Some(parse_meta(name, path, &content))
        })
        .collect();
    tasks.sort_by(|a, b| a.name.cmp(&b.name));
    tasks
}

fn param_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*//\s*@param\s+([\w-]+)\s*:\s*(.+)$").unwrap())
}

/// Parse `// @param name: description` header comments from script content.
pub fn parse_meta(name: String, path: PathBuf, content: &str) -> TaskMeta {
    let mut params = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with("//") {
            break;
        }
        if let Some(caps) = param_re().captures(trimmed) {
            params.push(TaskParam {
                name: caps[1].to_string(),
                description: caps[2].trim().to_string(),
            });
        }
    }
    TaskMeta { name, path, params }
}

pub struct TaskInvocation {
    pub task_name: String,
    pub params: HashMap<String, String>,
}

/// Parse `/taskname k=v k=v` into a `TaskInvocation`.
/// Returns `None` if the input doesn't start with `/` or has no task name.
pub fn parse_invocation(input: &str) -> Option<TaskInvocation> {
    let without_slash = input.trim_start().strip_prefix('/')?;
    let mut tokens = without_slash.split_whitespace();
    let task_name = tokens.next()?.to_string();
    if task_name.is_empty() {
        return None;
    }
    let mut params = HashMap::new();
    for token in tokens {
        if let Some((k, v)) = token.split_once('=') {
            params.insert(k.to_string(), v.to_string());
        }
    }
    Some(TaskInvocation { task_name, params })
}

/// Everything the engine needs to drive async calls from sync Rhai callbacks.
pub struct TaskContext {
    pub conn_cfg: ConnConfig,
    pub task_dir: PathBuf,
    pub depth: u8,
}

/// Run a named task. Called from a `tokio::task::spawn_blocking` thread so
/// `tokio_handle.block_on(future)` is safe to call inside Rhai callbacks.
///
/// Returns a JSON string suitable for storing as `Block.result`.
pub fn run_task(ctx: &TaskContext, name: &str, params: &HashMap<String, String>) -> String {
    let path = ctx.task_dir.join(format!("{}.rhai", name));
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!([{"error": format!("task '{}' not found: {}", name, e)}])
                .to_string()
        }
    };

    let engine = build_engine(ctx);
    let mut scope = rhai::Scope::new();

    let mut params_map = rhai::Map::new();
    for (k, v) in params {
        params_map.insert(k.as_str().into(), rhai::Dynamic::from(v.clone()));
    }
    scope.push("params", params_map);

    match engine.eval_with_scope::<rhai::Dynamic>(&mut scope, &content) {
        Ok(val) => match rhai::serde::from_dynamic::<serde_json::Value>(&val) {
            Ok(json) => serde_json::to_string(&json)
                .unwrap_or_else(|e| serde_json::json!([{"error": e.to_string()}]).to_string()),
            Err(e) => serde_json::json!([{"error": format!("script returned non-serializable value: {}", e)}]).to_string(),
        },
        Err(e) => serde_json::json!([{"error": e.to_string()}]).to_string(),
    }
}

fn build_engine(ctx: &TaskContext) -> rhai::Engine {
    let mut engine = rhai::Engine::new();

    // query(surql) -> Dynamic
    // Drives the async reqwest call synchronously.
    // We build a fresh single-threaded runtime rather than calling Handle::block_on
    // so there is no risk of the "Cannot call block_on inside an async context" panic
    // that can occur even on spawn_blocking threads in some tokio configurations.
    let cfg_q = ctx.conn_cfg.clone();
    engine.register_fn("query", move |surql: &str| -> rhai::Dynamic {
        // block_in_place (called in cmds.rs) marks the thread as blocking,
        // so Handle::current().block_on() is safe here — no deadlock risk.
        let result_str = tokio::runtime::Handle::current()
            .block_on(crate::db::query(&cfg_q, surql))
            .unwrap_or_else(|e| serde_json::json!([{"error": e}]).to_string());
        println!("query result: {}", result_str);
        let json_val: serde_json::Value = serde_json::from_str(&result_str)
            .unwrap_or(serde_json::Value::String(result_str));
        rhai::serde::to_dynamic(json_val).unwrap_or(rhai::Dynamic::UNIT)
    });

    // shell(cmd, args) -> String
    // Uses explicit arg array — never shell interpolation — to prevent injection.
    engine.register_fn("shell", |cmd: &str, args: rhai::Array| -> String {
        let str_args: Vec<String> = args.into_iter().map(|a| a.to_string()).collect();
        println!("cmd: {} args: {:?}", cmd, str_args);
        let output = std::process::Command::new(cmd)
            .args(&str_args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_else(|e| format!("shell_error: {}", e));
        println!("shell-output: {}", output);
        output.trim().to_string()
    });

    // run_task(name, params) -> Dynamic
    // Allows scripts to invoke other scripts with an explicit params map.
    // Depth-capped at 8 to break cycles.
    let task_dir_rt = ctx.task_dir.clone();
    let cfg_rt = ctx.conn_cfg.clone();
    let depth = ctx.depth;
    engine.register_fn("run_task", move |name: &str, rhai_params: rhai::Map| -> rhai::Dynamic {
        if depth >= 8 {
            return rhai::Dynamic::from(
                "run_task: max recursion depth (8) exceeded".to_string(),
            );
        }
        let child_ctx = TaskContext {
            conn_cfg: cfg_rt.clone(),
            task_dir: task_dir_rt.clone(),
            depth: depth + 1,
        };
        let params: HashMap<String, String> = rhai_params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let result_str = run_task(&child_ctx, name, &params);
        let json_val: serde_json::Value = serde_json::from_str(&result_str)
            .unwrap_or(serde_json::Value::String(result_str));
        rhai::serde::to_dynamic(json_val).unwrap_or(rhai::Dynamic::UNIT)
    });

    engine
}
