use crate::bridge::{
    Block, BlockState, PasteResult, PlaceholderValue, Settings, Suggestion, Suggestions, TaskMeta,
};
use crate::state::DataM;
use std::collections::HashMap;
use tauri::State;

#[tauri::command]
pub(crate) async fn blocks(handle: State<'_, DataM>) -> Result<Vec<Block>, ()> {
    Ok(handle.lock().unwrap().blocks.clone())
}

#[tauri::command]
pub(crate) async fn submit_block(
    id: String,
    query: String,
    handle: State<'_, DataM>,
) -> Result<Vec<Block>, ()> {
    let result = if query.trim_start().starts_with('/') {
        // Task invocation path
        let invocation = hyphae::task::parse_invocation(&query).ok_or(())?;
        let task_name = invocation.task_name;

        let (registry, conn_cfg, params) = {
            let data = handle.lock().unwrap();

            // Resolve @placeholders in param values before passing to the task.
            let prefix = &data.settings.placeholder_prefix;
            let mut sorted_values = data.values.clone();
            sorted_values.sort_by(|a, b| b.name.len().cmp(&a.name.len()));

            let mut params: HashMap<String, String> = invocation.params;
            for val_field in params.values_mut() {
                for sv in &sorted_values {
                    let token = format!("{}{}", prefix, sv.name);
                    *val_field = val_field.replace(&token, &sv.value);
                }
            }

            let conn_cfg = hyphae::db::ConnConfig {
                endpoint: data.settings.surreal_endpoint.clone(),
                namespace: data.settings.surreal_namespace.clone(),
                database: data.settings.surreal_database.clone(),
                username: data.settings.surreal_username.clone(),
                password: data.settings.surreal_password.clone(),
            };

            (std::sync::Arc::clone(&data.registry), conn_cfg, params)
        };

        let ctx = hyphae::task::TaskRunContext {
            conn_cfg: &conn_cfg,
            registry: &registry,
            depth: 0,
        };
        tokio::task::block_in_place(|| registry.run(&ctx, &task_name, &params))
    } else {
        // SurrealDB query path — unchanged
        let (cfg, resolved) = {
            let data = handle.lock().unwrap();

            let cfg = hyphae::db::ConnConfig {
                endpoint: data.settings.surreal_endpoint.clone(),
                namespace: data.settings.surreal_namespace.clone(),
                database: data.settings.surreal_database.clone(),
                username: data.settings.surreal_username.clone(),
                password: data.settings.surreal_password.clone(),
            };

            let prefix = &data.settings.placeholder_prefix;
            let mut sorted_values = data.values.clone();
            sorted_values.sort_by(|a, b| b.name.len().cmp(&a.name.len()));

            let mut resolved = query.clone();
            for val in &sorted_values {
                let token = format!("{}{}", prefix, val.name);
                resolved = resolved.replace(&token, &val.value);
            }

            (cfg, resolved)
        };

        hyphae::db::query(&cfg, &resolved)
            .await
            .unwrap_or_else(|e| serde_json::json!([{ "error": e }]).to_string())
    };

    let mut data = handle.lock().unwrap();
    if let Some(block) = data.blocks.iter_mut().find(|b| b.id == id) {
        block.query = query.clone();
        block.state = BlockState::Done;
        block.result = Some(result);
    }
    let new_id = data.new_id();
    data.blocks.push(Block {
        id: new_id,
        query: String::new(),
        state: BlockState::Composing,
        result: None,
    });
    Ok(data.blocks.clone())
}

#[tauri::command]
pub(crate) async fn suggestions(handle: State<'_, DataM>) -> Result<Suggestions, ()> {
    let data = handle.lock().unwrap();
    let mut suggestions = data.suggestions.clone();
    // Mirror saved placeholder values as placeholder suggestions
    suggestions.placeholders = data
        .values
        .iter()
        .map(|v| crate::bridge::Suggestion {
            text: format!("{}{}", data.settings.placeholder_prefix, v.name),
            metadata: "placeholder".to_string(),
        })
        .collect();
    Ok(suggestions)
}

#[tauri::command]
pub(crate) async fn save_value(
    name: String,
    value: String,
    handle: State<'_, DataM>,
) -> Result<Vec<PlaceholderValue>, ()> {
    let mut data = handle.lock().unwrap();
    if let Some(existing) = data.values.iter_mut().find(|v| v.name == name) {
        existing.value = value;
    } else {
        data.values.push(PlaceholderValue { name, value });
    }
    Ok(data.values.clone())
}

#[tauri::command]
pub(crate) async fn delete_value(
    name: String,
    handle: State<'_, DataM>,
) -> Result<Vec<PlaceholderValue>, ()> {
    let mut data = handle.lock().unwrap();
    data.values.retain(|v| v.name != name);
    Ok(data.values.clone())
}

#[tauri::command]
pub(crate) async fn rename_value(
    old_name: String,
    new_name: String,
    handle: State<'_, DataM>,
) -> Result<Vec<PlaceholderValue>, ()> {
    let mut data = handle.lock().unwrap();
    if let Some(v) = data.values.iter_mut().find(|v| v.name == old_name) {
        v.name = new_name;
    }
    Ok(data.values.clone())
}

#[tauri::command]
pub(crate) async fn get_values(handle: State<'_, DataM>) -> Result<Vec<PlaceholderValue>, ()> {
    Ok(handle.lock().unwrap().values.clone())
}

#[tauri::command]
pub(crate) async fn get_settings(handle: State<'_, DataM>) -> Result<Settings, ()> {
    Ok(handle.lock().unwrap().settings.clone())
}

#[tauri::command]
pub(crate) async fn update_settings(
    settings: Settings,
    handle: State<'_, DataM>,
) -> Result<Settings, ()> {
    let mut data = handle.lock().unwrap();
    data.settings = settings;
    data.save_settings();
    Ok(data.settings.clone())
}

#[tauri::command]
pub(crate) async fn suggest_name(_context: String) -> Result<String, ()> {
    Ok(hyphae::namer::generate_random())
}

#[tauri::command]
pub(crate) async fn paste_value(
    _context: String,
    value: String,
    handle: State<'_, DataM>,
) -> Result<PasteResult, ()> {
    let mut data = handle.lock().unwrap();
    let mut name = hyphae::namer::generate_random();
    // Avoid collisions
    while data.values.iter().any(|v| v.name == name) {
        name = hyphae::namer::generate_random();
    }
    data.values.push(PlaceholderValue {
        name: name.clone(),
        value,
    });
    Ok(PasteResult {
        name,
        values: data.values.clone(),
    })
}

/// Score a partial word against a candidate for autocomplete.
/// Prefix matches score highest (0.8–1.0), substring matches score mid (0.5–0.7),
/// fuzzy similarity fills the rest so typos still surface results.
fn completion_score(word: &str, candidate: &str) -> f64 {
    if word.is_empty() {
        return 0.0;
    }
    let w = word.to_lowercase();
    let c = candidate.to_lowercase();
    if c.starts_with(&w) {
        // Reward full prefix matches; longer remaining tail = slightly lower
        0.8 + 0.2 * (w.len() as f64 / c.len() as f64)
    } else if c.contains(&w) {
        0.5
    } else {
        strsim::jaro_winkler(&w, &c) * 0.6 // scale down pure fuzzy so prefix always wins
    }
}

#[tauri::command]
pub(crate) async fn filter_suggestions(
    word: String,
    handle: State<'_, DataM>,
) -> Result<Suggestions, ()> {
    let data = handle.lock().unwrap();
    let prefix = &data.settings.placeholder_prefix;

    // Build placeholder list from saved values
    let placeholders: Vec<Suggestion> = data
        .values
        .iter()
        .map(|v| Suggestion {
            text: format!("{}{}", prefix, v.name),
            metadata: "placeholder".to_string(),
        })
        .collect();

    // If the word starts with the placeholder prefix, only suggest placeholders
    if word.starts_with(prefix.as_str()) {
        let query = &word[prefix.len()..];
        let mut scored: Vec<(f64, Suggestion)> = placeholders
            .iter()
            .map(|s| {
                let name = s.text.trim_start_matches(prefix.as_str());
                let score = completion_score(query, name);
                (score, s.clone())
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(Suggestions {
            placeholders: scored.into_iter().take(4).map(|(_, s)| s).collect(),
            schema: vec![],
            other: vec![],
        });
    }

    // General mode: score everything together, return top 4 by relevance.
    // When word is empty, keywords come first (schema pool), then placeholders.
    let mut all: Vec<(f64, Suggestion)> = vec![];
    for (i, s) in data.suggestions.schema.iter().enumerate() {
        let score = if word.is_empty() {
            1.0 - (i as f64 * 0.001) // preserve seeded order when idle
        } else {
            completion_score(&word, &s.text)
        };
        all.push((score, s.clone()));
    }
    for s in &placeholders {
        let name = s.text.trim_start_matches(prefix.as_str());
        let score = if word.is_empty() {
            0.4
        } else {
            completion_score(&word, name)
        };
        all.push((score, s.clone()));
    }
    for s in &data.suggestions.other {
        let score = if word.is_empty() {
            0.3
        } else {
            completion_score(&word, &s.text)
        };
        all.push((score, s.clone()));
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let ranked: Vec<Suggestion> = all.into_iter().take(4).map(|(_, s)| s).collect();

    Ok(Suggestions {
        placeholders: vec![],
        schema: ranked,
        other: vec![],
    })
}

#[tauri::command]
pub(crate) async fn list_tasks(handle: State<'_, DataM>) -> Result<Vec<TaskMeta>, ()> {
    let data = handle.lock().unwrap();
    Ok(data.registry.list().into_iter().map(TaskMeta::from).collect())
}

#[tauri::command]
pub(crate) async fn reload_tasks(handle: State<'_, DataM>) -> Result<Vec<TaskMeta>, ()> {
    let data = handle.lock().unwrap();
    Ok(data.registry.list().into_iter().map(TaskMeta::from).collect())
}

/// Return task-mode completions based on full input and cursor position.
///
/// - One token (cursor on task name): scores task names → `{ text: "/name", metadata: "task" }`
/// - Multiple tokens (cursor on param): scores task's declared params → `{ text: "name=", metadata: description }`
#[tauri::command]
pub(crate) async fn filter_task_suggestions(
    input: String,
    cursor: usize,
    handle: State<'_, DataM>,
) -> Result<Suggestions, ()> {
    let data = handle.lock().unwrap();
    let tasks = data.registry.list();

    let trimmed = input.trim_start();
    let leading = input.len() - trimmed.len();
    let effective_cursor = cursor.saturating_sub(leading);
    let up_to_cursor = &trimmed[..effective_cursor.min(trimmed.len())];
    let without_slash = up_to_cursor.strip_prefix('/').unwrap_or(up_to_cursor);

    let tokens: Vec<&str> = without_slash.split_whitespace().collect();
    let ends_with_space = up_to_cursor.ends_with(|c: char| c.is_whitespace());

    // Mode 1: still typing the task name
    if tokens.len() <= 1 && !ends_with_space {
        let partial = tokens.first().copied().unwrap_or("");
        let mut scored: Vec<(f64, &hyphae::task::TaskMeta)> = tasks
            .iter()
            .map(|t| {
                let score = if partial.is_empty() {
                    1.0
                } else {
                    completion_score(partial, &t.name)
                };
                (score, t)
            })
            .filter(|(s, _)| partial.is_empty() || *s > 0.1)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let suggestions: Vec<Suggestion> = scored
            .into_iter()
            .take(4)
            .map(|(_, t)| Suggestion {
                text: format!("/{}", t.name),
                metadata: "task".to_string(),
            })
            .collect();
        return Ok(Suggestions {
            schema: suggestions,
            placeholders: vec![],
            other: vec![],
        });
    }

    // Mode 2: typing params — complete param names for the matched task
    let task_name = tokens.first().copied().unwrap_or("");
    let Some(task) = tasks.iter().find(|t| t.name == task_name) else {
        return Ok(Suggestions::default());
    };

    // Collect params already provided (key side of k=v tokens)
    let used: std::collections::HashSet<&str> = tokens[1..]
        .iter()
        .filter_map(|t| t.split_once('=').map(|(k, _)| k))
        .collect();

    // Partial key: last token if it doesn't contain `=` and cursor isn't after a space
    let partial_key = if ends_with_space {
        ""
    } else {
        tokens
            .last()
            .and_then(|t| if t.contains('=') { None } else { Some(*t) })
            .unwrap_or("")
    };

    let mut scored: Vec<(f64, &hyphae::task::TaskParam)> = task
        .params
        .iter()
        .filter(|p| !used.contains(p.name.as_str()))
        .map(|p| {
            let score = if partial_key.is_empty() {
                1.0
            } else {
                completion_score(partial_key, &p.name)
            };
            (score, p)
        })
        .filter(|(s, _)| partial_key.is_empty() || *s > 0.1)
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let suggestions: Vec<Suggestion> = scored
        .into_iter()
        .take(4)
        .map(|(_, p)| Suggestion {
            text: format!("{}=", p.name),
            metadata: p.description.clone(),
        })
        .collect();

    Ok(Suggestions {
        schema: suggestions,
        placeholders: vec![],
        other: vec![],
    })
}

/// Fetch schema from SurrealDB (INFO FOR DB + INFO FOR TABLE) and populate
/// the `other` suggestions group with table and field names.
/// Called from the frontend when the user opens settings or on demand.
#[tauri::command]
pub(crate) async fn refresh_schema(
    handle: State<'_, DataM>,
) -> Result<Vec<crate::bridge::Suggestion>, String> {
    let settings = handle.lock().unwrap().settings.clone();

    let cfg = hyphae::db::ConnConfig {
        endpoint: settings.surreal_endpoint,
        namespace: settings.surreal_namespace,
        database: settings.surreal_database,
        username: settings.surreal_username,
        password: settings.surreal_password,
    };

    let completions = hyphae::db::fetch_schema(&cfg).await?;

    let schema_suggestions: Vec<crate::bridge::Suggestion> = completions
        .to_suggestions()
        .into_iter()
        .map(|(text, metadata)| crate::bridge::Suggestion { text, metadata })
        .collect();

    handle.lock().unwrap().suggestions.other = schema_suggestions.clone();
    Ok(schema_suggestions)
}
