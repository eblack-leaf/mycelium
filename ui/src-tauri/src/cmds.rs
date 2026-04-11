use crate::bridge::{
    Block, BlockState, PasteResult, PlaceholderValue, Settings, Suggestion, Suggestions,
};
use crate::state::DataM;
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
    let (cfg, resolved) = {
        let data = handle.lock().unwrap();

        let cfg = hyphae::db::ConnConfig {
            endpoint: data.settings.surreal_endpoint.clone(),
            namespace: data.settings.surreal_namespace.clone(),
            database: data.settings.surreal_database.clone(),
            username: data.settings.surreal_username.clone(),
            password: data.settings.surreal_password.clone(),
        };

        // Substitute @placeholder tokens with saved values.
        // Sort by name length descending so `@user-id` is replaced before `@user`.
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

    // Run the query — fall back to an error string as the result so the block
    // still completes rather than leaving the user in Executing state.
    let result = hyphae::db::query(&cfg, &resolved)
        .await
        .unwrap_or_else(|e| serde_json::json!([{ "error": e }]).to_string());

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
