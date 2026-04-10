use crate::bridge::{Block, BlockState, PasteResult, PlaceholderValue, Settings, Suggestion, Suggestions};
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
    let mut data = handle.lock().unwrap();
    if let Some(block) = data.blocks.iter_mut().find(|b| b.id == id) {
        block.query = query.clone();
        block.state = BlockState::Done;
        block.result = Some(mock_result(&query));
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
            metadata: v.value.chars().take(24).collect(),
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
    Ok(data.settings.clone())
}

fn slugify(context: &str) -> String {
    let slug: String = context
        .chars()
        .take(32)
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    let mut name = String::new();
    let mut last_dash = false;
    for c in slug.chars() {
        if c == '-' {
            if !last_dash { name.push(c); }
            last_dash = true;
        } else {
            name.push(c);
            last_dash = false;
        }
    }
    name
}

#[tauri::command]
pub(crate) async fn suggest_name(context: String) -> Result<String, ()> {
    Ok(slugify(&context))
}

#[tauri::command]
pub(crate) async fn paste_value(
    context: String,
    value: String,
    handle: State<'_, DataM>,
) -> Result<PasteResult, ()> {
    let mut data = handle.lock().unwrap();
    let base = slugify(&context);
    let mut name = base.clone();
    let mut n = 2u32;
    while data.values.iter().any(|v| v.name == name) {
        name = format!("{}-{}", base, n);
        n += 1;
    }
    data.values.push(PlaceholderValue { name: name.clone(), value });
    Ok(PasteResult { name, values: data.values.clone() })
}

/// Score a partial word against a candidate for autocomplete.
/// Prefix matches score highest (0.8–1.0), substring matches score mid (0.5–0.7),
/// fuzzy similarity fills the rest so typos still surface results.
fn completion_score(word: &str, candidate: &str) -> f64 {
    if word.is_empty() { return 0.0; }
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
    let placeholders: Vec<Suggestion> = data.values.iter()
        .map(|v| Suggestion {
            text: format!("{}{}", prefix, v.name),
            metadata: "placeholder".to_string(),
        })
        .collect();

    // If the word starts with the placeholder prefix, only suggest placeholders
    if word.starts_with(prefix.as_str()) {
        let query = &word[prefix.len()..];
        let mut scored: Vec<(f64, Suggestion)> = placeholders.iter()
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
        let score = if word.is_empty() { 0.4 } else { completion_score(&word, name) };
        all.push((score, s.clone()));
    }
    // TODO: schema pool from DB INFO traversal or file-based schema sources
    // will be injected into data.suggestions.other here once implemented
    for s in &data.suggestions.other {
        let score = if word.is_empty() { 0.3 } else { completion_score(&word, &s.text) };
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

fn mock_result(query: &str) -> String {
    serde_json::json!([
        {
            "id": "user:abc123",
            "name": "Alice",
            "email": "alice@example.com",
            "age": 30,
            "meta": { "query": query, "note": "mock result" }
        },
        {
            "id": "user:def456",
            "name": "Bob",
            "email": "bob@example.com",
            "age": 25,
            "meta": { "query": query, "note": "mock result" }
        }
    ])
    .to_string()
}
