use crate::bridge::{Block, BlockState, PasteResult, PlaceholderValue, Settings, Suggestions};
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
