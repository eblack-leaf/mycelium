use tauri::{State};
use crate::bridge::{Block, Suggestion, Suggestions};
use crate::state::DataM;

#[tauri::command]
pub(crate) async fn blocks(handle: State<'_, DataM>) -> Result<Vec<Block>, ()> {
    Ok(vec![])
}
#[tauri::command]
pub(crate) async fn suggestions(handle: State<'_, DataM>) -> Result<Suggestions, ()> {
    Ok(Suggestions {
        placeholders: vec![],
        ids: vec![],
        schema: vec![],
    })
}