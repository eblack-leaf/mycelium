use crate::bridge::{Block, Suggestions};
use crate::state::DataM;
use tauri::State;

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