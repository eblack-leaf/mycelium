use tauri::{State};
use crate::bridge::{Block, Suggestion};
use crate::state::DataM;

#[tauri::command]
pub(crate) async fn blocks(handle: State<'_, DataM>) -> Result<Vec<Block>, ()> {
    Ok(vec![])
}
#[tauri::command]
pub(crate) async fn placeholders(handle: State<'_, DataM>) -> Result<Vec<Suggestion>, ()> {
    Ok(vec![])
}
#[tauri::command]
pub(crate) async fn ids(handle: State<'_, DataM>) -> Result<Vec<Suggestion>, ()> {
    Ok(vec![])
}
#[tauri::command]
pub(crate) async fn schema(handle: State<'_, DataM>) -> Result<Vec<Suggestion>, ()> {
    Ok(vec![])
}