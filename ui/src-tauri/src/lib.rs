use crate::state::{Data, DataM};
use std::path::PathBuf;
use tauri::Manager;

mod bridge;
mod cmds;
mod state;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            cmds::blocks,
            cmds::submit_block,
            cmds::suggestions,
            cmds::save_value,
            cmds::delete_value,
            cmds::rename_value,
            cmds::get_values,
            cmds::get_settings,
            cmds::update_settings,
            cmds::suggest_name,
            cmds::paste_value,
            cmds::filter_suggestions,
            cmds::refresh_schema,
            cmds::list_tasks,
            cmds::reload_tasks,
            cmds::filter_task_suggestions,
        ])
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            std::fs::create_dir_all(&data_dir).ok();
            app.manage(DataM::new(Data::new(data_dir)));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
