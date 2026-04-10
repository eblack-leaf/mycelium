use crate::state::{Data, DataM};

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
        ])
        .manage(DataM::new(Data::new()))
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
