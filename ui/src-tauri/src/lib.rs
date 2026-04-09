use crate::state::{Data, DataM};

mod cmds;
mod bridge;
mod state;
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![cmds::placeholders, cmds::ids, cmds::schema, cmds::blocks])
        .manage(DataM::new(Data::new()))
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
