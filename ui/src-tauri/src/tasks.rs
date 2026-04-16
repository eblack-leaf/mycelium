use hyphae::task::{Task, TaskParam, TaskRegistry, TaskRunContext};
use std::collections::HashMap;

pub fn build_registry() -> TaskRegistry {
    TaskRegistry::new().register(TotpTask)
}

// ---------------------------------------------------------------------------
// TOTP — generate the current code for a base32-encoded secret
// ---------------------------------------------------------------------------

struct TotpTask;

impl Task for TotpTask {
    fn name(&self) -> &str {
        "totp"
    }

    fn params(&self) -> Vec<TaskParam> {
        vec![TaskParam {
            name: "name".into(),
            description: "name field to match in totp_secret table".into(),
        }]
    }

    fn run(&self, ctx: &TaskRunContext<'_>, params: &HashMap<String, String>) -> String {
        let name = match params.get("name") {
            Some(n) => n.as_str(),
            None => return serde_json::json!([{"error": "missing param: name"}]).to_string(),
        };

        let name_lit = serde_json::to_string(name).unwrap_or_default();
        let surql = format!("LET $name = {name_lit}; SELECT secret FROM totp_secret WHERE name = $name;");
        let raw = tokio::runtime::Handle::current()
            .block_on(hyphae::db::query(ctx.conn_cfg, &surql))
            .unwrap_or_else(|e| serde_json::json!([{"error": e}]).to_string());

        let rows: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => return serde_json::json!([{"error": e.to_string()}]).to_string(),
        };

        let secret = match rows
            .get(1)
            .and_then(|r| r.get(0))
            .and_then(|r| r.get("secret"))
            .and_then(|s| s.as_str())
        {
            Some(s) => s.to_uppercase(),
            None => return serde_json::json!([{"error": format!("no totp_secret found for name '{}'", name)}]).to_string(),
        };

        let totp = match totp_rs::TOTP::new(
            totp_rs::Algorithm::SHA1,
            6,
            1,
            30,
            totp_rs::Secret::Encoded(secret).to_bytes().unwrap_or_default(),
        ) {
            Ok(t) => t,
            Err(e) => return serde_json::json!([{"error": e.to_string()}]).to_string(),
        };

        match totp.generate_current() {
            Ok(code) => serde_json::json!([{"code": code}]).to_string(),
            Err(e) => serde_json::json!([{"error": e.to_string()}]).to_string(),
        }
    }
}
