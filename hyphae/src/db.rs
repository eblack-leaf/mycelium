/// SurrealDB HTTP client — minimal tool for running SurrealQL over the REST API.
///
/// POST <endpoint>/sql
///   Authorization: Basic <base64(user:pass)>
///   surreal-ns: <namespace>
///   surreal-db: <database>
///   Accept:       application/json
///   Content-Type: text/plain
///   body: <SurrealQL query>
///
/// Response: [{ "status": "OK", "result": <value>, "time": "..." }]

use crate::schema::{parse_db_info, parse_table_info, SchemaCompletions};
use reqwest::Client;

pub struct ConnConfig {
    pub endpoint:  String,  // any of: ws:// wss:// http:// https:// or bare host:port
    pub namespace: String,
    pub database:  String,
    pub username:  String,
    pub password:  String,
}

impl ConnConfig {
    /// Normalise the endpoint to an http(s):// base URL regardless of what the user typed.
    fn http_base(&self) -> String {
        let ep = self.endpoint.trim_end_matches('/');
        if ep.starts_with("wss://") {
            ep.replacen("wss://", "https://", 1)
        } else if ep.starts_with("ws://") {
            ep.replacen("ws://", "http://", 1)
        } else if ep.starts_with("http://") || ep.starts_with("https://") {
            ep.to_string()
        } else {
            // bare host:port
            format!("http://{}", ep)
        }
    }

    fn sql_url(&self) -> String { format!("{}/sql", self.http_base()) }
}

/// Run one SurrealQL statement, return the raw JSON body.
/// Returns Err with a descriptive message on any transport or HTTP error.
pub async fn query(cfg: &ConnConfig, surql: &str) -> Result<String, String> {
    let url = cfg.sql_url();

    let resp = Client::new()
        .post(&url)
        .header("surreal-ns", &cfg.namespace)
        .header("surreal-db", &cfg.database)
        .header("Accept", "application/json")
        .header("Content-Type", "text/plain")
        .basic_auth(&cfg.username, Some(&cfg.password))
        .body(surql.to_string())
        .send()
        .await
        .map_err(|e| format!("POST {} — {}", url, e))?;

    let status = resp.status();
    let body   = resp.text().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(format!("HTTP {} from {} — {}", status.as_u16(), url,
            body.chars().take(200).collect::<String>()));
    }

    Ok(body)
}

/// Fetch schema by running INFO FOR DB + INFO FOR TABLE for each table.
pub async fn fetch_schema(cfg: &ConnConfig) -> Result<SchemaCompletions, String> {
    let db_json = query(cfg, "INFO FOR DB").await?;

    let Some(db_info) = parse_db_info(&db_json) else {
        // Parsed OK but result wasn't DbInfo shaped — return what we have
        return Ok(SchemaCompletions { table_names: vec![], field_names: vec![] });
    };

    let mut table_infos = Vec::new();
    for name in db_info.tables.keys() {
        match query(cfg, &format!("INFO FOR TABLE {}", name)).await {
            Ok(json) => {
                if let Some(ti) = parse_table_info(name, &json) {
                    table_infos.push(ti);
                }
            }
            Err(_) => {} // best-effort; skip tables that error
        }
    }

    Ok(SchemaCompletions::from_db(&db_info, &table_infos))
}
