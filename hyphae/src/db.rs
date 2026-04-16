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
use crate::schema::{SchemaCompletions, parse_db_info, parse_table_info};
use reqwest::Client;

#[derive(Clone)]
pub struct ConnConfig {
    pub endpoint: String, // any of: ws:// wss:// http:// https:// or bare host:port
    pub namespace: String,
    pub database: String,
    pub username: String,
    pub password: String,
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

    fn sql_url(&self) -> String {
        format!("{}/sql", self.http_base())
    }
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
    let body = resp.text().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        return Err(format!(
            "HTTP {} from {} — {}",
            status.as_u16(),
            url,
            body.chars().take(200).collect::<String>()
        ));
    }

    Ok(unwrap_surreal_envelope(&body))
}

/// Unwrap the SurrealDB response envelope:
///   [{ "status": "OK", "result": <value>, "time": "..." }, ...]
/// Single query  → serialize `result` value directly.
/// Multiple queries → serialize array of `result` values.
/// Anything else → return body unchanged.
fn unwrap_surreal_envelope(body: &str) -> String {
    let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(body) else {
        return body.to_string();
    };
    let is_envelope = arr.iter().all(|v| {
        v.get("status").is_some() && v.get("result").is_some()
    });
    if !is_envelope {
        return body.to_string();
    }
    let results: Vec<&serde_json::Value> = arr.iter().map(|v| &v["result"]).collect();
    if results.len() == 1 {
        serde_json::to_string(results[0]).unwrap_or_else(|_| body.to_string())
    } else {
        serde_json::to_string(&results).unwrap_or_else(|_| body.to_string())
    }
}

/// Fetch schema by running INFO FOR DB + INFO FOR TABLE for each table.
/// Returns both flat completions and the structured table list.
pub async fn fetch_schema(cfg: &ConnConfig) -> Result<(SchemaCompletions, Vec<crate::schema::TableInfo>), String> {
    let db_json = query(cfg, "INFO FOR DB").await?;

    let Some(db_info) = parse_db_info(&db_json) else {
        return Ok((SchemaCompletions { table_names: vec![], field_names: vec![] }, vec![]));
    };

    let mut table_infos = Vec::new();
    for name in db_info.tables.keys() {
        let mut ti = match query(cfg, &format!("INFO FOR TABLE {}", name)).await {
            Ok(json) => parse_table_info(name, &json)
                .unwrap_or_else(|| crate::schema::TableInfo { name: name.clone(), ..Default::default() }),
            Err(_) => crate::schema::TableInfo { name: name.clone(), ..Default::default() },
        };

        // Schemaless tables have no defined fields — infer from a sample record
        if ti.fields.is_empty() {
            if let Ok(json) = query(cfg, &format!("SELECT * FROM {} LIMIT 1", name)).await {
                if let Ok(serde_json::Value::Array(rows)) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(obj) = rows.first().and_then(|r| r.as_object()) {
                        for key in obj.keys() {
                            if key != "id" {
                                ti.fields.insert(key.clone(), serde_json::Value::Null);
                            }
                        }
                    }
                }
            }
        }

        table_infos.push(ti);
    }

    let completions = SchemaCompletions::from_db(&db_info, &table_infos);
    Ok((completions, table_infos))
}
