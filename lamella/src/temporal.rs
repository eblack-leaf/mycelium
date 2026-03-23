use crate::query::{TemporalExpr, ValueRef};

/// Scan NL left-to-right for temporal phrases and return them as ValueRef::Temporal
/// in occurrence order. Used to populate cond_values / asgn_values at inference.
///
/// Recognises:
///   today, yesterday
///   N days/weeks/months ago
///   last week, last month
///   last N days/weeks/months
///   YYYY-MM-DD
pub fn extract_temporal_values(nl: &str) -> Vec<ValueRef> {
    let lower = nl.to_lowercase();
    let mut results: Vec<(usize, ValueRef)> = Vec::new();

    // ISO date: YYYY-MM-DD
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i + 9 < lower.len() {
        if bytes[i..i+4].iter().all(|b| b.is_ascii_digit())
            && bytes[i+4] == b'-'
            && bytes[i+5..i+7].iter().all(|b| b.is_ascii_digit())
            && bytes[i+7] == b'-'
            && bytes[i+8..i+10].iter().all(|b| b.is_ascii_digit())
        {
            results.push((i, ValueRef::Temporal(TemporalExpr::Iso(nl[i..i+10].to_string()))));
            i += 10;
            continue;
        }
        i += 1;
    }

    // Word-based patterns — collect all matches with their byte positions
    let words: Vec<(usize, &str)> = lower
        .split_whitespace()
        .scan(0usize, |pos, w| {
            let start = lower[*pos..].find(w).map(|o| *pos + o).unwrap_or(*pos);
            *pos = start + w.len();
            Some((start, w))
        })
        .collect();

    let n = words.len();
    let mut skip_next = 0usize;

    for idx in 0..n {
        if idx < skip_next { continue; }
        let (pos, w) = words[idx];

        // today
        if w == "today" {
            results.push((pos, ValueRef::Temporal(TemporalExpr::Today)));
            continue;
        }

        // yesterday
        if w == "yesterday" {
            results.push((pos, ValueRef::Temporal(TemporalExpr::Yesterday)));
            continue;
        }

        // last week / last month
        if w == "last" && idx + 1 < n {
            let next = words[idx + 1].1;
            match next {
                "week"  => { results.push((pos, ValueRef::Temporal(TemporalExpr::WeeksAgo(1)))); skip_next = idx + 2; continue; }
                "month" => { results.push((pos, ValueRef::Temporal(TemporalExpr::MonthsAgo(1)))); skip_next = idx + 2; continue; }
                _ => {}
            }
            // last N days/weeks/months
            if let Ok(n_val) = next.parse::<u32>() {
                if idx + 2 < n {
                    let unit = words[idx + 2].1.trim_end_matches('s');
                    let expr = match unit {
                        "day"   => Some(TemporalExpr::DaysAgo(n_val)),
                        "week"  => Some(TemporalExpr::WeeksAgo(n_val)),
                        "month" => Some(TemporalExpr::MonthsAgo(n_val)),
                        _ => None,
                    };
                    if let Some(e) = expr {
                        results.push((pos, ValueRef::Temporal(e)));
                        skip_next = idx + 3;
                        continue;
                    }
                }
            }
        }

        // N days/weeks/months ago
        if let Ok(n_val) = w.parse::<u32>() {
            if idx + 2 < n && words[idx + 2].1 == "ago" {
                let unit = words[idx + 1].1.trim_end_matches('s');
                let expr = match unit {
                    "day"   => Some(TemporalExpr::DaysAgo(n_val)),
                    "week"  => Some(TemporalExpr::WeeksAgo(n_val)),
                    "month" => Some(TemporalExpr::MonthsAgo(n_val)),
                    _ => None,
                };
                if let Some(e) = expr {
                    results.push((pos, ValueRef::Temporal(e)));
                    skip_next = idx + 3;
                    continue;
                }
            }
        }
    }

    // Sort by position and return
    results.sort_by_key(|(pos, _)| *pos);
    results.into_iter().map(|(_, v)| v).collect()
}
