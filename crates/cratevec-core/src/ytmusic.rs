use std::collections::HashSet;

use reqwest::Client;
use serde_json::{json, Value};
use tracing::{info, warn};

const YTM_CLIENT_VERSION: &str = "1.20241118.01.00";
const YTM_SONG_FILTER_PARAMS: &str = "EgWKAQIIAWoKEAkQBRAKEAMQBA%3D%3D";

fn ytm_context() -> Value {
    json!({
        "client": {
            "clientName": "WEB_REMIX",
            "clientVersion": YTM_CLIENT_VERSION,
            "hl": "en",
            "gl": "DE"
        }
    })
}

/// Fetches up to ~48 track-level song radio recommendations from YouTube Music's public
/// Innertube endpoint (zero API key, zero login) for a specific `(artist, title)` seed.
pub async fn fetch_song_radio(
    client: &Client,
    artist: &str,
    title: &str,
) -> Vec<(String, String)> {
    let query = format!("{} {}", artist, title);
    let video_id = match search_song_video_id(client, &query, artist, title).await {
        Some(vid) => vid,
        None => {
            warn!("YT Music: no videoId found for '{} - {}'", artist, title);
            return Vec::new();
        }
    };

    let next_body = json!({
        "context": ytm_context(),
        "videoId": video_id,
        "playlistId": format!("RDAMVM{}", video_id)
    });

    let resp = match client
        .post("https://music.youtube.com/youtubei/v1/next?prettyPrint=false")
        .header("Content-Type", "application/json")
        .header("Origin", "https://music.youtube.com")
        .header("Referer", "https://music.youtube.com/")
        .json(&next_body)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            warn!("YT Music /next HTTP {} for '{}'", query, r.status());
            return Vec::new();
        }
        Err(e) => {
            warn!("YT Music /next request failed for '{}': {}", query, e);
            return Vec::new();
        }
    };

    let json_val: Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut raw_tracks: Vec<(String, String)> = Vec::new();
    collect_panel_tracks(&json_val, &mut raw_tracks);

    let seed_artist_low = normalize_token(artist);
    let seed_title_low = normalize_token(clean_base_title(title));

    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<(String, String)> = Vec::new();

    for (cand_artist, cand_title) in raw_tracks {
        let a_clean = cand_artist.trim();
        let t_clean = cand_title.trim();
        if a_clean.is_empty() || t_clean.is_empty() {
            continue;
        }
        let na = normalize_token(a_clean);
        let nt = normalize_token(clean_base_title(t_clean));
        if na.is_empty() || nt.is_empty() {
            continue;
        }
        // Skip the seed song itself (always returned at index 0 of RDAMVM radio)
        if na == seed_artist_low && nt == seed_title_low {
            continue;
        }
        let key = format!("{}::{}", na, nt);
        if seen.insert(key) {
            out.push((a_clean.to_string(), t_clean.to_string()));
        }
    }

    info!(
        "YT Music Song-Radio for '{} - {}' (vid={}): {} track-level neighbors",
        artist,
        title,
        video_id,
        out.len()
    );
    out
}

async fn search_song_video_id(
    client: &Client,
    query: &str,
    artist: &str,
    title: &str,
) -> Option<String> {
    // Try filtered song search first, then unfiltered fallback
    for params in [Some(YTM_SONG_FILTER_PARAMS), None] {
        let mut body = json!({
            "context": ytm_context(),
            "query": query
        });
        if let Some(p) = params {
            body["params"] = json!(p);
        }

        let resp = match client
            .post("https://music.youtube.com/youtubei/v1/search?prettyPrint=false")
            .header("Content-Type", "application/json")
            .header("Origin", "https://music.youtube.com")
            .header("Referer", "https://music.youtube.com/")
            .json(&body)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };

        let val: Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut candidates: Vec<(String, String, String)> = Vec::new();
        collect_search_songs(&val, &mut candidates);

        if !candidates.is_empty() {
            let target_artist = normalize_token(artist);
            let target_title = normalize_token(clean_base_title(title));

            let best = candidates
                .iter()
                .max_by(|a, b| {
                    let sa = score_search_hit(&a.1, &a.2, &target_artist, &target_title);
                    let sb = score_search_hit(&b.1, &b.2, &target_artist, &target_title);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(vid, _, _)| vid.clone());

            if best.is_some() {
                return best;
            }
        }

        if let Some(vid) = find_first_video_id(&val) {
            return Some(vid);
        }
    }
    None
}

fn score_search_hit(cand_artist: &str, cand_title: &str, target_a: &str, target_t: &str) -> f32 {
    let na = normalize_token(cand_artist);
    let nt = normalize_token(clean_base_title(cand_title));
    let mut score = 0.0f32;
    if nt == target_t {
        score += 4.0;
    } else if nt.contains(target_t) || target_t.contains(&nt) {
        score += 2.2;
    }
    if na == target_a {
        score += 3.5;
    } else if na.contains(target_a) || target_a.contains(&na) {
        score += 1.8;
    }
    let low_title = cand_title.to_lowercase();
    if low_title.contains("remix")
        || low_title.contains("slowed")
        || low_title.contains("sped up")
        || low_title.contains("karaoke")
        || low_title.contains("live")
    {
        score -= 2.0;
    }
    score
}

fn collect_search_songs(val: &Value, out: &mut Vec<(String, String, String)>) {
    match val {
        Value::Object(map) => {
            if let Some(renderer) = map.get("musicResponsiveListItemRenderer") {
                if let Some(vid) = find_first_video_id(renderer) {
                    let (title, artist) = extract_list_item_title_artist(renderer);
                    if !title.is_empty() {
                        out.push((vid, artist, title));
                    }
                }
            }
            for v in map.values() {
                collect_search_songs(v, out);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_search_songs(item, out);
            }
        }
        _ => {}
    }
}

fn extract_list_item_title_artist(renderer: &Value) -> (String, String) {
    let cols = match renderer.get("flexColumns").and_then(|v| v.as_array()) {
        Some(c) => c,
        None => return (String::new(), String::new()),
    };

    let title = cols
        .first()
        .and_then(|c| c.get("musicResponsiveListItemFlexColumnRenderer"))
        .and_then(|r| r.get("text"))
        .map(extract_runs_text)
        .unwrap_or_default();

    let artist = cols
        .get(1)
        .and_then(|c| c.get("musicResponsiveListItemFlexColumnRenderer"))
        .and_then(|r| r.get("text"))
        .and_then(|t| t.get("runs"))
        .and_then(|r| r.as_array())
        .and_then(|runs| {
            // Skip "Song" / bullet prefixes if present
            for run in runs {
                let txt = run.get("text").and_then(|x| x.as_str()).unwrap_or("").trim();
                if txt.is_empty()
                    || txt == "•"
                    || txt == "·"
                    || txt.eq_ignore_ascii_case("song")
                    || txt.eq_ignore_ascii_case("titel")
                    || txt.eq_ignore_ascii_case("video")
                {
                    continue;
                }
                return Some(txt.to_string());
            }
            None
        })
        .unwrap_or_default();

    (title, artist)
}

fn collect_panel_tracks(val: &Value, out: &mut Vec<(String, String)>) {
    match val {
        Value::Object(map) => {
            if let Some(p) = map.get("playlistPanelVideoRenderer") {
                let title = p.get("title").map(extract_runs_text).unwrap_or_default();
                let artist = p
                    .get("shortBylineText")
                    .or_else(|| p.get("longBylineText"))
                    .and_then(|b| b.get("runs"))
                    .and_then(|r| r.as_array())
                    .and_then(|runs| runs.first())
                    .and_then(|first| first.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();

                if !title.is_empty() && !artist.is_empty() {
                    out.push((artist, title));
                }
            }
            for v in map.values() {
                collect_panel_tracks(v, out);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_panel_tracks(item, out);
            }
        }
        _ => {}
    }
}

fn extract_runs_text(v: &Value) -> String {
    v.get("runs")
        .and_then(|r| r.as_array())
        .map(|runs| {
            runs.iter()
                .filter_map(|r| r.get("text").and_then(|t| t.as_str()))
                .collect::<String>()
        })
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn find_first_video_id(val: &Value) -> Option<String> {
    match val {
        Value::Object(map) => {
            if let Some(Value::String(s)) = map.get("videoId") {
                if s.len() == 11 {
                    return Some(s.clone());
                }
            }
            for v in map.values() {
                if let Some(found) = find_first_video_id(v) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(arr) => {
            for item in arr {
                if let Some(found) = find_first_video_id(item) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn clean_base_title(title: &str) -> &str {
    title
        .split('(')
        .next()
        .unwrap_or(title)
        .split('[')
        .next()
        .unwrap_or(title)
        .split(" - ")
        .next()
        .unwrap_or(title)
        .trim()
}

fn normalize_token(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}
