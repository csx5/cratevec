use anyhow::{anyhow, Result};
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SpotifyOEmbed {
    title: String,
}

/// Checks if a query string is a Spotify URL (`open.spotify.com/...` or `spotify:track:...`).
pub fn is_spotify_url(input: &str) -> bool {
    let trimmed = input.trim();
    trimmed.contains("open.spotify.com/") || trimmed.starts_with("spotify:")
}

/// Resolves a public Spotify track or playlist URL into one or more `"Artist - Title"` search strings
/// without requiring any Spotify Developer API key.
pub async fn resolve_spotify_input(client: &Client, input: &str) -> Result<Vec<String>> {
    let trimmed = input.trim();

    // Extract resource type (track / playlist / album) and 22-char base62 ID
    let re = Regex::new(r"(?:open\.spotify\.com/(?:intl-[a-z]+/)?|spotify:)(track|playlist|album)[/:]([A-Za-z0-9]{22})")?;
    let caps = re
        .captures(trimmed)
        .ok_or_else(|| anyhow!("Could not parse Spotify URL: {}", trimmed))?;

    let kind = caps.get(1).map(|m| m.as_str()).unwrap_or("track");
    let id = caps.get(2).map(|m| m.as_str()).unwrap_or("");

    let embed_url = format!("https://open.spotify.com/embed/{}/{}", kind, id);
    let html = client
        .get(&embed_url)
        .header(
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
        )
        .send()
        .await?
        .text()
        .await
        .unwrap_or_default();

    // Try extracting from __NEXT_DATA__ JSON in the public embed HTML
    if let Some(json_str) = extract_next_data(&html) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
            let entity = &val["props"]["pageProps"]["state"]["data"]["entity"];
            if kind == "track" {
                let title = entity["title"].as_str().unwrap_or_default();
                let mut artists = Vec::new();
                if let Some(arr) = entity["artists"].as_array() {
                    for a in arr {
                        if let Some(name) = a["name"].as_str() {
                            artists.push(name);
                        }
                    }
                }
                if !title.is_empty() && !artists.is_empty() {
                    return Ok(vec![format!("{} - {}", artists.join(" "), title)]);
                }
            } else if let Some(track_list) = entity["trackList"].as_array() {
                let mut results = Vec::new();
                for item in track_list.iter().take(8) {
                    let t = item["title"].as_str().unwrap_or_default();
                    let sub = item["subtitle"].as_str().unwrap_or_default();
                    if !t.is_empty() && !sub.is_empty() {
                        results.push(format!("{} - {}", sub.replace('\u{a0}', " "), t));
                    }
                }
                if !results.is_empty() {
                    return Ok(results);
                }
            }
        }
    }

    // Fallback to Spotify's public oEmbed endpoint
    let canonical = format!("https://open.spotify.com/{}/{}", kind, id);
    let oembed_url = format!(
        "https://open.spotify.com/oembed?url={}",
        urlencoding::encode(&canonical)
    );
    let resp = client.get(&oembed_url).send().await?;
    if resp.status().is_success() {
        if let Ok(oem) = resp.json::<SpotifyOEmbed>().await {
            if !oem.title.is_empty() {
                return Ok(vec![oem.title]);
            }
        }
    }

    Err(anyhow!("Could not resolve Spotify URL metadata"))
}

fn extract_next_data(html: &str) -> Option<&str> {
    let marker = r#"<script id="__NEXT_DATA__" type="application/json">"#;
    let start = html.find(marker)? + marker.len();
    let rest = &html[start..];
    let end = rest.find("</script>")?;
    Some(&rest[..end])
}
