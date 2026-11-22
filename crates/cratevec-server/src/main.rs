use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use axum::{
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use cratevec_core::{
    models::{MixerWeights, SwipeDirection},
    CrateVecEngine,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::info;

#[derive(Clone)]
struct AppState {
    engine: CrateVecEngine,
    audio_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
}

#[derive(Deserialize)]
struct SeedRequest {
    query: String,
}

#[derive(Deserialize)]
struct SwipeRequest {
    track_id: String,
    direction: SwipeDirection,
}

#[derive(Deserialize)]
struct FocusRequest {
    track_id: String,
}

#[derive(Deserialize)]
struct CorridorRequest {
    corridor_id: String,
}

#[derive(Deserialize)]
struct AudioProxyQuery {
    #[serde(default)]
    id: String,
    #[serde(default)]
    artist: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
}

#[derive(Serialize)]
struct StatusResp {
    status: &'static str,
    version: &'static str,
    cached_tracks: usize,
    has_active_session: bool,
}

#[derive(Serialize)]
struct ErrorResp {
    error: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("cratevec=info,symphonia=error,warn")
        .init();

    let engine = CrateVecEngine::new(None);

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        let initial_seed = args.join(" ");
        info!("Pre-warming initial session with seed: {}", initial_seed);
        if let Err(e) = engine.start_session(&initial_seed).await {
            tracing::warn!("Initial seed pre-warm failed: {}", e);
        }
    }

    let state = AppState {
        engine,
        audio_cache: Arc::new(RwLock::new(HashMap::new())),
    };

    let ui_dir = resolve_ui_dir();
    info!("Serving UI from {:?}", ui_dir);

    let app = Router::new()
        .route("/api/status", get(api_status))
        .route("/api/search", get(api_search))
        .route("/api/session/start", post(api_start_session))
        .route("/api/session/bridge", post(api_bridge_seed))
        .route("/api/session/swipe", post(api_swipe))
        .route("/api/session/undo", post(api_undo))
        .route("/api/session/negative/remove", post(api_remove_negative))
        .route("/api/session/crate/remove", post(api_remove_crate))
        .route("/api/session/more", post(api_load_more))
        .route("/api/session/focus", post(api_focus_track))
        .route("/api/session/corridor", post(api_select_corridor))
        .route("/api/session/weights", post(api_update_weights))
        .route("/api/session/snapshot", get(api_get_snapshot))
        .route("/api/audio/proxy", get(api_audio_proxy))
        .fallback_service(ServeDir::new(ui_dir))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port: u16 = std::env::var("CRATEVEC_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7420);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("cratevec hybrid server listening on http://127.0.0.1:{}", port);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn resolve_ui_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("CRATEVEC_UI_DIR") {
        let p = PathBuf::from(custom);
        if p.join("index.html").exists() {
            return p;
        }
    }

    let mut candidates = vec![
        PathBuf::from("ui"),
        PathBuf::from("../ui"),
        PathBuf::from("../../ui"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent();
        for _ in 0..4 {
            if let Some(dir) = cur {
                candidates.push(dir.join("ui"));
                cur = dir.parent();
            }
        }
    }

    for c in candidates {
        if c.join("index.html").exists() {
            return c;
        }
    }
    PathBuf::from("ui")
}

async fn api_status(State(state): State<AppState>) -> impl IntoResponse {
    let cached_tracks = state.engine.harvester().store().count();
    let has_active_session = state.engine.get_snapshot().is_some();
    Json(StatusResp {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        cached_tracks,
        has_active_session,
    })
}

async fn api_search(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Response {
    match state.engine.search_tracks(&params.q, 8).await {
        Ok(list) => Json(list).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_start_session(
    State(state): State<AppState>,
    Json(req): Json<SeedRequest>,
) -> Response {
    match state.engine.start_session(&req.query).await {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_bridge_seed(
    State(state): State<AppState>,
    Json(req): Json<SeedRequest>,
) -> Response {
    match state.engine.add_bridge_seed(&req.query).await {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_swipe(
    State(state): State<AppState>,
    Json(req): Json<SwipeRequest>,
) -> Response {
    match state.engine.swipe(&req.track_id, req.direction).await {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_undo(State(state): State<AppState>) -> Response {
    match state.engine.undo_last_swipe() {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_remove_negative(
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Response {
    match state.engine.remove_negative_anchor(&req.track_id) {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_remove_crate(
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Response {
    match state.engine.remove_crate_track(&req.track_id) {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_load_more(State(state): State<AppState>) -> Response {
    match state.engine.load_more_candidates().await {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_focus_track(
    State(state): State<AppState>,
    Json(req): Json<FocusRequest>,
) -> Response {
    match state.engine.focus_track(&req.track_id) {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_select_corridor(
    State(state): State<AppState>,
    Json(req): Json<CorridorRequest>,
) -> Response {
    match state.engine.select_corridor(&req.corridor_id) {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_update_weights(
    State(state): State<AppState>,
    Json(weights): Json<MixerWeights>,
) -> Response {
    match state.engine.update_weights(weights) {
        Ok(snap) => Json(snap).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResp {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_get_snapshot(State(state): State<AppState>) -> Response {
    match state.engine.get_snapshot() {
        Some(snap) => Json(snap).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResp {
                error: "Keine aktive Session".to_string(),
            }),
        )
            .into_response(),
    }
}

async fn api_audio_proxy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<AudioProxyQuery>,
) -> Response {
    let cache_key = if !q.id.is_empty() {
        q.id.clone()
    } else {
        format!("{}::{}", q.artist, q.title)
    };

    // 1. Check in-memory audio byte cache first for zero-latency seeking & switching
    let cached_bytes = state
        .audio_cache
        .read()
        .ok()
        .and_then(|m| m.get(&cache_key).cloned());

    let audio_bytes = if let Some(b) = cached_bytes {
        b
    } else {
        match fetch_preview_bytes(&state, &q).await {
            Some(b) => {
                if let Ok(mut cache) = state.audio_cache.write() {
                    if cache.len() >= 90 {
                        cache.clear();
                    }
                    cache.insert(cache_key, b.clone());
                }
                b
            }
            None => {
                return (StatusCode::BAD_GATEWAY, "Audio preview unavailable").into_response()
            }
        }
    };

    let total_len = audio_bytes.len();
    if total_len == 0 {
        return (StatusCode::BAD_GATEWAY, "Empty audio buffer").into_response();
    }

    let content_type = if total_len > 8 && &audio_bytes[4..8] == b"ftyp" {
        "audio/mp4"
    } else {
        "audio/mpeg"
    };

    // 2. Support HTTP Range requests (206 Partial Content) so Firefox/Zen Browser can seek to Drop-Sniper timestamps instantly
    if let Some(range_val) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) {
        if let Some((start, end)) = parse_byte_range(range_val, total_len) {
            let slice = audio_bytes[start..=end].to_vec();
            let slice_len = slice.len();
            return (
                StatusCode::PARTIAL_CONTENT,
                [
                    (header::CONTENT_TYPE, content_type.to_string()),
                    (header::ACCEPT_RANGES, "bytes".to_string()),
                    (
                        header::CONTENT_RANGE,
                        format!("bytes {}-{}/{}", start, end, total_len),
                    ),
                    (header::CONTENT_LENGTH, slice_len.to_string()),
                    (header::CACHE_CONTROL, "no-store".to_string()),
                ],
                slice,
            )
                .into_response();
        }
    }

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (header::ACCEPT_RANGES, "bytes".to_string()),
            (header::CONTENT_LENGTH, total_len.to_string()),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        audio_bytes,
    )
        .into_response()
}

async fn fetch_preview_bytes(state: &AppState, q: &AudioProxyQuery) -> Option<Vec<u8>> {
    let client = state.engine.harvester().client();

    if !q.url.is_empty() {
        if let Ok(resp) = client.get(&q.url).send().await {
            if resp.status().is_success() {
                if let Ok(b) = resp.bytes().await {
                    if b.len() > 1024 {
                        return Some(b.to_vec());
                    }
                }
            }
        }
    }

    // Refresh expired CDN signature via track ID or artist+title
    let fresh_url = state
        .engine
        .harvester()
        .refresh_preview_url(&q.id, &q.artist, &q.title)
        .await?;

    let resp = client.get(&fresh_url).send().await.ok()?;
    if resp.status().is_success() {
        let b = resp.bytes().await.ok()?;
        if b.len() > 1024 {
            return Some(b.to_vec());
        }
    }
    None
}

fn parse_byte_range(header_val: &str, total_len: usize) -> Option<(usize, usize)> {
    let range_spec = header_val.strip_prefix("bytes=")?;
    let (start_str, end_str) = range_spec.split_once('-')?;
    let start: usize = start_str.trim().parse().ok()?;
    if start >= total_len {
        return None;
    }
    let end: usize = if end_str.trim().is_empty() {
        total_len - 1
    } else {
        end_str
            .trim()
            .parse::<usize>()
            .ok()?
            .min(total_len - 1)
    };
    if start > end {
        return None;
    }
    Some((start, end))
}
