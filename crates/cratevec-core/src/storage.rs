use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::models::{AnalyzedTrack, HistoryEntry, LibrarySnapshot, SavedTrackEntry};

#[derive(Clone)]
pub struct VectorStore {
    path: PathBuf,
    by_id: Arc<RwLock<HashMap<String, AnalyzedTrack>>>,
    by_dedup: Arc<RwLock<HashMap<String, String>>>,
}

impl VectorStore {
    pub fn new(custom_path: Option<PathBuf>) -> Self {
        let path = custom_path.unwrap_or_else(|| {
            let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            p.push("data");
            let _ = fs::create_dir_all(&p);
            p.push("vector_cache_v5.json");
            p
        });

        let mut map_id = HashMap::new();
        let mut map_dedup = HashMap::new();

        if let Ok(raw) = fs::read_to_string(&path) {
            if let Ok(tracks) = serde_json::from_str::<Vec<AnalyzedTrack>>(&raw) {
                for t in tracks {
                    map_dedup.insert(t.meta.dedup_key.clone(), t.meta.id.clone());
                    map_id.insert(t.meta.id.clone(), t);
                }
            }
        }

        Self {
            path,
            by_id: Arc::new(RwLock::new(map_id)),
            by_dedup: Arc::new(RwLock::new(map_dedup)),
        }
    }

    pub fn get_by_id(&self, id: &str) -> Option<AnalyzedTrack> {
        self.by_id.read().ok()?.get(id).cloned()
    }

    pub fn get_by_dedup(&self, dedup_key: &str) -> Option<AnalyzedTrack> {
        let id = self.by_dedup.read().ok()?.get(dedup_key).cloned()?;
        self.get_by_id(&id)
    }

    pub fn insert(&self, track: AnalyzedTrack) {
        if let (Ok(mut ids), Ok(mut dedups)) = (self.by_id.write(), self.by_dedup.write()) {
            dedups.insert(track.meta.dedup_key.clone(), track.meta.id.clone());
            ids.insert(track.meta.id.clone(), track);
        }
    }

    pub fn all_tracks(&self) -> Vec<AnalyzedTrack> {
        self.by_id
            .read()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn count(&self) -> usize {
        self.by_id.read().map(|m| m.len()).unwrap_or(0)
    }

    pub fn flush(&self) {
        if let Ok(ids) = self.by_id.read() {
            let list: Vec<&AnalyzedTrack> = ids.values().collect();
            if let Ok(json) = serde_json::to_string(&list) {
                if let Some(parent) = self.path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::write(&self.path, json);
            }
        }
    }
}

/// Persistent cross-session storage for listening/swipe history and separately bookmarked ("Gemerkt") tracks.
#[derive(Clone)]
pub struct LibraryStore {
    path: PathBuf,
    data: Arc<RwLock<LibrarySnapshot>>,
}

impl LibraryStore {
    pub fn new(custom_path: Option<PathBuf>) -> Self {
        let path = custom_path.unwrap_or_else(|| {
            let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            p.push("data");
            let _ = fs::create_dir_all(&p);
            p.push("library_v1.json");
            p
        });

        let snapshot = fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<LibrarySnapshot>(&raw).ok())
            .unwrap_or_default();

        Self {
            path,
            data: Arc::new(RwLock::new(snapshot)),
        }
    }

    pub fn snapshot(&self) -> LibrarySnapshot {
        self.data.read().map(|d| d.clone()).unwrap_or_default()
    }

    pub fn record_history(&self, track: AnalyzedTrack, action: &str, seed_context: &str) {
        let ts = now_ms();
        if let Ok(mut d) = self.data.write() {
            // Avoid consecutive exact duplicate entries within the same action
            if let Some(first) = d.history.first() {
                if first.track.meta.id == track.meta.id && first.action == action {
                    return;
                }
            }
            d.history.insert(
                0,
                HistoryEntry {
                    track,
                    action: action.to_string(),
                    seed_context: seed_context.to_string(),
                    timestamp_ms: ts,
                },
            );
            if d.history.len() > 400 {
                d.history.truncate(400);
            }
        }
        self.flush();
    }

    pub fn clear_history(&self) {
        if let Ok(mut d) = self.data.write() {
            d.history.clear();
        }
        self.flush();
    }

    /// Toggles a track in the separate persistent "Gemerkt" vault. Returns `true` if now saved, `false` if removed.
    pub fn toggle_saved(&self, track: AnalyzedTrack) -> bool {
        let is_now_saved = if let Ok(mut d) = self.data.write() {
            if let Some(pos) = d.saved.iter().position(|e| {
                e.track.meta.id == track.meta.id || e.track.meta.dedup_key == track.meta.dedup_key
            }) {
                d.saved.remove(pos);
                false
            } else {
                d.saved.insert(
                    0,
                    SavedTrackEntry {
                        track,
                        saved_at_ms: now_ms(),
                    },
                );
                true
            }
        } else {
            false
        };
        self.flush();
        is_now_saved
    }

    pub fn remove_saved(&self, track_id: &str) -> bool {
        let removed = if let Ok(mut d) = self.data.write() {
            let before = d.saved.len();
            d.saved.retain(|e| e.track.meta.id != track_id);
            d.saved.len() != before
        } else {
            false
        };
        if removed {
            self.flush();
        }
        removed
    }

    pub fn find_track(&self, track_id: &str) -> Option<AnalyzedTrack> {
        let d = self.data.read().ok()?;
        if let Some(e) = d.saved.iter().find(|e| e.track.meta.id == track_id) {
            return Some(e.track.clone());
        }
        if let Some(e) = d.history.iter().find(|e| e.track.meta.id == track_id) {
            return Some(e.track.clone());
        }
        None
    }

    fn flush(&self) {
        if let Ok(d) = self.data.read() {
            if let Ok(json) = serde_json::to_string_pretty(&*d) {
                if let Some(parent) = self.path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let _ = fs::write(&self.path, json);
            }
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
