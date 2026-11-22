use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::models::AnalyzedTrack;

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
            p.push("vector_cache.json");
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
