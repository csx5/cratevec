pub mod dsp;
pub mod harvester;
pub mod models;
pub mod session;
pub mod spotify;
pub mod storage;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};

use crate::harvester::Harvester;
use crate::models::{MixerWeights, SessionSnapshot, SwipeDirection, TrackMetadata};
use crate::session::DiscoverySession;
use crate::storage::VectorStore;

#[derive(Clone)]
pub struct CrateVecEngine {
    harvester: Harvester,
    active_session: Arc<RwLock<Option<DiscoverySession>>>,
    global_seen_dedups: Arc<RwLock<HashSet<String>>>,
}

impl CrateVecEngine {
    pub fn new(cache_path: Option<PathBuf>) -> Self {
        let store = VectorStore::new(cache_path);
        let harvester = Harvester::new(store);
        Self {
            harvester,
            active_session: Arc::new(RwLock::new(None)),
            global_seen_dedups: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn harvester(&self) -> &Harvester {
        &self.harvester
    }

    pub async fn search_tracks(&self, query: &str, limit: usize) -> Result<Vec<TrackMetadata>> {
        self.harvester.search_tracks(query, limit).await
    }

    /// Starts a new discovery session from one or more seed queries (or a Spotify URL),
    /// remembering previously seen/played tracks so new sessions don't repeat the same tracks.
    pub async fn start_session(&self, query: &str) -> Result<SessionSnapshot> {
        let seeds = self.harvester.resolve_seeds(query).await?;

        let prev_seen: HashSet<String> = self
            .global_seen_dedups
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();

        let pool = self
            .harvester
            .harvest_pool_for_seeds(&seeds, 56, &prev_seen)
            .await;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let session_id = format!("sess_{}", ts);

        let session = DiscoverySession::new(session_id, seeds.clone(), pool, prev_seen);
        let snapshot = session.snapshot();

        if let Ok(mut g) = self.global_seen_dedups.write() {
            for s in &seeds {
                g.insert(s.meta.dedup_key.clone());
            }
            if let Some(card) = &snapshot.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        if let Ok(mut guard) = self.active_session.write() {
            *guard = Some(session);
        }

        Ok(snapshot)
    }

    /// Adds a bridge seed song to the active session and merges its acoustic neighborhood.
    pub async fn add_bridge_seed(&self, query: &str) -> Result<SessionSnapshot> {
        let seed = self.harvester.resolve_single_seed(query).await?;
        let prev_seen: HashSet<String> = self
            .global_seen_dedups
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();

        let new_candidates = self
            .harvester
            .harvest_pool_for_seeds(std::slice::from_ref(&seed), 30, &prev_seen)
            .await;

        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;

        session.add_bridge_seed(seed.clone(), new_candidates);
        let snapshot = session.snapshot();

        if let Ok(mut g) = self.global_seen_dedups.write() {
            g.insert(seed.meta.dedup_key);
            if let Some(card) = &snapshot.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        Ok(snapshot)
    }

    /// Processes a 4-way swipe (Right = Like, Left = Skip, Up = Super-Seed, Down = Kill Vibe).
    pub async fn swipe(
        &self,
        track_id: &str,
        direction: SwipeDirection,
    ) -> Result<SessionSnapshot> {
        let super_seed = {
            let mut guard = self
                .active_session
                .write()
                .map_err(|_| anyhow!("Session lock poisoned"))?;
            let session = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
            session.apply_swipe(track_id, direction)
        };

        if let Some(seed_track) = super_seed {
            let prev_seen: HashSet<String> = self
                .global_seen_dedups
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            let extra = self
                .harvester
                .harvest_pool_for_seeds(std::slice::from_ref(&seed_track), 24, &prev_seen)
                .await;
            if let Ok(mut guard) = self.active_session.write() {
                if let Some(session) = guard.as_mut() {
                    session.extend_pool(extra);
                }
            }
        }

        let snap = self
            .get_snapshot()
            .ok_or_else(|| anyhow!("Keine aktive Session"))?;

        if let Ok(mut g) = self.global_seen_dedups.write() {
            if let Some(card) = &snap.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        Ok(snap)
    }

    /// Focuses a specific candidate track from the queue or 2D map so it immediately becomes the active card.
    pub fn focus_track(&self, track_id: &str) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.focus_track(track_id);
        let snap = session.snapshot();

        if let Ok(mut g) = self.global_seen_dedups.write() {
            if let Some(card) = &snap.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        Ok(snap)
    }

    pub fn update_weights(&self, weights: MixerWeights) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.update_weights(weights);
        Ok(session.snapshot())
    }

    pub fn get_snapshot(&self) -> Option<SessionSnapshot> {
        self.active_session
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.snapshot()))
    }
}
