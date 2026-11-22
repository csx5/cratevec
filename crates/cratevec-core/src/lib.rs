pub mod dsp;
pub mod harvester;
pub mod models;
pub mod session;
pub mod spotify;
pub mod storage;
pub mod trainer;
pub mod ytmusic;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};

use crate::harvester::Harvester;
use crate::models::{
    AnalyzedTrack, LibrarySnapshot, MixerWeights, SessionSnapshot, SwipeDirection, TrackMetadata,
};
use crate::session::DiscoverySession;
use crate::storage::{LibraryStore, VectorStore};
use crate::trainer::{TrainedMetricModel, TrainingReport};

#[derive(Clone)]
pub struct CrateVecEngine {
    harvester: Harvester,
    library: LibraryStore,
    active_session: Arc<RwLock<Option<DiscoverySession>>>,
    global_seen_dedups: Arc<RwLock<HashSet<String>>>,
    metric_model: Arc<RwLock<TrainedMetricModel>>,
}

impl CrateVecEngine {
    pub fn new(cache_path: Option<PathBuf>) -> Self {
        let store = VectorStore::new(cache_path);
        let library = LibraryStore::new(None);
        let harvester = Harvester::new(store);
        let model = TrainedMetricModel::load_or_default(None);
        Self {
            harvester,
            library,
            active_session: Arc::new(RwLock::new(None)),
            global_seen_dedups: Arc::new(RwLock::new(HashSet::new())),
            metric_model: Arc::new(RwLock::new(model)),
        }
    }

    pub fn harvester(&self) -> &Harvester {
        &self.harvester
    }

    pub async fn search_tracks(&self, query: &str, limit: usize) -> Result<Vec<TrackMetadata>> {
        self.harvester.search_tracks(query, limit).await
    }

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

        let model = self
            .metric_model
            .read()
            .map(|m| m.clone())
            .unwrap_or_default();

        let session = DiscoverySession::new(session_id, seeds.clone(), pool, prev_seen, model);
        let snapshot = session.snapshot();

        for s in &seeds {
            self.library
                .record_history(s.clone(), "seed", &format!("{} - {}", s.meta.artist, s.meta.title));
        }

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

        self.library.record_history(
            seed.clone(),
            "seed",
            &format!("Bridge: {} - {}", seed.meta.artist, seed.meta.title),
        );

        if let Ok(mut g) = self.global_seen_dedups.write() {
            g.insert(seed.meta.dedup_key);
            if let Some(card) = &snapshot.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        Ok(snapshot)
    }

    pub async fn swipe(
        &self,
        track_id: &str,
        direction: SwipeDirection,
    ) -> Result<SessionSnapshot> {
        let (super_seed, auto_replenish_seed, swiped_track, seed_ctx) = {
            let mut guard = self
                .active_session
                .write()
                .map_err(|_| anyhow!("Session lock poisoned"))?;
            let session = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
            let swiped = session.pool.iter().find(|t| t.meta.id == track_id).cloned();
            let ctx = session
                .seeds
                .first()
                .map(|s| format!("{} - {}", s.meta.artist, s.meta.title))
                .unwrap_or_default();
            let res = session.apply_swipe(track_id, direction);
            if let Ok(mut mm) = self.metric_model.write() {
                *mm = session.metric_model.clone();
            }

            let auto_seed = if res.is_none()
                && (session.remaining_unseen_count() < 12 || session.likes_since_replenish >= 4)
            {
                session.likes_since_replenish = 0;
                session.latest_liked_or_seed()
            } else {
                None
            };

            (res, auto_seed, swiped, ctx)
        };

        if let Some(track) = swiped_track {
            let action = match direction {
                SwipeDirection::Right => "right",
                SwipeDirection::Left => "left",
                SwipeDirection::Up => "up",
                SwipeDirection::Down => "down",
            };
            self.library.record_history(track, action, &seed_ctx);
        }

        let harvest_target = super_seed
            .map(|t| (t, 26usize))
            .or_else(|| auto_replenish_seed.map(|t| (t, 20usize)));

        if let Some((seed_track, count)) = harvest_target {
            let prev_seen: HashSet<String> = self
                .global_seen_dedups
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            let extra = self
                .harvester
                .harvest_pool_for_seeds(std::slice::from_ref(&seed_track), count, &prev_seen)
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

    pub fn undo_last_swipe(&self) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        if !session.undo_last_swipe() {
            return Err(anyhow!("Kein vorheriger Swipe zum Rueckgaengigmachen"));
        }
        if let Ok(mut mm) = self.metric_model.write() {
            *mm = session.metric_model.clone();
        }
        Ok(session.snapshot())
    }

    pub fn remove_negative_anchor(&self, track_id: &str) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.remove_negative_anchor(track_id);
        Ok(session.snapshot())
    }

    pub fn remove_crate_track(&self, track_id: &str) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.remove_crate_track(track_id);
        Ok(session.snapshot())
    }

    pub async fn load_more_candidates(&self) -> Result<SessionSnapshot> {
        let seed_track = {
            let guard = self
                .active_session
                .read()
                .map_err(|_| anyhow!("Session lock poisoned"))?;
            let session = guard
                .as_ref()
                .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
            session
                .latest_liked_or_seed()
                .ok_or_else(|| anyhow!("Kein Seed zum Nachladen vorhanden"))?
        };

        let prev_seen: HashSet<String> = self
            .global_seen_dedups
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();

        let extra = self
            .harvester
            .harvest_pool_for_seeds(std::slice::from_ref(&seed_track), 28, &prev_seen)
            .await;

        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.likes_since_replenish = 0;
        session.extend_pool(extra);
        Ok(session.snapshot())
    }

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

    pub fn select_corridor(&self, corridor_id: &str) -> Result<SessionSnapshot> {
        let mut guard = self
            .active_session
            .write()
            .map_err(|_| anyhow!("Session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Keine aktive Session vorhanden"))?;
        session.select_corridor(corridor_id);
        let snap = session.snapshot();

        if let Ok(mut g) = self.global_seen_dedups.write() {
            if let Some(card) = &snap.current_card {
                g.insert(card.track.meta.dedup_key.clone());
            }
        }

        Ok(snap)
    }

    pub fn get_snapshot(&self) -> Option<SessionSnapshot> {
        self.active_session
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.snapshot()))
    }

    pub fn get_library(&self) -> LibrarySnapshot {
        self.library.snapshot()
    }

    pub fn clear_history(&self) -> LibrarySnapshot {
        self.library.clear_history();
        self.library.snapshot()
    }

    pub fn toggle_save_track(&self, track_id: &str) -> Result<(bool, LibrarySnapshot)> {
        let track = self
            .find_any_track(track_id)
            .ok_or_else(|| anyhow!("Track nicht gefunden: {}", track_id))?;
        let is_saved = self.library.toggle_saved(track);
        Ok((is_saved, self.library.snapshot()))
    }

    pub fn remove_saved_track(&self, track_id: &str) -> LibrarySnapshot {
        self.library.remove_saved(track_id);
        self.library.snapshot()
    }

    fn find_any_track(&self, track_id: &str) -> Option<AnalyzedTrack> {
        if let Ok(guard) = self.active_session.read() {
            if let Some(session) = guard.as_ref() {
                if let Some(t) = session.pool.iter().find(|t| t.meta.id == track_id) {
                    return Some(t.clone());
                }
                if let Some(t) = session.seeds.iter().find(|t| t.meta.id == track_id) {
                    return Some(t.clone());
                }
                if let Some(t) = session.liked_crate.iter().find(|t| t.meta.id == track_id) {
                    return Some(t.clone());
                }
                if let Some(t) = session
                    .negative_anchors
                    .iter()
                    .find(|t| t.meta.id == track_id)
                {
                    return Some(t.clone());
                }
            }
        }
        if let Some(t) = self.library.find_track(track_id) {
            return Some(t);
        }
        self.harvester.store().get_by_id(track_id)
    }

    /// Runs Contrastive Triplet Margin Training over all cached vectors (optionally harvesting additional
    /// multi-genre reference clusters first) and updates the live 48-D Mahalanobis metric model.
    pub async fn train_metric_model(
        &self,
        seed_clusters: &[&str],
        epochs: usize,
    ) -> Result<TrainingReport> {
        let mut used_dedups = HashSet::new();
        let mut labeled_clusters: Vec<Vec<crate::models::AnalyzedTrack>> = Vec::new();

        for &q in seed_clusters {
            if let Ok(seeds) = self.harvester.resolve_seeds(q).await {
                for s in &seeds {
                    used_dedups.insert(s.meta.dedup_key.clone());
                }
                let mut cluster = self
                    .harvester
                    .harvest_pool_for_seeds(&seeds, 20, &used_dedups)
                    .await;
                for c in &cluster {
                    used_dedups.insert(c.meta.dedup_key.clone());
                }
                cluster.extend(seeds);
                if cluster.len() >= 4 {
                    labeled_clusters.push(cluster);
                }
            }
        }

        let mut model = self
            .metric_model
            .read()
            .map(|m| m.clone())
            .unwrap_or_default();

        let report = if labeled_clusters.len() >= 2 {
            model.train_on_labeled_clusters(&labeled_clusters, epochs)
        } else {
            let corpus = self.harvester.store().all_tracks();
            model.train_on_corpus(&corpus, epochs)
        };

        if let Ok(mut mm) = self.metric_model.write() {
            *mm = model.clone();
        }
        if let Ok(mut sess_guard) = self.active_session.write() {
            if let Some(sess) = sess_guard.as_mut() {
                sess.metric_model = model;
            }
        }

        Ok(report)
    }
}
