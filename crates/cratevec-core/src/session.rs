use std::collections::HashSet;

use crate::dsp::{cosine_similarity, cosine_to_percentage};
use crate::models::{
    AnalyzedTrack, MixerWeights, ScoredCandidate, SessionSnapshot, SonicMapPoint, SwipeDirection,
    VibeCorridor,
};
use crate::trainer::TrainedMetricModel;

#[derive(Debug, Clone)]
pub struct DiscoverySession {
    pub session_id: String,
    pub seeds: Vec<AnalyzedTrack>,
    pub pool: Vec<AnalyzedTrack>,
    pub liked_crate: Vec<AnalyzedTrack>,
    pub negative_anchors: Vec<AnalyzedTrack>,
    pub seen_ids: HashSet<String>,
    pub seen_dedups: HashSet<String>,
    pub prev_session_dedups: HashSet<String>,
    pub pinned_track_id: Option<String>,
    pub active_corridor: String,
    pub weights: MixerWeights,
    pub metric_model: TrainedMetricModel,
    pub target_beat: Vec<f32>,
    pub target_vocal: Vec<f32>,
    pub target_atmo: Vec<f32>,
    pub target_bpm: f32,
    pub target_aggression: f32,
    pub target_darkness: f32,
}

impl DiscoverySession {
    pub fn new(
        session_id: String,
        mut seeds: Vec<AnalyzedTrack>,
        pool: Vec<AnalyzedTrack>,
        prev_session_dedups: HashSet<String>,
        metric_model: TrainedMetricModel,
    ) -> Self {
        let mut seen_ids = HashSet::new();
        let mut seen_dedups = HashSet::new();
        for s in &seeds {
            seen_ids.insert(s.meta.id.clone());
            seen_dedups.insert(s.meta.dedup_key.clone());
        }

        let mut deduped_pool = Vec::new();
        let mut pool_dedups = HashSet::new();
        for t in pool {
            if !seen_dedups.contains(&t.meta.dedup_key)
                && pool_dedups.insert(t.meta.dedup_key.clone())
            {
                deduped_pool.push(t);
            }
        }

        whiten_tracks(&mut seeds, &mut deduped_pool);

        let (target_beat, target_vocal, target_atmo, target_bpm, target_aggression, target_darkness) =
            compute_seed_centroid(&seeds);

        Self {
            session_id,
            seeds,
            pool: deduped_pool,
            liked_crate: Vec::new(),
            negative_anchors: Vec::new(),
            seen_ids,
            seen_dedups,
            prev_session_dedups,
            pinned_track_id: None,
            active_corridor: "all".to_string(),
            weights: MixerWeights::default(),
            metric_model,
            target_beat,
            target_vocal,
            target_atmo,
            target_bpm,
            target_aggression,
            target_darkness,
        }
    }

    pub fn extend_pool(&mut self, mut tracks: Vec<AnalyzedTrack>) {
        let existing_dedups: HashSet<String> =
            self.pool.iter().map(|t| t.meta.dedup_key.clone()).collect();

        for t in &mut tracks {
            l2_normalize_in_place(&mut t.dsp.v_beat);
            l2_normalize_in_place(&mut t.dsp.v_vocal);
            l2_normalize_in_place(&mut t.dsp.v_atmo);
        }

        for t in tracks {
            if !self.seen_dedups.contains(&t.meta.dedup_key)
                && !existing_dedups.contains(&t.meta.dedup_key)
            {
                self.pool.push(t);
            }
        }
    }

    pub fn add_bridge_seed(&mut self, seed: AnalyzedTrack, new_candidates: Vec<AnalyzedTrack>) {
        self.seen_ids.insert(seed.meta.id.clone());
        self.seen_dedups.insert(seed.meta.dedup_key.clone());
        if !self
            .seeds
            .iter()
            .any(|s| s.meta.dedup_key == seed.meta.dedup_key)
        {
            self.seeds.push(seed);
        }
        let (tb, tv, ta, bpm, aggr, dark) = compute_seed_centroid(&self.seeds);
        self.target_beat = tb;
        self.target_vocal = tv;
        self.target_atmo = ta;
        self.target_bpm = bpm;
        self.target_aggression = aggr;
        self.target_darkness = dark;
        self.pinned_track_id = None;
        self.extend_pool(new_candidates);
    }

    pub fn update_weights(&mut self, weights: MixerWeights) {
        self.weights = weights;
    }

    /// Switches the active Vibe-Corridor ("all", "rhythm_bass", "vocal_flow", "atmo_sample")
    /// and aligns the stem mixer weights and deck ordering to that sonic corridor.
    pub fn select_corridor(&mut self, corridor_id: &str) {
        self.active_corridor = corridor_id.to_string();
        self.pinned_track_id = None;
        match corridor_id {
            "rhythm_bass" => {
                self.weights.beat = 0.64;
                self.weights.vocal = 0.18;
                self.weights.atmo = 0.18;
            }
            "vocal_flow" => {
                self.weights.beat = 0.20;
                self.weights.vocal = 0.62;
                self.weights.atmo = 0.18;
            }
            "atmo_sample" => {
                self.weights.beat = 0.18;
                self.weights.vocal = 0.20;
                self.weights.atmo = 0.62;
            }
            _ => {
                self.active_corridor = "all".to_string();
                self.weights.beat = 0.38;
                self.weights.vocal = 0.31;
                self.weights.atmo = 0.31;
            }
        }
    }

    pub fn focus_track(&mut self, track_id: &str) {
        if self.pool.iter().any(|t| t.meta.id == track_id) {
            self.pinned_track_id = Some(track_id.to_string());
        }
    }

    pub fn apply_swipe(
        &mut self,
        track_id: &str,
        direction: SwipeDirection,
    ) -> Option<AnalyzedTrack> {
        let track = self.pool.iter().find(|t| t.meta.id == track_id).cloned()?;

        self.seen_ids.insert(track.meta.id.clone());
        self.seen_dedups.insert(track.meta.dedup_key.clone());
        if self.pinned_track_id.as_deref() == Some(track_id) {
            self.pinned_track_id = None;
        }

        self.metric_model.online_update_from_swipe(
            &self.target_beat,
            &self.target_vocal,
            &self.target_atmo,
            &track,
            direction,
        );

        match direction {
            SwipeDirection::Right => {
                self.target_beat = rocchio_step(&self.target_beat, &track.dsp.v_beat, 0.26);
                self.target_vocal = rocchio_step(&self.target_vocal, &track.dsp.v_vocal, 0.26);
                self.target_atmo = rocchio_step(&self.target_atmo, &track.dsp.v_atmo, 0.26);
                self.target_bpm = self.target_bpm * 0.8 + track.dsp.bpm * 0.2;
                self.target_aggression =
                    self.target_aggression * 0.82 + track.dsp.aggression_index * 0.18;
                self.target_darkness =
                    self.target_darkness * 0.82 + track.dsp.darkness_index * 0.18;
                if !self
                    .liked_crate
                    .iter()
                    .any(|t| t.meta.dedup_key == track.meta.dedup_key)
                {
                    self.liked_crate.push(track);
                }
                None
            }
            SwipeDirection::Left => {
                self.target_beat = rocchio_step(&self.target_beat, &track.dsp.v_beat, -0.09);
                self.target_vocal = rocchio_step(&self.target_vocal, &track.dsp.v_vocal, -0.09);
                self.target_atmo = rocchio_step(&self.target_atmo, &track.dsp.v_atmo, -0.09);
                None
            }
            SwipeDirection::Up => {
                self.target_beat = rocchio_step(&self.target_beat, &track.dsp.v_beat, 0.55);
                self.target_vocal = rocchio_step(&self.target_vocal, &track.dsp.v_vocal, 0.55);
                self.target_atmo = rocchio_step(&self.target_atmo, &track.dsp.v_atmo, 0.55);
                self.target_bpm = self.target_bpm * 0.5 + track.dsp.bpm * 0.5;
                self.target_aggression =
                    self.target_aggression * 0.55 + track.dsp.aggression_index * 0.45;
                self.target_darkness =
                    self.target_darkness * 0.55 + track.dsp.darkness_index * 0.45;
                if !self
                    .liked_crate
                    .iter()
                    .any(|t| t.meta.dedup_key == track.meta.dedup_key)
                {
                    self.liked_crate.push(track.clone());
                }
                if !self
                    .seeds
                    .iter()
                    .any(|s| s.meta.dedup_key == track.meta.dedup_key)
                {
                    self.seeds.push(track.clone());
                }
                Some(track)
            }
            SwipeDirection::Down => {
                let (sb, sv, sa, _, _, _) = compute_seed_centroid(&self.seeds);
                self.target_beat = gram_schmidt_repel(&self.target_beat, &sb, &track.dsp.v_beat);
                self.target_vocal =
                    gram_schmidt_repel(&self.target_vocal, &sv, &track.dsp.v_vocal);
                self.target_atmo = gram_schmidt_repel(&self.target_atmo, &sa, &track.dsp.v_atmo);
                if !self
                    .negative_anchors
                    .iter()
                    .any(|n| n.meta.dedup_key == track.meta.dedup_key)
                {
                    self.negative_anchors.push(track);
                }
                None
            }
        }
    }

    pub fn ranked_candidates(&self) -> Vec<ScoredCandidate> {
        let (w_beat, w_vocal, w_atmo) = self.weights.normalized_triplet();

        let mut scored: Vec<ScoredCandidate> = self
            .pool
            .iter()
            .filter(|t| {
                !self.seen_ids.contains(&t.meta.id)
                    && !self.seen_dedups.contains(&t.meta.dedup_key)
            })
            .map(|track| self.score_single_candidate(track, w_beat, w_vocal, w_atmo))
            .collect();

        if !scored.is_empty() {
            let n = scored.len() as f32;
            let mean_v = scored.iter().map(|c| c.vocal_similarity).sum::<f32>() / n;
            let mean_aggr = scored
                .iter()
                .map(|c| c.track.dsp.aggression_index)
                .sum::<f32>()
                / n;

            for c in &mut scored {
                let aggr = c.track.dsp.aggression_index;
                let dark = c.track.dsp.darkness_index;
                let dv = c.vocal_similarity - mean_v;

                // Partition candidates into 3 distinct musical corridors:
                // 1. "rhythm_bass" -> HART & AGGRESSIV (Hard-hitting, high aggression & dark pressure)
                // 2. "atmo_sample" -> CHILL & LAID-BACK (Smooth, relaxed energy & warm sample/harmony)
                // 3. "vocal_flow"  -> FLOW & VOCAL-FOKUS (Strongest vocal timbre & cadence lock)
                let tag = if aggr >= mean_aggr + 0.06 && (aggr + dark * 0.5) >= 0.80 {
                    "rhythm_bass"
                } else if aggr <= mean_aggr - 0.05 {
                    "atmo_sample"
                } else if dv >= 0.0 {
                    "vocal_flow"
                } else if aggr >= mean_aggr {
                    "rhythm_bass"
                } else {
                    "atmo_sample"
                };

                c.corridor_tag = tag.to_string();
                if let Some(first_badge) = c.explanation.first_mut() {
                    *first_badge = match tag {
                        "rhythm_bass" => "Vibe: Hart & Aggressiv".to_string(),
                        "vocal_flow" => "Vibe: Flow & Vocal-Lock".to_string(),
                        _ => "Vibe: Chill & Laid-Back".to_string(),
                    };
                }

                match self.active_corridor.as_str() {
                    "rhythm_bass" => {
                        // Explicitly boost hard, aggressive, dark tracks and penalize soft/chill tracks
                        let boost = (aggr - 0.50) * 32.0 + (dark - 0.50) * 14.0;
                        c.total_score =
                            ((c.total_score + boost).clamp(5.0, 99.4) * 10.0).round() / 10.0;
                    }
                    "atmo_sample" => {
                        // Explicitly boost chill, smooth, laid-back tracks and penalize aggressive tracks
                        let boost = (0.52 - aggr) * 32.0 + (c.atmo_similarity - 60.0) * 0.18;
                        c.total_score =
                            ((c.total_score + boost).clamp(5.0, 99.4) * 10.0).round() / 10.0;
                    }
                    "vocal_flow" => {
                        let boost = if c.corridor_tag == "vocal_flow" {
                            6.5
                        } else {
                            -3.5
                        };
                        c.total_score =
                            ((c.total_score + boost).clamp(5.0, 99.4) * 10.0).round() / 10.0;
                    }
                    _ => {}
                }
            }
        }

        scored.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        diversify_artist_order(&mut scored);

        if let Some(pin_id) = &self.pinned_track_id {
            if let Some(pos) = scored.iter().position(|c| &c.track.meta.id == pin_id) {
                if pos > 0 {
                    let pinned = scored.remove(pos);
                    scored.insert(0, pinned);
                }
            }
        }

        scored
    }

    fn classify_corridor(&self, beat_pct: f32, vocal_pct: f32, atmo_pct: f32) -> &'static str {
        if beat_pct >= vocal_pct + 1.2 && beat_pct >= atmo_pct - 0.5 {
            "rhythm_bass"
        } else if vocal_pct >= beat_pct - 0.5 && vocal_pct >= atmo_pct + 0.5 {
            "vocal_flow"
        } else {
            "atmo_sample"
        }
    }

    fn build_vibe_corridors(&self, ranked: &[ScoredCandidate]) -> Vec<VibeCorridor> {
        let mut beat_artists = Vec::new();
        let mut vocal_artists = Vec::new();
        let mut atmo_artists = Vec::new();
        let mut beat_count = 0usize;
        let mut vocal_count = 0usize;
        let mut atmo_count = 0usize;

        for c in ranked {
            match c.corridor_tag.as_str() {
                "rhythm_bass" => {
                    beat_count += 1;
                    if beat_artists.len() < 3 && !beat_artists.contains(&c.track.meta.artist) {
                        beat_artists.push(c.track.meta.artist.clone());
                    }
                }
                "vocal_flow" => {
                    vocal_count += 1;
                    if vocal_artists.len() < 3 && !vocal_artists.contains(&c.track.meta.artist) {
                        vocal_artists.push(c.track.meta.artist.clone());
                    }
                }
                _ => {
                    atmo_count += 1;
                    if atmo_artists.len() < 3 && !atmo_artists.contains(&c.track.meta.artist) {
                        atmo_artists.push(c.track.meta.artist.clone());
                    }
                }
            }
        }

        let all_artists: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|c| c.track.meta.artist.clone())
            .collect();

        vec![
            VibeCorridor {
                id: "all".to_string(),
                title: "SEED-VIBE (EXAKT)".to_string(),
                description: "Exakter Energie-, Haerte- & Klang-Match zum Seed".to_string(),
                sample_artists: if all_artists.is_empty() {
                    "—".to_string()
                } else {
                    all_artists.join(", ")
                },
                track_count: ranked.len(),
                active: self.active_corridor == "all",
            },
            VibeCorridor {
                id: "rhythm_bass".to_string(),
                title: "HART & AGGRESSIV".to_string(),
                description: "Hoher Druck, harte Transienten & duesterer Sub-Bass".to_string(),
                sample_artists: if beat_artists.is_empty() {
                    "—".to_string()
                } else {
                    beat_artists.join(", ")
                },
                track_count: beat_count,
                active: self.active_corridor == "rhythm_bass",
            },
            VibeCorridor {
                id: "vocal_flow".to_string(),
                title: "FLOW & VOCAL-LOCK".to_string(),
                description: "Stimmfarbe, Artikulations-Cadence & Rap-Delivery".to_string(),
                sample_artists: if vocal_artists.is_empty() {
                    "—".to_string()
                } else {
                    vocal_artists.join(", ")
                },
                track_count: vocal_count,
                active: self.active_corridor == "vocal_flow",
            },
            VibeCorridor {
                id: "atmo_sample".to_string(),
                title: "CHILL & LAID-BACK".to_string(),
                description: "Entspannter Groove, Sample-Waerme & ruhigere Energie".to_string(),
                sample_artists: if atmo_artists.is_empty() {
                    "—".to_string()
                } else {
                    atmo_artists.join(", ")
                },
                track_count: atmo_count,
                active: self.active_corridor == "atmo_sample",
            },
        ]
    }

    fn score_single_candidate(
        &self,
        track: &AnalyzedTrack,
        w_beat: f32,
        w_vocal: f32,
        w_atmo: f32,
    ) -> ScoredCandidate {
        let raw_beat = self
            .metric_model
            .sim_beat(&self.target_beat, &track.dsp.v_beat);
        let raw_vocal = self
            .metric_model
            .sim_vocal(&self.target_vocal, &track.dsp.v_vocal);
        let raw_atmo = self
            .metric_model
            .sim_atmo(&self.target_atmo, &track.dsp.v_atmo);

        let beat_pct = cosine_to_percentage(raw_beat);
        let vocal_pct = cosine_to_percentage(raw_vocal);
        let atmo_pct = cosine_to_percentage(raw_atmo);

        let corridor_tag = self.classify_corridor(beat_pct, vocal_pct, atmo_pct);

        // 1. Hybrid Linear + Harmonic Mean with Bottleneck Penalty
        let s_lin = w_beat * beat_pct + w_vocal * vocal_pct + w_atmo * atmo_pct;
        let s_harm = 1.0
            / (w_beat / (beat_pct + 10.0)
                + w_vocal / (vocal_pct + 10.0)
                + w_atmo / (atmo_pct + 10.0))
            - 10.0;
        let bottleneck = w_beat * (46.0 - beat_pct).max(0.0)
            + w_vocal * (46.0 - vocal_pct).max(0.0)
            + w_atmo * (46.0 - atmo_pct).max(0.0);

        let mut total = 0.72 * s_lin + 0.28 * s_harm - 0.35 * bottleneck;

        // 2. Scene Consensus Affinity Bonus (rewards candidates verified across multiple scene sources)
        total += (track.meta.scene_affinity - 0.72) * 15.0;

        // 3. Arousal-Valence Vibe-Alignment Gatekeeper (Aggression/Hardness & Darkness/Mood match)
        // Ensures an aggressive/dark track isn't matched with a calm/chill or bright party/carnival track (and vice versa).
        let aggr_diff = (track.dsp.aggression_index - self.target_aggression).abs();
        let dark_diff = (track.dsp.darkness_index - self.target_darkness).abs();

        if aggr_diff <= 0.12 && dark_diff <= 0.14 {
            total += 4.2;
        } else {
            if aggr_diff > 0.14 {
                total -= ((aggr_diff - 0.14) * 44.0).clamp(0.0, 26.0);
            }
            if dark_diff > 0.15 {
                total -= ((dark_diff - 0.15) * 40.0).clamp(0.0, 24.0);
            }
        }

        // 4. Continuous Multi-Octave Gaussian Tempo Kernel coupled to Beat weight w_beat
        let bpm_diff = (track.dsp.bpm - self.target_bpm)
            .abs()
            .min((track.dsp.bpm * 2.0 - self.target_bpm).abs())
            .min((track.dsp.bpm - self.target_bpm * 2.0).abs());
        let gauss_bpm = (-0.5 * (bpm_diff / 9.5).powi(2)).exp();
        let bpm_bonus = (1.6 + 4.2 * w_beat) * (2.0 * gauss_bpm - 1.0);
        total += bpm_bonus;

        // 5. Bridge-Mode Symmetry Bonus
        if self.seeds.len() >= 2 {
            let mut seed_sims = Vec::with_capacity(self.seeds.len());
            for s in &self.seeds {
                let sb = self.metric_model.sim_beat(&s.dsp.v_beat, &track.dsp.v_beat);
                let sv = self
                    .metric_model
                    .sim_vocal(&s.dsp.v_vocal, &track.dsp.v_vocal);
                let sa = self.metric_model.sim_atmo(&s.dsp.v_atmo, &track.dsp.v_atmo);
                seed_sims.push(w_beat * sb + w_vocal * sv + w_atmo * sa);
            }
            let min_sim = seed_sims
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let mean_sim = seed_sims.iter().sum::<f32>() / seed_sims.len() as f32;
            let std_sim = (seed_sims
                .iter()
                .map(|&x| (x - mean_sim) * (x - mean_sim))
                .sum::<f32>()
                / seed_sims.len() as f32)
                .sqrt();
            let bridge_mod = 6.2 * min_sim - 4.8 * std_sim;
            total += bridge_mod;
        }

        // 6. Underground slider modifier
        let obscurity = 1.0 - track.meta.popularity_score;
        let ug_mod = self.weights.underground * (obscurity - 0.45) * 16.0;
        total += ug_mod;

        // 7. Cross-session repeat penalty
        if self.prev_session_dedups.contains(&track.meta.dedup_key) {
            total -= 18.0;
        }

        // 8. Session exploration jitter (-1.1 .. +1.1)
        total += session_track_jitter(&self.session_id, &track.meta.dedup_key);

        // 9. Mixer-weighted Negative Anchor Penalty ("Kill Vibe" filter)
        let mut neg_penalty = 0.0f32;
        for neg in &self.negative_anchors {
            let n_beat = self
                .metric_model
                .sim_beat(&neg.dsp.v_beat, &track.dsp.v_beat);
            let n_vocal = self
                .metric_model
                .sim_vocal(&neg.dsp.v_vocal, &track.dsp.v_vocal);
            let n_atmo = self
                .metric_model
                .sim_atmo(&neg.dsp.v_atmo, &track.dsp.v_atmo);
            let n_comb = w_beat * n_beat + w_vocal * n_vocal + w_atmo * n_atmo;
            if n_comb > 0.28 {
                let p = (42.0 * (n_comb - 0.28).powf(1.3)).clamp(0.0, 36.0);
                if p > neg_penalty {
                    neg_penalty = p;
                }
            }
        }
        total -= neg_penalty;

        let total_score = (total.clamp(5.0, 99.4) * 10.0).round() / 10.0;

        let corridor_label = match corridor_tag {
            "rhythm_bass" => "Korridor: Beat & Groove",
            "vocal_flow" => "Korridor: Flow & Vocals",
            _ => "Korridor: Sample & Atmo",
        };

        let mut explanation = Vec::with_capacity(6);
        explanation.push(corridor_label.to_string());
        explanation.push(format!(
            "Haerte/Aggr: {:.0}% | Duesterkeit: {:.0}%",
            track.dsp.aggression_index * 100.0,
            track.dsp.darkness_index * 100.0
        ));
        explanation.push(format!(
            "{:.1} BPM (Drop @ {:.1}s)",
            track.dsp.bpm, track.dsp.drop_start_sec
        ));
        explanation.push(format!(
            "Beat: {:.0}% | Vocals: {:.0}% | Atmo: {:.0}%",
            beat_pct, vocal_pct, atmo_pct
        ));
        explanation.push(format!(
            "Szene-Konsens: {:.0}% ({})",
            track.meta.scene_affinity * 100.0,
            track.meta.discovery_channel
        ));
        if obscurity >= 0.65 {
            explanation.push("Underground-Cut".to_string());
        }

        let (map_x, map_y) = self.project_to_2d(
            track, raw_beat, raw_vocal, raw_atmo, w_beat, w_vocal, w_atmo,
        );

        ScoredCandidate {
            track: track.clone(),
            total_score,
            beat_similarity: beat_pct,
            vocal_similarity: vocal_pct,
            atmo_similarity: atmo_pct,
            negative_penalty: (neg_penalty * 10.0).round() / 10.0,
            corridor_tag: corridor_tag.to_string(),
            explanation,
            map_x,
            map_y,
        }
    }

    fn project_to_2d(
        &self,
        track: &AnalyzedTrack,
        raw_beat: f32,
        raw_vocal: f32,
        raw_atmo: f32,
        w_beat: f32,
        w_vocal: f32,
        w_atmo: f32,
    ) -> (f32, f32) {
        let n_s = self.seeds.len().max(1) as f32;
        let mean_glitch: f32 =
            self.seeds.iter().map(|s| s.dsp.glitch_brightness).sum::<f32>() / n_s;
        let mean_dark: f32 = self.seeds.iter().map(|s| s.dsp.darkness_index).sum::<f32>() / n_s;
        let mean_sub: f32 = self.seeds.iter().map(|s| s.dsp.sub_bass_ratio).sum::<f32>() / n_s;
        let mean_dist: f32 = self.seeds.iter().map(|s| s.dsp.bass_distortion).sum::<f32>() / n_s;
        let mean_vocal: f32 = self.seeds.iter().map(|s| s.dsp.vocal_presence).sum::<f32>() / n_s;

        let ux = ((track.dsp.bpm - self.target_bpm) / 28.0) * 0.35
            + (track.dsp.glitch_brightness - mean_glitch) * 1.15
            + (track.dsp.v_beat.get(7).copied().unwrap_or(0.0)
                + track.dsp.v_beat.get(8).copied().unwrap_or(0.0)
                - self.target_beat.get(7).copied().unwrap_or(0.0)
                - self.target_beat.get(8).copied().unwrap_or(0.0))
                * 0.35;

        let uy = (track.dsp.darkness_index - mean_dark) * 1.10
            + (track.dsp.sub_bass_ratio - mean_sub) * 1.35
            + (track.dsp.bass_distortion - mean_dist) * 0.45
            - (track.dsp.vocal_presence - mean_vocal) * 1.15
            + (raw_atmo - raw_vocal) * 0.30;

        let mut rho_x = 0.0f32;
        let mut rho_y = 0.0f32;
        for i in 0..8 {
            let db_even = track.dsp.v_beat.get(2 * i).copied().unwrap_or(0.0)
                - self.target_beat.get(2 * i).copied().unwrap_or(0.0);
            let da_even = track.dsp.v_atmo.get(2 * i).copied().unwrap_or(0.0)
                - self.target_atmo.get(2 * i).copied().unwrap_or(0.0);
            rho_x += db_even + da_even;

            let db_odd = track.dsp.v_beat.get(2 * i + 1).copied().unwrap_or(0.0)
                - self.target_beat.get(2 * i + 1).copied().unwrap_or(0.0);
            let dv_odd = track.dsp.v_vocal.get(2 * i + 1).copied().unwrap_or(0.0)
                - self.target_vocal.get(2 * i + 1).copied().unwrap_or(0.0);
            rho_y += db_odd - dv_odd;
        }
        rho_x *= 0.25;
        rho_y *= 0.25;

        let tx = ux + 0.22 * rho_x;
        let ty = uy + 0.22 * rho_y;
        let r_tilde = (tx * tx + ty * ty + 1e-4).sqrt();

        let weighted_sim =
            (w_beat * raw_beat + w_vocal * raw_vocal + w_atmo * raw_atmo).clamp(-0.85, 0.96);
        let r_sim = ((1.0 - weighted_sim).max(0.02) * 0.65)
            .sqrt()
            .clamp(0.10, 0.92);
        let r_final = (0.55 * r_sim + 0.45 * (1.35 * r_tilde).tanh()).clamp(0.08, 0.92);

        let x = (r_final * (tx / r_tilde)).clamp(-0.92, 0.92);
        let y = (r_final * (ty / r_tilde)).clamp(-0.92, 0.92);
        ((x * 1000.0).round() / 1000.0, (y * 1000.0).round() / 1000.0)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        let ranked = self.ranked_candidates();
        let vibe_corridors = self.build_vibe_corridors(&ranked);
        let current_card = ranked.first().cloned();
        let queue_preview: Vec<ScoredCandidate> = ranked.iter().skip(1).take(12).cloned().collect();

        let mut map_points = Vec::new();
        let (w_beat, w_vocal, w_atmo) = self.weights.normalized_triplet();

        let n_seeds = self.seeds.len().max(1);
        for s in &self.seeds {
            let (sx, sy) = if n_seeds == 1 {
                (0.0, 0.0)
            } else {
                let sb = cosine_similarity(&self.target_beat, &s.dsp.v_beat);
                let sv = cosine_similarity(&self.target_vocal, &s.dsp.v_vocal);
                let sa = cosine_similarity(&self.target_atmo, &s.dsp.v_atmo);
                self.project_to_2d(s, sb, sv, sa, w_beat, w_vocal, w_atmo)
            };
            map_points.push(SonicMapPoint {
                id: s.meta.id.clone(),
                title: s.meta.title.clone(),
                artist: s.meta.artist.clone(),
                cover_url: s.meta.cover_url.clone(),
                preview_url: s.meta.preview_url.clone(),
                drop_start_sec: s.dsp.drop_start_sec,
                bpm: s.dsp.bpm,
                total_score: 100.0,
                beat_similarity: 100.0,
                vocal_similarity: 100.0,
                atmo_similarity: 100.0,
                corridor_tag: "seed".to_string(),
                x: sx,
                y: sy,
                kind: "seed".to_string(),
            });
        }

        for l in &self.liked_crate {
            if self.seeds.iter().any(|s| s.meta.id == l.meta.id) {
                continue;
            }
            let sc = self.score_single_candidate(l, w_beat, w_vocal, w_atmo);
            map_points.push(SonicMapPoint {
                id: l.meta.id.clone(),
                title: l.meta.title.clone(),
                artist: l.meta.artist.clone(),
                cover_url: l.meta.cover_url.clone(),
                preview_url: l.meta.preview_url.clone(),
                drop_start_sec: l.dsp.drop_start_sec,
                bpm: l.dsp.bpm,
                total_score: sc.total_score,
                beat_similarity: sc.beat_similarity,
                vocal_similarity: sc.vocal_similarity,
                atmo_similarity: sc.atmo_similarity,
                corridor_tag: sc.corridor_tag,
                x: sc.map_x,
                y: sc.map_y,
                kind: "liked".to_string(),
            });
        }

        for c in ranked.iter().take(45) {
            map_points.push(SonicMapPoint {
                id: c.track.meta.id.clone(),
                title: c.track.meta.title.clone(),
                artist: c.track.meta.artist.clone(),
                cover_url: c.track.meta.cover_url.clone(),
                preview_url: c.track.meta.preview_url.clone(),
                drop_start_sec: c.track.dsp.drop_start_sec,
                bpm: c.track.dsp.bpm,
                total_score: c.total_score,
                beat_similarity: c.beat_similarity,
                vocal_similarity: c.vocal_similarity,
                atmo_similarity: c.atmo_similarity,
                corridor_tag: c.corridor_tag.clone(),
                x: c.map_x,
                y: c.map_y,
                kind: "candidate".to_string(),
            });
        }

        for n in &self.negative_anchors {
            let sc = self.score_single_candidate(n, w_beat, w_vocal, w_atmo);
            map_points.push(SonicMapPoint {
                id: n.meta.id.clone(),
                title: n.meta.title.clone(),
                artist: n.meta.artist.clone(),
                cover_url: n.meta.cover_url.clone(),
                preview_url: n.meta.preview_url.clone(),
                drop_start_sec: n.dsp.drop_start_sec,
                bpm: n.dsp.bpm,
                total_score: sc.total_score,
                beat_similarity: sc.beat_similarity,
                vocal_similarity: sc.vocal_similarity,
                atmo_similarity: sc.atmo_similarity,
                corridor_tag: sc.corridor_tag,
                x: sc.map_x,
                y: sc.map_y,
                kind: "negative".to_string(),
            });
        }

        SessionSnapshot {
            session_id: self.session_id.clone(),
            seeds: self.seeds.clone(),
            liked_crate: self.liked_crate.clone(),
            negative_anchors: self.negative_anchors.clone(),
            seen_count: self.seen_ids.len().saturating_sub(self.seeds.len()),
            pool_size: ranked.len(),
            weights: self.weights,
            active_corridor: self.active_corridor.clone(),
            vibe_corridors,
            current_card,
            queue_preview,
            map_points,
        }
    }
}

fn gram_schmidt_repel(target: &[f32], seed_anchor: &[f32], neg: &[f32]) -> Vec<f32> {
    let dim = target.len();
    if dim == 0 || seed_anchor.len() != dim || neg.len() != dim {
        return target.to_vec();
    }

    let proj_seed = cosine_similarity(neg, seed_anchor).max(0.0);
    let mut d = Vec::with_capacity(dim);
    let mut d_norm_sq = 0.0f32;
    for i in 0..dim {
        let val = neg[i] - proj_seed * seed_anchor[i];
        d_norm_sq += val * val;
        d.push(val);
    }

    let d_norm = d_norm_sq.sqrt();
    if d_norm < 1e-4 {
        return target.to_vec();
    }
    for x in &mut d {
        *x /= d_norm;
    }

    let sim_t_n = cosine_similarity(target, neg);
    let alpha = 0.28 * (1.0 - sim_t_n).clamp(0.15, 1.2);
    let proj_t_d = cosine_similarity(target, &d).max(0.0);

    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        out.push(target[i] - (proj_t_d + alpha) * d[i]);
    }
    l2_normalize_in_place(&mut out);
    out
}

fn diversify_artist_order(scored: &mut Vec<ScoredCandidate>) {
    let len = scored.len().min(16);
    if len < 3 {
        return;
    }
    for i in 1..len {
        let prev_artist = scored[i - 1].track.meta.artist.to_lowercase();
        let cur_artist = scored[i].track.meta.artist.to_lowercase();
        if prev_artist == cur_artist {
            if let Some(swap_idx) = ((i + 1)..scored.len())
                .find(|&j| scored[j].track.meta.artist.to_lowercase() != prev_artist)
            {
                scored.swap(i, swap_idx);
            }
        }
    }
}

fn session_track_jitter(session_id: &str, dedup_key: &str) -> f32 {
    let mut h = 0x811c9dc5u32;
    for b in session_id.bytes().chain(dedup_key.bytes()) {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    let norm = (h % 1000) as f32 / 1000.0;
    (norm - 0.5) * 2.2
}

fn compute_seed_centroid(seeds: &[AnalyzedTrack]) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, f32) {
    if seeds.is_empty() {
        return (vec![0.0; 16], vec![0.0; 16], vec![0.0; 16], 135.0, 0.50, 0.50);
    }
    let dim = 16usize;
    let mut tb = vec![0.0f32; dim];
    let mut tv = vec![0.0f32; dim];
    let mut ta = vec![0.0f32; dim];
    let mut bpm_sum = 0.0f32;
    let mut aggr_sum = 0.0f32;
    let mut dark_sum = 0.0f32;

    for s in seeds {
        for i in 0..dim {
            tb[i] += s.dsp.v_beat.get(i).copied().unwrap_or(0.0);
            tv[i] += s.dsp.v_vocal.get(i).copied().unwrap_or(0.0);
            ta[i] += s.dsp.v_atmo.get(i).copied().unwrap_or(0.0);
        }
        bpm_sum += s.dsp.bpm;
        aggr_sum += s.dsp.aggression_index;
        dark_sum += s.dsp.darkness_index;
    }

    l2_normalize_in_place(&mut tb);
    l2_normalize_in_place(&mut tv);
    l2_normalize_in_place(&mut ta);

    let n = seeds.len() as f32;
    (tb, tv, ta, bpm_sum / n, aggr_sum / n, dark_sum / n)
}

fn rocchio_step(target: &[f32], vec: &[f32], alpha: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(target.len());
    for i in 0..target.len() {
        let t = target[i];
        let v = vec.get(i).copied().unwrap_or(0.0);
        out.push(t + alpha * v);
    }
    l2_normalize_in_place(&mut out);
    out
}

fn l2_normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v {
        *x /= norm;
    }
}

/// Empirical Pool-Relative Z-Score Whitening:
/// Centers all 48 acoustic dimensions across the session's active seeds + candidate pool
/// to exact mean 0.0 and variance 1.0 before L2-normalizing, guaranteeing that no dimension
/// can ever saturate and that aggressive vs. calm / dark vs. bright tracks have opposite signs.
fn whiten_tracks(seeds: &mut [AnalyzedTrack], pool: &mut [AnalyzedTrack]) {
    let total = seeds.len() + pool.len();
    if total < 4 {
        for t in seeds.iter_mut().chain(pool.iter_mut()) {
            l2_normalize_in_place(&mut t.dsp.v_beat);
            l2_normalize_in_place(&mut t.dsp.v_vocal);
            l2_normalize_in_place(&mut t.dsp.v_atmo);
        }
        return;
    }

    let n = total as f32;
    let mut mean_b = [0.0f32; 16];
    let mut mean_v = [0.0f32; 16];
    let mut mean_a = [0.0f32; 16];
    let mut mean_aggr = 0.0f32;
    let mut mean_dark = 0.0f32;

    for t in seeds.iter().chain(pool.iter()) {
        for j in 0..16 {
            mean_b[j] += t.dsp.v_beat.get(j).copied().unwrap_or(0.0);
            mean_v[j] += t.dsp.v_vocal.get(j).copied().unwrap_or(0.0);
            mean_a[j] += t.dsp.v_atmo.get(j).copied().unwrap_or(0.0);
        }
        mean_aggr += t.dsp.aggression_index;
        mean_dark += t.dsp.darkness_index;
    }
    for j in 0..16 {
        mean_b[j] /= n;
        mean_v[j] /= n;
        mean_a[j] /= n;
    }
    mean_aggr /= n;
    mean_dark /= n;

    let mut std_b = [0.0f32; 16];
    let mut std_v = [0.0f32; 16];
    let mut std_a = [0.0f32; 16];
    let mut var_aggr = 0.0f32;
    let mut var_dark = 0.0f32;

    for t in seeds.iter().chain(pool.iter()) {
        for j in 0..16 {
            let db = t.dsp.v_beat.get(j).copied().unwrap_or(0.0) - mean_b[j];
            let dv = t.dsp.v_vocal.get(j).copied().unwrap_or(0.0) - mean_v[j];
            let da = t.dsp.v_atmo.get(j).copied().unwrap_or(0.0) - mean_a[j];
            std_b[j] += db * db;
            std_v[j] += dv * dv;
            std_a[j] += da * da;
        }
        let dag = t.dsp.aggression_index - mean_aggr;
        let ddk = t.dsp.darkness_index - mean_dark;
        var_aggr += dag * dag;
        var_dark += ddk * ddk;
    }
    for j in 0..16 {
        std_b[j] = (std_b[j] / n).sqrt().max(0.04);
        std_v[j] = (std_v[j] / n).sqrt().max(0.04);
        std_a[j] = (std_a[j] / n).sqrt().max(0.04);
    }
    let std_aggr = (var_aggr / n).sqrt().max(0.04);
    let std_dark = (var_dark / n).sqrt().max(0.04);

    for t in seeds.iter_mut().chain(pool.iter_mut()) {
        for j in 0..16 {
            if let Some(vb) = t.dsp.v_beat.get_mut(j) {
                *vb = ((*vb - mean_b[j]) / std_b[j]).clamp(-2.6, 2.6);
            }
            if let Some(vv) = t.dsp.v_vocal.get_mut(j) {
                *vv = ((*vv - mean_v[j]) / std_v[j]).clamp(-2.6, 2.6);
            }
            if let Some(va) = t.dsp.v_atmo.get_mut(j) {
                *va = ((*va - mean_a[j]) / std_a[j]).clamp(-2.6, 2.6);
            }
        }
        l2_normalize_in_place(&mut t.dsp.v_beat);
        l2_normalize_in_place(&mut t.dsp.v_vocal);
        l2_normalize_in_place(&mut t.dsp.v_atmo);

        let z_aggr = (t.dsp.aggression_index - mean_aggr) / std_aggr;
        let z_dark = (t.dsp.darkness_index - mean_dark) / std_dark;

        // Combine physical baseline with pool-calibrated z-score for sharp discrimination
        let cal_aggr = 0.25 * t.dsp.aggression_index + 0.75 * (0.50 + 0.45 * (0.85 * z_aggr).tanh());
        let cal_dark = 0.25 * t.dsp.darkness_index + 0.75 * (0.50 + 0.45 * (0.85 * z_dark).tanh());

        t.dsp.aggression_index = (cal_aggr.clamp(0.04, 0.96) * 100.0).round() / 100.0;
        t.dsp.darkness_index = (cal_dark.clamp(0.04, 0.96) * 100.0).round() / 100.0;
    }
}
