use std::collections::HashSet;

use crate::dsp::{cosine_similarity, cosine_to_percentage};
use crate::models::{
    AnalyzedTrack, MixerWeights, ScoredCandidate, SessionSnapshot, SonicMapPoint, SwipeDirection,
};

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
    pub weights: MixerWeights,
    pub target_beat: Vec<f32>,
    pub target_vocal: Vec<f32>,
    pub target_atmo: Vec<f32>,
    pub target_bpm: f32,
}

impl DiscoverySession {
    pub fn new(
        session_id: String,
        seeds: Vec<AnalyzedTrack>,
        pool: Vec<AnalyzedTrack>,
        prev_session_dedups: HashSet<String>,
    ) -> Self {
        let mut seen_ids = HashSet::new();
        let mut seen_dedups = HashSet::new();
        for s in &seeds {
            seen_ids.insert(s.meta.id.clone());
            seen_dedups.insert(s.meta.dedup_key.clone());
        }

        let (target_beat, target_vocal, target_atmo, target_bpm) = compute_seed_centroid(&seeds);

        let mut session = Self {
            session_id,
            seeds,
            pool: Vec::new(),
            liked_crate: Vec::new(),
            negative_anchors: Vec::new(),
            seen_ids,
            seen_dedups,
            prev_session_dedups,
            pinned_track_id: None,
            weights: MixerWeights::default(),
            target_beat,
            target_vocal,
            target_atmo,
            target_bpm,
        };
        session.extend_pool(pool);
        session
    }

    pub fn extend_pool(&mut self, tracks: Vec<AnalyzedTrack>) {
        let existing_dedups: HashSet<String> =
            self.pool.iter().map(|t| t.meta.dedup_key.clone()).collect();

        for t in tracks {
            if !self.seen_dedups.contains(&t.meta.dedup_key)
                && !existing_dedups.contains(&t.meta.dedup_key)
            {
                self.pool.push(t);
            }
        }
    }

    /// Add an additional seed track (Bridge Mode) and recalculate the target vector centroid.
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
        let (tb, tv, ta, bpm) = compute_seed_centroid(&self.seeds);
        self.target_beat = tb;
        self.target_vocal = tv;
        self.target_atmo = ta;
        self.target_bpm = bpm;
        self.pinned_track_id = None;
        self.extend_pool(new_candidates);
    }

    pub fn update_weights(&mut self, weights: MixerWeights) {
        self.weights = weights;
    }

    /// Directly focus/select a specific candidate from the queue or 2D Sonic Map as the active card.
    pub fn focus_track(&mut self, track_id: &str) {
        if self.pool.iter().any(|t| t.meta.id == track_id) {
            self.pinned_track_id = Some(track_id.to_string());
        }
    }

    /// Apply a 4-way swipe action on a track and update the live Rocchio session vectors.
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

        match direction {
            SwipeDirection::Right => {
                self.target_beat = rocchio_step(&self.target_beat, &track.dsp.v_beat, 0.26);
                self.target_vocal = rocchio_step(&self.target_vocal, &track.dsp.v_vocal, 0.26);
                self.target_atmo = rocchio_step(&self.target_atmo, &track.dsp.v_atmo, 0.26);
                self.target_bpm = self.target_bpm * 0.8 + track.dsp.bpm * 0.2;
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
                self.target_beat = rocchio_step(&self.target_beat, &track.dsp.v_beat, -0.24);
                self.target_vocal = rocchio_step(&self.target_vocal, &track.dsp.v_vocal, -0.24);
                self.target_atmo = rocchio_step(&self.target_atmo, &track.dsp.v_atmo, -0.24);
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

        scored.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Artist diversity pass: avoid back-to-back tracks by the exact same artist in the top queue
        diversify_artist_order(&mut scored);

        // If the user explicitly clicked a track in the queue or 2D map, place it at index 0
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

    fn score_single_candidate(
        &self,
        track: &AnalyzedTrack,
        w_beat: f32,
        w_vocal: f32,
        w_atmo: f32,
    ) -> ScoredCandidate {
        let raw_beat = cosine_similarity(&self.target_beat, &track.dsp.v_beat);
        let raw_vocal = cosine_similarity(&self.target_vocal, &track.dsp.v_vocal);
        let raw_atmo = cosine_similarity(&self.target_atmo, &track.dsp.v_atmo);

        let beat_pct = cosine_to_percentage(raw_beat);
        let vocal_pct = cosine_to_percentage(raw_vocal);
        let atmo_pct = cosine_to_percentage(raw_atmo);

        let mut total = w_beat * beat_pct + w_vocal * vocal_pct + w_atmo * atmo_pct;

        // Tempo alignment (supports direct match and half/double time)
        let bpm_diff = (track.dsp.bpm - self.target_bpm)
            .abs()
            .min((track.dsp.bpm * 2.0 - self.target_bpm).abs())
            .min((track.dsp.bpm - self.target_bpm * 2.0).abs());
        if bpm_diff <= 6.0 {
            total += 2.2;
        } else if bpm_diff > 28.0 {
            total -= ((bpm_diff - 28.0) * 0.15).min(5.0);
        }

        // Underground slider modifier (-1.0 mainstream to +1.0 deep underground)
        // Strongly penalizes overplayed chart tracks when underground > 0
        let obscurity = 1.0 - track.meta.popularity_score;
        let ug_mod = self.weights.underground * (obscurity - 0.45) * 16.0;
        total += ug_mod;

        // Cross-session repeat penalty: if a track was already seen in an earlier session, push it down
        if self.prev_session_dedups.contains(&track.meta.dedup_key) {
            total -= 18.0;
        }

        // Session-specific exploration jitter (-1.4 .. +1.4) so near-ties vary organically between runs
        let jitter = session_track_jitter(&self.session_id, &track.meta.dedup_key);
        total += jitter;

        // Negative Anchor Penalty ("Kill Vibe" filter)
        let mut neg_penalty = 0.0f32;
        for neg in &self.negative_anchors {
            let n_beat = cosine_similarity(&neg.dsp.v_beat, &track.dsp.v_beat);
            let n_vocal = cosine_similarity(&neg.dsp.v_vocal, &track.dsp.v_vocal);
            let n_atmo = cosine_similarity(&neg.dsp.v_atmo, &track.dsp.v_atmo);
            let n_comb = (n_beat + n_vocal + n_atmo) / 3.0;
            if n_comb > 0.35 {
                let p = ((n_comb - 0.35) * 38.0).clamp(0.0, 35.0);
                if p > neg_penalty {
                    neg_penalty = p;
                }
            }
        }
        total -= neg_penalty;

        let total_score = (total.clamp(5.0, 99.4) * 10.0).round() / 10.0;

        let mut explanation = Vec::with_capacity(5);
        explanation.push(format!(
            "{:.0} BPM (Drop @ {:.1}s)",
            track.dsp.bpm, track.dsp.drop_start_sec
        ));
        explanation.push(format!(
            "Beat/808: {:.0}% | Vocals: {:.0}% | Atmo: {:.0}%",
            beat_pct, vocal_pct, atmo_pct
        ));
        if track.dsp.bass_distortion >= 0.45 {
            explanation.push(format!(
                "Hohe 808-Saettigung ({:.0}%)",
                track.dsp.bass_distortion * 100.0
            ));
        }
        if obscurity >= 0.65 {
            explanation.push("Underground-Release".to_string());
        }
        explanation.push(format!("Quelle: {}", track.meta.discovery_channel));

        let (map_x, map_y) = self.project_to_2d(track, raw_beat, raw_vocal, raw_atmo);

        ScoredCandidate {
            track: track.clone(),
            total_score,
            beat_similarity: beat_pct,
            vocal_similarity: vocal_pct,
            atmo_similarity: atmo_pct,
            negative_penalty: (neg_penalty * 10.0).round() / 10.0,
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
    ) -> (f32, f32) {
        let seed_dsp = &self.seeds[0].dsp;
        let dx_drive = ((track.dsp.bpm - seed_dsp.bpm) / 35.0) * 0.45
            + (track.dsp.glitch_brightness - seed_dsp.glitch_brightness) * 1.4
            + (track.dsp.v_beat.first().copied().unwrap_or(0.0)
                - self.target_beat.first().copied().unwrap_or(0.0))
                * 0.6;

        let dy_depth = (track.dsp.darkness_index - seed_dsp.darkness_index) * 1.3
            + (track.dsp.sub_bass_ratio - seed_dsp.sub_bass_ratio) * 1.8
            - (track.dsp.vocal_presence - seed_dsp.vocal_presence) * 1.1
            + (raw_atmo - raw_vocal) * 0.45;

        let avg_sim = ((raw_beat + raw_vocal + raw_atmo) / 3.0).clamp(-0.8, 0.95);
        let target_radius = (1.0 - ((avg_sim + 0.8) / 1.75)).clamp(0.14, 0.90);

        let angle = dy_depth.atan2(dx_drive);
        let x = (target_radius * angle.cos()).clamp(-0.92, 0.92);
        let y = (target_radius * angle.sin()).clamp(-0.92, 0.92);
        ((x * 1000.0).round() / 1000.0, (y * 1000.0).round() / 1000.0)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        let ranked = self.ranked_candidates();
        let current_card = ranked.first().cloned();
        let queue_preview: Vec<ScoredCandidate> = ranked.iter().skip(1).take(12).cloned().collect();

        let mut map_points = Vec::new();

        let n_seeds = self.seeds.len().max(1);
        for (idx, s) in self.seeds.iter().enumerate() {
            let (sx, sy) = if n_seeds == 1 {
                (0.0, 0.0)
            } else {
                let a = (idx as f32 / n_seeds as f32) * std::f32::consts::TAU;
                (0.08 * a.cos(), 0.08 * a.sin())
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
                x: sx,
                y: sy,
                kind: "seed".to_string(),
            });
        }

        let (w_beat, w_vocal, w_atmo) = self.weights.normalized_triplet();
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
            current_card,
            queue_preview,
            map_points,
        }
    }
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
            // Find the next track by a different artist and swap
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
    let norm = (h % 1000) as f32 / 1000.0; // 0.0 .. 1.0
    (norm - 0.5) * 2.8 // -1.4 .. +1.4
}

fn compute_seed_centroid(seeds: &[AnalyzedTrack]) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    if seeds.is_empty() {
        return (vec![0.0; 16], vec![0.0; 16], vec![0.0; 16], 135.0);
    }
    let dim = 16usize;
    let mut tb = vec![0.0f32; dim];
    let mut tv = vec![0.0f32; dim];
    let mut ta = vec![0.0f32; dim];
    let mut bpm_sum = 0.0f32;

    for s in seeds {
        for i in 0..dim {
            tb[i] += s.dsp.v_beat.get(i).copied().unwrap_or(0.0);
            tv[i] += s.dsp.v_vocal.get(i).copied().unwrap_or(0.0);
            ta[i] += s.dsp.v_atmo.get(i).copied().unwrap_or(0.0);
        }
        bpm_sum += s.dsp.bpm;
    }

    l2_normalize_in_place(&mut tb);
    l2_normalize_in_place(&mut tv);
    l2_normalize_in_place(&mut ta);

    (tb, tv, ta, bpm_sum / seeds.len() as f32)
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
