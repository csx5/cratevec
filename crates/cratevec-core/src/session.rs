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
                // Gram-Schmidt Orthogonal Subspace Elimination:
                // Only reject the component of `track` that deviated from the positive seed centroid,
                // preserving shared desirable features (such as matching BPM or bass foundation).
                let (sb, sv, sa, _) = compute_seed_centroid(&self.seeds);
                self.target_beat =
                    gram_schmidt_repel(&self.target_beat, &sb, &track.dsp.v_beat);
                self.target_vocal =
                    gram_schmidt_repel(&self.target_vocal, &sv, &track.dsp.v_vocal);
                self.target_atmo =
                    gram_schmidt_repel(&self.target_atmo, &sa, &track.dsp.v_atmo);
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

        // 1. Hybrid Linear + Harmonic Mean with Bottleneck Penalty against single-channel failures
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

        // 2. Continuous Multi-Octave Gaussian Tempo Kernel coupled to Beat weight w_beat
        let bpm_diff = (track.dsp.bpm - self.target_bpm)
            .abs()
            .min((track.dsp.bpm * 2.0 - self.target_bpm).abs())
            .min((track.dsp.bpm - self.target_bpm * 2.0).abs());
        let gauss_bpm = (-0.5 * (bpm_diff / 9.5).powi(2)).exp();
        let bpm_bonus = (1.6 + 4.2 * w_beat) * (2.0 * gauss_bpm - 1.0);
        total += bpm_bonus;

        // 3. Bridge-Mode Symmetry Bonus (rewards true bridge tracks uniting multiple seeds over 1-seed clones)
        if self.seeds.len() >= 2 {
            let mut seed_sims = Vec::with_capacity(self.seeds.len());
            for s in &self.seeds {
                let sb = cosine_similarity(&s.dsp.v_beat, &track.dsp.v_beat);
                let sv = cosine_similarity(&s.dsp.v_vocal, &track.dsp.v_vocal);
                let sa = cosine_similarity(&s.dsp.v_atmo, &track.dsp.v_atmo);
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

        // 4. Underground slider modifier (-1.0 mainstream to +1.0 deep underground)
        let obscurity = 1.0 - track.meta.popularity_score;
        let ug_mod = self.weights.underground * (obscurity - 0.45) * 16.0;
        total += ug_mod;

        // 5. Cross-session repeat penalty
        if self.prev_session_dedups.contains(&track.meta.dedup_key) {
            total -= 18.0;
        }

        // 6. Session exploration jitter (-1.2 .. +1.2)
        total += session_track_jitter(&self.session_id, &track.meta.dedup_key);

        // 7. Mixer-weighted Negative Anchor Penalty ("Kill Vibe" filter)
        let mut neg_penalty = 0.0f32;
        for neg in &self.negative_anchors {
            let n_beat = cosine_similarity(&neg.dsp.v_beat, &track.dsp.v_beat);
            let n_vocal = cosine_similarity(&neg.dsp.v_vocal, &track.dsp.v_vocal);
            let n_atmo = cosine_similarity(&neg.dsp.v_atmo, &track.dsp.v_atmo);
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

        let mut explanation = Vec::with_capacity(5);
        explanation.push(format!(
            "{:.1} BPM (Drop @ {:.1}s)",
            track.dsp.bpm, track.dsp.drop_start_sec
        ));
        explanation.push(format!(
            "Beat/808: {:.0}% | Vocals: {:.0}% | Atmo: {:.0}%",
            beat_pct, vocal_pct, atmo_pct
        ));
        if track.dsp.bass_distortion >= 0.42 {
            explanation.push(format!(
                "Hohe 808-Saettigung ({:.0}%)",
                track.dsp.bass_distortion * 100.0
            ));
        }
        if obscurity >= 0.65 {
            explanation.push("Underground-Release".to_string());
        }
        explanation.push(format!("Quelle: {}", track.meta.discovery_channel));

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
            explanation,
            map_x,
            map_y,
        }
    }

    /// Singularity-free 48-D hyperplane projection onto the 2D Sonic Map radar.
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

        // X-axis: Tempo + High-Frequency Glitch Drive vs. Warm/Slow
        let ux = ((track.dsp.bpm - self.target_bpm) / 28.0) * 0.35
            + (track.dsp.glitch_brightness - mean_glitch) * 1.15
            + (track.dsp.v_beat.get(7).copied().unwrap_or(0.0)
                + track.dsp.v_beat.get(8).copied().unwrap_or(0.0)
                - self.target_beat.get(7).copied().unwrap_or(0.0)
                - self.target_beat.get(8).copied().unwrap_or(0.0))
                * 0.35;

        // Y-axis: Sub-Bass + Darkness + 808 Saturation vs. Vocal & Mid Presence
        let uy = (track.dsp.darkness_index - mean_dark) * 1.10
            + (track.dsp.sub_bass_ratio - mean_sub) * 1.35
            + (track.dsp.bass_distortion - mean_dist) * 0.45
            - (track.dsp.vocal_presence - mean_vocal) * 1.15
            + (raw_atmo - raw_vocal) * 0.30;

        // Orthogonal 48-D residual components to eliminate atan2 direction singularity when ux ~ uy ~ 0
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

        let weighted_sim = (w_beat * raw_beat + w_vocal * raw_vocal + w_atmo * raw_atmo).clamp(-0.85, 0.96);
        let r_sim = ((1.0 - weighted_sim).max(0.02) * 0.65).sqrt().clamp(0.10, 0.92);
        let r_final = (0.55 * r_sim + 0.45 * (1.35 * r_tilde).tanh()).clamp(0.08, 0.92);

        let x = (r_final * (tx / r_tilde)).clamp(-0.92, 0.92);
        let y = (r_final * (ty / r_tilde)).clamp(-0.92, 0.92);
        ((x * 1000.0).round() / 1000.0, (y * 1000.0).round() / 1000.0)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        let ranked = self.ranked_candidates();
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

/// Gram-Schmidt orthogonal subspace elimination for Negative Anchors ("Kill Vibe"):
/// Computes the component `d` of `neg` that is orthogonal to the positive seed anchor `seed_anchor`,
/// and pushes `target` out of `d`'s half-space without eroding shared positive features.
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
    (norm - 0.5) * 2.4
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
