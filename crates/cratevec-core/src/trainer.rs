use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::models::{AnalyzedTrack, SwipeDirection};

/// Pre-trained and online-adaptable 48-D diagonal Mahalanobis metric model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedMetricModel {
    pub version: u32,
    /// 16 learned feature weights for v_beat (mean = 1.0)
    pub w_beat_dims: [f32; 16],
    /// 16 learned feature weights for v_vocal (mean = 1.0)
    pub w_vocal_dims: [f32; 16],
    /// 16 learned feature weights for v_atmo (mean = 1.0)
    pub w_atmo_dims: [f32; 16],
    pub total_training_steps: usize,
    pub online_swipe_updates: usize,
    pub baseline_triplet_acc: f32,
    pub trained_triplet_acc: f32,
}

impl Default for TrainedMetricModel {
    fn default() -> Self {
        // Pre-trained prior weights emphasizing octave-folded BPM phase, sub-bass pressure,
        // 808 harmonic saturation, lower vocal formants, delta-MFCC articulation,
        // minor/dark harmonic intervals, and high-frequency glitch spectral contrast.
        Self {
            version: 1,
            w_beat_dims: [
                1.22, 1.18, 1.18, 1.38, 1.12, 0.86, 1.34, 1.08, 1.04, 0.92, 0.84, 0.82, 0.94, 0.72,
                0.68, 0.68,
            ],
            w_vocal_dims: [
                1.32, 1.26, 1.14, 1.04, 0.92, 0.82, 0.74, 0.68, 1.24, 1.16, 0.96, 0.84, 1.28, 1.12,
                0.76, 0.72,
            ],
            w_atmo_dims: [
                1.36, 0.96, 1.24, 1.08, 0.82, 0.94, 1.14, 1.18, 0.74, 0.88, 1.12, 1.06, 0.78, 0.92,
                0.96, 0.82,
            ],
            total_training_steps: 0,
            online_swipe_updates: 0,
            baseline_triplet_acc: 0.0,
            trained_triplet_acc: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub num_tracks: usize,
    pub num_clusters: usize,
    pub num_triplets: usize,
    pub epochs: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub baseline_accuracy: f32,
    pub trained_accuracy: f32,
    pub top_beat_features: Vec<(String, f32)>,
    pub top_vocal_features: Vec<(String, f32)>,
    pub top_atmo_features: Vec<(String, f32)>,
}

pub const BEAT_DIM_NAMES: [&str; 16] = [
    "Octave-Folded BPM",
    "Log2-BPM Phase Sin",
    "Log2-BPM Phase Cos",
    "Sub-Bass Ratio (20-90Hz)",
    "Kick Punch Ratio (90-180Hz)",
    "Low-Mid Ratio (180-360Hz)",
    "808 Harmonic Saturation",
    "Low-Band Onset Flux",
    "High-Band Onset Flux",
    "Tempo Autocorrelation Strength",
    "Triplet Syncopation Ratio",
    "Double-Time Subdivision",
    "Sub-Bass Crest Factor",
    "Dynamic RMS Variance",
    "Sub-Octave Spectral Contrast",
    "Bass-Octave Spectral Contrast",
];

pub const VOCAL_DIM_NAMES: [&str; 16] = [
    "Vocal MFCC-1 (Formant Tilt)",
    "Vocal MFCC-2 (Open/Closed)",
    "Vocal MFCC-3 (Nasal/Chest)",
    "Vocal MFCC-4 (Throat/Presence)",
    "Vocal MFCC-5",
    "Vocal MFCC-6",
    "Vocal MFCC-7",
    "Vocal MFCC-8",
    "Delta-MFCC-1 (Articulation Speed)",
    "Delta-MFCC-2 (Flow Cadence)",
    "Delta-MFCC-3 (Syllable Chop)",
    "Delta-MFCC-4 (Timbre Flux)",
    "Vocal Core Presence (360-2.4kHz)",
    "Vocal Bite Ratio (2.4-4.8kHz)",
    "Vocal Envelope Modulation",
    "Mid-Range Spectral Contrast",
];

pub const ATMO_DIM_NAMES: [&str; 16] = [
    "Harmonic Darkness Ratio",
    "2nd Interval Tension",
    "Minor 3rd Interval Weight",
    "Major 3rd Interval Weight",
    "4th/5th Power Intervals",
    "Tritone Dissonance",
    "Minor 6th/7th Intervals",
    "Spectral Centroid Brightness",
    "Spectral Centroid Variance",
    "Spectral Flatness (Noise/Synth)",
    "High-Synth Ratio (4.8-9.5kHz)",
    "Glitch Air Ratio (9.5-16.5kHz)",
    "Low-Mid Spectral Contrast",
    "Upper-Mid Spectral Contrast",
    "High-Freq Glitch Contrast",
    "Chroma Harmonic Entropy",
];

impl TrainedMetricModel {
    pub fn load_or_default(custom_path: Option<PathBuf>) -> Self {
        let path = Self::resolve_path(custom_path);
        if let Ok(raw) = fs::read_to_string(&path) {
            if let Ok(mut model) = serde_json::from_str::<Self>(&raw) {
                for x in &mut model.w_beat_dims {
                    *x = x.clamp(0.65, 1.50);
                }
                for x in &mut model.w_vocal_dims {
                    *x = x.clamp(0.65, 1.50);
                }
                for x in &mut model.w_atmo_dims {
                    *x = x.clamp(0.65, 1.50);
                }
                normalize_weights_16(&mut model.w_beat_dims);
                normalize_weights_16(&mut model.w_vocal_dims);
                normalize_weights_16(&mut model.w_atmo_dims);
                return model;
            }
        }
        Self::default()
    }

    pub fn save(&self, custom_path: Option<PathBuf>) {
        let path = Self::resolve_path(custom_path);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, json);
        }
    }

    fn resolve_path(custom_path: Option<PathBuf>) -> PathBuf {
        custom_path.unwrap_or_else(|| {
            let mut p = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            p.push("data");
            let _ = fs::create_dir_all(&p);
            p.push("metric_model.json");
            p
        })
    }

    /// Diagonal Mahalanobis weighted cosine similarity for 16-D sub-vectors.
    pub fn sim_beat(&self, a: &[f32], b: &[f32]) -> f32 {
        weighted_cosine_similarity(a, b, &self.w_beat_dims)
    }

    pub fn sim_vocal(&self, a: &[f32], b: &[f32]) -> f32 {
        weighted_cosine_similarity(a, b, &self.w_vocal_dims)
    }

    pub fn sim_atmo(&self, a: &[f32], b: &[f32]) -> f32 {
        weighted_cosine_similarity(a, b, &self.w_atmo_dims)
    }

    /// Online continual learning step triggered on every user swipe.
    /// Adjusts the 48-D feature importance weights based on which dimensions agreed or disagreed
    /// between the session target vector and the swiped track.
    pub fn online_update_from_swipe(
        &mut self,
        target_beat: &[f32],
        target_vocal: &[f32],
        target_atmo: &[f32],
        track: &AnalyzedTrack,
        direction: SwipeDirection,
    ) {
        let (lr, is_positive) = match direction {
            SwipeDirection::Right => (0.032f32, true),
            SwipeDirection::Up => (0.058f32, true),
            SwipeDirection::Left => (0.020f32, false),
            SwipeDirection::Down => (0.052f32, false),
        };

        online_update_subvector(
            &mut self.w_beat_dims,
            target_beat,
            &track.dsp.v_beat,
            lr,
            is_positive,
        );
        online_update_subvector(
            &mut self.w_vocal_dims,
            target_vocal,
            &track.dsp.v_vocal,
            lr,
            is_positive,
        );
        online_update_subvector(
            &mut self.w_atmo_dims,
            target_atmo,
            &track.dsp.v_atmo,
            lr,
            is_positive,
        );

        self.online_swipe_updates += 1;
        self.save(None);
    }

    /// Offline Contrastive Metric Learning via Supervised Triplet Margin Gradient Descent
    /// over ground-truth musical scene clusters.
    pub fn train_on_labeled_clusters(
        &mut self,
        clusters: &[Vec<AnalyzedTrack>],
        epochs: usize,
    ) -> TrainingReport {
        let mut tracks: Vec<AnalyzedTrack> = Vec::new();
        let mut assignments: Vec<usize> = Vec::new();

        for (c_idx, cluster) in clusters.iter().enumerate() {
            for t in cluster {
                tracks.push(t.clone());
                assignments.push(c_idx);
            }
        }

        if tracks.len() < 12 {
            return TrainingReport {
                num_tracks: tracks.len(),
                num_clusters: clusters.len(),
                num_triplets: 0,
                epochs: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
                baseline_accuracy: 0.0,
                trained_accuracy: 0.0,
                top_beat_features: Vec::new(),
                top_vocal_features: Vec::new(),
                top_atmo_features: Vec::new(),
            };
        }

        self.optimize_triplets(&tracks, &assignments, clusters.len(), epochs)
    }

    pub fn train_on_corpus(&mut self, tracks: &[AnalyzedTrack], epochs: usize) -> TrainingReport {
        if tracks.len() < 12 {
            return TrainingReport {
                num_tracks: tracks.len(),
                num_clusters: 0,
                num_triplets: 0,
                epochs: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
                baseline_accuracy: 0.0,
                trained_accuracy: 0.0,
                top_beat_features: Vec::new(),
                top_vocal_features: Vec::new(),
                top_atmo_features: Vec::new(),
            };
        }

        let k_clusters = 6.min(tracks.len() / 4).max(2);
        let assignments = assign_acoustic_clusters(tracks, k_clusters);
        self.optimize_triplets(tracks, &assignments, k_clusters, epochs)
    }

    fn optimize_triplets(
        &mut self,
        tracks: &[AnalyzedTrack],
        assignments: &[usize],
        num_clusters: usize,
        epochs: usize,
    ) -> TrainingReport {
        let mut triplets: Vec<(usize, usize, usize)> = Vec::new();
        let n = tracks.len();
        for a in 0..n {
            let ca = assignments[a];
            let mut pos_candidates: Vec<(f32, usize)> = Vec::new();
            let mut neg_candidates: Vec<(f32, usize)> = Vec::new();

            for b in 0..n {
                if a == b {
                    continue;
                }
                let raw_sim = unweighted_48d_sim(&tracks[a], &tracks[b]);
                if assignments[b] == ca {
                    pos_candidates.push((raw_sim, b));
                } else {
                    neg_candidates.push((raw_sim, b));
                }
            }

            pos_candidates.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
            neg_candidates.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

            // Pair the core intra-scene positives against inter-scene negatives across the spectrum
            let neg_step = (neg_candidates.len() / 6).max(1);
            for &(_, p_idx) in pos_candidates.iter().take(5) {
                for (_, n_idx) in neg_candidates.iter().step_by(neg_step).take(6) {
                    triplets.push((a, p_idx, *n_idx));
                }
            }
        }

        let margin = 0.16f32;
        let l2_reg = 0.001f32;

        // Start from uniform baseline [1.0; 16] to measure true learning gain on this corpus
        let uniform = [1.0f32; 16];
        let (initial_loss, baseline_acc) =
            evaluate_triplets(tracks, &triplets, &uniform, &uniform, &uniform, margin);

        self.w_beat_dims = uniform;
        self.w_vocal_dims = uniform;
        self.w_atmo_dims = uniform;

        for ep in 0..epochs {
            let lr = 0.08 * (1.0 - (ep as f32 / epochs.max(1) as f32) * 0.65);
            let mut grad_b = [0.0f32; 16];
            let mut grad_v = [0.0f32; 16];
            let mut grad_a = [0.0f32; 16];
            let mut active_count = 0.0f32;

            for &(a_idx, p_idx, n_idx) in &triplets {
                let ta = &tracks[a_idx].dsp;
                let tp = &tracks[p_idx].dsp;
                let tn = &tracks[n_idx].dsp;

                let sim_p = 0.38 * self.sim_beat(&ta.v_beat, &tp.v_beat)
                    + 0.31 * self.sim_vocal(&ta.v_vocal, &tp.v_vocal)
                    + 0.31 * self.sim_atmo(&ta.v_atmo, &tp.v_atmo);

                let sim_n = 0.38 * self.sim_beat(&ta.v_beat, &tn.v_beat)
                    + 0.31 * self.sim_vocal(&ta.v_vocal, &tn.v_vocal)
                    + 0.31 * self.sim_atmo(&ta.v_atmo, &tn.v_atmo);

                if margin + sim_n - sim_p > 0.0 {
                    active_count += 1.0;
                    accumulate_subvector_triplet_grad(
                        &mut grad_b,
                        &self.w_beat_dims,
                        &ta.v_beat,
                        &tp.v_beat,
                        &tn.v_beat,
                        0.38,
                    );
                    accumulate_subvector_triplet_grad(
                        &mut grad_v,
                        &self.w_vocal_dims,
                        &ta.v_vocal,
                        &tp.v_vocal,
                        &tn.v_vocal,
                        0.31,
                    );
                    accumulate_subvector_triplet_grad(
                        &mut grad_a,
                        &self.w_atmo_dims,
                        &ta.v_atmo,
                        &tp.v_atmo,
                        &tn.v_atmo,
                        0.31,
                    );
                }
            }

            if active_count > 0.0 {
                // RMS-normalize gradient per subvector so step size is invariant to feature scale
                let rms_b = (grad_b.iter().map(|g| g * g).sum::<f32>() / 16.0)
                    .sqrt()
                    .max(1e-6);
                let rms_v = (grad_v.iter().map(|g| g * g).sum::<f32>() / 16.0)
                    .sqrt()
                    .max(1e-6);
                let rms_a = (grad_a.iter().map(|g| g * g).sum::<f32>() / 16.0)
                    .sqrt()
                    .max(1e-6);

                for i in 0..16 {
                    let gb = (grad_b[i] / rms_b) + l2_reg * (self.w_beat_dims[i] - 1.0);
                    let gv = (grad_v[i] / rms_v) + l2_reg * (self.w_vocal_dims[i] - 1.0);
                    let ga = (grad_a[i] / rms_a) + l2_reg * (self.w_atmo_dims[i] - 1.0);

                    self.w_beat_dims[i] = (self.w_beat_dims[i] - lr * gb).clamp(0.65, 1.50);
                    self.w_vocal_dims[i] = (self.w_vocal_dims[i] - lr * gv).clamp(0.65, 1.50);
                    self.w_atmo_dims[i] = (self.w_atmo_dims[i] - lr * ga).clamp(0.65, 1.50);
                }
                normalize_weights_16(&mut self.w_beat_dims);
                normalize_weights_16(&mut self.w_vocal_dims);
                normalize_weights_16(&mut self.w_atmo_dims);
            }
        }

        let (final_loss, trained_acc) = evaluate_triplets(
            tracks,
            &triplets,
            &self.w_beat_dims,
            &self.w_vocal_dims,
            &self.w_atmo_dims,
            margin,
        );

        self.total_training_steps += epochs * triplets.len();
        self.baseline_triplet_acc = baseline_acc;
        self.trained_triplet_acc = trained_acc;
        self.save(None);

        TrainingReport {
            num_tracks: tracks.len(),
            num_clusters,
            num_triplets: triplets.len(),
            epochs,
            initial_loss,
            final_loss,
            baseline_accuracy: baseline_acc,
            trained_accuracy: trained_acc,
            top_beat_features: top_named_weights(&self.w_beat_dims, &BEAT_DIM_NAMES, 5),
            top_vocal_features: top_named_weights(&self.w_vocal_dims, &VOCAL_DIM_NAMES, 5),
            top_atmo_features: top_named_weights(&self.w_atmo_dims, &ATMO_DIM_NAMES, 5),
        }
    }
}

pub fn weighted_cosine_similarity(a: &[f32], b: &[f32], w: &[f32; 16]) -> f32 {
    if a.len() < 16 || b.len() < 16 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..16 {
        let wi = w[i].max(0.05);
        dot += wi * a[i] * b[i];
        na += wi * a[i] * a[i];
        nb += wi * b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-6);
    (dot / denom).clamp(-1.0, 1.0)
}

fn online_update_subvector(
    w: &mut [f32; 16],
    target: &[f32],
    cand: &[f32],
    lr: f32,
    is_positive: bool,
) {
    if target.len() < 16 || cand.len() < 16 {
        return;
    }
    let mut prod = [0.0f32; 16];
    let mut mean_prod = 0.0f32;
    for i in 0..16 {
        prod[i] = target[i] * cand[i];
        mean_prod += prod[i];
    }
    mean_prod /= 16.0;

    for i in 0..16 {
        let centered = prod[i] - mean_prod;
        let delta = if is_positive {
            lr * centered
        } else {
            -lr * centered
        };
        w[i] = (w[i] + delta).clamp(0.30, 2.80);
    }
    normalize_weights_16(w);
}

fn accumulate_subvector_triplet_grad(
    grad: &mut [f32; 16],
    w: &[f32; 16],
    a: &[f32],
    p: &[f32],
    n: &[f32],
    channel_weight: f32,
) {
    if a.len() < 16 || p.len() < 16 || n.len() < 16 {
        return;
    }
    let sim_p = weighted_cosine_similarity(a, p, w);
    let sim_n = weighted_cosine_similarity(a, n, w);

    let mut na = 0.0f32;
    let mut np = 0.0f32;
    let mut nn = 0.0f32;
    for i in 0..16 {
        let wi = w[i].max(0.05);
        na += wi * a[i] * a[i];
        np += wi * p[i] * p[i];
        nn += wi * n[i] * n[i];
    }
    na = na.max(1e-5);
    np = np.max(1e-5);
    nn = nn.max(1e-5);
    let denom_p = (na * np).sqrt();
    let denom_n = (na * nn).sqrt();

    for i in 0..16 {
        // Exact analytical derivative d(sim)/d(w_i) for weighted cosine similarity
        let d_sim_p = (a[i] * p[i]) / denom_p - 0.5 * sim_p * (a[i] * a[i] / na + p[i] * p[i] / np);
        let d_sim_n = (a[i] * n[i]) / denom_n - 0.5 * sim_n * (a[i] * a[i] / na + n[i] * n[i] / nn);
        // Loss = margin + sim_n - sim_p => dLoss/dw_i = d_sim_n - d_sim_p
        grad[i] += channel_weight * (d_sim_n - d_sim_p);
    }
}

fn evaluate_triplets(
    tracks: &[AnalyzedTrack],
    triplets: &[(usize, usize, usize)],
    wb: &[f32; 16],
    wv: &[f32; 16],
    wa: &[f32; 16],
    margin: f32,
) -> (f32, f32) {
    if triplets.is_empty() {
        return (0.0, 100.0);
    }
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for &(a_idx, p_idx, n_idx) in triplets {
        let ta = &tracks[a_idx].dsp;
        let tp = &tracks[p_idx].dsp;
        let tn = &tracks[n_idx].dsp;

        let sim_p = 0.38 * weighted_cosine_similarity(&ta.v_beat, &tp.v_beat, wb)
            + 0.31 * weighted_cosine_similarity(&ta.v_vocal, &tp.v_vocal, wv)
            + 0.31 * weighted_cosine_similarity(&ta.v_atmo, &tp.v_atmo, wa);

        let sim_n = 0.38 * weighted_cosine_similarity(&ta.v_beat, &tn.v_beat, wb)
            + 0.31 * weighted_cosine_similarity(&ta.v_vocal, &tn.v_vocal, wv)
            + 0.31 * weighted_cosine_similarity(&ta.v_atmo, &tn.v_atmo, wa);

        let loss = (margin + sim_n - sim_p).max(0.0);
        total_loss += loss;
        if sim_p > sim_n {
            correct += 1;
        }
    }

    let avg_loss = total_loss / triplets.len() as f32;
    let acc = (correct as f32 / triplets.len() as f32) * 100.0;
    (avg_loss, acc)
}

fn assign_acoustic_clusters(tracks: &[AnalyzedTrack], k: usize) -> Vec<usize> {
    let n = tracks.len();
    // Initialize centroids via furthest-point k-means++ seeding
    let mut centroids: Vec<usize> = vec![0];
    while centroids.len() < k {
        let mut best_idx = 0;
        let mut best_min_dist = -1.0f32;
        for i in 0..n {
            let min_d = centroids
                .iter()
                .map(|&c| 1.0 - unweighted_48d_sim(&tracks[i], &tracks[c]))
                .fold(f32::INFINITY, f32::min);
            if min_d > best_min_dist {
                best_min_dist = min_d;
                best_idx = i;
            }
        }
        centroids.push(best_idx);
    }

    let mut assignments = vec![0usize; n];
    for i in 0..n {
        let mut best_c = 0usize;
        let mut best_sim = f32::MIN;
        for (c_idx, &medoid_idx) in centroids.iter().enumerate() {
            let sim = unweighted_48d_sim(&tracks[i], &tracks[medoid_idx]);
            if sim > best_sim {
                best_sim = sim;
                best_c = c_idx;
            }
        }
        assignments[i] = best_c;
    }
    assignments
}

fn unweighted_48d_sim(a: &AnalyzedTrack, b: &AnalyzedTrack) -> f32 {
    let u = [1.0f32; 16];
    0.38 * weighted_cosine_similarity(&a.dsp.v_beat, &b.dsp.v_beat, &u)
        + 0.31 * weighted_cosine_similarity(&a.dsp.v_vocal, &b.dsp.v_vocal, &u)
        + 0.31 * weighted_cosine_similarity(&a.dsp.v_atmo, &b.dsp.v_atmo, &u)
}

fn normalize_weights_16(w: &mut [f32; 16]) {
    let sum: f32 = w.iter().sum::<f32>().max(1e-5);
    let scale = 16.0 / sum;
    for x in w.iter_mut() {
        *x *= scale;
    }
}

fn top_named_weights(w: &[f32; 16], names: &[&str; 16], top_k: usize) -> Vec<(String, f32)> {
    let mut pairs: Vec<(String, f32)> = (0..16).map(|i| (names[i].to_string(), w[i])).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().take(top_k).collect()
}
