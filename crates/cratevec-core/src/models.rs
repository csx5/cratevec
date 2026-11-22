use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    pub id: String,
    pub dedup_key: String,
    pub title: String,
    pub artist: String,
    pub artist_id: Option<String>,
    pub album: String,
    pub cover_url: String,
    pub preview_url: String,
    pub external_url: String,
    pub spotify_search_url: String,
    pub source: String,
    pub duration_sec: u32,
    pub popularity_rank: u32,
    /// Normalized 0.0 (deep underground) to 1.0 (global mainstream chart hit)
    pub popularity_score: f32,
    pub discovery_channel: String,
    /// Multi-source scene graph consensus score (0.0 .. 1.0): measures how strongly
    /// this candidate is anchored in the seed track's real musical neighborhood.
    #[serde(default = "default_scene_affinity")]
    pub scene_affinity: f32,
}

fn default_scene_affinity() -> f32 {
    0.75
}

fn default_aggression() -> f32 {
    0.50
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticFingerprint {
    pub bpm: f32,
    pub drop_start_sec: f32,
    pub rms_db: f32,
    pub sub_bass_ratio: f32,
    pub bass_distortion: f32,
    pub vocal_presence: f32,
    pub glitch_brightness: f32,
    pub darkness_index: f32,
    /// 0.0 (calm, chill, laid-back, soft transients) .. 1.0 (aggressive, hard-hitting, high vocal bite & attack)
    #[serde(default = "default_aggression")]
    pub aggression_index: f32,
    /// 64 normalized amplitude peaks (0.0..1.0) across the 30s preview for instant waveform UI rendering
    pub waveform_peaks: Vec<f32>,
    /// 16-dim L2-normalized vector: Rhythm, BPM, sub-bass pressure, 808 harmonic distortion, transient punch
    pub v_beat: Vec<f32>,
    /// 16-dim L2-normalized vector: Vocal-band Mel filterbank cepstrum (300-4500 Hz), Delta-MFCCs, formants
    pub v_vocal: Vec<f32>,
    /// 16-dim L2-normalized vector: Harmonic chroma, spectral contrast, synth flatness, high-freq glitch texture
    pub v_atmo: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzedTrack {
    pub meta: TrackMetadata,
    pub dsp: AcousticFingerprint,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MixerWeights {
    pub beat: f32,
    pub vocal: f32,
    pub atmo: f32,
    /// -1.0 (Mainstream bias) .. 0.0 (Neutral) .. +1.0 (Deep Underground bias)
    pub underground: f32,
}

impl Default for MixerWeights {
    fn default() -> Self {
        Self {
            beat: 0.38,
            vocal: 0.31,
            atmo: 0.31,
            underground: 0.35,
        }
    }
}

impl MixerWeights {
    pub fn normalized_triplet(&self) -> (f32, f32, f32) {
        let sum = (self.beat.max(0.01) + self.vocal.max(0.01) + self.atmo.max(0.01)).max(0.03);
        (
            self.beat.max(0.01) / sum,
            self.vocal.max(0.01) / sum,
            self.atmo.max(0.01) / sum,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeCorridor {
    pub id: String,
    pub title: String,
    pub description: String,
    pub sample_artists: String,
    pub track_count: usize,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCandidate {
    pub track: AnalyzedTrack,
    pub total_score: f32,
    pub beat_similarity: f32,
    pub vocal_similarity: f32,
    pub atmo_similarity: f32,
    pub negative_penalty: f32,
    pub corridor_tag: String,
    pub explanation: Vec<String>,
    /// 2D Sonic Map coordinate X (-1.0 .. +1.0)
    pub map_x: f32,
    /// 2D Sonic Map coordinate Y (-1.0 .. +1.0)
    pub map_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SwipeDirection {
    Right,
    Left,
    Up,
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub session_id: String,
    pub seeds: Vec<AnalyzedTrack>,
    pub liked_crate: Vec<AnalyzedTrack>,
    pub negative_anchors: Vec<AnalyzedTrack>,
    pub seen_count: usize,
    pub pool_size: usize,
    pub weights: MixerWeights,
    pub active_corridor: String,
    pub vibe_corridors: Vec<VibeCorridor>,
    pub current_card: Option<ScoredCandidate>,
    pub queue_preview: Vec<ScoredCandidate>,
    pub map_points: Vec<SonicMapPoint>,
    #[serde(default)]
    pub can_undo: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonicMapPoint {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub cover_url: String,
    pub preview_url: String,
    pub drop_start_sec: f32,
    pub bpm: f32,
    pub total_score: f32,
    pub beat_similarity: f32,
    pub vocal_similarity: f32,
    pub atmo_similarity: f32,
    pub corridor_tag: String,
    pub x: f32,
    pub y: f32,
    pub kind: String, // "seed" | "candidate" | "liked" | "negative"
}

pub fn make_dedup_key(artist: &str, title: &str) -> String {
    let clean_artist: String = artist
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    let base_title = title
        .split('(')
        .next()
        .unwrap_or(title)
        .split('-')
        .next()
        .unwrap_or(title);
    let clean_title: String = base_title
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    format!("{}::{}", clean_artist, clean_title)
}
