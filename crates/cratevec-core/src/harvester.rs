use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

use crate::dsp::{analyze_audio_bytes, cosine_similarity};
use crate::models::{make_dedup_key, AnalyzedTrack, TrackMetadata};
use crate::spotify::{is_spotify_url, resolve_spotify_input};
use crate::storage::VectorStore;
use crate::ytmusic;

#[derive(Clone)]
pub struct Harvester {
    client: Client,
    store: VectorStore,
}

#[derive(Debug, Deserialize)]
struct DeezerSearchResp<T> {
    data: Option<Vec<T>>,
}

#[derive(Debug, Clone, Deserialize)]
struct DeezerTrack {
    id: u64,
    title: String,
    #[serde(default)]
    duration: u32,
    #[serde(default)]
    rank: u32,
    #[serde(default)]
    preview: String,
    #[serde(default)]
    link: String,
    artist: Option<DeezerArtist>,
    album: Option<DeezerAlbum>,
}

#[derive(Debug, Clone, Deserialize)]
struct DeezerArtist {
    id: u64,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct DeezerAlbum {
    title: String,
    #[serde(default)]
    cover_medium: String,
    #[serde(default)]
    cover_big: String,
}

#[derive(Debug, Clone, Deserialize)]
struct DeezerPlaylistItem {
    id: u64,
    #[serde(default)]
    title: String,
}

#[derive(Debug, Deserialize)]
struct ItunesSearchResp {
    #[serde(default)]
    results: Vec<ItunesTrack>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ItunesTrack {
    #[serde(default)]
    track_id: u64,
    #[serde(default)]
    track_name: String,
    #[serde(default)]
    artist_name: String,
    #[serde(default)]
    artist_id: Option<u64>,
    #[serde(default)]
    collection_name: String,
    #[serde(default)]
    preview_url: String,
    #[serde(default)]
    artwork_url100: String,
    #[serde(default)]
    track_view_url: String,
    #[serde(default)]
    track_time_millis: u64,
}

#[derive(Debug, Clone, Default)]
struct SceneArtistStats {
    artist_id: u64,
    in_first_hop_graph: bool,
    in_second_hop_graph: bool,
    in_artist_radio: bool,
    playlist_occurrences: usize,
    consensus_score: f32,
}

struct SessionRng {
    state: u64,
}

impl SessionRng {
    fn new(seed_str: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut h = 0xcbf29ce484222325u64 ^ nanos;
        for b in seed_str.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Self { state: h.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

impl Harvester {
    pub fn new(store: VectorStore) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(12))
            .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0")
            .build()
            .expect("Failed to build HTTP client");
        Self { client, store }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    pub async fn resolve_seeds(&self, query: &str) -> Result<Vec<AnalyzedTrack>> {
        let queries = if is_spotify_url(query) {
            resolve_spotify_input(&self.client, query).await?
        } else {
            query
                .split([';', '|'])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        let mut seeds = Vec::new();
        for q in queries.into_iter().take(5) {
            if let Ok(seed) = self.resolve_single_seed(&q).await {
                seeds.push(seed);
            }
        }

        if seeds.is_empty() {
            return Err(anyhow!(
                "Kein Track mit Audio-Preview gefunden fuer: {}",
                query
            ));
        }

        self.store.flush();
        Ok(seeds)
    }

    pub async fn search_tracks(&self, query: &str, limit: usize) -> Result<Vec<TrackMetadata>> {
        let q = if is_spotify_url(query) {
            resolve_spotify_input(&self.client, query)
                .await?
                .into_iter()
                .next()
                .unwrap_or_else(|| query.to_string())
        } else {
            query.trim().to_string()
        };

        if q.is_empty() {
            return Ok(Vec::new());
        }

        let (artist_hint, title_hint) = split_artist_title(&q);
        let mut candidates = Vec::new();
        if let (Some(a), Some(t)) = (&artist_hint, &title_hint) {
            let structured = format!("artist:\"{}\" track:\"{}\"", a, t);
            candidates.extend(
                self.search_deezer_tracks_limit(&structured, "Direct Search", 8)
                    .await,
            );
        }
        candidates.extend(self.search_deezer_tracks_limit(&q, "Direct Search", 15).await);
        if candidates.len() < 4 {
            let itunes = self.search_itunes_tracks(&q, "Direct Search").await;
            candidates.extend(itunes);
        }

        let query_lower = q.to_lowercase();
        let tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2)
            .collect();

        candidates.sort_by(|a, b| {
            let sa = seed_match_score(a, &tokens, artist_hint.as_deref(), title_hint.as_deref());
            let sb = seed_match_score(b, &tokens, artist_hint.as_deref(), title_hint.as_deref());
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for c in candidates {
            if !c.preview_url.is_empty() && seen.insert(c.dedup_key.clone()) {
                out.push(c);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    pub async fn resolve_single_seed(&self, raw_query: &str) -> Result<AnalyzedTrack> {
        let trimmed = raw_query.trim();

        if let Some(dz_id) = trimmed.strip_prefix("dz:") {
            if let Some(meta) = self.fetch_single_deezer_track(dz_id, "Seed").await {
                if let Ok(analyzed) = self.analyze_or_get_cached(meta).await {
                    info!(
                        "Resolved exact seed ID {}: {} - {} ({:.1} BPM, drop @ {:.1}s)",
                        trimmed,
                        analyzed.meta.artist,
                        analyzed.meta.title,
                        analyzed.dsp.bpm,
                        analyzed.dsp.drop_start_sec
                    );
                    return Ok(analyzed);
                }
            }
        }

        let (artist_hint, title_hint) = split_artist_title(trimmed);
        let mut results = Vec::new();

        if let (Some(a), Some(t)) = (&artist_hint, &title_hint) {
            let structured = format!("artist:\"{}\" track:\"{}\"", a, t);
            results.extend(self.search_deezer_tracks(&structured, "Seed").await);
            let swapped = format!("artist:\"{}\" track:\"{}\"", t, a);
            results.extend(self.search_deezer_tracks(&swapped, "Seed").await);
        }

        results.extend(self.search_deezer_tracks(trimmed, "Seed").await);
        results.extend(self.search_itunes_tracks(trimmed, "Seed").await);

        if results.is_empty() {
            return Err(anyhow!("Keine Treffer fuer '{}'", trimmed));
        }

        let query_lower = trimmed.to_lowercase();
        let tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2)
            .collect();

        results.sort_by(|a, b| {
            let score_a =
                seed_match_score(a, &tokens, artist_hint.as_deref(), title_hint.as_deref());
            let score_b =
                seed_match_score(b, &tokens, artist_hint.as_deref(), title_hint.as_deref());
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for meta in results.into_iter().take(6) {
            match self.analyze_or_get_cached(meta.clone()).await {
                Ok(analyzed) => {
                    info!(
                        "Resolved seed: {} - {} ({:.1} BPM, drop @ {:.1}s)",
                        analyzed.meta.artist,
                        analyzed.meta.title,
                        analyzed.dsp.bpm,
                        analyzed.dsp.drop_start_sec
                    );
                    return Ok(analyzed);
                }
                Err(e) => {
                    warn!(
                        "Seed candidate failed DSP: {} - {} [{}]: {}",
                        meta.artist, meta.title, meta.source, e
                    );
                }
            }
        }

        Err(anyhow!("Audio-Analyse fuer '{}' fehlgeschlagen", trimmed))
    }

    async fn fetch_single_deezer_track(&self, dz_id: &str, channel: &str) -> Option<TrackMetadata> {
        let url = format!("https://api.deezer.com/track/{}", dz_id);
        let resp = self.client.get(&url).send().await.ok()?;
        let dt = resp.json::<DeezerTrack>().await.ok()?;
        self.convert_deezer_track(dt, channel, 1.0)
    }

    pub async fn refresh_preview_url(
        &self,
        track_id: &str,
        artist: &str,
        title: &str,
    ) -> Option<String> {
        if let Some(dz_id) = track_id.strip_prefix("dz:") {
            let url = format!("https://api.deezer.com/track/{}", dz_id);
            if let Ok(resp) = self.client.get(&url).send().await {
                if let Ok(dt) = resp.json::<DeezerTrack>().await {
                    if !dt.preview.is_empty() {
                        return Some(dt.preview);
                    }
                }
            }
        }
        let q = format!("{} {}", artist, title);
        let tracks = self.search_deezer_tracks(&q, "Refresh").await;
        tracks
            .into_iter()
            .find(|t| !t.preview_url.is_empty())
            .map(|t| t.preview_url)
    }

    pub async fn harvest_pool_for_seeds(
        &self,
        seeds: &[AnalyzedTrack],
        target_pool_size: usize,
        global_seen_dedups: &HashSet<String>,
    ) -> Vec<AnalyzedTrack> {
        let mut raw_fresh: Vec<TrackMetadata> = Vec::new();
        let mut raw_previously_seen: Vec<TrackMetadata> = Vec::new();
        let mut seed_dedups: HashSet<String> =
            seeds.iter().map(|s| s.meta.dedup_key.clone()).collect();

        for seed in seeds {
            let seed_candidates = self.collect_metadata_around_seed(seed).await;
            for c in seed_candidates {
                if global_seen_dedups.contains(&c.dedup_key) {
                    raw_previously_seen.push(c);
                } else {
                    raw_fresh.push(c);
                }
            }
        }

        let mut combined_raw = raw_fresh;
        combined_raw.extend(raw_previously_seen);

        let mut unique_candidates: Vec<TrackMetadata> = Vec::new();
        let mut artist_counts: HashMap<String, usize> = HashMap::new();

        for cand in combined_raw {
            if cand.preview_url.is_empty() {
                continue;
            }
            if seed_dedups.contains(&cand.dedup_key) {
                continue;
            }

            let title_low = cand.title.to_lowercase();
            let artist_low = cand.artist.to_lowercase();
            let seed_is_instrumental = seeds
                .iter()
                .any(|s| s.meta.title.to_lowercase().contains("instrumental"));
            if title_low.contains("karaoke")
                || title_low.contains("ringtone")
                || title_low.contains("8-bit")
                || title_low.contains("lullaby")
                || (!seed_is_instrumental
                    && (title_low.contains("(instrumental")
                        || title_low.contains("[instrumental")
                        || title_low.ends_with(" instrumental")))
                || artist_low.contains("karaoke")
                || artist_low.contains("tribute")
                || artist_low.contains("querbeat")
                || artist_low.contains("kasalla")
                || artist_low.contains("cat ballou")
                || title_low.contains("komm ran mit bier")
            {
                continue;
            }

            let cand_base_title = title_low
                .split('(')
                .next()
                .unwrap_or(&title_low)
                .split('-')
                .next()
                .unwrap_or(&title_low)
                .trim();
            let is_seed_reupload = seeds.iter().any(|s| {
                let s_base = s.meta.title.to_lowercase();
                let s_base = s_base.split('(').next().unwrap_or(&s_base).trim();
                cand_base_title == s_base
                    && (title_low.contains("slowed")
                        || title_low.contains("sped")
                        || title_low.contains("reverb")
                        || artist_low.contains("backfromparadise"))
            });
            if is_seed_reupload {
                continue;
            }

            let artist_norm: String = artist_low.chars().filter(|c| c.is_alphanumeric()).collect();
            let seed_artist_norm: String = seeds[0]
                .meta
                .artist
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();

            let is_seed_artist = artist_norm == seed_artist_norm
                || (seed_artist_norm.len() >= 4 && artist_norm.contains(&seed_artist_norm));
            let artist_bucket = if is_seed_artist {
                seed_artist_norm.clone()
            } else {
                artist_norm
            };
            let max_for_artist = if is_seed_artist { 1 } else { 2 };
            let count = artist_counts.entry(artist_bucket).or_insert(0);
            if *count >= max_for_artist {
                continue;
            }

            *count += 1;
            seed_dedups.insert(cand.dedup_key.clone());
            unique_candidates.push(cand);

            if unique_candidates.len() >= target_pool_size {
                break;
            }
        }

        info!(
            "Harvested {} scene-verified candidate metadata records; running parallel Rust DSP analysis...",
            unique_candidates.len()
        );

        let analyzed: Vec<AnalyzedTrack> = stream::iter(unique_candidates.into_iter())
            .map(|meta| {
                let harvester = self.clone();
                async move {
                    match harvester.analyze_or_get_cached(meta.clone()).await {
                        Ok(a) => Some(a),
                        Err(e) => {
                            warn!("DSP failed for {} - {}: {}", meta.artist, meta.title, e);
                            None
                        }
                    }
                }
            })
            .buffer_unordered(12)
            .filter_map(|x| async move { x })
            .collect()
            .await;

        let mut filtered = Vec::with_capacity(analyzed.len());
        for track in analyzed {
            let same_artist_as_seed = seeds
                .iter()
                .any(|s| s.meta.artist.eq_ignore_ascii_case(&track.meta.artist));
            if same_artist_as_seed {
                let sim_beat = cosine_similarity(&seeds[0].dsp.v_beat, &track.dsp.v_beat);
                let sim_atmo = cosine_similarity(&seeds[0].dsp.v_atmo, &track.dsp.v_atmo);
                if sim_beat < -0.15 && sim_atmo < -0.15 {
                    continue;
                }
            }
            filtered.push(track);
        }

        self.store.flush();
        filtered
    }

    async fn resolve_deezer_artist_id(&self, seed: &AnalyzedTrack) -> Option<String> {
        if let Some(aid) = &seed.meta.artist_id {
            if let Some(dz_aid) = aid.strip_prefix("dz_artist:") {
                return Some(dz_aid.to_string());
            }
        }
        let tracks = self
            .search_deezer_tracks_limit(&seed.meta.artist, "Artist Lookup", 5)
            .await;
        for t in tracks {
            if t.artist.eq_ignore_ascii_case(&seed.meta.artist) {
                if let Some(aid) = t.artist_id {
                    if let Some(dz_aid) = aid.strip_prefix("dz_artist:") {
                        return Some(dz_aid.to_string());
                    }
                }
            }
        }
        None
    }

    /// Track-Level Song-Radio + Scene-Consensus Harvester (v0.6.0):
    /// Primary source: YouTube Music track-level song radio (`RDAMVM<videoId>`), which finds
    /// neighbors tailored to the EXACT song vibe (e.g. aggressive vs. chill tracks of the same artist)
    /// rather than artist-average catalog popularity.
    /// Secondary supplement / fallback: Deezer 1st-hop related artists & artist radio (and verified
    /// playlists only when Track-Radio is sparse).
    async fn collect_metadata_around_seed(&self, seed: &AnalyzedTrack) -> Vec<TrackMetadata> {
        let mut rng = SessionRng::new(&format!("{}::{}", seed.meta.artist, seed.meta.title));

        // --- STEP 0 (PRIMARY): Track-Level Song-Radio (YouTube Music -> Deezer/iTunes Preview) ---
        let mut ch_track_radio = self.fetch_track_radio_candidates(seed, &mut rng).await;
        let track_radio_artists: HashSet<String> = ch_track_radio
            .iter()
            .map(|t| {
                t.artist
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect()
            })
            .collect();

        let track_radio_strong = ch_track_radio.len() >= 18;

        let mut scene_registry: HashMap<u64, SceneArtistStats> = HashMap::new();
        let mut ch_related_deep_cuts: Vec<TrackMetadata> = Vec::new();
        let mut ch_artist_radio: Vec<TrackMetadata> = Vec::new();
        let mut ch_verified_playlists: Vec<TrackMetadata> = Vec::new();
        let mut ch_second_hop: Vec<TrackMetadata> = Vec::new();

        let dz_aid_opt = self.resolve_deezer_artist_id(seed).await;
        let seed_aid_num: u64 = dz_aid_opt
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // --- STEP 1: 1st-Hop Official Related Artists & Artist Radio (Supplement / Fallback) ---
        let mut related_artists: Vec<DeezerArtist> = Vec::new();
        if let Some(dz_aid) = &dz_aid_opt {
            related_artists = self.fetch_deezer_related_artists(dz_aid).await;
            for ra in &related_artists {
                let entry = scene_registry.entry(ra.id).or_insert_with(|| SceneArtistStats {
                    artist_id: ra.id,
                    ..Default::default()
                });
                entry.in_first_hop_graph = true;
                entry.consensus_score += 3.5;
            }

            let mut radio = self
                .fetch_deezer_track_list(
                    &format!("https://api.deezer.com/artist/{}/radio?limit=24", dz_aid),
                    "Artist Radio Graph",
                )
                .await;
            for rt in &mut radio {
                let a_norm: String = rt
                    .artist
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                rt.scene_affinity = if track_radio_artists.contains(&a_norm) {
                    0.96
                } else if track_radio_strong {
                    0.76
                } else {
                    0.86
                };
                if let Some(aid) = extract_dz_aid(&rt.artist_id) {
                    if aid != seed_aid_num {
                        let entry = scene_registry.entry(aid).or_insert_with(|| SceneArtistStats {
                            artist_id: aid,
                            ..Default::default()
                        });
                        if !entry.in_artist_radio {
                            entry.in_artist_radio = true;
                            entry.consensus_score += 2.6;
                        }
                    }
                }
            }
            rng.shuffle(&mut radio);
            if seed.meta.popularity_rank < 800_000 {
                radio.retain(|t| t.popularity_rank < 880_000);
            }
            ch_artist_radio.extend(radio);
        }

        // --- STEP 2: Verified User Playlists (Only when Track-Radio is sparse to avoid festival/party bleed) ---
        let mut playlist_candidate_buffer: Vec<(u64, usize, TrackMetadata)> = Vec::new();
        if !track_radio_strong {
            let raw_pl_batches = self
                .fetch_verified_seed_playlists(&seed.meta.artist, &seed.meta.title, &mut rng)
                .await;

            let seed_artist_low = seed.meta.artist.to_lowercase();
            for batch in raw_pl_batches {
                let seed_pos = match batch
                    .iter()
                    .position(|t| t.artist.to_lowercase() == seed_artist_low)
                {
                    Some(pos) => pos,
                    None => continue,
                };

                let mut seen_artists_in_this_pl = HashSet::new();
                for (idx, t) in batch.into_iter().enumerate() {
                    if t.artist.to_lowercase() == seed_artist_low || t.popularity_rank > 840_000 {
                        continue;
                    }
                    let dist = idx.abs_diff(seed_pos);
                    if dist > 20 {
                        continue;
                    }
                    if let Some(aid) = extract_dz_aid(&t.artist_id) {
                        if seen_artists_in_this_pl.insert(aid) {
                            let entry =
                                scene_registry.entry(aid).or_insert_with(|| SceneArtistStats {
                                    artist_id: aid,
                                    ..Default::default()
                                });
                            entry.playlist_occurrences += 1;
                            let pos_bonus = if dist <= 6 { 1.8 } else { 1.2 };
                            entry.consensus_score += pos_bonus;
                        }
                        playlist_candidate_buffer.push((aid, dist, t));
                    }
                }
            }
        }

        // --- STEP 3: Expand 2nd-Hop Related Artists only when sparse ---
        let mut top_scene_aids: Vec<(f32, u64)> = scene_registry
            .values()
            .filter(|s| s.artist_id != seed_aid_num && s.consensus_score >= 2.4)
            .map(|s| (s.consensus_score, s.artist_id))
            .collect();
        top_scene_aids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if !track_radio_strong && related_artists.len() < 6 {
            let expand_aids: Vec<u64> = top_scene_aids.iter().take(6).map(|&(_, id)| id).collect();
            let rel2_futures = expand_aids.into_iter().map(|aid| {
                let this = self.clone();
                async move {
                    this.fetch_deezer_related_artists(&aid.to_string()).await
                }
            });
            let rel2_batches: Vec<Vec<DeezerArtist>> =
                stream::iter(rel2_futures).buffer_unordered(6).collect().await;
            for batch in rel2_batches {
                for ra in batch.into_iter().take(8) {
                    if ra.id == seed_aid_num {
                        continue;
                    }
                    let entry = scene_registry.entry(ra.id).or_insert_with(|| SceneArtistStats {
                        artist_id: ra.id,
                        ..Default::default()
                    });
                    entry.in_second_hop_graph = true;
                    entry.consensus_score += 1.4;
                }
            }
        }

        // --- STEP 4: Filter Verified Playlist Tracks (fallback mode) ---
        playlist_candidate_buffer.sort_by_key(|&(_, dist, _)| dist);
        for (aid, _dist, mut t) in playlist_candidate_buffer {
            if let Some(stats) = scene_registry.get(&aid) {
                let is_verified_scene = stats.in_first_hop_graph
                    || stats.in_second_hop_graph
                    || stats.in_artist_radio
                    || stats.playlist_occurrences >= 2;
                if is_verified_scene {
                    t.scene_affinity = (stats.consensus_score / 5.0).clamp(0.45, 0.92);
                    ch_verified_playlists.push(t);
                }
            }
        }

        // --- STEP 5: Fetch Deep Cuts from 1st-Hop Related Artists (smaller slice when Track-Radio is strong) ---
        rng.shuffle(&mut related_artists);
        let rel1_limit = if track_radio_strong { 8 } else { 14 };
        let rel1_ids: Vec<u64> = related_artists.iter().take(rel1_limit).map(|a| a.id).collect();
        let rel1_futures = rel1_ids.into_iter().map(|aid| {
            let this = self.clone();
            async move {
                let url = format!("https://api.deezer.com/artist/{}/top?limit=10", aid);
                (aid, this.fetch_deezer_track_list(&url, "Related Artist Deep Cut").await)
            }
        });
        let rel1_batches: Vec<(u64, Vec<TrackMetadata>)> =
            stream::iter(rel1_futures).buffer_unordered(8).collect().await;

        for (aid, mut batch) in rel1_batches {
            if batch.len() >= 4 {
                batch.remove(0);
            }
            let base_aff = scene_registry
                .get(&aid)
                .map(|s| (s.consensus_score / 4.5).clamp(0.72, 0.95))
                .unwrap_or(0.78);
            for t in &mut batch {
                let a_norm: String = t
                    .artist
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                t.scene_affinity = if track_radio_artists.contains(&a_norm) {
                    (base_aff + 0.12).min(1.0)
                } else if track_radio_strong {
                    base_aff.min(0.78)
                } else {
                    base_aff
                };
            }
            rng.shuffle(&mut batch);
            ch_related_deep_cuts.extend(batch.into_iter().take(2));
        }

        // --- STEP 6: 2nd-Hop Deep Cuts (only when Track-Radio is sparse) ---
        if !track_radio_strong {
            let mut second_hop_candidates: Vec<(f32, u64)> = scene_registry
                .values()
                .filter(|s| {
                    s.artist_id != seed_aid_num
                        && !s.in_first_hop_graph
                        && s.consensus_score >= 2.2
                })
                .map(|s| (s.consensus_score, s.artist_id))
                .collect();
            second_hop_candidates
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let sh_ids: Vec<(f32, u64)> = second_hop_candidates.into_iter().take(12).collect();
            let sh_futures = sh_ids.into_iter().map(|(c_score, aid)| {
                let this = self.clone();
                async move {
                    let url = format!("https://api.deezer.com/artist/{}/top?limit=10", aid);
                    let list = this
                        .fetch_deezer_track_list(&url, "Scene Consensus Graph")
                        .await;
                    (c_score, list)
                }
            });
            let sh_batches: Vec<(f32, Vec<TrackMetadata>)> =
                stream::iter(sh_futures).buffer_unordered(8).collect().await;

            for (c_score, mut batch) in sh_batches {
                if batch.len() >= 4 {
                    batch.remove(0);
                }
                batch.retain(|t| t.popularity_rank < 840_000);
                let aff = (c_score / 4.5).clamp(0.55, 0.92);
                for t in &mut batch {
                    t.scene_affinity = aff;
                }
                rng.shuffle(&mut batch);
                ch_second_hop.extend(batch.into_iter().take(2));
            }
        }

        // --- Boost Track-Radio items whose artist is also verified in Deezer's scene graph ---
        for t in &mut ch_track_radio {
            if let Some(aid) = extract_dz_aid(&t.artist_id) {
                if scene_registry.contains_key(&aid) {
                    t.scene_affinity = (t.scene_affinity + 0.05).min(1.0);
                }
            }
        }

        // --- Weighted Interleaving: Track-Level Radio dominates (~75%), Deezer supplements ---
        let mut interleaved = Vec::with_capacity(96);
        let mut idx_tr = 0usize;
        let mut idx_rel = 0usize;
        let mut idx_ar = 0usize;
        let mut idx_sh = 0usize;
        let mut idx_pl = 0usize;

        for round in 0..40 {
            let tr_take = if track_radio_strong { 3 } else { 2 };
            for _ in 0..tr_take {
                if let Some(t) = ch_track_radio.get(idx_tr) {
                    interleaved.push(t.clone());
                    idx_tr += 1;
                }
            }
            if round % 2 == 0 {
                if let Some(t) = ch_related_deep_cuts.get(idx_rel) {
                    interleaved.push(t.clone());
                    idx_rel += 1;
                }
            } else if let Some(t) = ch_artist_radio.get(idx_ar) {
                interleaved.push(t.clone());
                idx_ar += 1;
            }
            if !track_radio_strong {
                if let Some(t) = ch_second_hop.get(idx_sh) {
                    interleaved.push(t.clone());
                    idx_sh += 1;
                }
                if let Some(t) = ch_verified_playlists.get(idx_pl) {
                    interleaved.push(t.clone());
                    idx_pl += 1;
                }
            }
            if idx_tr >= ch_track_radio.len()
                && idx_rel >= ch_related_deep_cuts.len()
                && idx_ar >= ch_artist_radio.len()
                && idx_sh >= ch_second_hop.len()
                && idx_pl >= ch_verified_playlists.len()
            {
                break;
            }
        }

        interleaved
    }

    /// Resolves track-level song radio pairs from YouTube Music into playable Deezer/iTunes previews.
    async fn fetch_track_radio_candidates(
        &self,
        seed: &AnalyzedTrack,
        rng: &mut SessionRng,
    ) -> Vec<TrackMetadata> {
        let radio_pairs =
            ytmusic::fetch_song_radio(&self.client, &seed.meta.artist, &seed.meta.title).await;
        if radio_pairs.is_empty() {
            return Vec::new();
        }

        // Preserve top-tier ordering while adding light within-tier freshness across sessions
        let mut indexed: Vec<(usize, String, String)> = radio_pairs
            .into_iter()
            .take(44)
            .enumerate()
            .map(|(i, (a, t))| (i, a, t))
            .collect();
        if indexed.len() > 14 {
            rng.shuffle(&mut indexed[..14]);
            rng.shuffle(&mut indexed[14..]);
        } else {
            rng.shuffle(&mut indexed);
        }

        let futures = indexed.into_iter().map(|(orig_rank, artist, title)| {
            let this = self.clone();
            async move {
                let mut meta = this.resolve_pair_to_preview(&artist, &title).await?;
                meta.discovery_channel = "Track-Radio".to_string();
                meta.scene_affinity = if orig_rank < 12 {
                    0.97
                } else if orig_rank < 26 {
                    0.93
                } else {
                    0.88
                };
                Some(meta)
            }
        });

        stream::iter(futures)
            .buffer_unordered(12)
            .filter_map(|x| async move { x })
            .collect()
            .await
    }

    async fn resolve_pair_to_preview(&self, artist: &str, title: &str) -> Option<TrackMetadata> {
        let base_title = title
            .split('(')
            .next()
            .unwrap_or(title)
            .split('[')
            .next()
            .unwrap_or(title)
            .trim();

        let structured = format!("artist:\"{}\" track:\"{}\"", artist, base_title);
        let mut hits = self
            .search_deezer_tracks_limit(&structured, "Track-Radio", 4)
            .await;

        if hits.is_empty() {
            let plain = format!("{} {}", artist, base_title);
            hits = self
                .search_deezer_tracks_limit(&plain, "Track-Radio", 5)
                .await;
        }

        if hits.is_empty() {
            let plain = format!("{} {}", artist, base_title);
            hits = self.search_itunes_tracks(&plain, "Track-Radio").await;
        }

        if hits.is_empty() {
            return None;
        }

        let query_lower = format!("{} {}", artist, base_title).to_lowercase();
        let tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= 2)
            .collect();

        hits.sort_by(|a, b| {
            let sa = seed_match_score(a, &tokens, Some(artist), Some(base_title));
            let sb = seed_match_score(b, &tokens, Some(artist), Some(base_title));
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = hits.into_iter().find(|t| !t.preview_url.is_empty())?;
        // Basic sanity check: ensure at least artist or title shares a meaningful token
        let best_a = best.artist.to_lowercase();
        let best_t = best.title.to_lowercase();
        let target_a = artist.to_lowercase();
        let target_t = base_title.to_lowercase();
        let artist_ok = target_a
            .split_whitespace()
            .filter(|w| w.len() >= 2)
            .any(|w| best_a.contains(w) || best_t.contains(w));
        let title_ok = target_t
            .split_whitespace()
            .filter(|w| w.len() >= 2)
            .any(|w| best_t.contains(w));
        if artist_ok || title_ok {
            Some(best)
        } else {
            None
        }
    }

    async fn fetch_deezer_related_artists(&self, dz_artist_id: &str) -> Vec<DeezerArtist> {
        let url = format!(
            "https://api.deezer.com/artist/{}/related?limit=25",
            dz_artist_id
        );
        let resp = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let parsed = match resp.json::<DeezerSearchResp<DeezerArtist>>().await {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        parsed.data.unwrap_or_default()
    }

    async fn fetch_verified_seed_playlists(
        &self,
        artist: &str,
        title: &str,
        rng: &mut SessionRng,
    ) -> Vec<Vec<TrackMetadata>> {
        let queries = [format!("{} {}", artist, title), artist.to_string()];

        let banned_playlist_words = [
            "top 100",
            "top 50",
            "hits",
            "classics",
            "best of",
            "workout",
            "gym",
            "charts",
            "radio",
            "party",
            "mainstream",
            "essentials",
            "all out",
            "mega hit",
            "billboard",
            "festival",
            "hurricane",
            "southside",
            "deichbrand",
            "karneval",
            "fasching",
            "saufen",
            "malle",
            "favoriten",
            "liebling",
        ];

        let mut playlist_ids = Vec::new();
        for q in &queries {
            let url = format!(
                "https://api.deezer.com/search/playlist?q={}&limit=12",
                urlencoding::encode(q)
            );
            if let Ok(resp) = self.client.get(&url).send().await {
                if let Ok(parsed) = resp.json::<DeezerSearchResp<DeezerPlaylistItem>>().await {
                    if let Some(list) = parsed.data {
                        for pl in list {
                            let pl_title_low = pl.title.to_lowercase();
                            if banned_playlist_words
                                .iter()
                                .any(|&bw| pl_title_low.contains(bw))
                            {
                                continue;
                            }
                            if !playlist_ids.contains(&pl.id) {
                                playlist_ids.push(pl.id);
                            }
                        }
                    }
                }
            }
            if playlist_ids.len() >= 10 {
                break;
            }
        }

        rng.shuffle(&mut playlist_ids);

        let pl_futures = playlist_ids.into_iter().take(7).map(|pid| {
            let this = self.clone();
            async move {
                let url = format!("https://api.deezer.com/playlist/{}/tracks?limit=75", pid);
                this.fetch_deezer_track_list(&url, "Verified Scene Playlist")
                    .await
            }
        });

        stream::iter(pl_futures).buffer_unordered(7).collect().await
    }

    async fn analyze_or_get_cached(&self, mut meta: TrackMetadata) -> Result<AnalyzedTrack> {
        if let Some(mut cached) = self
            .store
            .get_by_id(&meta.id)
            .or_else(|| self.store.get_by_dedup(&meta.dedup_key))
        {
            if !meta.preview_url.is_empty() {
                cached.meta.preview_url = meta.preview_url;
            }
            cached.meta.discovery_channel = meta.discovery_channel;
            cached.meta.scene_affinity = meta.scene_affinity;
            return Ok(cached);
        }

        if meta.preview_url.is_empty() {
            return Err(anyhow!("Empty preview URL"));
        }

        let bytes = self
            .client
            .get(&meta.preview_url)
            .send()
            .await?
            .bytes()
            .await?
            .to_vec();

        let dsp = tokio::task::spawn_blocking(move || analyze_audio_bytes(&bytes)).await??;

        if meta.spotify_search_url.is_empty() {
            meta.spotify_search_url = format!(
                "https://open.spotify.com/search/{}",
                urlencoding::encode(&format!("{} {}", meta.artist, meta.title))
            );
        }

        let analyzed = AnalyzedTrack { meta, dsp };
        self.store.insert(analyzed.clone());
        Ok(analyzed)
    }

    async fn search_deezer_tracks(&self, query: &str, channel: &str) -> Vec<TrackMetadata> {
        self.search_deezer_tracks_limit(query, channel, 15).await
    }

    async fn search_deezer_tracks_limit(
        &self,
        query: &str,
        channel: &str,
        limit: usize,
    ) -> Vec<TrackMetadata> {
        let url = format!(
            "https://api.deezer.com/search?q={}&limit={}",
            urlencoding::encode(query),
            limit
        );
        self.fetch_deezer_track_list(&url, channel).await
    }

    async fn fetch_deezer_track_list(&self, url: &str, channel: &str) -> Vec<TrackMetadata> {
        let resp = match self.client.get(url).send().await {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let parsed = match resp.json::<DeezerSearchResp<DeezerTrack>>().await {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut out = Vec::new();
        for dt in parsed.data.unwrap_or_default() {
            if let Some(meta) = self.convert_deezer_track(dt, channel, 0.78) {
                out.push(meta);
            }
        }
        out
    }

    fn convert_deezer_track(
        &self,
        dt: DeezerTrack,
        channel: &str,
        scene_affinity: f32,
    ) -> Option<TrackMetadata> {
        if dt.preview.is_empty() {
            return None;
        }
        let artist_name = dt
            .artist
            .as_ref()
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        let artist_id = dt.artist.as_ref().map(|a| format!("dz_artist:{}", a.id));
        let album_title = dt
            .album
            .as_ref()
            .map(|a| a.title.clone())
            .unwrap_or_default();
        let cover = dt
            .album
            .as_ref()
            .map(|a| {
                if !a.cover_big.is_empty() {
                    a.cover_big.clone()
                } else {
                    a.cover_medium.clone()
                }
            })
            .unwrap_or_default();

        let pop_score = ((dt.rank as f32 / 950_000.0).max(0.0).powf(1.35)).clamp(0.01, 0.99);
        let dedup_key = make_dedup_key(&artist_name, &dt.title);
        let spotify_search_url = format!(
            "https://open.spotify.com/search/{}",
            urlencoding::encode(&format!("{} {}", artist_name, dt.title))
        );

        Some(TrackMetadata {
            id: format!("dz:{}", dt.id),
            dedup_key,
            title: dt.title,
            artist: artist_name,
            artist_id,
            album: album_title,
            cover_url: cover,
            preview_url: dt.preview,
            external_url: dt.link,
            spotify_search_url,
            source: "deezer".to_string(),
            duration_sec: dt.duration,
            popularity_rank: dt.rank,
            popularity_score: pop_score,
            discovery_channel: channel.to_string(),
            scene_affinity,
        })
    }

    async fn search_itunes_tracks(&self, query: &str, channel: &str) -> Vec<TrackMetadata> {
        let url = format!(
            "https://itunes.apple.com/search?term={}&entity=song&limit=12",
            urlencoding::encode(query)
        );
        let resp = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let parsed = match resp.json::<ItunesSearchResp>().await {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut out = Vec::new();
        for (idx, it) in parsed.results.into_iter().enumerate() {
            if it.preview_url.is_empty() || it.track_name.is_empty() {
                continue;
            }
            let cover = it.artwork_url100.replace("100x100bb", "500x500bb");
            let dedup_key = make_dedup_key(&it.artist_name, &it.track_name);
            let pop_score = (0.58 - (idx as f32 * 0.035)).clamp(0.15, 0.70);
            let spotify_search_url = format!(
                "https://open.spotify.com/search/{}",
                urlencoding::encode(&format!("{} {}", it.artist_name, it.track_name))
            );

            out.push(TrackMetadata {
                id: format!("it:{}", it.track_id),
                dedup_key,
                title: it.track_name,
                artist: it.artist_name,
                artist_id: it.artist_id.map(|id| format!("it_artist:{}", id)),
                album: it.collection_name,
                cover_url: cover,
                preview_url: it.preview_url,
                external_url: it.track_view_url,
                spotify_search_url,
                source: "itunes".to_string(),
                duration_sec: (it.track_time_millis / 1000) as u32,
                popularity_rank: (pop_score * 500_000.0) as u32,
                popularity_score: pop_score,
                discovery_channel: channel.to_string(),
                scene_affinity: 0.70,
            });
        }
        out
    }
}

fn extract_dz_aid(opt: &Option<String>) -> Option<u64> {
    opt.as_deref()?
        .strip_prefix("dz_artist:")?
        .parse::<u64>()
        .ok()
}

fn split_artist_title(raw: &str) -> (Option<String>, Option<String>) {
    for sep in [" - ", " – ", " — ", " by ", " von "] {
        if let Some((left, right)) = raw.split_once(sep) {
            let l = left.trim();
            let r = right.trim();
            if !l.is_empty() && !r.is_empty() {
                if sep == " by " || sep == " von " {
                    return (Some(r.to_string()), Some(l.to_string()));
                }
                return (Some(l.to_string()), Some(r.to_string()));
            }
        }
    }
    let words: Vec<&str> = raw.split_whitespace().collect();
    if words.len() == 2 {
        return (Some(words[0].to_string()), Some(words[1].to_string()));
    }
    (None, None)
}

fn seed_match_score(
    meta: &TrackMetadata,
    tokens: &[&str],
    artist_hint: Option<&str>,
    title_hint: Option<&str>,
) -> f32 {
    let a_low = meta.artist.to_lowercase();
    let t_low = meta.title.to_lowercase();
    let mut score = 0.0f32;
    let mut matched_tokens = 0usize;

    for &tok in tokens {
        if a_low == tok || t_low == tok {
            score += 6.0;
            matched_tokens += 1;
        } else if a_low.contains(tok) || t_low.contains(tok) {
            score += 3.0;
            matched_tokens += 1;
        }
    }

    if !tokens.is_empty() && matched_tokens == tokens.len() {
        score += 22.0;
    }

    if let (Some(ah), Some(th)) = (artist_hint, title_hint) {
        let ah_l = ah.to_lowercase();
        let th_l = th.to_lowercase();
        if (a_low == ah_l && t_low == th_l) || (a_low == th_l && t_low == ah_l) {
            score += 30.0;
        } else if (a_low == ah_l && t_low.contains(&th_l))
            || (a_low == th_l && t_low.contains(&ah_l))
        {
            score += 18.0;
        }
    }

    if (t_low.contains("slowed") || t_low.contains("sped up") || t_low.contains("instrumental"))
        && !tokens
            .iter()
            .any(|&t| t == "slowed" || t == "sped" || t == "instrumental")
    {
        score -= 4.5;
    }

    score
}
