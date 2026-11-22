use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, warn};

use crate::dsp::{analyze_audio_bytes, cosine_similarity};
use crate::models::{make_dedup_key, AcousticFingerprint, AnalyzedTrack, TrackMetadata};
use crate::spotify::{is_spotify_url, resolve_spotify_input};
use crate::storage::VectorStore;

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
        self.convert_deezer_track(dt, channel)
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

        // 1. Local VectorStore KNN Mining (< 1ms acoustic lookup across all previously analyzed tracks)
        let knn_matches = self.mine_local_vector_knn(seeds, &seed_dedups, global_seen_dedups, 8);
        for km in knn_matches {
            raw_fresh.push(km.meta);
        }

        // 2. Network Graph Harvesting around each seed
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
            if title_low.contains("karaoke")
                || title_low.contains("ringtone")
                || title_low.contains("8-bit")
                || title_low.contains("lullaby")
                || artist_low.contains("karaoke")
                || artist_low.contains("tribute")
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

            let max_for_artist = if artist_norm == seed_artist_norm { 1 } else { 2 };
            let count = artist_counts.entry(artist_norm).or_insert(0);
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
            "Harvested {} unique candidate metadata records; running parallel Rust DSP analysis...",
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

    /// Scans the local VectorStore in < 1ms for previously analyzed tracks with high 48-D cosine similarity.
    fn mine_local_vector_knn(
        &self,
        seeds: &[AnalyzedTrack],
        seed_dedups: &HashSet<String>,
        global_seen: &HashSet<String>,
        limit: usize,
    ) -> Vec<AnalyzedTrack> {
        let all = self.store.all_tracks();
        if all.is_empty() || seeds.is_empty() {
            return Vec::new();
        }
        let s = &seeds[0].dsp;
        let mut scored: Vec<(f32, AnalyzedTrack)> = all
            .into_iter()
            .filter(|t| {
                !seed_dedups.contains(&t.meta.dedup_key)
                    && !global_seen.contains(&t.meta.dedup_key)
            })
            .filter_map(|mut t| {
                let sb = cosine_similarity(&s.v_beat, &t.dsp.v_beat);
                let sv = cosine_similarity(&s.v_vocal, &t.dsp.v_vocal);
                let sa = cosine_similarity(&s.v_atmo, &t.dsp.v_atmo);
                let comb = 0.38 * sb + 0.31 * sv + 0.31 * sa;
                if comb >= 0.44 {
                    t.meta.discovery_channel = "Local Vector KNN".to_string();
                    Some((comb, t))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(_, t)| t).collect()
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

    /// Genre-agnostic multi-hop discovery around a seed track:
    /// Channel A: 1st-Hop Artist Radio & Related Artist Deep Cuts
    /// Channel B: Curated User Playlists with Positional Locality around the seed
    /// Channel C: Exact 2nd-Hop Artist Graph via `dz_artist:{id}` of co-occurring & related artists
    /// Channel D: Emergency Sonic Archetype Probes (active primarily when underground graph is sparse)
    async fn collect_metadata_around_seed(&self, seed: &AnalyzedTrack) -> Vec<TrackMetadata> {
        let mut rng = SessionRng::new(&format!("{}::{}", seed.meta.artist, seed.meta.title));

        let mut ch_artist_graph: Vec<TrackMetadata> = Vec::new();
        let mut ch_related_deep_cuts: Vec<TrackMetadata> = Vec::new();
        let mut ch_playlist_vibe: Vec<TrackMetadata> = Vec::new();
        let mut ch_second_hop: Vec<TrackMetadata> = Vec::new();
        let mut ch_sonic_cluster: Vec<TrackMetadata> = Vec::new();

        // --- CHANNEL A: Direct Artist Radio + 1st-Hop Related Artists (Deep Cuts) ---
        let mut related_artists: Vec<DeezerArtist> = Vec::new();
        if let Some(dz_aid) = self.resolve_deezer_artist_id(seed).await {
            let mut radio = self
                .fetch_deezer_track_list(
                    &format!("https://api.deezer.com/artist/{}/radio?limit=25", dz_aid),
                    "Artist Radio Graph",
                )
                .await;
            rng.shuffle(&mut radio);
            if seed.meta.popularity_rank < 800_000 {
                radio.retain(|t| t.popularity_rank < 880_000);
            }
            ch_artist_graph.extend(radio);

            related_artists = self.fetch_deezer_related_artists(&dz_aid).await;
            rng.shuffle(&mut related_artists);

            let rel_ids: Vec<u64> = related_artists.iter().take(12).map(|a| a.id).collect();
            let rel_futures = rel_ids.into_iter().map(|aid| {
                let this = self.clone();
                async move {
                    let url = format!("https://api.deezer.com/artist/{}/top?limit=12", aid);
                    this.fetch_deezer_track_list(&url, "Related Artist Deep Cut")
                        .await
                }
            });
            let rel_batches: Vec<Vec<TrackMetadata>> =
                stream::iter(rel_futures).buffer_unordered(8).collect().await;

            for mut batch in rel_batches {
                if batch.len() >= 4 {
                    batch.remove(0);
                }
                rng.shuffle(&mut batch);
                ch_related_deep_cuts.extend(batch.into_iter().take(2));
            }
        }

        // --- CHANNEL B: Curated Playlists with Positional Locality ---
        let pl_tracks = self
            .mine_curated_playlist_cooccurrences(&seed.meta.artist, &seed.meta.title, &mut rng)
            .await;

        // Extract exact Deezer artist IDs (`dz_artist:{id}`) from playlist tracks & radio tracks for 2nd-hop graph traversal!
        let mut co_artist_ids: Vec<u64> = Vec::new();
        let mut seen_aid = HashSet::new();
        for pt in pl_tracks.iter().chain(ch_artist_graph.iter()) {
            if !pt.artist.eq_ignore_ascii_case(&seed.meta.artist) && pt.popularity_rank < 800_000 {
                if let Some(aid_str) = &pt.artist_id {
                    if let Some(id_num) = aid_str
                        .strip_prefix("dz_artist:")
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        if seen_aid.insert(id_num) {
                            co_artist_ids.push(id_num);
                        }
                    }
                }
            }
        }
        ch_playlist_vibe.extend(pl_tracks);

        // --- CHANNEL C: Exact 2nd-Hop Artist Graph via `dz_artist:{id}` ---
        rng.shuffle(&mut co_artist_ids);
        let second_hop_futures = co_artist_ids.into_iter().take(8).map(|aid| {
            let this = self.clone();
            async move {
                let url = format!("https://api.deezer.com/artist/{}/top?limit=10", aid);
                this.fetch_deezer_track_list(&url, "2nd-Hop Underground Graph")
                    .await
            }
        });
        let second_hop_batches: Vec<Vec<TrackMetadata>> = stream::iter(second_hop_futures)
            .buffer_unordered(8)
            .collect()
            .await;
        for mut batch in second_hop_batches {
            if batch.len() >= 4 {
                batch.remove(0);
            }
            batch.retain(|t| t.popularity_rank < 840_000);
            rng.shuffle(&mut batch);
            ch_second_hop.extend(batch.into_iter().take(2));
        }

        // --- CHANNEL D: Supplementary Sonic Micro-Cluster Probes ---
        // Only activate heavily when the direct related-artist graph is sparse (e.g. underground artist like PIXY)
        let organic_count = ch_related_deep_cuts.len()
            + ch_artist_graph.len()
            + ch_playlist_vibe.len()
            + ch_second_hop.len();
        let probe_limit = if related_artists.len() < 4 || organic_count < 28 {
            12
        } else {
            4
        };

        let mut archetype_queries = build_sonic_archetype_probes(&seed.meta, &seed.dsp);
        rng.shuffle(&mut archetype_queries);

        let probe_futures = archetype_queries
            .into_iter()
            .take(probe_limit)
            .map(|(artist_probe, label)| {
                let this = self.clone();
                async move {
                    let raw = this
                        .search_deezer_tracks_limit(&artist_probe, &label, 10)
                        .await;
                    let probe_low = artist_probe.to_lowercase();
                    raw.into_iter()
                        .filter(|t| {
                            t.artist.to_lowercase().contains(&probe_low)
                                && t.popularity_rank < 850_000
                        })
                        .collect::<Vec<TrackMetadata>>()
                }
            });
        let probe_batches: Vec<Vec<TrackMetadata>> =
            stream::iter(probe_futures).buffer_unordered(8).collect().await;
        for mut batch in probe_batches {
            rng.shuffle(&mut batch);
            ch_sonic_cluster.extend(batch.into_iter().take(2));
        }

        // --- Round-Robin Interleaving across all channels ---
        let mut interleaved = Vec::with_capacity(85);
        let max_len = ch_related_deep_cuts
            .len()
            .max(ch_artist_graph.len())
            .max(ch_second_hop.len())
            .max(ch_playlist_vibe.len())
            .max(ch_sonic_cluster.len());

        for i in 0..max_len {
            if let Some(t) = ch_related_deep_cuts.get(i) {
                interleaved.push(t.clone());
            }
            if let Some(t) = ch_artist_graph.get(i) {
                interleaved.push(t.clone());
            }
            if let Some(t) = ch_second_hop.get(i) {
                interleaved.push(t.clone());
            }
            if let Some(t) = ch_playlist_vibe.get(i) {
                interleaved.push(t.clone());
            }
            if let Some(t) = ch_sonic_cluster.get(i) {
                interleaved.push(t.clone());
            }
        }

        interleaved
    }

    async fn fetch_deezer_related_artists(&self, dz_artist_id: &str) -> Vec<DeezerArtist> {
        let url = format!(
            "https://api.deezer.com/artist/{}/related?limit=20",
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

    async fn mine_curated_playlist_cooccurrences(
        &self,
        artist: &str,
        title: &str,
        rng: &mut SessionRng,
    ) -> Vec<TrackMetadata> {
        let mut out = Vec::new();
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
        ];

        let mut playlist_ids = Vec::new();
        for q in &queries {
            let url = format!(
                "https://api.deezer.com/search/playlist?q={}&limit=10",
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
            if playlist_ids.len() >= 6 {
                break;
            }
        }

        rng.shuffle(&mut playlist_ids);

        let pl_futures = playlist_ids.into_iter().take(4).map(|pid| {
            let this = self.clone();
            async move {
                let url = format!("https://api.deezer.com/playlist/{}/tracks?limit=50", pid);
                this.fetch_deezer_track_list(&url, "Playlist Co-Occurrence")
                    .await
            }
        });

        let batches: Vec<Vec<TrackMetadata>> =
            stream::iter(pl_futures).buffer_unordered(4).collect().await;

        let artist_low = artist.to_lowercase();
        for batch in batches {
            // Find the position where the seed artist appears in this playlist to prioritize immediate neighbors!
            let seed_pos = batch
                .iter()
                .position(|t| t.artist.to_lowercase() == artist_low);

            let mut filtered: Vec<(usize, TrackMetadata)> = batch
                .into_iter()
                .enumerate()
                .filter(|(_, t)| t.popularity_rank < 820_000)
                .collect();

            if let Some(sp) = seed_pos {
                filtered.sort_by_key(|(idx, _)| idx.abs_diff(sp));
            } else {
                rng.shuffle(&mut filtered);
            }

            out.extend(filtered.into_iter().take(6).map(|(_, t)| t));
        }

        out
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
            if let Some(meta) = self.convert_deezer_track(dt, channel) {
                out.push(meta);
            }
        }
        out
    }

    fn convert_deezer_track(&self, dt: DeezerTrack, channel: &str) -> Option<TrackMetadata> {
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

        // Smooth power-law popularity normalization across the entire [0 .. 950,000] rank spectrum
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
            });
        }
        out
    }
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

fn build_sonic_archetype_probes(
    meta: &TrackMetadata,
    dsp: &AcousticFingerprint,
) -> Vec<(String, String)> {
    let mut probes: Vec<(String, String)> = Vec::new();

    if dsp.sub_bass_ratio >= 0.18
        && dsp.glitch_brightness >= 0.50
        && (dsp.bass_distortion >= 0.18 || dsp.darkness_index >= 0.65)
    {
        let glitch_trap_probes = [
            "2hollis",
            "Jane Remover",
            "ericdoa",
            "glaive",
            "midwxst",
            "Snow Strippers",
            "brakence",
            "kets4eki",
            "asteria",
            "Odetari",
            "Lumi Athena",
            "6arelyhuman",
            "OsamaSon",
            "Nettspend",
            "Crystal Castles",
            "Pastel Ghost",
            "Ayesha Erotica",
            "Suzy Sheer",
            "underscores",
            "Kanii",
            "Nimstarr",
        ];
        for p in glitch_trap_probes {
            if !p.eq_ignore_ascii_case(&meta.artist) {
                probes.push((
                    p.to_string(),
                    "DSP Cluster: Glitchpop / Dark Trap".to_string(),
                ));
            }
        }
    }

    if dsp.darkness_index >= 0.55 && dsp.vocal_presence <= 0.34 {
        for p in [
            "Mareux",
            "Gesaffelstein",
            "Boy Harsher",
            "Sidewalks and Skeletons",
            "Mr.Kitty",
            "Perturbator",
            "TR/ST",
            "Health",
            "crim3s",
            "IC3PEAK",
            "Lebanon Hanover",
            "Molchat Doma",
        ] {
            if !p.eq_ignore_ascii_case(&meta.artist) {
                probes.push((
                    p.to_string(),
                    "DSP Cluster: Dark Synth / Coldwave".to_string(),
                ));
            }
        }
    }

    if dsp.vocal_presence >= 0.36 && dsp.bass_distortion >= 0.35 && dsp.sub_bass_ratio < 0.22 {
        for p in [
            "Deftones",
            "Loathe",
            "Wisp",
            "Superheaven",
            "Narrow Head",
            "Fleshwater",
            "Nine Inch Nails",
            "Julie",
            "Glare",
            "Whirr",
            "Quannnic",
        ] {
            if !p.eq_ignore_ascii_case(&meta.artist) {
                probes.push((
                    p.to_string(),
                    "DSP Cluster: Heavy Texture / Shoegaze".to_string(),
                ));
            }
        }
    }

    if dsp.sub_bass_ratio >= 0.24 && dsp.darkness_index >= 0.50 {
        for p in [
            "Destroy Lonely",
            "Bladee",
            "Yung Lean",
            "Ecco2k",
            "BONES",
            "Night Lovell",
            "Lancey Foux",
            "Homixide Gang",
            "Bktherula",
            "Hardrock",
            "Cochise",
            "SoFaygo",
            "JPEGMAFIA",
            "Denzel Curry",
            "Kenny Mason",
            "Paris Texas",
        ] {
            if !p.eq_ignore_ascii_case(&meta.artist) {
                probes.push((
                    p.to_string(),
                    "DSP Cluster: Underground Rap / Cloud 808".to_string(),
                ));
            }
        }
    }

    probes
}
