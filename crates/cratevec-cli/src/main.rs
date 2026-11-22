use anyhow::Result;
use cratevec_core::CrateVecEngine;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("cratevec=info,symphonia=error,warn")
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(|s| s.as_str()) == Some("train") {
        let engine = CrateVecEngine::new(None);
        let seed_clusters: Vec<&str> = vec![
            "PIXY - LEGACY",
            "Playboi Carti - Rockstar Made",
            "Gesaffelstein - Pursuit",
            "Deftones - Change (In the House of Flies)",
            "Kendrick Lamar - XXX.",
        ];

        println!("==================================================================");
        println!(" CRATEVEC METRIC TRAINER // 48-D Contrastive Triplet Learning");
        println!("==================================================================");
        let report = engine.train_metric_model(&seed_clusters, 140).await?;

        println!(
            " Trainings-Korpus   : {} Tracks | {} Akustische Cluster | {} Triplets",
            report.num_tracks, report.num_clusters, report.num_triplets
        );
        println!(
            " Triplet Margin Loss: {:.4} -> {:.4} ({} Epochen)",
            report.initial_loss, report.final_loss, report.epochs
        );
        println!(
            " Separation Accuracy: {:.2}% (Ungewichtet) -> {:.2}% (Trainiertes Mahalanobis-Modell)",
            report.baseline_accuracy, report.trained_accuracy
        );
        println!("------------------------------------------------------------------");
        println!(" Top gelernte Beat/808-Gewichte:");
        for (name, w) in &report.top_beat_features {
            println!("   * {:<34} : {:.3}x", name, w);
        }
        println!(" Top gelernte Vocal/MFCC-Gewichte:");
        for (name, w) in &report.top_vocal_features {
            println!("   * {:<34} : {:.3}x", name, w);
        }
        println!(" Top gelernte Atmo/Harmonik-Gewichte:");
        for (name, w) in &report.top_atmo_features {
            println!("   * {:<34} : {:.3}x", name, w);
        }
        println!("==================================================================");
        return Ok(());
    }

    let query = if args.is_empty() {
        "PIXY - LEGACY".to_string()
    } else {
        args.join(" ")
    };

    println!("==================================================================");
    println!(" CRATEVEC DSP ENGINE | Seed-Query: {}", query);
    println!("==================================================================");

    let engine = CrateVecEngine::new(None);
    let snapshot = engine.start_session(&query).await?;

    for (idx, seed) in snapshot.seeds.iter().enumerate() {
        println!(
            "[SEED #{}] {} - {} ({})",
            idx + 1,
            seed.meta.artist,
            seed.meta.title,
            seed.meta.album
        );
        println!(
            "  DSP Telemetry : {:.1} BPM | Drop @ {:.1}s | Sub-Bass: {:.0}% | 808-Dist: {:.0}% | Haerte/Aggr: {:.0}% | Darkness: {:.0}%",
            seed.dsp.bpm,
            seed.dsp.drop_start_sec,
            seed.dsp.sub_bass_ratio * 100.0,
            seed.dsp.bass_distortion * 100.0,
            seed.dsp.aggression_index * 100.0,
            seed.dsp.darkness_index * 100.0
        );
    }

    println!("------------------------------------------------------------------");
    println!(" Erkannte Vibe-Korridore:");
    for (idx, vc) in snapshot.vibe_corridors.iter().enumerate() {
        println!(
            "   [0{}] {:<26} ({:02} Tracks) | {}",
            idx + 1,
            vc.title,
            vc.track_count,
            vc.sample_artists
        );
    }
    println!("------------------------------------------------------------------");
    println!(
        " Analysierter Kandidaten-Pool: {} Tracks (Top Matches nach DSP & Szene-Konsens)",
        snapshot.pool_size
    );
    println!("------------------------------------------------------------------");

    let mut top = Vec::new();
    if let Some(first) = snapshot.current_card {
        top.push(first);
    }
    top.extend(snapshot.queue_preview);

    for (i, c) in top.iter().enumerate() {
        println!(
            "{:02}. [{:4.1}%] {} - {}",
            i + 1,
            c.total_score,
            c.track.meta.artist,
            c.track.meta.title
        );
        println!(
            "     Beat: {:4.1}% | Vocals: {:4.1}% | Atmo: {:4.1}% | {:.1} BPM | Drop @ {:.1}s | {}",
            c.beat_similarity,
            c.vocal_similarity,
            c.atmo_similarity,
            c.track.dsp.bpm,
            c.track.dsp.drop_start_sec,
            c.track.meta.discovery_channel
        );
    }

    Ok(())
}
