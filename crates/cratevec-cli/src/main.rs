use anyhow::Result;
use cratevec_core::CrateVecEngine;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("cratevec=info,symphonia=error,warn")
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
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
            "  DSP Telemetry : {:.1} BPM | Drop @ {:.1}s | Sub-Bass: {:.0}% | 808-Distortion: {:.0}% | Glitch: {:.0}% | Darkness: {:.0}%",
            seed.dsp.bpm,
            seed.dsp.drop_start_sec,
            seed.dsp.sub_bass_ratio * 100.0,
            seed.dsp.bass_distortion * 100.0,
            seed.dsp.glitch_brightness * 100.0,
            seed.dsp.darkness_index * 100.0
        );
    }

    println!("------------------------------------------------------------------");
    println!(
        " Analysierter Kandidaten-Pool: {} Tracks (Top Matches nach DSP-Vektor)",
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
