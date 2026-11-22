use std::f32::consts::PI;
use std::io::Cursor;

use anyhow::{anyhow, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::models::AcousticFingerprint;

/// Decode an in-memory MP3/AAC/M4A/OGG/WAV buffer into mono f32 PCM samples and sample rate.
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if bytes.len() < 128 {
        return Err(anyhow!("Audio buffer too small ({} bytes)", bytes.len()));
    }

    let mut hint = Hint::new();
    if bytes.len() > 8 && &bytes[4..8] == b"ftyp" {
        hint.with_extension("m4a");
        hint.mime_type("audio/mp4");
    } else if bytes.starts_with(b"OggS") {
        hint.with_extension("ogg");
        hint.mime_type("audio/ogg");
    } else if bytes.starts_with(b"RIFF") {
        hint.with_extension("wav");
        hint.mime_type("audio/wav");
    } else {
        hint.with_extension("mp3");
        hint.mime_type("audio/mpeg");
    }

    // For MP3 streams (such as Deezer CDN 30s previews), strip any ID3v2 header block
    // (which in preview clips often contains truncated/oversized frame headers that cause
    // symphonia-metadata's ID3v2 parser to fail with "out of bounds") and align to the first MPEG sync frame.
    let slice = if bytes.len() > 8 && &bytes[4..8] != b"ftyp" && !bytes.starts_with(b"OggS") && !bytes.starts_with(b"RIFF") {
        strip_id3_and_sync_mp3(bytes)
    } else {
        bytes
    };

    let cursor = Cursor::new(slice.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!("Audio format probe failed (total_len={}, stripped_len={}): {}", bytes.len(), slice.len(), e))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("No audio track found in preview stream"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!("Codec initialization failed: {}", e))?;

    let mut mono_samples: Vec<f32> = Vec::with_capacity(sample_rate as usize * 30);

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::ResetRequired) => break,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => append_mono_samples(&decoded, &mut mono_samples),
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if mono_samples.len() < (sample_rate as usize * 2) {
        return Err(anyhow!(
            "Decoded audio too short ({} samples)",
            mono_samples.len()
        ));
    }

    Ok((mono_samples, sample_rate))
}

fn append_mono_samples(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let channels = b.spec().channels.count().max(1);
            let frames = b.frames();
            for f in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[f];
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let channels = b.spec().channels.count().max(1);
            let frames = b.frames();
            for f in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[f] as f32 / 32768.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S32(b) => {
            let channels = b.spec().channels.count().max(1);
            let frames = b.frames();
            for f in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[f] as f32 / 2147483648.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U8(b) => {
            let channels = b.spec().channels.count().max(1);
            let frames = b.frames();
            for f in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += (b.chan(c)[f] as f32 - 128.0) / 128.0;
                }
                out.push(sum / channels as f32);
            }
        }
        _ => {}
    }
}

/// Full DSP extraction: Drop-Sniper beat drop detection, 64-bar waveform, and 3 centered L2 vectors.
pub fn analyze_audio_bytes(bytes: &[u8]) -> Result<AcousticFingerprint> {
    let (samples, sample_rate) = decode_audio_bytes(bytes)?;
    Ok(analyze_pcm(&samples, sample_rate))
}

pub fn analyze_pcm(samples: &[f32], sample_rate: u32) -> AcousticFingerprint {
    let sr = sample_rate.max(8000) as f32;

    // 1. Peak-normalize signal for level-invariant timbre comparison while preserving raw RMS dB
    let mut global_energy_sq = 0.0f64;
    let mut max_abs = 1e-6f32;
    for &s in samples {
        global_energy_sq += (s as f64) * (s as f64);
        let a = s.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let global_rms = ((global_energy_sq / samples.len().max(1) as f64).sqrt()) as f32;
    let rms_db = 20.0 * (global_rms.max(1e-5)).log10();

    let norm_gain = 1.0 / max_abs.max(0.05);

    // 2. 64-bin waveform peaks for UI rendering
    let num_peaks = 64usize;
    let chunk_size = (samples.len() / num_peaks).max(1);
    let mut waveform_peaks = Vec::with_capacity(num_peaks);
    let mut max_bar = 1e-5f32;
    for i in 0..num_peaks {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(samples.len());
        if start >= end {
            waveform_peaks.push(0.1);
            continue;
        }
        let mut sq = 0.0f32;
        for &s in &samples[start..end] {
            let v = s * norm_gain;
            sq += v * v;
        }
        let bar = (sq / (end - start) as f32).sqrt();
        if bar > max_bar {
            max_bar = bar;
        }
        waveform_peaks.push(bar);
    }
    for bar in &mut waveform_peaks {
        *bar = ((*bar / max_bar).clamp(0.06, 1.0) * 100.0).round() / 100.0;
    }

    // 3. STFT computation (N = 2048, hop = 1024)
    let n_fft = 2048usize;
    let hop = 1024usize;
    let n_bins = n_fft / 2;
    let df = sr / n_fft as f32;

    let hann: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos()))
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_buf = vec![Complex { re: 0.0, im: 0.0 }; n_fft];

    let num_frames = samples.len().saturating_sub(n_fft) / hop + 1;
    let mut frame_rms = Vec::with_capacity(num_frames);
    let mut sub_bass_env = Vec::with_capacity(num_frames); // 20 - 90 Hz
    let mut kick_punch_env = Vec::with_capacity(num_frames); // 90 - 180 Hz
    let mut low_mid_env = Vec::with_capacity(num_frames); // 180 - 360 Hz
    let mut vocal_core_env = Vec::with_capacity(num_frames); // 360 - 2400 Hz
    let mut vocal_bite_env = Vec::with_capacity(num_frames); // 2400 - 4800 Hz
    let mut synth_high_env = Vec::with_capacity(num_frames); // 4800 - 9500 Hz
    let mut glitch_air_env = Vec::with_capacity(num_frames); // 9500 - 16500 Hz

    let mut onset_low_flux = Vec::with_capacity(num_frames);
    let mut onset_high_flux = Vec::with_capacity(num_frames);
    let mut spectral_centroids = Vec::with_capacity(num_frames);
    let mut spectral_flatness_vec = Vec::with_capacity(num_frames);

    // 16 Mel-spaced vocal/mid filter centers (260 Hz .. 5200 Hz)
    let vocal_band_edges: [f32; 17] = [
        260.0, 340.0, 430.0, 540.0, 660.0, 800.0, 960.0, 1150.0, 1370.0, 1620.0, 1920.0, 2260.0,
        2660.0, 3130.0, 3680.0, 4350.0, 5200.0,
    ];
    let mut vocal_mel_sum = [0.0f64; 16];
    let mut vocal_mel_sq = [0.0f64; 16];

    // 12 pitch-class chroma accumulator
    let mut chroma_acc = [0.0f64; 12];
    // 6-octave spectral contrast accumulator
    let mut octave_contrast_acc = [0.0f64; 6];
    let octave_bounds: [(f32, f32); 6] = [
        (40.0, 120.0),
        (120.0, 320.0),
        (320.0, 850.0),
        (850.0, 2200.0),
        (2200.0, 5800.0),
        (5800.0, 14500.0),
    ];

    let mut prev_low_e = 0.0f32;
    let mut prev_high_e = 0.0f32;
    let mut active_frames = 0usize;

    for frame_idx in 0..num_frames {
        let offset = frame_idx * hop;
        if offset + n_fft > samples.len() {
            break;
        }

        let mut f_sq = 0.0f32;
        for i in 0..n_fft {
            let s = samples[offset + i] * norm_gain;
            f_sq += s * s;
            fft_buf[i] = Complex {
                re: s * hann[i],
                im: 0.0,
            };
        }
        let f_rms = (f_sq / n_fft as f32).sqrt();
        frame_rms.push(f_rms);

        fft.process(&mut fft_buf);

        let mut e_sub = 0.0f32;
        let mut e_kick = 0.0f32;
        let mut e_lowmid = 0.0f32;
        let mut e_vcore = 0.0f32;
        let mut e_vbite = 0.0f32;
        let mut e_shigh = 0.0f32;
        let mut e_gair = 0.0f32;

        let mut weighted_freq = 0.0f32;
        let mut total_mag = 1e-6f32;
        let mut log_mag_sum = 0.0f32;
        let mut mag_count = 0usize;

        let mut frame_vocal_mel = [0.0f32; 16];
        let mut mags = vec![0.0f32; n_bins];

        for bin in 1..n_bins {
            let freq = bin as f32 * df;
            if freq > 16800.0 {
                break;
            }
            let c = fft_buf[bin];
            let mag = (c.re * c.re + c.im * c.im).sqrt() / (n_fft as f32);
            mags[bin] = mag;
            let power = mag * mag;

            weighted_freq += freq * mag;
            total_mag += mag;
            if freq >= 100.0 && freq <= 14000.0 {
                log_mag_sum += (mag + 1e-7).ln();
                mag_count += 1;
            }

            if freq < 90.0 {
                e_sub += power;
            } else if freq < 180.0 {
                e_kick += power;
            } else if freq < 360.0 {
                e_lowmid += power;
            } else if freq < 2400.0 {
                e_vcore += power;
            } else if freq < 4800.0 {
                e_vbite += power;
            } else if freq < 9500.0 {
                e_shigh += power;
            } else {
                e_gair += power;
            }

            // Vocal mel bands
            if freq >= vocal_band_edges[0] && freq <= vocal_band_edges[16] {
                for b in 0..16 {
                    if freq >= vocal_band_edges[b] && freq < vocal_band_edges[b + 1] {
                        frame_vocal_mel[b] += power;
                        break;
                    }
                }
            }

            // Chroma accumulation (110 Hz .. 3520 Hz)
            if freq >= 110.0 && freq <= 3520.0 && f_rms > 0.03 {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                let pitch_class = (midi.round() as i32).rem_euclid(12) as usize;
                chroma_acc[pitch_class] += mag as f64;
            }
        }

        let sub_amp = e_sub.sqrt();
        let kick_amp = e_kick.sqrt();
        let lowmid_amp = e_lowmid.sqrt();
        let vcore_amp = e_vcore.sqrt();
        let vbite_amp = e_vbite.sqrt();
        let shigh_amp = e_shigh.sqrt();
        let gair_amp = e_gair.sqrt();

        sub_bass_env.push(sub_amp);
        kick_punch_env.push(kick_amp);
        low_mid_env.push(lowmid_amp);
        vocal_core_env.push(vcore_amp);
        vocal_bite_env.push(vbite_amp);
        synth_high_env.push(shigh_amp);
        glitch_air_env.push(gair_amp);

        let low_e = sub_amp + kick_amp;
        let high_e = shigh_amp + gair_amp + vbite_amp * 0.4;
        onset_low_flux.push((low_e - prev_low_e).max(0.0));
        onset_high_flux.push((high_e - prev_high_e).max(0.0));
        prev_low_e = low_e;
        prev_high_e = high_e;

        let centroid = (weighted_freq / total_mag).clamp(100.0, 12000.0);
        spectral_centroids.push(centroid);

        let geom_mean = (log_mag_sum / mag_count.max(1) as f32).exp();
        let arith_mean = (total_mag / mag_count.max(1) as f32).max(1e-7);
        let flatness = (geom_mean / arith_mean).clamp(0.0, 1.0);
        spectral_flatness_vec.push(flatness);

        if f_rms > 0.035 {
            active_frames += 1;
            let frame_v_tot = frame_vocal_mel.iter().sum::<f32>().max(1e-6);
            for b in 0..16 {
                let log_v = ((frame_vocal_mel[b] / frame_v_tot) + 1e-5).ln() as f64;
                vocal_mel_sum[b] += log_v;
                vocal_mel_sq[b] += log_v * log_v;
            }

            // Octave spectral contrast
            for (o_idx, &(f_lo, f_hi)) in octave_bounds.iter().enumerate() {
                let b_lo = ((f_lo / df).floor() as usize).max(1);
                let b_hi = ((f_hi / df).ceil() as usize).min(n_bins - 1);
                if b_hi > b_lo + 2 {
                    let mut slice: Vec<f32> = mags[b_lo..=b_hi].to_vec();
                    slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let q_idx = (slice.len() / 5).max(1);
                    let valley = slice[..q_idx].iter().sum::<f32>() / q_idx as f32;
                    let peak = slice[slice.len() - q_idx..].iter().sum::<f32>() / q_idx as f32;
                    let contrast = ((peak + 1e-5) / (valley + 1e-5)).ln();
                    octave_contrast_acc[o_idx] += contrast as f64;
                }
            }
        }
    }

    let n_f = frame_rms.len().max(1);
    let act_f = active_frames.max(1) as f64;
    let frames_per_sec = sr / hop as f32;

    // 4. DROP-SNIPER: Locate exact beat drop / chorus entry timestamp
    let drop_start_sec = detect_drop_timestamp(
        &frame_rms,
        &sub_bass_env,
        &kick_punch_env,
        &onset_low_flux,
        frames_per_sec,
    );

    // 5. BPM & Rhythm autocorrelation
    let (bpm, tempo_strength, syncopation_ratio, double_time_ratio) =
        estimate_bpm_and_groove(&onset_low_flux, &onset_high_flux, frames_per_sec);

    // 6. Band energy ratios & 808 distortion analysis
    let mean_sub = mean_f32(&sub_bass_env);
    let mean_kick = mean_f32(&kick_punch_env);
    let mean_lowmid = mean_f32(&low_mid_env);
    let mean_vcore = mean_f32(&vocal_core_env);
    let mean_vbite = mean_f32(&vocal_bite_env);
    let mean_shigh = mean_f32(&synth_high_env);
    let mean_gair = mean_f32(&glitch_air_env);

    let total_band_amp =
        (mean_sub + mean_kick + mean_lowmid + mean_vcore + mean_vbite + mean_shigh + mean_gair)
            .max(1e-5);

    let sub_bass_ratio = (mean_sub + mean_kick * 0.6) / total_band_amp;
    let punch_ratio = mean_kick / total_band_amp;
    let lowmid_ratio = mean_lowmid / total_band_amp;
    let vocal_presence = (mean_vcore + mean_vbite * 0.8) / total_band_amp;
    let vocal_bite_ratio = mean_vbite / (mean_vcore + mean_vbite + 1e-5);
    let synth_high_ratio = mean_shigh / total_band_amp;
    let air_ratio = mean_gair / total_band_amp;

    // 808 harmonic distortion: when sub-bass hits hard, how strong are upper bass harmonics (90-360 Hz)?
    let mut dist_acc = 0.0f32;
    let mut dist_count = 0usize;
    let sub_thresh = mean_sub * 1.15;
    for i in 0..n_f {
        if sub_bass_env[i] > sub_thresh && sub_bass_env[i] > 1e-4 {
            let harm_ratio = (kick_punch_env[i] + low_mid_env[i] * 0.85) / (sub_bass_env[i] + 1e-4);
            dist_acc += harm_ratio;
            dist_count += 1;
        }
    }
    let raw_distortion = if dist_count > 0 {
        dist_acc / dist_count as f32
    } else {
        (mean_kick + mean_lowmid) / (mean_sub + 1e-4)
    };
    let bass_distortion = (raw_distortion / 2.4).clamp(0.0, 1.0);

    let low_flux_density = mean_f32(&onset_low_flux) / (mean_sub + mean_kick + 1e-4);
    let high_flux_density = mean_f32(&onset_high_flux) / (mean_shigh + mean_gair + 1e-4);
    let sub_crest = max_f32(&sub_bass_env) / (mean_sub + 1e-4);
    let rms_variance = std_dev_f32(&frame_rms) / (mean_f32(&frame_rms) + 1e-4);

    let mean_centroid = mean_f32(&spectral_centroids);
    let std_centroid = std_dev_f32(&spectral_centroids);
    let mean_flatness = mean_f32(&spectral_flatness_vec);

    let mut oct_contrasts = [0.0f32; 6];
    for i in 0..6 {
        oct_contrasts[i] = (octave_contrast_acc[i] / act_f) as f32;
    }

    let glitch_brightness = ((synth_high_ratio * 3.2
        + air_ratio * 4.5
        + high_flux_density * 0.9
        + oct_contrasts[4] * 0.08)
        .clamp(0.0, 1.0)
        * 100.0)
        .round()
        / 100.0;

    // Chroma rotation to root pitch (key-invariant harmonic structure)
    let mut root_pc = 0usize;
    let mut max_c = -1.0f64;
    let chroma_sum = chroma_acc.iter().sum::<f64>().max(1e-6);
    for (i, &c) in chroma_acc.iter().enumerate() {
        if c > max_c {
            max_c = c;
            root_pc = i;
        }
    }
    let mut aligned_chroma = [0.0f32; 12];
    for i in 0..12 {
        aligned_chroma[i] = (chroma_acc[(root_pc + i) % 12] / chroma_sum) as f32;
    }
    // Minor / phrygian / dark harmonic intervals (b2 = 1, b3 = 3, tritone = 6, b6 = 8, b7 = 10)
    let dark_harmony = aligned_chroma[1] * 1.3
        + aligned_chroma[3] * 1.4
        + aligned_chroma[6] * 1.2
        + aligned_chroma[8] * 1.2
        + aligned_chroma[10] * 0.9;
    let bright_harmony = aligned_chroma[4] * 1.4 + aligned_chroma[7] * 0.8 + aligned_chroma[9] * 1.1;
    let harmonic_darkness = (dark_harmony / (dark_harmony + bright_harmony + 1e-5)).clamp(0.0, 1.0);

    let darkness_index = (((sub_bass_ratio * 1.6 + harmonic_darkness * 0.7)
        - (mean_centroid / 8000.0) * 0.35)
        .clamp(0.0, 1.0)
        * 100.0)
        .round()
        / 100.0;

    // 7. Construct the 3 centered, z-scored, L2-normalized 16-dimensional vectors

    // --- Vector 1: v_beat (16 dims: BPM, groove, sub-bass pressure, 808 distortion, kick transients) ---
    // Note: For BPM, we include both linear tempo and octave-folded tempo (since 70 BPM and 140 BPM share half-time trap feel)
    let folded_bpm = if bpm < 105.0 { bpm * 2.0 } else { bpm };
    let beat_raw: [(f32, f32, f32); 16] = [
        (folded_bpm, 138.0, 18.0),
        ((folded_bpm * 2.0 * PI / 60.0).sin(), 0.0, 0.7),
        ((folded_bpm * 2.0 * PI / 60.0).cos(), 0.0, 0.7),
        (sub_bass_ratio, 0.22, 0.09),
        (punch_ratio, 0.14, 0.06),
        (lowmid_ratio, 0.14, 0.05),
        (bass_distortion, 0.48, 0.20),
        (low_flux_density, 0.28, 0.11),
        (high_flux_density, 0.32, 0.12),
        (tempo_strength, 0.45, 0.18),
        (syncopation_ratio, 0.48, 0.16),
        (double_time_ratio, 0.52, 0.18),
        (sub_crest, 2.4, 0.7),
        (rms_variance, 0.32, 0.14),
        (oct_contrasts[0], 1.8, 0.6),
        (oct_contrasts[1], 1.9, 0.6),
    ];
    let v_beat = zscore_and_l2_normalize(&beat_raw);

    // --- Vector 2: v_vocal (16 dims: DCT-II MFCCs of the 300-5200 Hz vocal Mel bands + formant/compression metrics) ---
    let mut vocal_log_means = [0.0f32; 16];
    let mut vocal_log_vars = [0.0f32; 16];
    for b in 0..16 {
        let m = (vocal_mel_sum[b] / act_f) as f32;
        let v = ((vocal_mel_sq[b] / act_f) as f32 - m * m).max(0.0).sqrt();
        vocal_log_means[b] = m;
        vocal_log_vars[b] = v;
    }
    // Compute 12 DCT-II cepstral coefficients across the 16 vocal Mel bands
    let mut vocal_cepstrum = [0.0f32; 12];
    for k in 0..12 {
        let mut s = 0.0f32;
        for n in 0..16 {
            s += vocal_log_means[n] * (PI * (k + 1) as f32 * (2 * n + 1) as f32 / 32.0).cos();
        }
        vocal_cepstrum[k] = s * 0.25;
    }
    let vocal_modulation = vocal_log_vars.iter().sum::<f32>() / 16.0;
    let vocal_raw: [(f32, f32, f32); 16] = [
        (vocal_cepstrum[0], -0.4, 0.45),
        (vocal_cepstrum[1], -0.1, 0.35),
        (vocal_cepstrum[2], 0.0, 0.30),
        (vocal_cepstrum[3], 0.0, 0.25),
        (vocal_cepstrum[4], 0.0, 0.22),
        (vocal_cepstrum[5], 0.0, 0.20),
        (vocal_cepstrum[6], 0.0, 0.18),
        (vocal_cepstrum[7], 0.0, 0.16),
        (vocal_cepstrum[8], 0.0, 0.15),
        (vocal_cepstrum[9], 0.0, 0.14),
        (vocal_cepstrum[10], 0.0, 0.13),
        (vocal_cepstrum[11], 0.0, 0.12),
        (vocal_presence, 0.34, 0.10),
        (vocal_bite_ratio, 0.32, 0.11),
        (vocal_modulation, 0.55, 0.18),
        (oct_contrasts[3], 2.1, 0.55),
    ];
    let v_vocal = zscore_and_l2_normalize(&vocal_raw);

    // --- Vector 3: v_atmo (16 dims: Harmonic intervals, spectral contrast, synth flatness, high-freq glitch texture) ---
    let atmo_raw: [(f32, f32, f32); 16] = [
        (harmonic_darkness, 0.52, 0.14),
        (aligned_chroma[1] + aligned_chroma[2], 0.15, 0.06), // 2nd intervals (tension)
        (aligned_chroma[3], 0.09, 0.04),                     // Minor 3rd
        (aligned_chroma[4], 0.08, 0.04),                     // Major 3rd
        (aligned_chroma[5] + aligned_chroma[7], 0.19, 0.06), // 4th + 5th power intervals
        (aligned_chroma[6], 0.06, 0.03),                     // Tritone dissonance
        (aligned_chroma[8] + aligned_chroma[10], 0.16, 0.05), // Minor 6th + Minor 7th
        (mean_centroid / 1000.0, 2.3, 0.85),
        (std_centroid / 1000.0, 0.75, 0.32),
        (mean_flatness, 0.18, 0.09),
        (synth_high_ratio, 0.09, 0.045),
        (air_ratio, 0.04, 0.025),
        (oct_contrasts[2], 2.0, 0.5),
        (oct_contrasts[4], 2.2, 0.6),
        (oct_contrasts[5], 2.3, 0.65),
        (darkness_index, 0.52, 0.16),
    ];
    let v_atmo = zscore_and_l2_normalize(&atmo_raw);

    AcousticFingerprint {
        bpm: (bpm * 10.0).round() / 10.0,
        drop_start_sec,
        rms_db: (rms_db * 10.0).round() / 10.0,
        sub_bass_ratio: (sub_bass_ratio * 100.0).round() / 100.0,
        bass_distortion: (bass_distortion * 100.0).round() / 100.0,
        vocal_presence: (vocal_presence * 100.0).round() / 100.0,
        glitch_brightness,
        darkness_index,
        waveform_peaks,
        v_beat,
        v_vocal,
        v_atmo,
    }
}

fn detect_drop_timestamp(
    frame_rms: &[f32],
    sub_bass: &[f32],
    kick_punch: &[f32],
    onset_low: &[f32],
    frames_per_sec: f32,
) -> f32 {
    let n = frame_rms.len();
    let min_frame = (1.2 * frames_per_sec) as usize;
    let max_frame = ((20.5 * frames_per_sec) as usize).min(n.saturating_sub(40));
    if min_frame >= max_frame || n < 60 {
        return 0.0;
    }

    let lookahead = (2.8 * frames_per_sec) as usize;
    let lookbehind = (1.3 * frames_per_sec) as usize;

    let mut best_frame = min_frame;
    let mut best_score = f32::MIN;

    for f in min_frame..max_frame {
        let fwd_end = (f + lookahead).min(n);
        let bwd_start = f.saturating_sub(lookbehind);

        let fwd_len = (fwd_end - f).max(1) as f32;
        let bwd_len = (f - bwd_start).max(1) as f32;

        let mut fwd_energy = 0.0f32;
        for i in f..fwd_end {
            fwd_energy +=
                frame_rms[i] * 1.1 + sub_bass[i] * 1.8 + kick_punch[i] * 1.5 + onset_low[i] * 2.2;
        }
        fwd_energy /= fwd_len;

        let mut bwd_energy = 0.0f32;
        for i in bwd_start..f {
            bwd_energy += frame_rms[i] * 1.1 + sub_bass[i] * 1.5 + kick_punch[i] * 1.2;
        }
        bwd_energy /= bwd_len;

        let jump = (fwd_energy - bwd_energy * 0.72).max(0.0);
        let score = fwd_energy * 0.65 + jump * 0.55;
        if score > best_score {
            best_score = score;
            best_frame = f;
        }
    }

    let sec = (best_frame as f32 / frames_per_sec).clamp(0.0, 20.0);
    (sec * 10.0).round() / 10.0
}

fn estimate_bpm_and_groove(
    onset_low: &[f32],
    onset_high: &[f32],
    frames_per_sec: f32,
) -> (f32, f32, f32, f32) {
    let n = onset_low.len();
    if n < 80 {
        return (130.0, 0.5, 0.5, 0.5);
    }

    let mut combined = Vec::with_capacity(n);
    for i in 0..n {
        combined.push(onset_low[i] * 1.4 + onset_high[i] * 0.6);
    }
    let mean_c = mean_f32(&combined);
    for v in &mut combined {
        *v = (*v - mean_c).max(0.0);
    }

    let min_bpm = 88.0f32;
    let max_bpm = 180.0f32;
    let min_lag = ((60.0 / max_bpm) * frames_per_sec).round() as usize;
    let max_lag = ((60.0 / min_bpm) * frames_per_sec).round() as usize;

    let mut zero_lag_e = 1e-5f32;
    for &v in &combined {
        zero_lag_e += v * v;
    }

    let mut best_lag = ((60.0 / 138.0) * frames_per_sec).round() as usize;
    let mut best_corr = -1.0f32;

    for lag in min_lag.max(4)..=max_lag.min(n / 2) {
        let mut corr = 0.0f32;
        for i in 0..(n - lag) {
            corr += combined[i] * combined[i + lag];
        }
        // Also reinforce 2x lag (two-beat phrase consistency)
        if lag * 2 < n {
            let mut corr2 = 0.0f32;
            for i in 0..(n - lag * 2) {
                corr2 += combined[i] * combined[i + lag * 2];
            }
            corr = corr * 0.7 + corr2 * 0.3;
        }
        let norm_corr = corr / zero_lag_e;
        if norm_corr > best_corr {
            best_corr = norm_corr;
            best_lag = lag;
        }
    }

    let bpm = (60.0 * frames_per_sec / best_lag.max(1) as f32).clamp(85.0, 185.0);
    let tempo_strength = best_corr.clamp(0.05, 1.0);

    // Check off-beat / triplet syncopation energy at 0.5x and 0.75x lag
    let half_lag = (best_lag / 2).max(2);
    let triplet_lag = ((best_lag * 3) / 4).max(2);
    let mut half_corr = 0.0f32;
    let mut trip_corr = 0.0f32;
    for i in 0..(n - best_lag) {
        half_corr += combined[i] * combined[i + half_lag];
        trip_corr += combined[i] * combined[i + triplet_lag];
    }
    let double_time_ratio = (half_corr / (zero_lag_e * tempo_strength + 1e-5)).clamp(0.0, 1.0);
    let syncopation_ratio = (trip_corr / (zero_lag_e * tempo_strength + 1e-5)).clamp(0.0, 1.0);

    (bpm, tempo_strength, syncopation_ratio, double_time_ratio)
}

fn zscore_and_l2_normalize(features: &[(f32, f32, f32)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(features.len());
    let mut norm_sq = 0.0f32;
    for &(val, mean, std) in features {
        let z = ((val - mean) / std.max(1e-4)).clamp(-3.2, 3.2);
        norm_sq += z * z;
        out.push(z);
    }
    let norm = norm_sq.sqrt().max(1e-5);
    for v in &mut out {
        *v /= norm;
    }
    out
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-6);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Convert raw centered cosine similarity (-1.0 .. +1.0) into an intuitive 0..100% match score.
pub fn cosine_to_percentage(cos: f32) -> f32 {
    // With zero-mean z-scored 16-D vectors, random tracks average cos ~ 0.0 (50%),
    // acoustically aligned tracks sit around cos 0.35 .. 0.95 (75% .. 99%).
    let mapped = 50.0 + 50.0 * (cos * 1.35).tanh();
    (mapped.clamp(1.0, 99.6) * 10.0).round() / 10.0
}

fn mean_f32(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().sum::<f32>() / slice.len() as f32
}

fn max_f32(slice: &[f32]) -> f32 {
    slice.iter().copied().fold(0.0f32, f32::max)
}

fn std_dev_f32(slice: &[f32]) -> f32 {
    if slice.len() < 2 {
        return 0.0;
    }
    let m = mean_f32(slice);
    let var = slice.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / slice.len() as f32;
    var.sqrt()
}

fn strip_id3_and_sync_mp3(bytes: &[u8]) -> &[u8] {
    let mut offset = 0usize;
    while bytes.len() > offset + 10 && &bytes[offset..offset + 3] == b"ID3" {
        let size = (((bytes[offset + 6] & 0x7F) as usize) << 21)
            | (((bytes[offset + 7] & 0x7F) as usize) << 14)
            | (((bytes[offset + 8] & 0x7F) as usize) << 7)
            | ((bytes[offset + 9] & 0x7F) as usize);
        let next = offset + 10 + size;
        if next + 128 < bytes.len() {
            offset = next;
        } else {
            offset += 10;
            break;
        }
    }
    let remaining = &bytes[offset..];
    if let Some(sync_off) = find_mp3_sync_offset(remaining) {
        &remaining[sync_off..]
    } else {
        remaining
    }
}

fn find_mp3_sync_offset(bytes: &[u8]) -> Option<usize> {
    let limit = bytes.len().min(65536).saturating_sub(8);
    for i in 0..limit {
        if bytes[i] == 0xFF && (bytes[i + 1] & 0xE0) == 0xE0 {
            let version_id = (bytes[i + 1] >> 3) & 0x03;
            let layer = (bytes[i + 1] >> 1) & 0x03;
            let bitrate_idx = ((bytes[i + 2] >> 4) & 0x0F) as usize;
            let sr_idx = ((bytes[i + 2] >> 2) & 0x03) as usize;
            let padding = ((bytes[i + 2] >> 1) & 0x01) as usize;

            if version_id != 1 && layer == 1 && bitrate_idx != 0 && bitrate_idx != 15 && sr_idx != 3 {
                // Verify MPEG-1 Layer III frame length points to the next sync word
                if version_id == 3 {
                    const BITRATES: [usize; 16] = [
                        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
                    ];
                    const SAMPLE_RATES: [usize; 4] = [44100, 48000, 32000, 1];
                    let frame_len = (144 * BITRATES[bitrate_idx] * 1000) / SAMPLE_RATES[sr_idx] + padding;
                    if i + frame_len + 2 < bytes.len() {
                        if bytes[i + frame_len] == 0xFF && (bytes[i + frame_len + 1] & 0xE0) == 0xE0 {
                            return Some(i);
                        }
                    }
                } else {
                    return Some(i);
                }
            }
        }
    }
    None
}
