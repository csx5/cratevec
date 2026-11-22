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

    let slice = if bytes.len() > 8
        && &bytes[4..8] != b"ftyp"
        && !bytes.starts_with(b"OggS")
        && !bytes.starts_with(b"RIFF")
    {
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
        .map_err(|e| {
            anyhow!(
                "Audio format probe failed (total_len={}, stripped_len={}): {}",
                bytes.len(),
                slice.len(),
                e
            )
        })?;

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

    // 3. Pass 1: STFT & Per-Frame Feature Extraction (N = 2048, hop = 1024)
    let n_fft = 2048usize;
    let hop = 1024usize;
    let n_bins = n_fft / 2;
    let df = sr / n_fft as f32;
    let frames_per_sec = sr / hop as f32;

    let hann: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft - 1) as f32).cos()))
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_buf = vec![Complex { re: 0.0, im: 0.0 }; n_fft];

    // 18 Mel-spaced control frequencies for 16 overlapping triangular vocal filters (240 Hz .. 5400 Hz)
    let vocal_centers: [f32; 18] = [
        240.0, 310.0, 395.0, 495.0, 610.0, 745.0, 900.0, 1080.0, 1290.0, 1535.0, 1820.0, 2150.0,
        2535.0, 2985.0, 3500.0, 4100.0, 4720.0, 5400.0,
    ];
    let octave_bounds: [(f32, f32); 6] = [
        (40.0, 120.0),
        (120.0, 320.0),
        (320.0, 850.0),
        (850.0, 2200.0),
        (2200.0, 5800.0),
        (5800.0, 14500.0),
    ];

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

    let mut frame_vocal_logs: Vec<[f32; 16]> = Vec::with_capacity(num_frames);
    let mut frame_chromas: Vec<[f32; 12]> = Vec::with_capacity(num_frames);
    let mut frame_contrasts: Vec<[f32; 6]> = Vec::with_capacity(num_frames);

    let mut prev_low_e = 0.0f32;
    let mut prev_high_e = 0.0f32;
    let mut mags = vec![0.0f32; n_bins];

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

        let mut v_mel = [0.0f32; 16];
        let mut f_chroma = [0.0f32; 12];

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

            // Triangular overlapping Mel filterbank in vocal band
            if freq >= vocal_centers[0] && freq <= vocal_centers[17] {
                for b in 0..16 {
                    let f_left = vocal_centers[b];
                    let f_mid = vocal_centers[b + 1];
                    let f_right = vocal_centers[b + 2];
                    if freq >= f_left && freq <= f_right {
                        let tri_w = if freq <= f_mid {
                            (freq - f_left) / (f_mid - f_left).max(1.0)
                        } else {
                            (f_right - freq) / (f_right - f_mid).max(1.0)
                        };
                        v_mel[b] += power * tri_w.max(0.0);
                    }
                }
            }

            // Octave-normalized (1/f weighted) chroma accumulation (100 Hz .. 3520 Hz)
            if freq >= 100.0 && freq <= 3520.0 {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                let pitch_class = (midi.round() as i32).rem_euclid(12) as usize;
                let oct_norm_weight = 220.0 / freq;
                f_chroma[pitch_class] += mag * oct_norm_weight;
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

        // Normalize frame vocal log-Mel and mean-center across the 16 bands
        let v_tot = v_mel.iter().sum::<f32>().max(1e-6);
        let mut v_logs = [0.0f32; 16];
        let mut l_mean = 0.0f32;
        for b in 0..16 {
            let lv = ((v_mel[b] / v_tot) + 1e-5).ln();
            v_logs[b] = lv;
            l_mean += lv;
        }
        l_mean /= 16.0;
        for b in 0..16 {
            v_logs[b] -= l_mean;
        }
        frame_vocal_logs.push(v_logs);
        frame_chromas.push(f_chroma);

        // 6-octave spectral contrast per frame
        let mut f_oct = [0.0f32; 6];
        for (o_idx, &(f_lo, f_hi)) in octave_bounds.iter().enumerate() {
            let b_lo = ((f_lo / df).floor() as usize).max(1);
            let b_hi = ((f_hi / df).ceil() as usize).min(n_bins - 1);
            if b_hi > b_lo + 2 {
                let mut slice: Vec<f32> = mags[b_lo..=b_hi].to_vec();
                slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q_idx = (slice.len() / 5).max(1);
                let valley = slice[..q_idx].iter().sum::<f32>() / q_idx as f32;
                let peak = slice[slice.len() - q_idx..].iter().sum::<f32>() / q_idx as f32;
                f_oct[o_idx] = ((peak + 1e-8) / (valley + 1e-8)).ln().clamp(0.2, 4.5);
            }
        }
        frame_contrasts.push(f_oct);
    }

    let n_f = frame_rms.len().max(1);

    // 4. DROP-SNIPER FIRST: Locate exact beat drop / chorus entry timestamp (or 0.0s if already in chorus)
    let drop_start_sec = detect_drop_timestamp(
        &frame_rms,
        &sub_bass_env,
        &kick_punch_env,
        &onset_low_flux,
        frames_per_sec,
    );
    let drop_frame = ((drop_start_sec * frames_per_sec).round() as usize).min(n_f.saturating_sub(20));

    // 5. Drop-Gated Temporal Pooling: weight post-drop frames w_t = 1.0, pre-drop intro frames w_t = 0.20
    let mut w_sum = 0.0f32;
    let mut mean_sub = 0.0f32;
    let mut mean_kick = 0.0f32;
    let mut mean_lowmid = 0.0f32;
    let mut mean_vcore = 0.0f32;
    let mut mean_vbite = 0.0f32;
    let mut mean_shigh = 0.0f32;
    let mut mean_gair = 0.0f32;
    let mut mean_low_flux = 0.0f32;
    let mut mean_high_flux = 0.0f32;
    let mut mean_centroid = 0.0f32;
    let mut mean_flatness = 0.0f32;

    let mut vocal_mel_sum = [0.0f64; 16];
    let mut vocal_mel_sq = [0.0f64; 16];
    let mut chroma_acc = [0.0f64; 12];
    let mut oct_contrasts = [0.0f32; 6];
    let mut active_weight_sum = 0.0f64;

    // Also compute frame-wise first 4 MFCCs for Delta-MFCC articulation energy
    let mut prev2_mfcc4 = [0.0f32; 4];
    let mut prev1_mfcc4 = [0.0f32; 4];
    let mut delta_mfcc_sq = [0.0f64; 4];
    let mut delta_weight_sum = 0.0f64;

    for t in 0..n_f {
        let wt = if t >= drop_frame { 1.0f32 } else { 0.20f32 };
        w_sum += wt;
        mean_sub += wt * sub_bass_env[t];
        mean_kick += wt * kick_punch_env[t];
        mean_lowmid += wt * low_mid_env[t];
        mean_vcore += wt * vocal_core_env[t];
        mean_vbite += wt * vocal_bite_env[t];
        mean_shigh += wt * synth_high_env[t];
        mean_gair += wt * glitch_air_env[t];
        mean_low_flux += wt * onset_low_flux[t];
        mean_high_flux += wt * onset_high_flux[t];
        mean_centroid += wt * spectral_centroids[t];
        mean_flatness += wt * spectral_flatness_vec[t];

        if frame_rms[t] > 0.032 {
            let wd = wt as f64;
            active_weight_sum += wd;
            let v_logs = &frame_vocal_logs[t];
            for b in 0..16 {
                let lv = v_logs[b] as f64;
                vocal_mel_sum[b] += wd * lv;
                vocal_mel_sq[b] += wd * lv * lv;
            }

            // Compute first 4 DCT-II coefficients for this frame to measure temporal Delta-MFCCs
            let mut cur_mfcc4 = [0.0f32; 4];
            for k in 0..4 {
                let mut s = 0.0f32;
                for n in 0..16 {
                    s += v_logs[n] * (PI * (k + 1) as f32 * (2 * n + 1) as f32 / 32.0).cos();
                }
                cur_mfcc4[k] = s * 0.25;
            }
            if t >= 2 {
                delta_weight_sum += wd;
                for k in 0..4 {
                    let d = (cur_mfcc4[k] - prev2_mfcc4[k]) as f64;
                    delta_mfcc_sq[k] += wd * d * d;
                }
            }
            prev2_mfcc4 = prev1_mfcc4;
            prev1_mfcc4 = cur_mfcc4;

            // Tonality-weighted chroma accumulation (suppresses broadband drum transients)
            let tonality_w = (1.0 - spectral_flatness_vec[t]).clamp(0.05, 1.0);
            let chroma_w = wd * (tonality_w * tonality_w) as f64;
            for pc in 0..12 {
                chroma_acc[pc] += chroma_w * (frame_chromas[t][pc] as f64);
            }

            for o in 0..6 {
                oct_contrasts[o] += wt * frame_contrasts[t][o];
            }
        }
    }

    let w_norm = w_sum.max(1e-5);
    mean_sub /= w_norm;
    mean_kick /= w_norm;
    mean_lowmid /= w_norm;
    mean_vcore /= w_norm;
    mean_vbite /= w_norm;
    mean_shigh /= w_norm;
    mean_gair /= w_norm;
    mean_low_flux /= w_norm;
    mean_high_flux /= w_norm;
    mean_centroid /= w_norm;
    mean_flatness /= w_norm;

    let act_w = active_weight_sum.max(1e-5);
    for o in 0..6 {
        oct_contrasts[o] /= act_w as f32;
    }

    // 6. Parabolic Sub-Frame BPM & Groove Autocorrelation (evaluated primarily on post-drop frames)
    let post_low = if drop_frame + 120 < n_f {
        &onset_low_flux[drop_frame..]
    } else {
        &onset_low_flux[..]
    };
    let post_high = if drop_frame + 120 < n_f {
        &onset_high_flux[drop_frame..]
    } else {
        &onset_high_flux[..]
    };
    let (bpm, tempo_strength, syncopation_ratio, double_time_ratio) =
        estimate_bpm_and_groove(post_low, post_high, frames_per_sec);

    // 7. Band ratios & Sub-Bass Presence-Gated 808 Distortion
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

    // 808 harmonic distortion & low-end grit gated by true sub-bass presence (20-90 Hz)
    let mut dist_acc = 0.0f32;
    let mut dist_count = 0usize;
    let sub_thresh = mean_sub * 1.10;
    for i in drop_frame..n_f {
        if sub_bass_env[i] > sub_thresh && sub_bass_env[i] > 1e-4 {
            let harm_ratio =
                (kick_punch_env[i] + low_mid_env[i] * 0.90) / (sub_bass_env[i] + 1e-4);
            dist_acc += harm_ratio;
            dist_count += 1;
        }
    }
    let raw_distortion = if dist_count > 0 {
        dist_acc / dist_count as f32
    } else {
        (mean_kick + mean_lowmid) / (mean_sub + 1e-4)
    };
    let sub_presence_gate = ((sub_bass_ratio - 0.22) / 0.16).clamp(0.0, 1.0);
    let bass_distortion =
        (sub_presence_gate * ((raw_distortion - 0.18) / 0.95).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    let low_flux_density = mean_low_flux / (mean_sub + mean_kick + 1e-4);
    let high_flux_density = mean_high_flux / (mean_shigh + mean_gair + 1e-4);
    let sub_crest = max_f32(&sub_bass_env[drop_frame..]) / (mean_sub + 1e-4);
    let std_centroid = std_dev_f32(&spectral_centroids);

    let glitch_brightness = ((synth_high_ratio * 2.8
        + air_ratio * 3.8
        + high_flux_density * 0.75
        + (oct_contrasts[4] - 1.5).max(0.0) * 0.12)
        .clamp(0.0, 1.0)
        * 100.0)
        .round()
        / 100.0;

    // 8. Triad-Correlation Root Detection & Key-Invariant Chroma Structure + Chroma Entropy
    let chroma_sum = chroma_acc.iter().sum::<f64>().max(1e-6);
    let mut norm_chroma = [0.0f32; 12];
    for i in 0..12 {
        norm_chroma[i] = (chroma_acc[i] / chroma_sum) as f32;
    }

    let mut root_pc = 0usize;
    let mut best_triad_score = -1.0f32;
    for r in 0..12 {
        let tonic = norm_chroma[r];
        let fifth = norm_chroma[(r + 7) % 12];
        let third = norm_chroma[(r + 3) % 12].max(norm_chroma[(r + 4) % 12]);
        let score = 1.0 * tonic + 0.55 * fifth + 0.45 * third;
        if score > best_triad_score {
            best_triad_score = score;
            root_pc = r;
        }
    }

    let mut aligned_chroma = [0.0f32; 12];
    let mut chroma_entropy = 0.0f32;
    for i in 0..12 {
        let p = norm_chroma[(root_pc + i) % 12];
        aligned_chroma[i] = p;
        if p > 1e-5 {
            chroma_entropy -= p * p.ln();
        }
    }
    chroma_entropy = (chroma_entropy / (12.0f32).ln()).clamp(0.0, 1.0);

    let dark_harmony = aligned_chroma[1] * 1.35
        + aligned_chroma[3] * 1.40
        + aligned_chroma[6] * 1.30
        + aligned_chroma[8] * 1.20
        + aligned_chroma[10] * 0.85;
    let bright_harmony =
        aligned_chroma[4] * 1.55 + aligned_chroma[7] * 0.90 + aligned_chroma[9] * 1.25;
    let harmonic_darkness =
        (dark_harmony / (dark_harmony + bright_harmony + 1e-5)).clamp(0.0, 1.0);

    // Calibrated Darkness Index (0.0 = bright brass/party/major pop .. 1.0 = dark menacing sub-bass & minor/tritone)
    let sub_to_warm_body = (mean_sub / (mean_lowmid * 1.4 + mean_vcore * 0.7 + 1e-5)).clamp(0.2, 2.8);
    let sub_dark_score = ((sub_to_warm_body - 0.65) / 1.45).clamp(0.0, 1.0);
    let harm_dark_score = ((harmonic_darkness - 0.38) / 0.30).clamp(0.0, 1.0);
    let darkness_index = ((0.48 * sub_dark_score + 0.52 * harm_dark_score).clamp(0.04, 0.98)
        * 100.0)
        .round()
        / 100.0;

    // Calibrated Aggression & Hardness Index (0.0 = calm/chill/soft .. 1.0 = aggressive/hard-hitting/intense)
    // Normalized against total_band_amp (never dividing by mean_sub or mean_vcore, which previously penalized
    // heavy sub-bass beats and deep chest-voice rappers!).
    let dw = delta_weight_sum.max(1e-5);
    let d_mfcc0 = (delta_mfcc_sq[0] / dw).sqrt() as f32;
    let d_mfcc1 = (delta_mfcc_sq[1] / dw).sqrt() as f32;
    let d_mfcc2 = (delta_mfcc_sq[2] / dw).sqrt() as f32;
    let d_mfcc3 = (delta_mfcc_sq[3] / dw).sqrt() as f32;

    let abs_low_attack = (mean_low_flux / total_band_amp) * 4.2;
    let abs_high_attack = (mean_high_flux / total_band_amp) * 3.5;
    let abs_vocal_bite = (mean_vbite / total_band_amp) * 4.5;

    let bass_pressure_aggr =
        (((sub_bass_ratio * 1.15 + punch_ratio * 1.6 + bass_distortion * 0.75) - 0.42) / 0.40)
            .clamp(0.0, 1.0);
    let transient_aggr =
        (((abs_low_attack * 1.35 + abs_high_attack * 0.65) - 0.16) / 0.36).clamp(0.0, 1.0);
    let vocal_attack_aggr =
        (((abs_vocal_bite + (d_mfcc0 + d_mfcc1) * 2.2) - 0.22) / 0.42).clamp(0.0, 1.0);

    let aggression_index = ((0.38 * bass_pressure_aggr
        + 0.32 * transient_aggr
        + 0.30 * vocal_attack_aggr)
        .clamp(0.04, 0.98)
        * 100.0)
        .round()
        / 100.0;

    // 9. Construct the 3 centered, z-scored, L2-normalized 16-dimensional vectors
    // Empirically calibrated (mean, std) so zero-mean z-scores span [-2.5, +2.5] without saturation!

    // --- Vector 1: v_beat (16 dims) ---
    let oct_pos = (bpm / 95.0).log2().rem_euclid(1.0);
    let bpm_oct = 95.0 * 2.0f32.powf(oct_pos);
    let bpm_phase = 2.0 * PI * bpm.log2();

    let beat_raw: [(f32, f32, f32); 16] = [
        (bpm_oct, 134.0, 22.0),
        (bpm_phase.sin(), 0.0, 0.707),
        (bpm_phase.cos(), 0.0, 0.707),
        (sub_bass_ratio, 0.42, 0.11),
        (punch_ratio, 0.16, 0.048),
        (lowmid_ratio, 0.15, 0.045),
        (bass_distortion, 0.32, 0.22),
        (low_flux_density, 0.20, 0.075),
        (high_flux_density, 0.24, 0.085),
        (tempo_strength, 0.44, 0.16),
        (syncopation_ratio, 0.48, 0.16),
        (double_time_ratio, 0.52, 0.18),
        (sub_crest, 2.25, 0.60),
        (aggression_index, 0.50, 0.20),
        (oct_contrasts[0], 1.65, 0.55),
        (oct_contrasts[1], 1.85, 0.55),
    ];
    let v_beat = zscore_and_l2_normalize(&beat_raw);

    // --- Vector 2: v_vocal (16 dims) ---
    let mut vocal_log_means = [0.0f32; 16];
    let mut vocal_log_vars = [0.0f32; 16];
    for b in 0..16 {
        let m = (vocal_mel_sum[b] / act_w) as f32;
        let v = ((vocal_mel_sq[b] / act_w) as f32 - m * m).max(0.0).sqrt();
        vocal_log_means[b] = m;
        vocal_log_vars[b] = v;
    }
    let mut vocal_cepstrum = [0.0f32; 8];
    for k in 0..8 {
        let mut s = 0.0f32;
        for n in 0..16 {
            s += vocal_log_means[n] * (PI * (k + 1) as f32 * (2 * n + 1) as f32 / 32.0).cos();
        }
        vocal_cepstrum[k] = s * 0.25;
    }

    let vocal_modulation = vocal_log_vars.iter().sum::<f32>() / 16.0;
    let vocal_raw: [(f32, f32, f32); 16] = [
        (vocal_cepstrum[0], 0.35, 0.38),
        (vocal_cepstrum[1], -0.12, 0.24),
        (vocal_cepstrum[2], -0.04, 0.16),
        (vocal_cepstrum[3], 0.0, 0.12),
        (vocal_cepstrum[4], 0.0, 0.10),
        (vocal_cepstrum[5], 0.0, 0.08),
        (vocal_cepstrum[6], 0.0, 0.07),
        (vocal_cepstrum[7], 0.0, 0.06),
        (d_mfcc0, 0.11, 0.045),
        (d_mfcc1, 0.08, 0.035),
        (d_mfcc2, 0.06, 0.028),
        (d_mfcc3, 0.05, 0.024),
        (vocal_presence, 0.24, 0.075),
        (vocal_bite_ratio, 0.26, 0.085),
        (vocal_modulation, 0.36, 0.11),
        (aggression_index, 0.50, 0.20),
    ];
    let v_vocal = zscore_and_l2_normalize(&vocal_raw);

    // --- Vector 3: v_atmo (16 dims) ---
    let atmo_raw: [(f32, f32, f32); 16] = [
        (harmonic_darkness, 0.53, 0.085),
        (aligned_chroma[1] + aligned_chroma[2], 0.15, 0.042),
        (aligned_chroma[3], 0.09, 0.032),
        (aligned_chroma[4], 0.085, 0.032),
        (aligned_chroma[5] + aligned_chroma[7], 0.19, 0.045),
        (aligned_chroma[6], 0.065, 0.025),
        (aligned_chroma[8] + aligned_chroma[10], 0.16, 0.040),
        (mean_centroid / 1000.0, 1.95, 0.65),
        (std_centroid / 1000.0, 0.65, 0.25),
        (mean_flatness, 0.16, 0.075),
        (synth_high_ratio, 0.07, 0.035),
        (air_ratio, 0.035, 0.022),
        (oct_contrasts[2], 1.95, 0.48),
        (oct_contrasts[4], 2.15, 0.52),
        (darkness_index, 0.52, 0.20),
        (chroma_entropy, 0.93, 0.045),
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
        aggression_index,
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

    // Check if the preview already starts directly in the chorus/drop at t = 0.0s
    let mut composite_e: Vec<f32> = (0..n)
        .map(|i| frame_rms[i] * 1.1 + sub_bass[i] * 1.8 + kick_punch[i] * 1.5)
        .collect();
    let init_end = ((1.8 * frames_per_sec) as usize).min(n);
    let init_energy = mean_f32(&composite_e[..init_end]);

    composite_e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p85_energy = composite_e[(n * 85 / 100).min(n - 1)].max(1e-4);

    let lookahead = (2.8 * frames_per_sec) as usize;
    let lookbehind = (1.4 * frames_per_sec) as usize;

    let mut best_frame = min_frame;
    let mut best_score = f32::MIN;
    let mut max_rel_jump = 1.0f32;

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
            bwd_energy += frame_rms[i] * 1.1 + sub_bass[i] * 1.6 + kick_punch[i] * 1.3;
        }
        bwd_energy /= bwd_len;

        let rel_jump = fwd_energy / (bwd_energy + 0.04);
        if rel_jump > max_rel_jump {
            max_rel_jump = rel_jump;
        }

        let jump = (fwd_energy - bwd_energy * 0.82).max(0.0) * rel_jump.clamp(0.8, 2.5);
        let score = fwd_energy * 0.38 + jump * 0.62;
        if score > best_score {
            best_score = score;
            best_frame = f;
        }
    }

    // If the clip starts at >= 82% of P85 energy and has no pronounced build-up jump (> 1.42x), start at 0.0s
    if init_energy >= 0.82 * p85_energy && max_rel_jump < 1.42 {
        return 0.0;
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
        combined.push(onset_low[i] * 1.45 + onset_high[i] * 0.55);
    }
    let mean_c = mean_f32(&combined);
    for v in &mut combined {
        *v = (*v - mean_c).max(0.0);
    }

    let min_bpm = 84.0f32;
    let max_bpm = 184.0f32;
    let min_lag = ((60.0 / max_bpm) * frames_per_sec).floor() as usize;
    let max_lag = ((60.0 / min_bpm) * frames_per_sec).ceil() as usize;

    let mut zero_lag_e = 1e-5f32;
    for &v in &combined {
        zero_lag_e += v * v;
    }

    let max_eval_lag = (max_lag * 2 + 2).min(n / 2);
    let mut raw_corr = vec![0.0f32; max_eval_lag + 1];
    for lag in min_lag.max(3)..=max_eval_lag {
        let mut c = 0.0f32;
        for i in 0..(n - lag) {
            c += combined[i] * combined[i + lag];
        }
        raw_corr[lag] = c / zero_lag_e;
    }

    // Harmonic comb correlation with gentle log-Gaussian prior centered at 130 BPM
    let lo = min_lag.max(4);
    let hi = max_lag.min(max_eval_lag);
    let mut comb_corr = vec![0.0f32; hi + 1];
    let mut best_lag = ((60.0 / 135.0) * frames_per_sec).round() as usize;
    let mut best_score = -1.0f32;

    for lag in lo..=hi {
        let r1 = raw_corr[lag];
        let r2 = if lag * 2 <= max_eval_lag {
            raw_corr[lag * 2]
        } else {
            r1
        };
        let cand_bpm = 60.0 * frames_per_sec / lag as f32;
        let log_prior = (-0.5 * ((cand_bpm / 130.0).log2() / 0.65).powi(2)).exp();
        let score = (0.68 * r1 + 0.32 * r2) * (0.85 + 0.15 * log_prior);
        comb_corr[lag] = score;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    // Parabolic sub-frame interpolation around best_lag to remove integer-lag BPM quantization
    let interp_lag = if best_lag > lo && best_lag < hi {
        let y0 = comb_corr[best_lag - 1];
        let y1 = comb_corr[best_lag];
        let y2 = comb_corr[best_lag + 1];
        let denom = y0 - 2.0 * y1 + y2;
        if denom.abs() > 1e-6 {
            let delta = (0.5 * (y0 - y2) / denom).clamp(-0.48, 0.48);
            best_lag as f32 + delta
        } else {
            best_lag as f32
        }
    } else {
        best_lag as f32
    };

    let bpm = (60.0 * frames_per_sec / interp_lag.max(1.0)).clamp(84.0, 184.0);
    let tempo_strength = raw_corr.get(best_lag).copied().unwrap_or(0.5).clamp(0.05, 1.0);

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
    for &(val, mean, std) in features {
        let z = (val - mean) / std.max(1e-4);
        out.push(z);
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

/// Maps zero-mean pool-whitened cosine similarity (-1.0 .. +1.0) into a high-contrast 0..100% score.
pub fn cosine_to_percentage(cos: f32) -> f32 {
    let mapped = 54.0 + 45.0 * (cos * 1.65).tanh();
    (mapped.clamp(5.0, 99.4) * 10.0).round() / 10.0
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

            if version_id != 1
                && layer == 1
                && bitrate_idx != 0
                && bitrate_idx != 15
                && sr_idx != 3
            {
                if version_id == 3 {
                    const BITRATES: [usize; 16] = [
                        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
                    ];
                    const SAMPLE_RATES: [usize; 4] = [44100, 48000, 32000, 1];
                    let frame_len =
                        (144 * BITRATES[bitrate_idx] * 1000) / SAMPLE_RATES[sr_idx] + padding;
                    if i + frame_len + 2 < bytes.len()
                        && bytes[i + frame_len] == 0xFF
                        && (bytes[i + frame_len + 1] & 0xE0) == 0xE0
                    {
                        return Some(i);
                    }
                } else {
                    return Some(i);
                }
            }
        }
    }
    None
}
