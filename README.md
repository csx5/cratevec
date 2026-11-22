# cratevec

**cratevec** ist eine in Rust geschriebene Musik-Discovery- und Crate-Digging-Engine. Statt sich wie herkoemmliche Empfehlungs-Apps auf Playlist-Statistiken ("Nutzer, die X hoeren, hoeren auch Y") oder grobe Genre-Tags zu verlassen, laedt `cratevec` die 30-Sekunden-Audio-Previews potenzieller Kandidaten herunter und berechnet per Fast Fourier Transform (`rustfft` + `symphonia`) einen 48-dimensionalen akustischen Fingerabdruck.

Das Projekt funktioniert **out-of-the-box ohne API-Keys** (Zero-Config) und laeuft hybrid: lokal als Desktop-/CLI-Anwendung oder als Self-Hosted Server (`cratevec-server`) im Heimnetz und Browser.

---

## Kernfunktionen

1. **Reine Rust DSP-Pipeline (`crates/cratevec-core`):**
   * Dekodiert MP3/AAC-Streams direkt im Arbeitsspeicher und berechnet drei getrennte, z-normierte 16-D-Teilvektoren:
     * `v_beat` (20-360 Hz): BPM, Groove-Autokorrelation, Sub-Bass-Druck, 808-Verzerrungsgrad (Oberschwingungs-Saettigung) und Kick-Transienten.
     * `v_vocal` (260-5200 Hz): 12 Mel-Frequency Cepstral Coefficients (MFCCs via DCT-II), Formant-Praesenz und Stimm-Dynamik.
     * `v_atmo` (110-16500 Hz): Tonart-invariante 12-Bin-Chroma-Intervallstruktur (Moll/Phrygisch-Dichte), 6-Oktaven-Spektralkontrast, Spectral Flatness und High-Freq-Glitch-Dichte.
2. **Drop-Sniper (Automatischer Beat-Drop-Detektor):**
   * Analysiert den Energie-Sprung und den Low-Band Spectral Flux ueber das 30s-Preview und springt beim Anspielen automatisch auf die Sekunde, an der Beat oder Chorus voll einsetzen.
3. **Interaktiver Dreiecks-Mixer & Underground-Slider:**
   * Stufenlose Echtzeit-Gewichtung zwischen **Beat/808**, **Vocals** und **Atmosphaere** (Neu-Sortierung des gesamten Pools in unter 1 ms).
   * Popularitaets-Regler von `Mainstream` bis `Deep Crate` (bevorzugt unbekannte Underground-Releases mit hoher akustischer Aehnlichkeit).
4. **4-Wege-Swipe (Live Rocchio-Vektor-Steering):**
   * `Rechts (D / Right)`: In die Session-Crate legen & Zielvektor Richtung Song verschieben.
   * `Links (A / Left)`: Ueberspringen.
   * `Oben (W / Up)`: **Super-Seed** (macht den Song zum neuen Anker und laedt sofort dessen akustische Nachbarschaft nach).
   * `Unten (S / Down)`: **Kill Vibe** (setzt den Song als Negativ-Anker und schneidet diese gesamte Klangecke fuer den Rest der Session mathematisch ab).
5. **Bridge-Modus (Multi-Seed-Kreuzung) & 2D Sonic Map:**
   * Beliebig viele Start-Songs oder öffentliche `open.spotify.com`-Track-/Playlist-Links lassen sich ohne Login kombinieren, um Songs auf der exakten Schnittmenge zweier Klangwelten zu finden.
   * Die **2D Sonic Map** projiziert alle Kandidaten, Likes und Negativ-Anker auf ein interaktives Klangradar rund um den Seed-Song.

---

## Projektstruktur

* `crates/cratevec-core`: Reine Rust-Bibliothek (Harvester, Spotify-Embed-Resolver, Symphonia/RustFFT DSP-Analyzer, Rocchio-Session-Engine, lokaler Vektor-Cache).
* `crates/cratevec-cli`: Kommandozeilen-Interface fuer schnelle Terminal-Tests und Benchmarks.
* `crates/cratevec-server`: Hybrid-Server (`axum`) mit REST-API, CORS-freiem Audio-Stream-Proxy und Web/PWA-Oberflaeche.
* `crates/cratevec-tauri`: Tauri v2 Konfiguration fuer native Desktop- und Mobile-Builds.
* `ui/`: Minimalistisches, ablenkungsfreies Studio-Interface (Vanilla JS + HTML5 Canvas, keine externen Frontend-Frameworks noetig).

---

## Schnellstart

### 1. CLI-Analyse & Contrastive Metric Training im Terminal
```bash
# Seed-Analyse
cargo run --release -p cratevec-cli -- "PIXY - LEGACY"

# 48-D Mahalanobis Metric Learning (Offline Triplet-Margin-Training)
cargo run --release -p cratevec-cli -- train
```

### 2. Hybrid-Server & Web/Desktop-UI starten
```bash
cargo run --release -p cratevec-server
```
Danach im Browser oder als Desktop-WebApp oeffnen:
`http://127.0.0.1:7420`

### 3. Docker (Self-Hosted)
```bash
docker compose up --build -d
```

---

## Tastatur-Steuerung im UI

| Taste | Aktion |
| :--- | :--- |
| `Rechts` / `D` / `L` | Like (In Crate legen & Vektor annaehern) |
| `Links` / `A` / `H` | Skip |
| `Oben` / `W` / `K` | Super-Seed (Deep Dive um diesen Song) |
| `Unten` / `S` / `J` | Kill Vibe (Negativ-Filter setzen) |
| `Leertaste` | Play / Pause |
| `R` | Direkt zum erkannten Beat-Drop springen |
| `Z` / `Backspace` | Letzten Swipe rueckgaengig machen (Undo) |
| `B` | Aktuellen Song separat dauerhaft merken (Tab `04 // GEMERKT`) |
| `1` - `4` | Zwischen den 4 Vibe-Korridoren wechseln |
| `/` / `Strg+K` | Direkt in die Seed-Suche springen |
| `M` | Audio stummschalten (Mute) |
| `Esc` | Suche, Dropdowns, Modals oder Preview verlassen |
