(() => {
  "use strict";

  function getBaseUrl() {
    const custom = localStorage.getItem("cratevec_node_url");
    if (custom && custom.trim().length > 0) {
      return custom.trim().replace(/\/+$/, "");
    }
    return "";
  }

  function updateNodeLabel() {
    const custom = localStorage.getItem("cratevec_node_url");
    const label = document.getElementById("node-label");
    if (custom && custom.trim()) {
      label.textContent = custom.trim().replace(/^https?:\/\//, "");
    } else {
      label.textContent = "LOCAL :7420";
    }
  }

  async function apiFetch(path, options = {}) {
    const base = getBaseUrl();
    const res = await fetch(`${base}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        if (err.error) msg = err.error;
      } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  // --- Application State ---
  let state = {
    snapshot: null,
    weights: { beat: 0.38, vocal: 0.31, atmo: 0.31, underground: 0.35 },
    activeView: "deck",
    selectedMapPoint: null,
    isDraggingTriangle: false,
    weightTimer: null,
    lastTrackId: null,
    playbackToken: 0,
    seedSuggestTimer: null,
    bridgeSuggestTimer: null,
    selectedSeedId: null,
    selectedBridgeId: null,
    isSwiping: false,
    previewTrack: null,
    lastVolume: 0.85,
    library: { history: [], saved: [] },
    savedIds: new Set(),
    savedDedups: new Set(),
    historyFilter: "all",
    historyQuery: "",
    savedQuery: "",
    playingLibraryId: null,
  };

  // --- DOM References ---
  const seedForm = document.getElementById("seed-form");
  const seedInput = document.getElementById("seed-input");
  const seedSuggest = document.getElementById("seed-suggest");
  const bridgeForm = document.getElementById("bridge-form");
  const bridgeInput = document.getElementById("bridge-input");
  const bridgeSuggest = document.getElementById("bridge-suggest");
  const loadingOverlay = document.getElementById("loading-overlay");
  const loaderTitle = document.getElementById("loader-title");
  const loaderSub = document.getElementById("loader-sub");

  const deckEmpty = document.getElementById("deck-empty");
  const trackCard = document.getElementById("track-card");
  const audioPlayer = document.getElementById("audio-player");
  const waveformCanvas = document.getElementById("waveform-canvas");
  const triangleCanvas = document.getElementById("triangle-canvas");
  const sonicMapCanvas = document.getElementById("sonic-map-canvas");
  const chkAutoDrop = document.getElementById("chk-auto-drop");
  const btnUndo = document.getElementById("btn-undo");
  const btnLoadMore = document.getElementById("btn-load-more");
  const btnSaveCard = document.getElementById("btn-save-card");
  const sliderVolume = document.getElementById("slider-volume");
  const btnMute = document.getElementById("btn-mute");
  const previewBanner = document.getElementById("preview-banner");
  const previewBannerText = document.getElementById("preview-banner-text");
  const toastContainer = document.getElementById("toast-container");
  const tabHistoryCount = document.getElementById("tab-history-count");
  const tabSavedCount = document.getElementById("tab-saved-count");

  function isTrackSaved(meta) {
    if (!meta) return false;
    return state.savedIds.has(meta.id) || (meta.dedup_key && state.savedDedups.has(meta.dedup_key));
  }

  function applyLibrary(lib) {
    state.library = lib || { history: [], saved: [] };
    state.savedIds = new Set((state.library.saved || []).map((e) => e.track.meta.id));
    state.savedDedups = new Set((state.library.saved || []).map((e) => e.track.meta.dedup_key));

    const hLen = (state.library.history || []).length;
    const sLen = (state.library.saved || []).length;
    if (tabHistoryCount) tabHistoryCount.textContent = String(hLen);
    if (tabSavedCount) tabSavedCount.textContent = String(sLen);
    const savedTabBtn = document.querySelector('.tab-btn[data-view="saved"]');
    if (savedTabBtn) savedTabBtn.classList.toggle("has-items", sLen > 0);

    updateSaveCardButton();
    if (state.snapshot) {
      renderCrate(state.snapshot.liked_crate);
      renderQueue(state.snapshot.queue_preview);
    }
    renderHistory();
    renderSaved();
  }

  async function refreshLibrary() {
    try {
      const lib = await apiFetch("/api/library");
      applyLibrary(lib);
    } catch (_) {}
  }

  function updateSaveCardButton() {
    if (!btnSaveCard) return;
    const card = state.snapshot && state.snapshot.current_card;
    const saved = card && isTrackSaved(card.track.meta);
    btnSaveCard.classList.toggle("saved", !!saved);
    btnSaveCard.textContent = saved ? "GEMERKT [B]" : "MERKEN [B]";
  }

  async function toggleSaveTrack(trackId) {
    if (!trackId) return;
    try {
      const res = await apiFetch("/api/saved/toggle", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId }),
      });
      applyLibrary(res.library);
      showToast(
        res.saved ? "Song gemerkt (Tab 04 // GEMERKT)." : "Aus Gemerkt entfernt.",
        res.saved ? "ok" : "info"
      );
    } catch (err) {
      showToast(`Merken fehlgeschlagen: ${err.message}`, "error");
    }
  }

  async function removeSavedTrack(trackId) {
    if (!trackId) return;
    try {
      const lib = await apiFetch("/api/saved/remove", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId }),
      });
      applyLibrary(lib);
      showToast("Aus Gemerkt entfernt.", "info");
    } catch (err) {
      showToast(`Fehler: ${err.message}`, "error");
    }
  }

  async function clearHistory() {
    try {
      const lib = await apiFetch("/api/history/clear", { method: "POST" });
      applyLibrary(lib);
      showToast("Verlauf geleert.", "info");
    } catch (err) {
      showToast(`Fehler: ${err.message}`, "error");
    }
  }

  function switchView(view) {
    state.activeView = view;
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-view") === view);
    });
    ["deck", "map", "history", "saved"].forEach((v) => {
      const el = document.getElementById(`view-${v}`);
      if (el) el.classList.toggle("hidden", v !== view);
    });
    if (view === "map") drawSonicMap();
    if (view === "history") renderHistory();
    if (view === "saved") renderSaved();
  }

  function formatTimestamp(ms) {
    if (!ms) return "";
    const d = new Date(ms);
    const now = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (d.toDateString() === now.toDateString()) return `${hh}:${mm}`;
    const dd = String(d.getDate()).padStart(2, "0");
    const mo = String(d.getMonth() + 1).padStart(2, "0");
    return `${dd}.${mo}. ${hh}:${mm}`;
  }

  function actionLabel(action) {
    switch (action) {
      case "right":
        return "IN CRATE";
      case "up":
        return "SUPER-SEED";
      case "down":
        return "KILL";
      case "seed":
        return "SEED";
      default:
        return "SKIP";
    }
  }

  function renderHistory() {
    const container = document.getElementById("history-list");
    if (!container) return;
    const q = (state.historyQuery || "").trim().toLowerCase();
    const list = (state.library.history || []).filter((item) => {
      if (state.historyFilter !== "all" && item.action !== state.historyFilter) return false;
      if (!q) return true;
      const hay = `${item.track.meta.artist} ${item.track.meta.title} ${item.seed_context}`.toLowerCase();
      return hay.includes(q);
    });

    if (list.length === 0) {
      container.innerHTML = `<div class="empty-hint">Keine Verlaufseintraege fuer diesen Filter.</div>`;
      return;
    }

    container.innerHTML = list
      .map((item, idx) => {
        const t = item.track;
        const saved = isTrackSaved(t.meta);
        const isPlaying = state.playingLibraryId === `hist_${idx}_${t.meta.id}`;
        return `
        <div class="lib-row ${isPlaying ? "playing" : ""}" data-hist-idx="${idx}">
          <img src="${escapeHtml(t.meta.cover_url)}" alt="">
          <div class="lib-meta">
            <div class="lib-title">${escapeHtml(t.meta.artist)} - ${escapeHtml(t.meta.title)}</div>
            <div class="lib-sub">${t.dsp.bpm.toFixed(0)} BPM | Seed: ${escapeHtml(item.seed_context || "-")} | ${formatTimestamp(item.timestamp_ms)}</div>
          </div>
          <span class="action-pill act-${escapeHtml(item.action)}">${actionLabel(item.action)}</span>
          <div class="lib-actions">
            <button type="button" class="btn btn-small" data-lib-play="1">PLAY</button>
            <button type="button" class="btn btn-small btn-bookmark ${saved ? "saved" : ""}" data-lib-save="${escapeHtml(t.meta.id)}">${saved ? "GEMERKT" : "MERKEN"}</button>
            <button type="button" class="btn btn-small" data-lib-seed="${escapeHtml(t.meta.id)}" data-lib-label="${escapeHtml(t.meta.artist)} - ${escapeHtml(t.meta.title)}">SEED</button>
            <a href="${escapeHtml(t.meta.spotify_search_url)}" target="_blank" rel="noopener" class="ext-link">SPOTIFY</a>
          </div>
        </div>
      `;
      })
      .join("");

    container.querySelectorAll(".lib-row").forEach((row) => {
      const idx = parseInt(row.getAttribute("data-hist-idx"), 10);
      const item = list[idx];
      if (!item) return;
      const t = item.track;
      const playKey = `hist_${idx}_${t.meta.id}`;
      row.querySelector("[data-lib-play]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        state.playingLibraryId = playKey;
        loadAndPlayPreview(t.meta.id, t.meta.artist, t.meta.title, t.meta.preview_url, t.dsp.drop_start_sec || 0, true);
        renderHistory();
      });
      row.querySelector("[data-lib-save]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        toggleSaveTrack(t.meta.id);
      });
      row.querySelector("[data-lib-seed]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        switchView("deck");
        startSession(t.meta.id, `${t.meta.artist} - ${t.meta.title}`);
      });
    });
  }

  function renderSaved() {
    const container = document.getElementById("saved-list");
    if (!container) return;
    const q = (state.savedQuery || "").trim().toLowerCase();
    const list = (state.library.saved || []).filter((item) => {
      if (!q) return true;
      const hay = `${item.track.meta.artist} ${item.track.meta.title} ${item.track.meta.album}`.toLowerCase();
      return hay.includes(q);
    });

    if (list.length === 0) {
      container.innerHTML = `<div class="empty-hint">Noch keine Songs gemerkt. Druecke [B] oder klicke auf MERKEN bei einem Song.</div>`;
      return;
    }

    container.innerHTML = list
      .map((item, idx) => {
        const t = item.track;
        const isPlaying = state.playingLibraryId === `saved_${t.meta.id}`;
        return `
        <div class="lib-row ${isPlaying ? "playing" : ""}" data-saved-idx="${idx}">
          <img src="${escapeHtml(t.meta.cover_url)}" alt="">
          <div class="lib-meta">
            <div class="lib-title">${escapeHtml(t.meta.artist)} - ${escapeHtml(t.meta.title)}</div>
            <div class="lib-sub">${t.dsp.bpm.toFixed(0)} BPM | Drop @ ${(t.dsp.drop_start_sec || 0).toFixed(1)}s | Gemerkt: ${formatTimestamp(item.saved_at_ms)}</div>
          </div>
          <div class="lib-actions">
            <button type="button" class="btn btn-small" data-lib-play="1">PLAY</button>
            <button type="button" class="btn btn-small" data-lib-seed="${escapeHtml(t.meta.id)}">ALS SEED</button>
            <a href="${escapeHtml(t.meta.spotify_search_url)}" target="_blank" rel="noopener" class="ext-link">SPOTIFY</a>
            <button type="button" class="remove-x-btn" data-lib-del="${escapeHtml(t.meta.id)}" title="Aus Gemerkt entfernen">X</button>
          </div>
        </div>
      `;
      })
      .join("");

    container.querySelectorAll(".lib-row").forEach((row) => {
      const idx = parseInt(row.getAttribute("data-saved-idx"), 10);
      const item = list[idx];
      if (!item) return;
      const t = item.track;
      const playKey = `saved_${t.meta.id}`;
      row.querySelector("[data-lib-play]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        state.playingLibraryId = playKey;
        loadAndPlayPreview(t.meta.id, t.meta.artist, t.meta.title, t.meta.preview_url, t.dsp.drop_start_sec || 0, true);
        renderSaved();
      });
      row.querySelector("[data-lib-seed]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        switchView("deck");
        startSession(t.meta.id, `${t.meta.artist} - ${t.meta.title}`);
      });
      row.querySelector("[data-lib-del]").addEventListener("click", (e) => {
        e.currentTarget.blur();
        removeSavedTrack(t.meta.id);
      });
    });
  }

  function releaseFocus() {
    if (document.activeElement && typeof document.activeElement.blur === "function") {
      document.activeElement.blur();
    }
  }

  function isTextInputElement(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    const tag = (el.tagName || "").toUpperCase();
    if (tag === "TEXTAREA") return true;
    if (tag === "INPUT") {
      const type = (el.type || "text").toLowerCase();
      return !["range", "checkbox", "radio", "button", "submit", "reset"].includes(type);
    }
    return false;
  }

  function showToast(message, kind = "info") {
    if (!toastContainer) return;
    const item = document.createElement("div");
    item.className = `toast-item ${kind === "error" ? "toast-error" : kind === "ok" ? "toast-ok" : ""}`;
    item.textContent = message;
    toastContainer.appendChild(item);
    setTimeout(() => {
      item.remove();
    }, 3200);
  }

  function flashCard(direction) {
    const cls = `flash-${direction}`;
    trackCard.classList.remove("flash-left", "flash-right", "flash-up", "flash-down");
    trackCard.classList.add(cls);
    setTimeout(() => trackCard.classList.remove(cls), 220);
  }

  function enterPreviewMode(trackObj) {
    state.previewTrack = trackObj;
    if (previewBanner && previewBannerText) {
      previewBannerText.textContent = `PREVIEW: ${trackObj.meta.artist} - ${trackObj.meta.title}`;
      previewBanner.classList.remove("hidden");
    }
    document.getElementById("drop-time-label").textContent = `${(trackObj.dsp.drop_start_sec || 0).toFixed(1)}s`;
    loadAndPlayPreview(
      trackObj.meta.id,
      trackObj.meta.artist,
      trackObj.meta.title,
      trackObj.meta.preview_url,
      trackObj.dsp.drop_start_sec || 0,
      true
    );
    drawWaveform();
  }

  function exitPreviewMode(reloadCard = true) {
    state.previewTrack = null;
    if (previewBanner) previewBanner.classList.add("hidden");
    const card = state.snapshot && state.snapshot.current_card;
    if (card) {
      document.getElementById("drop-time-label").textContent = `${(card.track.dsp.drop_start_sec || 0).toFixed(1)}s`;
      drawWaveform();
      if (reloadCard) {
        loadAndPlayPreview(
          card.track.meta.id,
          card.track.meta.artist,
          card.track.meta.title,
          card.track.meta.preview_url,
          card.track.dsp.drop_start_sec || 0,
          true
        );
      }
    }
  }

  function applyVolume(vol) {
    const clamped = Math.max(0, Math.min(1, vol));
    audioPlayer.volume = clamped;
    audioPlayer.muted = clamped === 0;
    if (sliderVolume) sliderVolume.value = String(Math.round(clamped * 100));
    if (btnMute) btnMute.textContent = clamped === 0 ? "MUTE" : "VOL";
    try {
      localStorage.setItem("cratevec_volume", String(clamped));
    } catch (_) {}
  }

  function initVolume() {
    let saved = 0.85;
    try {
      const raw = parseFloat(localStorage.getItem("cratevec_volume"));
      if (isFinite(raw) && raw >= 0 && raw <= 1) saved = raw;
    } catch (_) {}
    state.lastVolume = saved > 0 ? saved : 0.85;
    applyVolume(saved);
  }

  function setLoading(active, title = "RUST DSP PIPELINE AKTIV", sub = "Analysiere Audio-Vektoren...") {
    loaderTitle.textContent = title;
    loaderSub.textContent = sub;
    if (active) {
      loadingOverlay.classList.remove("hidden");
    } else {
      loadingOverlay.classList.add("hidden");
    }
  }

  // --- Clean Audio Abort & Switching ---
  function stopAudioImmediate() {
    state.playbackToken += 1;
    audioPlayer.onloadedmetadata = null;
    audioPlayer.oncanplay = null;
    try {
      audioPlayer.pause();
      audioPlayer.removeAttribute("src");
      audioPlayer.load();
    } catch (_) {}
    document.getElementById("btn-play-toggle").textContent = "PLAY [SPACE]";
  }

  function hideDropdowns() {
    seedSuggest.classList.add("hidden");
    bridgeSuggest.classList.add("hidden");
  }

  // --- Session Lifecycle ---
  async function startSession(rawQuery, displayLabel = null) {
    const q = (rawQuery || "").trim();
    if (!q) return;
    hideDropdowns();
    releaseFocus();
    state.previewTrack = null;
    if (previewBanner) previewBanner.classList.add("hidden");
    stopAudioImmediate();

    if (displayLabel) {
      seedInput.value = displayLabel;
    }

    setLoading(
      true,
      "NEUE SESSION // TRACK-RADIO + RUST DSP",
      `Lade Track-Radio & berechne 48-D Audio-Vektoren fuer "${displayLabel || q}"...`
    );
    try {
      const snap = await apiFetch("/api/session/start", {
        method: "POST",
        body: JSON.stringify({ query: q }),
      });
      state.selectedSeedId = null;
      state.lastTrackId = null;
      if (snap.seeds && snap.seeds.length > 0) {
        seedInput.value = `${snap.seeds[0].meta.artist} - ${snap.seeds[0].meta.title}`;
      }
      applySnapshot(snap, true);
      refreshLibrary();
      releaseFocus();
    } catch (err) {
      showToast(`Analyse-Fehler: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }

  async function addBridgeSeed(rawQuery, displayLabel = null) {
    const q = (rawQuery || "").trim();
    if (!q) return;
    hideDropdowns();
    releaseFocus();
    if (!state.snapshot) {
      return startSession(q, displayLabel);
    }
    state.previewTrack = null;
    if (previewBanner) previewBanner.classList.add("hidden");
    stopAudioImmediate();
    setLoading(
      true,
      "BRIDGE-MODUS // VEKTOR-KREUZUNG",
      `Fuege zweiten Anker "${displayLabel || q}" in den Klangraum ein...`
    );
    try {
      const snap = await apiFetch("/api/session/bridge", {
        method: "POST",
        body: JSON.stringify({ query: q }),
      });
      state.selectedBridgeId = null;
      bridgeInput.value = "";
      state.lastTrackId = null;
      applySnapshot(snap, true);
      refreshLibrary();
      releaseFocus();
      showToast("Bridge-Anker hinzugefuegt.", "ok");
    } catch (err) {
      showToast(`Bridge-Fehler: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }

  async function sendSwipe(direction, explicitTrackId = null) {
    if (state.isSwiping) return;
    const card = state.snapshot && state.snapshot.current_card;
    const trackId = explicitTrackId || (card && card.track.meta.id);
    if (!trackId) return;

    state.isSwiping = true;
    state.previewTrack = null;
    if (previewBanner) previewBanner.classList.add("hidden");
    flashCard(direction);
    stopAudioImmediate();

    if (direction === "up") {
      setLoading(
        true,
        "SUPER-SEED DEEP DIVE",
        "Erweitere Kandidaten-Pool um Track-Radio-Nachbarn dieses Songs..."
      );
    }

    try {
      const snap = await apiFetch("/api/session/swipe", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId, direction }),
      });
      state.lastTrackId = null;
      applySnapshot(snap, true);
      refreshLibrary();
    } catch (err) {
      showToast(`Swipe fehlgeschlagen: ${err.message}`, "error");
    } finally {
      state.isSwiping = false;
      setLoading(false);
    }
  }

  async function undoLastSwipe() {
    if (state.isSwiping || !state.snapshot || !state.snapshot.can_undo) return;
    state.isSwiping = true;
    state.previewTrack = null;
    if (previewBanner) previewBanner.classList.add("hidden");
    stopAudioImmediate();
    try {
      const snap = await apiFetch("/api/session/undo", { method: "POST" });
      state.lastTrackId = null;
      applySnapshot(snap, true);
      showToast("Letzter Swipe rueckgaengig gemacht.", "info");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      state.isSwiping = false;
    }
  }

  async function removeNegativeAnchor(trackId) {
    if (!trackId) return;
    try {
      const snap = await apiFetch("/api/session/negative/remove", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId }),
      });
      applySnapshot(snap, false);
      showToast("Negativ-Filter entfernt.", "info");
    } catch (err) {
      showToast(`Fehler: ${err.message}`, "error");
    }
  }

  async function removeCrateTrack(trackId) {
    if (!trackId) return;
    try {
      const snap = await apiFetch("/api/session/crate/remove", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId }),
      });
      applySnapshot(snap, false);
      showToast("Track aus Crate entfernt.", "info");
    } catch (err) {
      showToast(`Fehler: ${err.message}`, "error");
    }
  }

  async function loadMoreCandidates() {
    if (!state.snapshot || state.isSwiping) return;
    releaseFocus();
    setLoading(
      true,
      "POOL ERWEITERN // TRACK-RADIO",
      "Suche frische Nachbarn fuer deinen aktuellen Vibe..."
    );
    try {
      const before = state.snapshot.pool_size || 0;
      const snap = await apiFetch("/api/session/more", { method: "POST" });
      const added = Math.max(0, (snap.pool_size || 0) - before);
      applySnapshot(snap, false);
      showToast(
        added > 0 ? `${added} neue Kandidaten in den Pool geladen.` : "Keine weiteren neuen Kandidaten gefunden.",
        added > 0 ? "ok" : "info"
      );
    } catch (err) {
      showToast(`Nachladen fehlgeschlagen: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }

  async function focusTrackFromQueue(trackId) {
    if (!trackId || !state.snapshot) return;
    stopAudioImmediate();
    try {
      const snap = await apiFetch("/api/session/focus", {
        method: "POST",
        body: JSON.stringify({ track_id: trackId }),
      });
      state.lastTrackId = null;
      applySnapshot(snap, true);
    } catch (err) {
      console.error("Focus track failed:", err);
    }
  }

  async function selectCorridor(corridorId) {
    if (!corridorId || !state.snapshot) return;
    stopAudioImmediate();
    try {
      const snap = await apiFetch("/api/session/corridor", {
        method: "POST",
        body: JSON.stringify({ corridor_id: corridorId }),
      });
      state.lastTrackId = null;
      applySnapshot(snap, true);
    } catch (err) {
      console.error("Select corridor failed:", err);
    }
  }

  function scheduleWeightsUpdate() {
    if (state.weightTimer) clearTimeout(state.weightTimer);
    state.weightTimer = setTimeout(async () => {
      if (!state.snapshot) return;
      try {
        const snap = await apiFetch("/api/session/weights", {
          method: "POST",
          body: JSON.stringify(state.weights),
        });
        applySnapshot(snap, false);
      } catch (err) {
        console.error("Weight update failed:", err);
      }
    }, 55);
  }

  // --- Autocomplete Search for Seed & Bridge Inputs ---
  function setupAutocomplete(inputEl, dropdownEl, isBridge) {
    let activeIdx = -1;

    function updateHighlight() {
      const items = dropdownEl.querySelectorAll(".suggest-item");
      items.forEach((el, idx) => {
        el.classList.toggle("active", idx === activeIdx);
      });
    }

    function pickItem(el) {
      if (!el) return;
      const id = el.getAttribute("data-id");
      const label = el.getAttribute("data-label");
      dropdownEl.classList.add("hidden");
      activeIdx = -1;
      inputEl.blur();
      if (isBridge) {
        addBridgeSeed(id, label);
      } else {
        startSession(id, label);
      }
    }

    inputEl.addEventListener("keydown", (e) => {
      const isOpen = !dropdownEl.classList.contains("hidden");
      const items = dropdownEl.querySelectorAll(".suggest-item");
      if (e.key === "Escape") {
        dropdownEl.classList.add("hidden");
        activeIdx = -1;
        inputEl.blur();
        return;
      }
      if (!isOpen || items.length === 0) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        activeIdx = (activeIdx + 1) % items.length;
        updateHighlight();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        activeIdx = (activeIdx - 1 + items.length) % items.length;
        updateHighlight();
      } else if (e.key === "Enter" && activeIdx >= 0 && items[activeIdx]) {
        e.preventDefault();
        pickItem(items[activeIdx]);
      }
    });

    inputEl.addEventListener("input", () => {
      activeIdx = -1;
      if (isBridge) state.selectedBridgeId = null;
      else state.selectedSeedId = null;

      const val = inputEl.value.trim();
      const timerKey = isBridge ? "bridgeSuggestTimer" : "seedSuggestTimer";
      if (state[timerKey]) clearTimeout(state[timerKey]);

      if (val.length < 2 || val.includes("open.spotify.com/")) {
        dropdownEl.classList.add("hidden");
        return;
      }

      state[timerKey] = setTimeout(async () => {
        try {
          const items = await apiFetch(`/api/search?q=${encodeURIComponent(val)}`);
          if (!items || items.length === 0 || document.activeElement !== inputEl) {
            dropdownEl.classList.add("hidden");
            return;
          }
          activeIdx = -1;
          dropdownEl.innerHTML = items
            .slice(0, 7)
            .map(
              (item) => `
            <div class="suggest-item" data-id="${escapeHtml(item.id)}" data-label="${escapeHtml(item.artist)} - ${escapeHtml(item.title)}">
              <img src="${escapeHtml(item.cover_url)}" alt="">
              <div class="suggest-meta">
                <div class="suggest-title">${escapeHtml(item.artist)} - ${escapeHtml(item.title)}</div>
                <div class="suggest-sub">${escapeHtml(item.album)}</div>
              </div>
            </div>
          `
            )
            .join("");
          dropdownEl.classList.remove("hidden");

          dropdownEl.querySelectorAll(".suggest-item").forEach((el) => {
            el.addEventListener("mousedown", (ev) => {
              ev.preventDefault();
              pickItem(el);
            });
          });
        } catch (_) {
          dropdownEl.classList.add("hidden");
        }
      }, 210);
    });

    inputEl.addEventListener("blur", () => {
      setTimeout(() => {
        dropdownEl.classList.add("hidden");
        activeIdx = -1;
      }, 160);
    });
  }

  setupAutocomplete(seedInput, seedSuggest, false);
  setupAutocomplete(bridgeInput, bridgeSuggest, true);

  // --- Render Snapshot ---
  function applySnapshot(snap, forceAutoPlay = false) {
    state.snapshot = snap;
    if (snap.weights) {
      state.weights = snap.weights;
    }

    document.getElementById("stat-pool").textContent = snap.pool_size;
    document.getElementById("stat-seen").textContent = snap.seen_count;
    document.getElementById("seed-count-badge").textContent = snap.seeds.length;
    document.getElementById("crate-count-badge").textContent = snap.liked_crate.length;
    document.getElementById("neg-count-badge").textContent = snap.negative_anchors.length;
    if (btnUndo) btnUndo.disabled = !snap.can_undo;

    renderSeeds(snap.seeds);
    renderNegativeAnchors(snap.negative_anchors);
    renderCrate(snap.liked_crate);
    renderQueue(snap.queue_preview);
    renderCorridors(snap.vibe_corridors);
    drawTriangleMixer();
    updateUndergroundUI();

    if (snap.current_card) {
      deckEmpty.classList.add("hidden");
      trackCard.classList.remove("hidden");
      const trackChanged = state.lastTrackId !== snap.current_card.track.meta.id;
      renderCurrentCard(snap.current_card, forceAutoPlay || trackChanged);
    } else {
      trackCard.classList.add("hidden");
      deckEmpty.classList.remove("hidden");
    }

    drawSonicMap();
  }

  function renderCorridors(corridors) {
    const bar = document.getElementById("corridor-bar");
    if (!bar) return;
    if (!corridors || corridors.length === 0) {
      bar.classList.add("hidden");
      return;
    }
    bar.classList.remove("hidden");
    bar.innerHTML = corridors
      .map(
        (c, idx) => `
      <button type="button" class="corridor-chip ${c.active ? "active" : ""}" data-corridor="${escapeHtml(c.id)}" title="${escapeHtml(c.description)} (Taste ${idx + 1})">
        <div class="corridor-head">
          <span class="corridor-title">0${idx + 1} // ${escapeHtml(c.title)}</span>
          <span class="corridor-count">[${c.track_count}]</span>
        </div>
        <div class="corridor-artists">${escapeHtml(c.sample_artists)}</div>
      </button>
    `
      )
      .join("");

    bar.querySelectorAll(".corridor-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.blur();
        const cid = btn.getAttribute("data-corridor");
        if (cid) selectCorridor(cid);
      });
    });
  }

  function renderSeeds(seeds) {
    const container = document.getElementById("seed-list");
    if (!seeds || seeds.length === 0) {
      container.innerHTML = `<div class="empty-hint">Noch keine Session gestartet.</div>`;
      return;
    }
    container.innerHTML = seeds
      .map(
        (s, idx) => `
      <div class="seed-item queue-clickable" data-seed-idx="${idx}" title="Klicken zum Anspielen des Seed-Tracks">
        <img src="${escapeHtml(s.meta.cover_url)}" alt="">
        <div class="seed-info">
          <div class="seed-title">${escapeHtml(s.meta.artist)} - ${escapeHtml(s.meta.title)}</div>
          <div class="seed-sub">${s.dsp.bpm.toFixed(0)} BPM | Drop ${s.dsp.drop_start_sec.toFixed(1)}s | H:${((s.dsp.aggression_index || 0.5) * 100).toFixed(0)}% D:${(s.dsp.darkness_index * 100).toFixed(0)}%</div>
        </div>
      </div>
    `
      )
      .join("");

    container.querySelectorAll(".seed-item").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.getAttribute("data-seed-idx"), 10);
        if (seeds[idx]) enterPreviewMode(seeds[idx]);
      });
    });
  }

  function renderNegativeAnchors(negatives) {
    const container = document.getElementById("negative-list");
    if (!negatives || negatives.length === 0) {
      container.innerHTML = `<div class="empty-hint">Swipe nach unten schneidet unerwuenschte Klang-Cluster ab.</div>`;
      return;
    }
    container.innerHTML = negatives
      .map(
        (n) => `
      <div class="neg-item">
        <span>${escapeHtml(n.meta.artist)} - ${escapeHtml(n.meta.title)}</span>
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="mono-badge">${n.dsp.bpm.toFixed(0)} BPM</span>
          <button type="button" class="remove-x-btn" data-remove-neg="${escapeHtml(n.meta.id)}" title="Filter entfernen">X</button>
        </div>
      </div>
    `
      )
      .join("");

    container.querySelectorAll("[data-remove-neg]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        btn.blur();
        removeNegativeAnchor(btn.getAttribute("data-remove-neg"));
      });
    });
  }

  function renderCrate(crate) {
    const container = document.getElementById("crate-list");
    if (!crate || crate.length === 0) {
      container.innerHTML = `<div class="empty-hint">Nach rechts oder oben geswipte Songs landen hier.</div>`;
      return;
    }
    const reversed = crate.slice().reverse();
    container.innerHTML = reversed
      .map((t, idx) => {
        const saved = isTrackSaved(t.meta);
        return `
      <div class="list-track queue-clickable" data-crate-idx="${idx}" title="Klicken zum Anspielen">
        <img src="${escapeHtml(t.meta.cover_url)}" alt="">
        <div class="list-track-meta">
          <div class="list-track-title">${escapeHtml(t.meta.artist)} - ${escapeHtml(t.meta.title)}</div>
          <div class="list-track-sub">${t.dsp.bpm.toFixed(0)} BPM | Drop @ ${t.dsp.drop_start_sec.toFixed(1)}s</div>
        </div>
        <button type="button" class="mini-save-btn ${saved ? "saved" : ""}" data-save-track="${escapeHtml(t.meta.id)}" title="Separat dauerhaft merken">M</button>
        <a href="${escapeHtml(t.meta.spotify_search_url)}" target="_blank" rel="noopener" class="ext-link" onclick="event.stopPropagation()">SPOTIFY</a>
        <button type="button" class="remove-x-btn" data-remove-crate="${escapeHtml(t.meta.id)}" title="Aus Crate entfernen">X</button>
      </div>
    `;
      })
      .join("");

    container.querySelectorAll(".list-track").forEach((el) => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.getAttribute("data-crate-idx"), 10);
        if (reversed[idx]) enterPreviewMode(reversed[idx]);
      });
    });

    container.querySelectorAll("[data-save-track]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        btn.blur();
        toggleSaveTrack(btn.getAttribute("data-save-track"));
      });
    });

    container.querySelectorAll("[data-remove-crate]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        btn.blur();
        removeCrateTrack(btn.getAttribute("data-remove-crate"));
      });
    });
  }

  function renderQueue(queue) {
    const container = document.getElementById("queue-list");
    if (!queue || queue.length === 0) {
      container.innerHTML = `<div class="empty-hint">Keine weiteren Kandidaten im Pool.</div>`;
      return;
    }
    container.innerHTML = queue
      .map((c, idx) => {
        const saved = isTrackSaved(c.track.meta);
        return `
      <div class="list-track queue-clickable" data-focus-id="${escapeHtml(c.track.meta.id)}" title="Klicken, um direkt zu diesem Song zu wechseln">
        <span class="mono-badge">${String(idx + 2).padStart(2, "0")}</span>
        <div class="list-track-meta">
          <div class="list-track-title">${escapeHtml(c.track.meta.artist)} - ${escapeHtml(c.track.meta.title)}</div>
          <div class="list-track-sub">B:${c.beat_similarity.toFixed(0)}% V:${c.vocal_similarity.toFixed(0)}% A:${c.atmo_similarity.toFixed(0)}% | ${c.track.dsp.bpm.toFixed(0)} BPM</div>
        </div>
        <button type="button" class="mini-save-btn ${saved ? "saved" : ""}" data-save-track="${escapeHtml(c.track.meta.id)}" title="Separat dauerhaft merken">M</button>
        <span class="list-track-score">${c.total_score.toFixed(1)}%</span>
      </div>
    `;
      })
      .join("");

    container.querySelectorAll("[data-save-track]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        btn.blur();
        toggleSaveTrack(btn.getAttribute("data-save-track"));
      });
    });

    container.querySelectorAll(".queue-clickable").forEach((el) => {
      el.addEventListener("click", () => {
        const tid = el.getAttribute("data-focus-id");
        if (tid) focusTrackFromQueue(tid);
      });
    });
  }

  function renderCurrentCard(card, shouldReloadAudio) {
    const { track } = card;
    document.getElementById("card-cover").src = track.meta.cover_url || "";
    document.getElementById("card-total-score").textContent = `${card.total_score.toFixed(1)}% MATCH`;
    document.getElementById("card-channel").textContent = track.meta.discovery_channel;
    document.getElementById("card-title").textContent = track.meta.title;
    document.getElementById("card-artist").textContent = track.meta.artist;
    document.getElementById("card-album").textContent = `${track.meta.album} // ${track.meta.source.toUpperCase()}`;
    document.getElementById("card-spotify-link").href = track.meta.spotify_search_url;
    document.getElementById("card-source-link").href = track.meta.external_url || track.meta.spotify_search_url;

    document.getElementById("bar-beat").style.width = `${Math.min(100, card.beat_similarity)}%`;
    document.getElementById("pct-beat").textContent = `${card.beat_similarity.toFixed(0)}%`;

    document.getElementById("bar-vocal").style.width = `${Math.min(100, card.vocal_similarity)}%`;
    document.getElementById("pct-vocal").textContent = `${card.vocal_similarity.toFixed(0)}%`;

    document.getElementById("bar-atmo").style.width = `${Math.min(100, card.atmo_similarity)}%`;
    document.getElementById("pct-atmo").textContent = `${card.atmo_similarity.toFixed(0)}%`;

    const badgesEl = document.getElementById("card-badges");
    badgesEl.innerHTML = (card.explanation || [])
      .map((b) => `<span class="tech-badge">${escapeHtml(b)}</span>`)
      .join("");

    document.getElementById("drop-time-label").textContent = `${track.dsp.drop_start_sec.toFixed(1)}s`;
    updateSaveCardButton();

    drawWaveform();

    if (shouldReloadAudio || state.lastTrackId !== track.meta.id) {
      state.lastTrackId = track.meta.id;
      loadAndPlayPreview(
        track.meta.id,
        track.meta.artist,
        track.meta.title,
        track.meta.preview_url,
        track.dsp.drop_start_sec,
        true
      );
    }
  }

  function loadAndPlayPreview(id, artist, title, url, dropSec, autoPlay) {
    stopAudioImmediate();
    const token = state.playbackToken;
    const base = getBaseUrl();
    const params = new URLSearchParams({
      id: id || "",
      artist: artist || "",
      title: title || "",
      url: url || "",
      _t: String(token),
    });

    audioPlayer.src = `${base}/api/audio/proxy?${params.toString()}`;
    audioPlayer.load();

    let hasSeeked = false;
    audioPlayer.onloadedmetadata = () => {
      if (token !== state.playbackToken) return;
      if (!hasSeeked && chkAutoDrop.checked && dropSec > 0 && dropSec < (audioPlayer.duration || 30) - 3) {
        hasSeeked = true;
        try {
          audioPlayer.currentTime = dropSec;
        } catch (_) {}
      }
      drawWaveform();
      if (autoPlay) {
        audioPlayer.play().catch(() => {});
      }
    };
  }

  // --- Waveform & Drop-Sniper Canvas ---
  function drawWaveform() {
    const ctx = waveformCanvas.getContext("2d");
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const activeTrack =
      state.previewTrack || (state.snapshot && state.snapshot.current_card && state.snapshot.current_card.track);
    if (!activeTrack) return;

    const peaks = activeTrack.dsp.waveform_peaks || [];
    const n = peaks.length || 64;
    const dur = audioPlayer.duration && isFinite(audioPlayer.duration) ? audioPlayer.duration : 30.0;
    const cur = audioPlayer.currentTime || 0.0;
    const dropSec = activeTrack.dsp.drop_start_sec || 0.0;

    const barW = w / n;
    const progressRatio = cur / dur;
    const dropRatio = dropSec / dur;

    for (let i = 0; i < n; i++) {
      const val = peaks[i] !== undefined ? peaks[i] : 0.25;
      const barH = Math.max(4, val * (h - 18));
      const x = i * barW;
      const y = (h - barH) / 2;
      const ratio = i / n;

      if (ratio <= progressRatio) {
        ctx.fillStyle = "#e6e7eb";
      } else if (ratio >= dropRatio) {
        ctx.fillStyle = "#4b5162";
      } else {
        ctx.fillStyle = "#2b2f3a";
      }
      ctx.fillRect(x + 1, y, Math.max(2, barW - 2), barH);
    }

    const dropX = Math.max(2, Math.min(w - 2, dropRatio * w));
    ctx.strokeStyle = "#d97724";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(dropX, 0);
    ctx.lineTo(dropX, h);
    ctx.stroke();

    ctx.fillStyle = "#d97724";
    ctx.font = "700 10px monospace";
    const label = `DROP ${dropSec.toFixed(1)}s`;
    const labelX = dropX + 55 > w ? dropX - 58 : dropX + 4;
    ctx.fillText(label, labelX, 11);

    const playX = Math.max(0, Math.min(w, progressRatio * w));
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, h);
    ctx.stroke();
  }

  waveformCanvas.addEventListener("click", (e) => {
    const rect = waveformCanvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const dur = audioPlayer.duration && isFinite(audioPlayer.duration) ? audioPlayer.duration : 30.0;
    audioPlayer.currentTime = ratio * dur;
    if (audioPlayer.paused) {
      audioPlayer.play().catch(() => {});
    }
    drawWaveform();
  });

  audioPlayer.addEventListener("timeupdate", () => {
    const cur = audioPlayer.currentTime || 0;
    const dur = audioPlayer.duration && isFinite(audioPlayer.duration) ? audioPlayer.duration : 30.0;
    document.getElementById("player-time").textContent = `${cur.toFixed(1).padStart(4, "0")}s / ${dur.toFixed(1)}s`;
    drawWaveform();
  });

  audioPlayer.addEventListener("play", () => {
    document.getElementById("btn-play-toggle").textContent = "PAUSE [SPACE]";
  });

  audioPlayer.addEventListener("pause", () => {
    document.getElementById("btn-play-toggle").textContent = "PLAY [SPACE]";
  });

  // --- Interactive Barycentric Triangle Mixer ---
  const triVerts = {
    top: { x: 115, y: 24 },
    left: { x: 24, y: 172 },
    right: { x: 206, y: 172 },
  };

  function drawTriangleMixer() {
    const ctx = triangleCanvas.getContext("2d");
    const w = triangleCanvas.width;
    const h = triangleCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const sum = Math.max(0.03, state.weights.beat + state.weights.vocal + state.weights.atmo);
    const wb = state.weights.beat / sum;
    const wv = state.weights.vocal / sum;
    const wa = state.weights.atmo / sum;

    document.getElementById("val-w-beat").textContent = `${Math.round(wb * 100)}%`;
    document.getElementById("val-w-vocal").textContent = `${Math.round(wv * 100)}%`;
    document.getElementById("val-w-atmo").textContent = `${Math.round(wa * 100)}%`;

    ctx.strokeStyle = "#2b2f3a";
    ctx.lineWidth = 1;
    const cx = (triVerts.top.x + triVerts.left.x + triVerts.right.x) / 3;
    const cy = (triVerts.top.y + triVerts.left.y + triVerts.right.y) / 3;
    [triVerts.top, triVerts.left, triVerts.right].forEach((v) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(v.x, v.y);
      ctx.stroke();
    });

    ctx.strokeStyle = "#4b5162";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(triVerts.top.x, triVerts.top.y);
    ctx.lineTo(triVerts.left.x, triVerts.left.y);
    ctx.lineTo(triVerts.right.x, triVerts.right.y);
    ctx.closePath();
    ctx.stroke();

    ctx.fillStyle = "#8b909c";
    ctx.font = "700 9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("BEAT / 808", triVerts.top.x, 14);
    ctx.textAlign = "left";
    ctx.fillText("VOCALS", 8, 188);
    ctx.textAlign = "right";
    ctx.fillText("ATMO", w - 8, 188);

    const px = wb * triVerts.top.x + wv * triVerts.left.x + wa * triVerts.right.x;
    const py = wb * triVerts.top.y + wv * triVerts.left.y + wa * triVerts.right.y;

    ctx.strokeStyle = "#d97724";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#d97724";
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  function updateWeightsFromTrianglePoint(clientX, clientY) {
    const rect = triangleCanvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * triangleCanvas.width;
    const y = ((clientY - rect.top) / rect.height) * triangleCanvas.height;

    const v0 = triVerts.top;
    const v1 = triVerts.left;
    const v2 = triVerts.right;

    const denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    let wb = ((v1.y - v2.y) * (x - v2.x) + (v2.x - v1.x) * (y - v2.y)) / denom;
    let wv = ((v2.y - v0.y) * (x - v2.x) + (v0.x - v2.x) * (y - v2.y)) / denom;
    let wa = 1.0 - wb - wv;

    wb = Math.max(0.04, wb);
    wv = Math.max(0.04, wv);
    wa = Math.max(0.04, wa);
    const sum = wb + wv + wa;

    state.weights.beat = wb / sum;
    state.weights.vocal = wv / sum;
    state.weights.atmo = wa / sum;

    drawTriangleMixer();
    scheduleWeightsUpdate();
  }

  triangleCanvas.addEventListener("mousedown", (e) => {
    state.isDraggingTriangle = true;
    updateWeightsFromTrianglePoint(e.clientX, e.clientY);
  });

  window.addEventListener("mousemove", (e) => {
    if (state.isDraggingTriangle) {
      updateWeightsFromTrianglePoint(e.clientX, e.clientY);
    }
  });

  window.addEventListener("mouseup", () => {
    state.isDraggingTriangle = false;
  });

  document.getElementById("btn-reset-weights").addEventListener("click", () => {
    state.weights.beat = 0.38;
    state.weights.vocal = 0.31;
    state.weights.atmo = 0.31;
    drawTriangleMixer();
    scheduleWeightsUpdate();
  });

  // --- Underground Slider ---
  const ugSlider = document.getElementById("slider-underground");
  function updateUndergroundUI() {
    const val = Math.round((state.weights.underground || 0) * 100);
    ugSlider.value = val;
    const badge = document.getElementById("val-underground");
    if (val > 5) {
      badge.textContent = `+${val}% UG`;
    } else if (val < -5) {
      badge.textContent = `${val}% MAIN`;
    } else {
      badge.textContent = "NEUTRAL";
    }
  }

  ugSlider.addEventListener("input", () => {
    state.weights.underground = parseInt(ugSlider.value, 10) / 100.0;
    updateUndergroundUI();
    scheduleWeightsUpdate();
  });

  // --- 2D Sonic Map Radar ---
  function drawSonicMap() {
    const ctx = sonicMapCanvas.getContext("2d");
    const w = sonicMapCanvas.width;
    const h = sonicMapCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 28;

    ctx.strokeStyle = "#1f222a";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1.0].forEach((r) => {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
      ctx.stroke();
    });

    ctx.beginPath();
    ctx.moveTo(16, cy);
    ctx.lineTo(w - 16, cy);
    ctx.moveTo(cx, 16);
    ctx.lineTo(cx, h - 16);
    ctx.stroke();

    ctx.fillStyle = "#5d626e";
    ctx.font = "600 9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("SUB-BASS & DARKNESS (+)", cx, 12);
    ctx.fillText("VOCAL & MID PRESENCE (-)", cx, h - 6);
    ctx.textAlign = "left";
    ctx.fillText("WARM / SLOW", 8, cy - 6);
    ctx.textAlign = "right";
    ctx.fillText("GLITCH / FAST DRIVE", w - 8, cy - 6);

    const points = (state.snapshot && state.snapshot.map_points) || [];
    for (const p of points) {
      const px = cx + p.x * radius;
      const py = cy - p.y * radius;

      if (p.kind === "seed") {
        ctx.fillStyle = "#d97724";
        ctx.fillRect(px - 6, py - 6, 12, 12);
      } else if (p.kind === "liked") {
        ctx.fillStyle = "#389e58";
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
      } else if (p.kind === "negative") {
        ctx.fillStyle = "#c44747";
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = p.total_score >= 76 ? "#e6e7eb" : "#6c7282";
        ctx.beginPath();
        ctx.arc(px, py, p.total_score >= 76 ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
      }

      if (state.selectedMapPoint && state.selectedMapPoint.id === p.id) {
        ctx.strokeStyle = "#d97724";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  sonicMapCanvas.addEventListener("click", (e) => {
    const points = (state.snapshot && state.snapshot.map_points) || [];
    if (!points.length) return;

    const rect = sonicMapCanvas.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * sonicMapCanvas.width;
    const my = ((e.clientY - rect.top) / rect.height) * sonicMapCanvas.height;

    const cx = sonicMapCanvas.width / 2;
    const cy = sonicMapCanvas.height / 2;
    const radius = Math.min(cx, cy) - 28;

    let best = null;
    let bestDist = 24;
    for (const p of points) {
      const px = cx + p.x * radius;
      const py = cy - p.y * radius;
      const d = Math.hypot(mx - px, my - py);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }

    if (best) {
      state.selectedMapPoint = best;
      document.getElementById("insp-title").textContent = `${best.artist} - ${best.title} (${best.total_score.toFixed(1)}%)`;
      document.getElementById("insp-sub").textContent = `${best.bpm.toFixed(0)} BPM | Beat: ${best.beat_similarity.toFixed(0)}% | Vocals: ${best.vocal_similarity.toFixed(0)}% | Atmo: ${best.atmo_similarity.toFixed(0)}%`;
      document.getElementById("btn-insp-save").disabled = false;
      document.getElementById("btn-insp-like").disabled = best.kind === "seed";
      document.getElementById("btn-insp-kill").disabled = best.kind === "seed";

      loadAndPlayPreview(
        best.id,
        best.artist,
        best.title,
        best.preview_url,
        best.drop_start_sec,
        true
      );
      drawSonicMap();
    }
  });

  document.getElementById("btn-insp-save").addEventListener("click", (e) => {
    e.currentTarget.blur();
    if (state.selectedMapPoint) {
      toggleSaveTrack(state.selectedMapPoint.id);
    }
  });

  document.getElementById("btn-insp-like").addEventListener("click", (e) => {
    e.currentTarget.blur();
    if (state.selectedMapPoint) {
      sendSwipe("right", state.selectedMapPoint.id);
    }
  });

  document.getElementById("btn-insp-kill").addEventListener("click", (e) => {
    e.currentTarget.blur();
    if (state.selectedMapPoint) {
      sendSwipe("down", state.selectedMapPoint.id);
    }
  });

  // --- View Switcher Tabs ---
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      btn.blur();
      const view = btn.getAttribute("data-view");
      if (view) switchView(view);
    });
  });

  // --- History & Saved Toolbar Listeners ---
  const historyFilters = document.getElementById("history-filters");
  if (historyFilters) {
    historyFilters.querySelectorAll("[data-filter]").forEach((btn) => {
      btn.addEventListener("click", () => {
        btn.blur();
        historyFilters.querySelectorAll("[data-filter]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.historyFilter = btn.getAttribute("data-filter") || "all";
        renderHistory();
      });
    });
  }

  const historySearch = document.getElementById("history-search");
  if (historySearch) {
    historySearch.addEventListener("input", () => {
      state.historyQuery = historySearch.value;
      renderHistory();
    });
  }

  const btnClearHistory = document.getElementById("btn-clear-history");
  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", (e) => {
      e.currentTarget.blur();
      clearHistory();
    });
  }

  const savedSearch = document.getElementById("saved-search");
  if (savedSearch) {
    savedSearch.addEventListener("input", () => {
      state.savedQuery = savedSearch.value;
      renderSaved();
    });
  }

  const btnSavedExpSpotify = document.getElementById("btn-saved-export-spotify");
  if (btnSavedExpSpotify) {
    btnSavedExpSpotify.addEventListener("click", (e) => {
      e.currentTarget.blur();
      const list = (state.library.saved || []).map((x) => x.track);
      if (!list.length) return showToast("Keine gemerkten Songs vorhanden.", "info");
      const lines = list.map((t) => `${t.meta.artist} - ${t.meta.title} | ${t.meta.spotify_search_url}`);
      navigator.clipboard.writeText(lines.join("\n")).then(() => {
        showToast(`${list.length} gemerkte Tracks + Spotify-Links kopiert.`, "ok");
      });
    });
  }

  const btnSavedExpM3u = document.getElementById("btn-saved-export-m3u");
  if (btnSavedExpM3u) {
    btnSavedExpM3u.addEventListener("click", (e) => {
      e.currentTarget.blur();
      const list = (state.library.saved || []).map((x) => x.track);
      if (!list.length) return showToast("Keine gemerkten Songs vorhanden.", "info");
      let m3u = "#EXTM3U\n";
      for (const t of list) {
        m3u += `#EXTINF:${t.meta.duration_sec || 180},${t.meta.artist} - ${t.meta.title}\n`;
        m3u += `${t.meta.external_url || t.meta.spotify_search_url}\n`;
      }
      downloadBlob(m3u, "cratevec_gemerkt.m3u8", "audio/x-mpegurl");
      showToast(`${list.length} gemerkte Tracks als .m3u8 exportiert.`, "ok");
    });
  }

  const btnSavedExpJson = document.getElementById("btn-saved-export-json");
  if (btnSavedExpJson) {
    btnSavedExpJson.addEventListener("click", (e) => {
      e.currentTarget.blur();
      const list = state.library.saved || [];
      if (!list.length) return showToast("Keine gemerkten Songs vorhanden.", "info");
      downloadBlob(JSON.stringify(list, null, 2), "cratevec_gemerkt.json", "application/json");
      showToast(`${list.length} gemerkte Tracks als JSON exportiert.`, "ok");
    });
  }

  if (btnSaveCard) {
    btnSaveCard.addEventListener("click", (e) => {
      e.currentTarget.blur();
      const card = state.snapshot && state.snapshot.current_card;
      if (card) toggleSaveTrack(card.track.meta.id);
    });
  }

  // --- Form Submissions & Controls ---
  seedForm.addEventListener("submit", (e) => {
    e.preventDefault();
    seedInput.blur();
    startSession(state.selectedSeedId || seedInput.value);
  });

  bridgeForm.addEventListener("submit", (e) => {
    e.preventDefault();
    bridgeInput.blur();
    addBridgeSeed(state.selectedBridgeId || bridgeInput.value);
  });

  document.querySelectorAll(".quick-seed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      btn.blur();
      const s = btn.getAttribute("data-seed");
      seedInput.value = s;
      startSession(s);
    });
  });

  document.getElementById("btn-play-toggle").addEventListener("click", (e) => {
    e.currentTarget.blur();
    if (audioPlayer.paused) {
      audioPlayer.play().catch(() => {});
    } else {
      audioPlayer.pause();
    }
  });

  document.getElementById("btn-jump-drop").addEventListener("click", (e) => {
    e.currentTarget.blur();
    const activeTrack =
      state.previewTrack || (state.snapshot && state.snapshot.current_card && state.snapshot.current_card.track);
    if (!activeTrack) return;
    audioPlayer.currentTime = activeTrack.dsp.drop_start_sec || 0;
    audioPlayer.play().catch(() => {});
  });

  if (btnUndo) {
    btnUndo.addEventListener("click", (e) => {
      e.currentTarget.blur();
      undoLastSwipe();
    });
  }

  if (btnLoadMore) {
    btnLoadMore.addEventListener("click", (e) => {
      e.currentTarget.blur();
      loadMoreCandidates();
    });
  }

  const btnExitPreview = document.getElementById("btn-exit-preview");
  if (btnExitPreview) {
    btnExitPreview.addEventListener("click", (e) => {
      e.currentTarget.blur();
      exitPreviewMode(true);
    });
  }

  if (sliderVolume) {
    sliderVolume.addEventListener("input", () => {
      const v = parseInt(sliderVolume.value, 10) / 100;
      if (v > 0) state.lastVolume = v;
      applyVolume(v);
    });
    sliderVolume.addEventListener("change", () => sliderVolume.blur());
  }

  if (btnMute) {
    btnMute.addEventListener("click", (e) => {
      e.currentTarget.blur();
      if (audioPlayer.volume > 0 && !audioPlayer.muted) {
        state.lastVolume = audioPlayer.volume;
        applyVolume(0);
      } else {
        applyVolume(state.lastVolume || 0.85);
      }
    });
  }

  chkAutoDrop.addEventListener("change", () => chkAutoDrop.blur());
  ugSlider.addEventListener("change", () => ugSlider.blur());

  document.getElementById("btn-swipe-left").addEventListener("click", (e) => {
    e.currentTarget.blur();
    sendSwipe("left");
  });
  document.getElementById("btn-swipe-right").addEventListener("click", (e) => {
    e.currentTarget.blur();
    sendSwipe("right");
  });
  document.getElementById("btn-swipe-up").addEventListener("click", (e) => {
    e.currentTarget.blur();
    sendSwipe("up");
  });
  document.getElementById("btn-swipe-down").addEventListener("click", (e) => {
    e.currentTarget.blur();
    sendSwipe("down");
  });

  window.addEventListener("keydown", (e) => {
    // Global Escape: close modal, dropdowns, exit preview or blur input
    if (e.key === "Escape") {
      if (!nodeModal.classList.contains("hidden")) {
        nodeModal.classList.add("hidden");
        releaseFocus();
        return;
      }
      hideDropdowns();
      if (isTextInputElement(document.activeElement)) {
        releaseFocus();
        return;
      }
      if (state.previewTrack) {
        exitPreviewMode(true);
        return;
      }
    }

    // Ctrl+K jumps straight to search even from inputs
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      seedInput.focus();
      seedInput.select();
      return;
    }

    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isTextInputElement(e.target) || isTextInputElement(document.activeElement)) return;
    if (!nodeModal.classList.contains("hidden")) return;

    if (e.key === "/") {
      e.preventDefault();
      seedInput.focus();
      seedInput.select();
    } else if (e.code === "Space") {
      e.preventDefault();
      releaseFocus();
      if (audioPlayer.paused) audioPlayer.play().catch(() => {});
      else audioPlayer.pause();
    } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D" || e.key === "l") {
      e.preventDefault();
      sendSwipe("right");
    } else if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A" || e.key === "h") {
      e.preventDefault();
      sendSwipe("left");
    } else if (e.key === "ArrowUp" || e.key === "w" || e.key === "W" || e.key === "k") {
      e.preventDefault();
      sendSwipe("up");
    } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S" || e.key === "j") {
      e.preventDefault();
      sendSwipe("down");
    } else if (e.key === "z" || e.key === "Z" || e.key === "Backspace") {
      e.preventDefault();
      undoLastSwipe();
    } else if (e.key === "b" || e.key === "B") {
      e.preventDefault();
      const card = state.snapshot && state.snapshot.current_card;
      if (card) toggleSaveTrack(card.track.meta.id);
    } else if (e.key === "m" || e.key === "M") {
      e.preventDefault();
      if (btnMute) btnMute.click();
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      document.getElementById("btn-jump-drop").click();
    } else if (["1", "2", "3", "4"].includes(e.key)) {
      const corridors = (state.snapshot && state.snapshot.vibe_corridors) || [];
      const idx = parseInt(e.key, 10) - 1;
      if (corridors[idx]) {
        e.preventDefault();
        selectCorridor(corridors[idx].id);
      }
    }
  });

  // --- Touch Swipe Support on Card ---
  let touchStartX = null;
  let touchStartY = null;
  trackCard.addEventListener("touchstart", (e) => {
    if (e.touches.length === 1) {
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
    }
  });

  trackCard.addEventListener("touchend", (e) => {
    if (touchStartX === null || touchStartY === null) return;
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    touchStartX = null;
    touchStartY = null;
    if (Math.hypot(dx, dy) < 55) return;

    if (Math.abs(dx) > Math.abs(dy)) {
      sendSwipe(dx > 0 ? "right" : "left");
    } else {
      sendSwipe(dy < 0 ? "up" : "down");
    }
  });

  // --- Crate Exports ---
  document.getElementById("btn-export-spotify").addEventListener("click", (e) => {
    e.currentTarget.blur();
    const crate = (state.snapshot && state.snapshot.liked_crate) || [];
    if (!crate.length) return showToast("Die Crate ist noch leer.", "info");
    const lines = crate.map(
      (t) => `${t.meta.artist} - ${t.meta.title} | ${t.meta.spotify_search_url}`
    );
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      showToast(`${crate.length} Tracks + Spotify-Suchlinks kopiert.`, "ok");
    });
  });

  document.getElementById("btn-export-m3u").addEventListener("click", (e) => {
    e.currentTarget.blur();
    const crate = (state.snapshot && state.snapshot.liked_crate) || [];
    if (!crate.length) return showToast("Die Crate ist noch leer.", "info");
    let m3u = "#EXTM3U\n";
    for (const t of crate) {
      m3u += `#EXTINF:${t.meta.duration_sec || 180},${t.meta.artist} - ${t.meta.title}\n`;
      m3u += `${t.meta.external_url || t.meta.spotify_search_url}\n`;
    }
    downloadBlob(m3u, "cratevec_session.m3u8", "audio/x-mpegurl");
    showToast(`${crate.length} Tracks als .m3u8 exportiert.`, "ok");
  });

  document.getElementById("btn-export-json").addEventListener("click", (e) => {
    e.currentTarget.blur();
    const crate = (state.snapshot && state.snapshot.liked_crate) || [];
    if (!crate.length) return showToast("Die Crate ist noch leer.", "info");
    const data = JSON.stringify(crate, null, 2);
    downloadBlob(data, "cratevec_session.json", "application/json");
    showToast(`${crate.length} Tracks als JSON exportiert.`, "ok");
  });

  function downloadBlob(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // --- Hybrid Node Modal ---
  const nodeModal = document.getElementById("node-modal");
  const inputNodeUrl = document.getElementById("input-node-url");

  document.getElementById("btn-node-config").addEventListener("click", () => {
    inputNodeUrl.value = localStorage.getItem("cratevec_node_url") || "";
    nodeModal.classList.remove("hidden");
  });

  document.getElementById("btn-close-modal").addEventListener("click", () => {
    nodeModal.classList.add("hidden");
  });

  document.getElementById("btn-save-node").addEventListener("click", async () => {
    const val = inputNodeUrl.value.trim();
    if (val) {
      localStorage.setItem("cratevec_node_url", val);
    } else {
      localStorage.removeItem("cratevec_node_url");
    }
    updateNodeLabel();
    nodeModal.classList.add("hidden");
    try {
      await apiFetch("/api/status");
      showToast(`Verbunden mit ${val || "LOCAL :7420"}.`, "ok");
    } catch (err) {
      showToast(`Server unter "${val || "lokal"}" nicht erreichbar: ${err.message}`, "error");
    }
  });

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // --- Initialization ---
  initVolume();
  updateNodeLabel();
  drawTriangleMixer();
  refreshLibrary();

  apiFetch("/api/session/snapshot")
    .then((snap) => {
      if (snap && snap.session_id) {
        if (snap.seeds && snap.seeds.length > 0) {
          seedInput.value = `${snap.seeds[0].meta.artist} - ${snap.seeds[0].meta.title}`;
        }
        applySnapshot(snap, false);
      }
    })
    .catch(() => {});
})();
