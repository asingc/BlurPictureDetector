"use strict";

// Page 5 — Album Tools: show a quick album summary, then let the user pick a
// destination folder and export kept photos (+ players.csv) — syncing
// tagged face crops into the system-wide .FaceReco database and rebuilding
// it first — with progress polling.
//
// Every long-running action below (deep regrade, jersey-colour LLM re-run,
// export, import-more, re-run face detection, burst-video, face-tagging
// CSV, per-player videos) shares the same "launch job -> poll server log
// into a modal -> react when done" UX via LogDialog (see common.js) —
// each `*Job` below is created once in the init block at the bottom of
// this file.
let deepRegradeJob, jerseyColorJob, exportJob, importMoreJob, rerunFacerecoJob;
let burstVideoJob, faceTaggingCsvJob, playerVideoJob;

let lastStarBreakdown = null;

let importMoreRunning = false;

let rerunFacerecoRunning = false;

let regrading = false;

let burstVideoRunning = false;

function setExportUIEnabled(enabled) {
  $("#exportFaceTaggingInput, #minStarsInput, #exportBtn").prop("disabled", !enabled);
  // Importing more images, re-running face detection, and exporting all
  // mutate/read the album at the same time — keep them mutually exclusive
  // from the UI side (each is already independently guarded server-side
  // too, via the shared processing_state lock for import-more/rerun).
  $("#importMoreBtn").prop("disabled", !enabled || importMoreRunning || rerunFacerecoRunning);
  $("#rerunFacerecoBtn").prop("disabled", !enabled || importMoreRunning || rerunFacerecoRunning);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled || importMoreRunning || rerunFacerecoRunning || regrading);
}

function setImportMoreUIEnabled(enabled) {
  importMoreRunning = !enabled;
  $("#importMoreBtn").prop("disabled", !enabled);
  $("#rerunFacerecoBtn").prop("disabled", !enabled);
  $("#exportBtn").prop("disabled", !enabled);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled);
}

function setRerunFacerecoUIEnabled(enabled) {
  rerunFacerecoRunning = !enabled;
  $("#rerunFacerecoBtn").prop("disabled", !enabled);
  $("#importMoreBtn").prop("disabled", !enabled);
  $("#exportBtn").prop("disabled", !enabled);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled);
}

// ------------------------------------------------------------------ //
// Adjust Blur Sensitivity — re-bucket Blur/Sharp using already-measured
// sharpness scores (see /api/album-tools/regrade-sensitivity), no re-import.
// ------------------------------------------------------------------ //
function applyRegradeSensitivityToUI(mode, customValue) {
  $(`input[name="regradeSensMode"][value="${mode || "medium"}"]`).prop("checked", true);
  const custom = customValue ?? 0.50;
  $("#regradeSensitivitySlider").val(custom);
  $("#regradeSensitivityValue").text(Number(custom).toFixed(2));
}

function readRegradeSensitivityFromUI() {
  const mode = $('input[name="regradeSensMode"]:checked').val() || "medium";
  const customValue = parseFloat($("#regradeSensitivitySlider").val()) || 0;
  return { mode, customValue };
}

function setRegradeUIEnabled(enabled) {
  regrading = !enabled;
  $('input[name="regradeSensMode"], #regradeSensitivitySlider').prop("disabled", !enabled);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled || importMoreRunning || rerunFacerecoRunning);
  $("#importMoreBtn").prop("disabled", !enabled);
  $("#rerunFacerecoBtn").prop("disabled", !enabled);
  $("#exportBtn").prop("disabled", !enabled);
}

// Deep regrade streams 1_prep_review.py's stdout into a non-closable modal
// (deepRegradeJob, created in the init block) that dismisses itself once
// the run finishes.

async function openExportDestination() {
  try {
    await apiPost("/api/album-tools/open-destination", {});
  } catch (err) {
    $("#exportStatus").text("Could not open folder: " + err.message);
  }
}

// Export progress dialog (exportJob, created in the init block) — a
// non-closable jQuery UI modal while the export is running, becoming
// closable once it finishes.

// Import-more and re-run-face-detection progress dialogs (importMoreJob /
// rerunFacerecoJob, created in the init block) — same pattern as export.

const STAR_TIERS = [5, 4, 3, 2, 1];

function renderStarBreakdown(breakdown, unrated) {
  const $el = $("#starBreakdown").empty();
  breakdown = breakdown || {};
  STAR_TIERS.forEach((n) => {
    const count = breakdown[String(n)] || 0;
    const $block = $("<div>", { class: `star-block star-block-${n}` });
    $block.append($("<span>", { class: "star-icons" }).text("★".repeat(n) + "☆".repeat(5 - n)));
    $block.append($("<span>", { class: "star-count" }).text(count));
    $el.append($block);
  });
  if (unrated) {
    const $block = $("<div>", { class: "star-block star-block-unrated", title: "Unrated" });
    $block.append($("<span>", { class: "star-icons" }).text("—"));
    $block.append($("<span>", { class: "star-count" }).text(unrated));
    $el.append($block);
  }
}

async function loadSummary() {
  try {
    const data = await apiGet("/api/album-tools/summary");
    $("#summaryAlbumName").text(data.name || "Album");
    lastStarBreakdown = data.starBreakdown || {};
    renderStarBreakdown(data.starBreakdown, data.unrated);
    $("#statFaces").text(data.facesDetected);
    $("#statPlayers").text(data.playersDetected);
    applyRegradeSensitivityToUI(data.sensitivityMode, data.sensitivityCustomValue);
    updateExportCount();
  } catch (err) {
    $("#summaryAlbumName").text("Album");
  }
}

// ------------------------------------------------------------------ //
// Team Jersey Colour override — see /api/album-tools/jersey-color. Rendered as
// round chip buttons ( "Auto" + each registered colour), same interaction
// pattern as the team/jersey-colour pickers on the Add Album page.
// ------------------------------------------------------------------ //
let jerseyColorOptions = [];
let selectedJerseyColor = "";
let jerseyColorControlsDisabled = false;

// Non-closable modal (jerseyColorJob, created in the init block) streaming
// this operation's output (mirrors deep regrade) — auto-dismisses on
// success, stays open with [x] enabled if it fails so the user can read
// the error.

function renderJerseyColorChips() {
  const $container = $("#jerseyColorPicker").empty();
  const makeChip = (label, value) => {
    const chip = $("<button>", {
      type: "button",
      class: "color-chip" + (selectedJerseyColor === value ? " selected" : ""),
      disabled: jerseyColorControlsDisabled,
    }).text(label);
    chip.on("click", () => { selectedJerseyColor = value; renderJerseyColorChips(); });
    $container.append(chip);
  };
  makeChip("Auto (detect)", "");
  jerseyColorOptions.forEach((color) => makeChip(color, color));
}

async function loadJerseyOptions() {
  try {
    const data = await apiGet("/api/album-tools/jersey-options");
    jerseyColorOptions = data.options || [];
    selectedJerseyColor = data.current || "";
    jerseyColorControlsDisabled = !!data.noTeam;
    renderJerseyColorChips();
    if (data.noTeam) {
      $("#jerseyColorPanel .row, #jerseyColorPanel .row.actions").find("input, button").prop("disabled", true);
      $("#jerseyColorStatus").text("This album has jersey-colour filtering disabled (--noteam).");
    }
    $("#jerseyColorDetected").text(
      data.current ? "" : (data.detected ? `Currently detected: ${data.detected}` : "")
    );
  } catch (err) {
    // Non-fatal — the panel just stays at its default "Auto" state.
  }
}

// Number of images that will be exported at the currently-selected minimum
// star rating, computed client-side from the star breakdown already loaded
// by loadSummary() — no extra round-trip needed when the selector changes.

function updateExportCount() {
  const minStars = parseInt($("#minStarsInput").val(), 10) || 3;
  let count = 0;
  if (lastStarBreakdown) {
    for (let n = minStars; n <= 5; n++) {
      count += lastStarBreakdown[String(n)] || 0;
    }
  }
  $("#exportCountLabel").text(`${count} image(s) will be exported`);
}

// ------------------------------------------------------------------ //
// Burst → MP4 (experimental) — render each detected photo burst (incl.
// blur/skipped shots) into its own MP4 via /api/album-tools/burst-video/*,
// saved to a user-chosen destination folder (same /api/browse-folder
// picker as Export/per-player videos below). Same background-job/poll
// pattern as the Export dialog above (burstVideoJob, created in the init
// block).
// ------------------------------------------------------------------ //
function readBurstMinFrames() {
  return Math.max(2, parseInt($("#burstMinFramesInput").val(), 10) || 3);
}

// Pre-fills the min-frames/fps/timing-mode controls from the last render's
// settings (persisted server-side in state.json — see api_burst_video_render
// in culling_app.py) instead of always resetting to the HTML defaults.
async function loadBurstVideoSettings() {
  try {
    const data = await apiGet("/api/album-tools/burst-video/settings");
    $("#burstMinFramesInput").val(data.minFrames);
    $("#burstFpsInput").val(data.fps);
    const mode = data.mode === "timestamp" ? "timestamp" : "fps";
    $(`input[name="burstTimingMode"][value="${mode}"]`).prop("checked", true);
    $("#burstFpsInput").prop("disabled", mode !== "fps");
  } catch (err) {
    // Keep the HTML defaults if this fails.
  }
}

async function refreshBurstSummary() {
  try {
    const data = await apiGet(`/api/album-tools/burst-video/summary?minFrames=${readBurstMinFrames()}`);
    const ffmpegNote = data.ffmpegAvailable ? "" : " (ffmpeg not found \u2014 using fallback encoder)";
    $("#burstSummaryLabel").text(
      `${data.qualifyingBursts} of ${data.totalBursts} burst(s) qualify (${data.qualifyingFrames} photo(s))${ffmpegNote}`
    );
  } catch (err) {
    $("#burstSummaryLabel").text("");
  }
}

// ------------------------------------------------------------------ //
// Export all face tagging to CSV — write a CSV of every tagged face
// (regardless of star rating/export status) to a user-chosen destination
// folder via /api/album-tools/face-tagging-csv/*. Same background-job/poll
// /modal pattern as the export & burst-video dialogs above
// (faceTaggingCsvJob, created in the init block).
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
// Per-player highlight videos — one MP4 per tagged player via
// /api/album-tools/player-video/*. Same background-job/poll/modal pattern
// as the burst-video dialog above (playerVideoJob, created in the init
// block).
// ------------------------------------------------------------------ //
async function refreshPlayerVideoSummary() {
  try {
    const data = await apiGet("/api/album-tools/player-video/summary");
    const ffmpegNote = data.ffmpegAvailable ? "" : " (ffmpeg not found \u2014 using fallback encoder)";
    $("#playerVideoSummaryLabel").text(`${data.playersFound} player(s), ${data.totalPhotos} tagged photo(s)${ffmpegNote}`);
  } catch (err) {
    $("#playerVideoSummaryLabel").text("");
  }
}

$(function () {
  loadSummary();
  loadJerseyOptions();

  deepRegradeJob = LogDialog.create({
    dialogId: "deepRegradeDialog",
    outputId: "deepRegradeOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    dialogOptions: { width: 720 },
    label: "deep regrade",
  });
  jerseyColorJob = LogDialog.create({
    dialogId: "jerseyColorDialog",
    outputId: "jerseyColorOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    dialogOptions: { width: 720 },
    label: "jersey colour apply",
  });
  exportJob = LogDialog.create({
    dialogId: "exportDialog",
    outputId: "exportOutput",
    statusUrl: (since) => `/api/album-tools/export-status?since=${since}`,
    label: "export",
  });
  importMoreJob = LogDialog.create({
    dialogId: "importMoreDialog",
    outputId: "importMoreOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    label: "import",
  });
  rerunFacerecoJob = LogDialog.create({
    dialogId: "rerunFacerecoDialog",
    outputId: "rerunFacerecoOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    label: "face detection re-run",
  });
  burstVideoJob = LogDialog.create({
    dialogId: "burstVideoDialog",
    outputId: "burstVideoOutput",
    statusUrl: (since) => `/api/album-tools/burst-video/status?since=${since}`,
    label: "burst-video rendering",
  });
  faceTaggingCsvJob = LogDialog.create({
    dialogId: "faceTaggingCsvDialog",
    outputId: "faceTaggingCsvOutput",
    statusUrl: (since) => `/api/album-tools/face-tagging-csv/status?since=${since}`,
    label: "face-tagging CSV export",
  });
  playerVideoJob = LogDialog.create({
    dialogId: "playerVideoDialog",
    outputId: "playerVideoOutput",
    statusUrl: (since) => `/api/album-tools/player-video/status?since=${since}`,
    label: "player-video rendering",
  });

  loadBurstVideoSettings().then(refreshBurstSummary);
  refreshPlayerVideoSummary();

  $("#minStarsInput").on("change", updateExportCount);

  $("#burstMinFramesInput").on("change", refreshBurstSummary);

  $('input[name="burstTimingMode"]').on("change", function () {
    $("#burstFpsInput").prop("disabled", $('input[name="burstTimingMode"]:checked').val() !== "fps");
  });

  $("#openBurstsFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/burst-video/open-folder", {});
    } catch (err) {
      $("#burstVideoStatus").text("Could not open folder: " + err.message);
    }
  });

  $("#renderBurstsBtn").on("click", async () => {
    $("#renderBurstsBtn").prop("disabled", true);
    $("#burstVideoStatus").text("Choose a destination folder…");
    $("#openBurstsFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for burst videos", context: "export" });
    } catch (err) {
      $("#burstVideoStatus").text("Browse failed: " + err.message);
      $("#renderBurstsBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#burstVideoStatus").text("");
      $("#renderBurstsBtn").prop("disabled", false);
      return;
    }

    burstVideoRunning = true;
    $("#burstVideoStatus").text("");
    burstVideoJob.reset();
    burstVideoJob.open("Rendering bursts\u2026");
    const mode = $('input[name="burstTimingMode"]:checked').val() || "fps";
    const fps = parseFloat($("#burstFpsInput").val()) || 6;
    const minFrames = readBurstMinFrames();
    try {
      await apiPost("/api/album-tools/burst-video/render", { destination: res.path, minFrames, mode, fps });
      burstVideoJob.startPolling((data) => {
        if (data.running) return;
        burstVideoRunning = false;
        $("#renderBurstsBtn").prop("disabled", false);
        const success = !data.error;
        burstVideoJob.setFinished(success ? "Rendering complete" : "Rendering failed");
        if (success) {
          $("#burstVideoStatus").text(data.result ? `Rendered ${data.result.rendered} video(s).` : "Done.");
          $("#openBurstsFolderBtn").show();
        } else {
          $("#burstVideoStatus").text("Failed: " + data.error);
        }
        refreshBurstSummary();
      });
    } catch (err) {
      $("#burstVideoStatus").text("Failed to start: " + err.message);
      $("#renderBurstsBtn").prop("disabled", false);
      burstVideoRunning = false;
      burstVideoJob.close();
    }
  });

  $("#regradeSensitivitySlider").on("input", function () {
    $("#regradeSensitivityValue").text(Number($(this).val()).toFixed(2));
  });

  $("#regradeBtn").on("click", async () => {
    setRegradeUIEnabled(false);
    $("#regradeStatus").text("Regrading…");
    const { mode, customValue } = readRegradeSensitivityFromUI();
    try {
      const res = await apiPost("/api/album-tools/regrade-sensitivity", {
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        mode: "shallow",
      });
      const parts = [];
      if (res.recovered) parts.push(`${res.recovered} moved to Sharp`);
      if (res.demoted) parts.push(`${res.demoted} moved to Blur`);
      if (res.teamColor) parts.push(`team colour ${res.teamColor}`);
      if (res.previewsRegenFailed) parts.push(`${res.previewsRegenFailed} preview(s) could not be refreshed (source photo unreadable)`);
      $("#regradeStatus").text(
        parts.length
          ? `Regraded ${res.imagesConsidered} image(s) — ${parts.join(", ")}.`
          : `Regraded ${res.imagesConsidered} image(s) — no changes.`
      );
      await loadSummary();
    } catch (err) {
      $("#regradeStatus").text("Regrade failed: " + err.message);
    } finally {
      setRegradeUIEnabled(true);
    }
  });

  $("#deepRegradeBtn").on("click", async () => {
    setRegradeUIEnabled(false);
    $("#regradeStatus").text("Starting deep regrade…");
    deepRegradeJob.reset();
    const { mode, customValue } = readRegradeSensitivityFromUI();
    try {
      await apiPost("/api/album-tools/regrade-sensitivity", {
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        mode: "deep",
      });
      deepRegradeJob.open();
      $("#regradeStatus").text("Deep regrade running…");
      deepRegradeJob.startPolling((data) => {
        if (data.running) return;
        setRegradeUIEnabled(true);
        deepRegradeJob.close();
        if (data.returnCode === 0) {
          $("#regradeStatus").text("Deep regrade complete.");
          loadSummary();
        } else {
          $("#regradeStatus").text(`Deep regrade failed (exit code ${data.returnCode}).`);
        }
      });
    } catch (err) {
      $("#regradeStatus").text("Deep regrade failed to start: " + err.message);
      setRegradeUIEnabled(true);
    }
  });

  $("#jerseyColorBtn").on("click", async () => {
    const teamColor = selectedJerseyColor;
    const rerunLlmCulling = $("#jerseyRerunLlmInput").is(":checked");
    $("#jerseyColorBtn, #jerseyRerunLlmInput").prop("disabled", true);
    jerseyColorControlsDisabled = true;
    renderJerseyColorChips();
    setRegradeUIEnabled(false);
    $("#jerseyColorStatus").text("Applying…");
    jerseyColorJob.reset();
    jerseyColorJob.open("Applying jersey colour…");
    try {
      const res = await apiPost("/api/album-tools/jersey-color", { teamColor, rerunLlmCulling });
      const parts = [];
      if (res.recovered) parts.push(`${res.recovered} moved to Sharp`);
      if (res.demoted) parts.push(`${res.demoted} moved to Blur`);
      if (res.starsRebaselined) parts.push(`${res.starsRebaselined} star rating(s) reset`);
      const summary =
        `Team colour: ${res.teamColor || "none detected"} (${res.pinned ? "pinned" : "auto"})` +
        (parts.length ? ` — ${parts.join(", ")}.` : " — no changes.");
      $("#jerseyColorStatus").text(summary);
      jerseyColorJob.appendLines([summary]);
      await loadSummary();
      await loadJerseyOptions();
      if (res.llmRerunStarted) {
        $("#jerseyColorStatus").append(" LLM re-cull running…");
        jerseyColorJob.appendLines(["Re-running LLM burst ranking…"]);
        jerseyColorJob.startPolling((data) => {
          if (data.running) return;
          $("#jerseyColorBtn, #jerseyRerunLlmInput").prop("disabled", false);
          jerseyColorControlsDisabled = false;
          renderJerseyColorChips();
          setRegradeUIEnabled(true);
          if (data.returnCode === 0) {
            $("#jerseyColorStatus").text("Team colour applied; LLM re-cull complete.");
            jerseyColorJob.close();
          } else {
            const msg = `LLM re-cull exited with code ${data.returnCode}.`;
            $("#jerseyColorStatus").text(msg);
            jerseyColorJob.appendLines([msg]);
            jerseyColorJob.setFinished("Jersey colour apply failed");
          }
          loadSummary();
        });
        return; // dialog + buttons stay until the background job finishes
      }
      jerseyColorJob.close();
    } catch (err) {
      $("#jerseyColorStatus").text("Failed: " + err.message);
      jerseyColorJob.appendLines(["Error: " + err.message]);
      jerseyColorJob.setFinished("Jersey colour apply failed");
    }
    $("#jerseyColorBtn, #jerseyRerunLlmInput").prop("disabled", false);
    jerseyColorControlsDisabled = false;
    renderJerseyColorChips();
    setRegradeUIEnabled(true);
  });

  $("#importMoreBtn").on("click", async () => {
    setImportMoreUIEnabled(false);
    $("#importMoreStatus").text("Choose a folder…");
    let path;
    try {
      const res = await apiPost("/api/browse-folder", { title: "Select folder to import more images from", context: "import" });
      path = res.path;
    } catch (err) {
      $("#importMoreStatus").text("Folder picker failed: " + err.message);
      setImportMoreUIEnabled(true);
      return;
    }
    if (!path) {
      // Cancelled — quietly return to the idle state.
      $("#importMoreStatus").text("");
      setImportMoreUIEnabled(true);
      return;
    }

    importMoreJob.reset();
    importMoreJob.open("Importing…");
    $("#importMoreStatus").text("Importing…");
    try {
      await apiPost("/api/import-more", { path });
      importMoreJob.startPolling((data) => {
        if (data.running) return;
        setImportMoreUIEnabled(true);
        const success = data.returnCode === 0;
        importMoreJob.setFinished(success ? "Import complete" : `Import failed (exit code ${data.returnCode})`);
        if (success) {
          $("#importMoreStatus").text("Import complete.");
          loadSummary();
        } else {
          $("#importMoreStatus").text(`Import exited with code ${data.returnCode}.`);
        }
      });
    } catch (err) {
      $("#importMoreStatus").text("Failed: " + err.message);
      setImportMoreUIEnabled(true);
      importMoreJob.close();
    }
  });

  $("#rerunFacerecoBtn").on("click", async () => {
    setRerunFacerecoUIEnabled(false);
    $("#rerunFacerecoStatus").text("Starting…");
    rerunFacerecoJob.reset();
    rerunFacerecoJob.open("Re-running face detection…");
    $("#rerunFacerecoStatus").text("Re-running face detection…");
    try {
      await apiPost("/api/album-tools/rerun-facereco", {});
      rerunFacerecoJob.startPolling((data) => {
        if (data.running) return;
        setRerunFacerecoUIEnabled(true);
        const success = data.returnCode === 0;
        rerunFacerecoJob.setFinished(success ? "Face detection complete" : `Face detection failed (exit code ${data.returnCode})`);
        if (success) {
          $("#rerunFacerecoStatus").text("Face detection re-run complete.");
          loadSummary();
        } else {
          $("#rerunFacerecoStatus").text(`Face detection re-run exited with code ${data.returnCode}.`);
        }
      });
    } catch (err) {
      $("#rerunFacerecoStatus").text("Failed: " + err.message);
      setRerunFacerecoUIEnabled(true);
      rerunFacerecoJob.close();
    }
  });

  $("#openDestBtn").on("click", openExportDestination);

  $("#exportBtn").on("click", async () => {
    setExportUIEnabled(false);
    $("#exportStatus").text("Choose a destination folder…");
    $("#openDestBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select export destination folder", context: "export" });
    } catch (err) {
      $("#exportStatus").text("Browse failed: " + err.message);
      setExportUIEnabled(true);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#exportStatus").text("");
      setExportUIEnabled(true);
      return;
    }

    const exportFaceTagging = $("#exportFaceTaggingInput").is(":checked");
    const minStars = parseInt($("#minStarsInput").val(), 10) || 3;
    $("#exportStatus").text("");
    $("#exportDialogStatus").text("Starting export…");
    exportJob.reset();
    exportJob.open("Exporting…");
    try {
      await apiPost("/api/album-tools/export", { destination: res.path, exportFaceTagging, minStars });
      exportJob.startPolling((data) => {
        if (data.totalImages) {
          $("#exportDialogStatus").text(
            `Copying photos: ${data.copiedImages}/${data.totalImages} · ` +
            `Faces: ${data.processedPlayers}/${data.totalPlayers} player(s)`
          );
        }
        if (data.running) return;
        setExportUIEnabled(true);
        if (data.error) {
          $("#exportDialogStatus").text("Export failed: " + data.error);
          exportJob.setFinished("Export failed");
        } else {
          $("#exportDialogStatus").text(`Done — ${data.copiedImages} photo(s) exported.`);
          exportJob.setFinished("Export complete");
          if (data.destDir) {
            $("#openDestBtn").show();
            openExportDestination();
          }
        }
      });
    } catch (err) {
      $("#exportDialogStatus").text("Failed: " + err.message);
      exportJob.setFinished("Export failed");
      setExportUIEnabled(true);
    }
  });

  $("#faceTaggingCsvBtn").on("click", async () => {
    $("#faceTaggingCsvBtn").prop("disabled", true);
    $("#faceTaggingCsvStatus").text("Choose a destination folder…");
    $("#openFaceTaggingCsvFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for face-tagging CSV", context: "export" });
    } catch (err) {
      $("#faceTaggingCsvStatus").text("Browse failed: " + err.message);
      $("#faceTaggingCsvBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#faceTaggingCsvStatus").text("");
      $("#faceTaggingCsvBtn").prop("disabled", false);
      return;
    }

    $("#faceTaggingCsvStatus").text("");
    $("#faceTaggingCsvDialogStatus").text("Starting export…");
    faceTaggingCsvJob.reset();
    faceTaggingCsvJob.open("Exporting…");
    try {
      await apiPost("/api/album-tools/face-tagging-csv/render", { destination: res.path });
      faceTaggingCsvJob.startPolling((data) => {
        if (data.running) return;
        $("#faceTaggingCsvBtn").prop("disabled", false);
        const success = !data.error;
        faceTaggingCsvJob.setFinished(success ? "Export complete" : "Export failed");
        if (success) {
          $("#faceTaggingCsvStatus").text(data.result ? `Wrote ${data.result.rows} row(s) to ${data.result.csvPath}` : "Done.");
          $("#openFaceTaggingCsvFolderBtn").show();
        } else {
          $("#faceTaggingCsvStatus").text("Failed: " + data.error);
        }
      });
    } catch (err) {
      $("#faceTaggingCsvDialogStatus").text("Failed: " + err.message);
      faceTaggingCsvJob.setFinished("Export failed");
      $("#faceTaggingCsvBtn").prop("disabled", false);
    }
  });

  $("#openFaceTaggingCsvFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/face-tagging-csv/open-folder", {});
    } catch (err) {
      $("#faceTaggingCsvStatus").text("Could not open folder: " + err.message);
    }
  });

  $("#renderPlayerVideosBtn").on("click", async () => {
    $("#renderPlayerVideosBtn").prop("disabled", true);
    $("#playerVideoStatus").text("Choose a destination folder…");
    $("#openPlayerVideoFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for player highlight videos", context: "export" });
    } catch (err) {
      $("#playerVideoStatus").text("Browse failed: " + err.message);
      $("#renderPlayerVideosBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#playerVideoStatus").text("");
      $("#renderPlayerVideosBtn").prop("disabled", false);
      return;
    }

    const fps = parseFloat($("#playerVideoFpsInput").val()) || 1;
    $("#playerVideoStatus").text("");
    playerVideoJob.reset();
    playerVideoJob.open("Rendering player videos…");
    try {
      await apiPost("/api/album-tools/player-video/render", { destination: res.path, fps });
      playerVideoJob.startPolling((data) => {
        if (data.running) return;
        $("#renderPlayerVideosBtn").prop("disabled", false);
        const success = !data.error;
        playerVideoJob.setFinished(success ? "Rendering complete" : "Rendering failed");
        if (success) {
          $("#playerVideoStatus").text(data.result ? `Rendered ${data.result.rendered} video(s).` : "Done.");
          $("#openPlayerVideoFolderBtn").show();
        } else {
          $("#playerVideoStatus").text("Failed: " + data.error);
        }
        refreshPlayerVideoSummary();
      });
    } catch (err) {
      $("#playerVideoStatus").text("Failed to start: " + err.message);
      $("#renderPlayerVideosBtn").prop("disabled", false);
      playerVideoJob.close();
    }
  });

  $("#openPlayerVideoFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/player-video/open-folder", {});
    } catch (err) {
      $("#playerVideoStatus").text("Could not open folder: " + err.message);
    }
  });
});

