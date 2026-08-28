"use strict";

// Page 5 — Apply: show a quick album summary, then let the user pick a
// destination folder and export kept photos (+ players.csv) — syncing
// tagged face crops into the system-wide .FaceReco database and rebuilding
// it first — with progress polling.

let exportPollTimer = null;
let exportPollSince = 0;
let lastStarBreakdown = null;

let importMorePollTimer = null;
let importMorePollSince = 0;
let importMoreRunning = false;

let rerunFacerecoPollTimer = null;
let rerunFacerecoPollSince = 0;
let rerunFacerecoRunning = false;

let regrading = false;

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
// sharpness scores (see /api/apply/regrade-sensitivity), no re-import.
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
// that dismisses itself once the run finishes.
let deepRegradePollTimer = null;
let deepRegradePollSince = 0;

function initDeepRegradeDialog() {
  $("#deepRegradeDialog").dialog({
    autoOpen: false,
    modal: true,
    closeOnEscape: false,
    draggable: false,
    resizable: false,
    width: 720,
  });
}

function openDeepRegradeDialog() {
  const $dialog = $("#deepRegradeDialog");
  $dialog.dialog("open");
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
}

function appendDeepRegradeLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("deepRegradeOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function pollDeepRegradeOutput() {
  let data;
  try {
    data = await apiGet(`/api/processing-output?since=${deepRegradePollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendDeepRegradeLines(data.lines);
  deepRegradePollSince = data.next;
  if (data.running) return;

  clearInterval(deepRegradePollTimer);
  deepRegradePollTimer = null;
  setRegradeUIEnabled(true);
  $("#deepRegradeDialog").dialog("close");
  if (data.returnCode === 0) {
    $("#regradeStatus").text("Deep regrade complete.");
    await loadSummary();
  } else {
    $("#regradeStatus").text(`Deep regrade failed (exit code ${data.returnCode}).`);
  }
}


async function openExportDestination() {
  try {
    await apiPost("/api/apply/open-destination", {});
  } catch (err) {
    $("#exportStatus").text("Could not open folder: " + err.message);
  }
}

function appendExportLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("exportOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// ------------------------------------------------------------------ //
// Export progress dialog — a non-closable jQuery UI modal while the
// export is running (mirrors add_album.js's processing dialog), becoming
// closable once it finishes.
// ------------------------------------------------------------------ //
function initExportDialog() {
  $("#exportDialog").dialog({
    autoOpen: false,
    modal: true,
    closeOnEscape: false,
    draggable: false,
    resizable: false,
    width: 640,
  });
}

function openExportDialog() {
  const $dialog = $("#exportDialog");
  $dialog.dialog("option", "title", "Exporting…");
  $dialog.dialog("option", "closeOnEscape", false);
  $dialog.dialog("open");
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
}

function finishExportDialog(success) {
  const $dialog = $("#exportDialog");
  $dialog.dialog("option", "title", success ? "Export complete" : "Export failed");
  $dialog.dialog("option", "closeOnEscape", true);
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
}

// ------------------------------------------------------------------ //
// Import-more progress dialog — same non-closable-while-running pattern
// as the export dialog above / add_album.js's processing dialog.
// ------------------------------------------------------------------ //
function initImportMoreDialog() {
  $("#importMoreDialog").dialog({
    autoOpen: false,
    modal: true,
    closeOnEscape: false,
    draggable: false,
    resizable: false,
    width: 640,
  });
}

function openImportMoreDialog() {
  const $dialog = $("#importMoreDialog");
  $dialog.dialog("option", "title", "Importing…");
  $dialog.dialog("option", "closeOnEscape", false);
  $dialog.dialog("open");
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
}

function appendImportMoreLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("importMoreOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function finishImportMoreDialog(returnCode) {
  const $dialog = $("#importMoreDialog");
  const success = returnCode === 0;
  $dialog.dialog("option", "title", success ? "Import complete" : `Import failed (exit code ${returnCode})`);
  $dialog.dialog("option", "closeOnEscape", true);
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
  if (success) {
    $("#importMoreStatus").text("Import complete.");
    loadSummary();
  } else {
    $("#importMoreStatus").text(`Import exited with code ${returnCode}.`);
  }
}

async function pollImportMoreOutput() {
  let data;
  try {
    data = await apiGet(`/api/processing-output?since=${importMorePollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendImportMoreLines(data.lines);
  importMorePollSince = data.next;
  if (!data.running) {
    clearInterval(importMorePollTimer);
    importMorePollTimer = null;
    setImportMoreUIEnabled(true);
    finishImportMoreDialog(data.returnCode);
  }
}

// ------------------------------------------------------------------ //
// Re-run face detection progress dialog — same pattern as import-more.
// ------------------------------------------------------------------ //
function initRerunFacerecoDialog() {
  $("#rerunFacerecoDialog").dialog({
    autoOpen: false,
    modal: true,
    closeOnEscape: false,
    draggable: false,
    resizable: false,
    width: 640,
  });
}

function openRerunFacerecoDialog() {
  const $dialog = $("#rerunFacerecoDialog");
  $dialog.dialog("option", "title", "Re-running face detection…");
  $dialog.dialog("option", "closeOnEscape", false);
  $dialog.dialog("open");
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
}

function appendRerunFacerecoLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("rerunFacerecoOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function finishRerunFacerecoDialog(returnCode) {
  const $dialog = $("#rerunFacerecoDialog");
  const success = returnCode === 0;
  $dialog.dialog("option", "title", success ? "Face detection complete" : `Face detection failed (exit code ${returnCode})`);
  $dialog.dialog("option", "closeOnEscape", true);
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
  if (success) {
    $("#rerunFacerecoStatus").text("Face detection re-run complete.");
    loadSummary();
  } else {
    $("#rerunFacerecoStatus").text(`Face detection re-run exited with code ${returnCode}.`);
  }
}

async function pollRerunFacerecoOutput() {
  let data;
  try {
    data = await apiGet(`/api/processing-output?since=${rerunFacerecoPollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendRerunFacerecoLines(data.lines);
  rerunFacerecoPollSince = data.next;
  if (!data.running) {
    clearInterval(rerunFacerecoPollTimer);
    rerunFacerecoPollTimer = null;
    setRerunFacerecoUIEnabled(true);
    finishRerunFacerecoDialog(data.returnCode);
  }
}

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
    const data = await apiGet("/api/apply/summary");
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
// Team Jersey Colour override — see /api/apply/jersey-color.
// ------------------------------------------------------------------ //
async function loadJerseyOptions() {
  try {
    const data = await apiGet("/api/apply/jersey-options");
    const $select = $("#jerseyColorSelect");
    $select.find("option:not(:first)").remove();
    (data.options || []).forEach((color) => {
      $select.append($("<option>", { value: color }).text(color));
    });
    $select.val(data.current || "");
    if (data.noTeam) {
      $("#jerseyColorPanel .row, #jerseyColorPanel .row.actions").find("select, input, button").prop("disabled", true);
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
async function pollJerseyLlmRerun() {
  let data;
  try {
    data = await apiGet("/api/processing-output?since=0");
  } catch (err) {
    setTimeout(pollJerseyLlmRerun, 500);
    return;
  }
  if (data.running) {
    setTimeout(pollJerseyLlmRerun, 500);
    return;
  }
  $("#jerseyColorStatus").text(
    data.returnCode === 0 ? "Team colour applied; LLM re-cull complete." : `LLM re-cull exited with code ${data.returnCode}.`
  );
  $("#jerseyColorBtn, #jerseyColorSelect, #jerseyRerunLlmInput").prop("disabled", false);
  setRegradeUIEnabled(true);
  loadSummary();
}

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

async function pollExportStatus() {
  let data;
  try {
    data = await apiGet(`/api/apply/export-status?since=${exportPollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendExportLines(data.lines);
  exportPollSince = data.next;
  if (data.totalImages) {
    $("#exportDialogStatus").text(
      `Copying photos: ${data.copiedImages}/${data.totalImages} · ` +
      `Faces: ${data.processedPlayers}/${data.totalPlayers} player(s)`
    );
  }
  if (!data.running) {
    clearInterval(exportPollTimer);
    exportPollTimer = null;
    setExportUIEnabled(true);
    if (data.error) {
      $("#exportDialogStatus").text("Export failed: " + data.error);
      finishExportDialog(false);
    } else {
      $("#exportDialogStatus").text(`Done — ${data.copiedImages} photo(s) exported.`);
      finishExportDialog(true);
      if (data.destDir) {
        $("#openDestBtn").show();
        openExportDestination();
      }
    }
  }
}

$(function () {
  loadSummary();
  loadJerseyOptions();
  initExportDialog();
  initImportMoreDialog();
  initRerunFacerecoDialog();
  initDeepRegradeDialog();

  $("#minStarsInput").on("change", updateExportCount);

  $("#regradeSensitivitySlider").on("input", function () {
    $("#regradeSensitivityValue").text(Number($(this).val()).toFixed(2));
  });

  $("#regradeBtn").on("click", async () => {
    setRegradeUIEnabled(false);
    $("#regradeStatus").text("Regrading…");
    const { mode, customValue } = readRegradeSensitivityFromUI();
    try {
      const res = await apiPost("/api/apply/regrade-sensitivity", {
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
    $("#deepRegradeOutput").val("");
    deepRegradePollSince = 0;
    const { mode, customValue } = readRegradeSensitivityFromUI();
    try {
      await apiPost("/api/apply/regrade-sensitivity", {
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        mode: "deep",
      });
      openDeepRegradeDialog();
      $("#regradeStatus").text("Deep regrade running…");
      deepRegradePollTimer = setInterval(pollDeepRegradeOutput, 500);
    } catch (err) {
      $("#regradeStatus").text("Deep regrade failed to start: " + err.message);
      setRegradeUIEnabled(true);
    }
  });

  $("#jerseyColorBtn").on("click", async () => {
    const teamColor = $("#jerseyColorSelect").val();
    const rerunLlmCulling = $("#jerseyRerunLlmInput").is(":checked");
    $("#jerseyColorBtn, #jerseyColorSelect, #jerseyRerunLlmInput").prop("disabled", true);
    setRegradeUIEnabled(false);
    $("#jerseyColorStatus").text("Applying…");
    try {
      const res = await apiPost("/api/apply/jersey-color", { teamColor, rerunLlmCulling });
      const parts = [];
      if (res.recovered) parts.push(`${res.recovered} moved to Sharp`);
      if (res.demoted) parts.push(`${res.demoted} moved to Blur`);
      if (res.starsRebaselined) parts.push(`${res.starsRebaselined} star rating(s) reset`);
      $("#jerseyColorStatus").text(
        `Team colour: ${res.teamColor || "none detected"} (${res.pinned ? "pinned" : "auto"})` +
        (parts.length ? ` — ${parts.join(", ")}.` : " — no changes.")
      );
      await loadSummary();
      await loadJerseyOptions();
      if (res.llmRerunStarted) {
        $("#jerseyColorStatus").append(" LLM re-cull running…");
        pollJerseyLlmRerun();
        return; // buttons stay disabled until the background job finishes
      }
    } catch (err) {
      $("#jerseyColorStatus").text("Failed: " + err.message);
    }
    $("#jerseyColorBtn, #jerseyColorSelect, #jerseyRerunLlmInput").prop("disabled", false);
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

    $("#importMoreOutput").val("");
    importMorePollSince = 0;
    openImportMoreDialog();
    $("#importMoreStatus").text("Importing…");
    try {
      await apiPost("/api/import-more", { path });
      importMorePollTimer = setInterval(pollImportMoreOutput, 500);
    } catch (err) {
      $("#importMoreStatus").text("Failed: " + err.message);
      setImportMoreUIEnabled(true);
      $("#importMoreDialog").dialog("close");
    }
  });

  $("#rerunFacerecoBtn").on("click", async () => {
    setRerunFacerecoUIEnabled(false);
    $("#rerunFacerecoStatus").text("Starting…");
    $("#rerunFacerecoOutput").val("");
    rerunFacerecoPollSince = 0;
    openRerunFacerecoDialog();
    $("#rerunFacerecoStatus").text("Re-running face detection…");
    try {
      await apiPost("/api/apply/rerun-facereco", {});
      rerunFacerecoPollTimer = setInterval(pollRerunFacerecoOutput, 500);
    } catch (err) {
      $("#rerunFacerecoStatus").text("Failed: " + err.message);
      setRerunFacerecoUIEnabled(true);
      $("#rerunFacerecoDialog").dialog("close");
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
    $("#exportOutput").val("");
    exportPollSince = 0;
    openExportDialog();
    try {
      await apiPost("/api/apply/export", { destination: res.path, exportFaceTagging, minStars });
      exportPollTimer = setInterval(pollExportStatus, 500);
    } catch (err) {
      $("#exportDialogStatus").text("Failed: " + err.message);
      finishExportDialog(false);
      setExportUIEnabled(true);
    }
  });
});

