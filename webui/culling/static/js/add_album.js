"use strict";

// Add Album page — create a new album (source folder, blur sensitivity,
// face recognition) and pick which team's settings to use for it.

let pollTimer = null;
let pollSince = 0;

// True while a processing run is in flight — used to disable the whole
// screen and to warn the user if they try to navigate away.
let isProcessing = false;

// Teams loaded from the server, and which one is currently selected for
// this album (used when Start Processing is clicked).
let teams = [];
let selectedTeamId = "";

// ------------------------------------------------------------------ //
// Team picker
// ------------------------------------------------------------------ //
function renderTeamPickerUI() {
  renderTeamPicker($("#teamPicker"), teams, {
    selectedId: selectedTeamId,
    onSelect: (team) => {
      selectedTeamId = team.id;
      renderTeamPickerUI();
    },
    onAddNew: () => { window.location.href = "/team"; },
    addLabel: "Create a New Team",
  });
}

async function loadTeams(preferredTeamId) {
  const data = await apiGet("/api/teams/query");
  teams = data.teams || [];
  const hasPreferred = preferredTeamId && teams.some((t) => t.id === preferredTeamId);
  selectedTeamId = hasPreferred ? preferredTeamId : (teams[0] ? teams[0].id : "");
  renderTeamPickerUI();
}

// ------------------------------------------------------------------ //
// Blur sensitivity
// ------------------------------------------------------------------ //
function applySensitivityToUI(mode, customValue) {
  $(`input[name="sensMode"][value="${mode || "medium"}"]`).prop("checked", true);
  const custom = customValue ?? 0.50;
  $("#sensitivityCustomSlider").val(custom);
  $("#sensitivityCustomValue").text(Number(custom).toFixed(2));
}

function readSensitivityFromUI() {
  const mode = $('input[name="sensMode"]:checked').val() || "medium";
  const customValue = parseFloat($("#sensitivityCustomSlider").val()) || 0;
  return { mode, customValue };
}

// ------------------------------------------------------------------ //
// Import state (last-used folder / sensitivity / face-recognition / team)
// ------------------------------------------------------------------ //
async function loadImportState() {
  const state = await apiGet("/api/import-state");
  if (state.lastImportPath) {
    $("#importPathInput").val(state.lastImportPath);
    $("#lastImportHint").text(
      state.lastImportPathExists
        ? "Last used: " + state.lastImportPath
        : "Last used (not found): " + state.lastImportPath
    );
  } else {
    $("#lastImportHint").text("");
  }
  applySensitivityToUI(state.sensitivityMode, state.sensitivityCustomValue);
  $("#recognizeFacesInput").prop("checked", state.recognizeFaces !== false);
  return state.selectedTeamId || "";
}

// ------------------------------------------------------------------ //
// Disable/re-enable the whole screen while a processing run is active.
// ------------------------------------------------------------------ //
function setImportUIEnabled(enabled) {
  isProcessing = !enabled;
  $("#createAlbumPanel").find("input, button").prop("disabled", !enabled);
  $("#teamPanel").find("button").prop("disabled", !enabled);
  $("#startProcessingBtn").prop("disabled", !enabled);
}

// ------------------------------------------------------------------ //
// Processing output polling
// ------------------------------------------------------------------ //
function appendOutputLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("processingOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function pollProcessingOutput() {
  let data;
  try {
    data = await apiGet(`/api/processing-output?since=${pollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendOutputLines(data.lines);
  pollSince = data.next;
  if (!data.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    setImportUIEnabled(true);
    if (data.returnCode === 0) {
      $("#importStatus").text("Processing complete. Continuing to Culling…");
      setTimeout(() => { window.location.href = "/review"; }, 800);
    } else {
      $("#importStatus").text(`Processing exited with code ${data.returnCode}.`);
    }
  }
}

// ------------------------------------------------------------------ //
// Drag & drop — the whole "Create New Album" box is a drop target.
//
// Browsers deliberately do not expose the absolute filesystem path of a
// dropped file/folder (a security measure to stop web pages probing the
// local filesystem), so we can only recover it when the non-standard
// File.path property happens to be present (e.g. some embedded/Electron
// browsers). Otherwise we tell the user what we detected and ask them to
// use Browse… instead, rather than silently failing.
// ------------------------------------------------------------------ //
function initDropZone() {
  const zone = document.getElementById("createAlbumPanel");
  if (!zone) return;
  let dragCounter = 0;

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    zone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); });
  });

  zone.addEventListener("dragenter", () => {
    dragCounter += 1;
    zone.classList.add("drag-active");
  });
  zone.addEventListener("dragleave", () => {
    dragCounter = Math.max(0, dragCounter - 1);
    if (dragCounter === 0) zone.classList.remove("drag-active");
  });
  zone.addEventListener("drop", (e) => {
    dragCounter = 0;
    zone.classList.remove("drag-active");
    const dt = e.dataTransfer;
    const item = dt && dt.items && dt.items[0];
    const entry = item && item.webkitGetAsEntry && item.webkitGetAsEntry();
    const file = item && item.getAsFile && item.getAsFile();
    const path = file && file.path; // non-standard; usually undefined in a normal browser tab

    if (path) {
      $("#importPathInput").val(path);
      $("#dropHint").text("");
      return;
    }
    if (entry && !entry.isDirectory) {
      $("#dropHint").text("Please drop a folder, not a file.");
      return;
    }
    const name = (entry && entry.name) || (file && file.name) || "the dropped item";
    $("#dropHint").text(
      `Browsers don't expose the full path of dropped folders for security reasons. ` +
      `Detected "${name}" — use Browse… above to select it, or type the path directly.`
    );
  });
}

$(function () {
  $("#browseFolderBtn").on("click", async () => {
    $("#browseFolderBtn").prop("disabled", true);
    try {
      const res = await apiPost("/api/browse-folder");
      if (res.path) $("#importPathInput").val(res.path);
    } catch (err) {
      alert("Browse failed: " + err.message);
    } finally {
      $("#browseFolderBtn").prop("disabled", false);
    }
  });

  $("#sensitivityCustomSlider").on("input", function () {
    $("#sensitivityCustomValue").text(Number($(this).val()).toFixed(2));
    $('input[name="sensMode"][value="custom"]').prop("checked", true);
  });

  $("#startProcessingBtn").on("click", async () => {
    const albumName = $("#albumNameInput").val().trim();
    if (!albumName) { $("#importStatus").text("Album name is required."); $("#albumNameInput").trigger("focus"); return; }
    const path = $("#importPathInput").val().trim();
    if (!path) { $("#importStatus").text("Choose a folder first."); return; }
    if (!selectedTeamId) { $("#importStatus").text("Select a team first."); return; }
    const { mode, customValue } = readSensitivityFromUI();
    const recognizeFaces = $("#recognizeFacesInput").is(":checked");
    setImportUIEnabled(false);
    $("#importStatus").text("Saving import path…");
    $("#processingSection").show();
    $("#processingOutput").val("");
    pollSince = 0;
    try {
      await apiPost("/api/import-path", { path });
      $("#importStatus").text("Processing…");
      await apiPost("/api/start-processing", {
        path,
        albumName,
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        recognizeFaces,
        teamId: selectedTeamId,
      });
      pollTimer = setInterval(pollProcessingOutput, 500);
    } catch (err) {
      $("#importStatus").text("Failed: " + err.message);
      setImportUIEnabled(true);
    }
  });

  window.addEventListener("beforeunload", (e) => {
    if (isProcessing) { e.preventDefault(); e.returnValue = ""; }
  });

  initDropZone();
  (async () => {
    const preferredTeamId = await loadImportState().catch(() => "");
    await loadTeams(preferredTeamId).catch(() => {});
  })();
});
