"use strict";

// Page 5 — Apply: show a quick album summary, then let the user pick a
// destination folder and export kept photos (+ players.csv) — syncing
// tagged face crops into the system-wide .FaceReco database and rebuilding
// it first — with progress polling.

let exportPollTimer = null;
let exportPollSince = 0;

function setExportUIEnabled(enabled) {
  $("#exportFaceTaggingInput, #exportBtn").prop("disabled", !enabled);
}

function appendExportLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("exportOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function loadSummary() {
  try {
    const data = await apiGet("/api/apply/summary");
    $("#summaryAlbumName").text(data.name || "Album");
    $("#statKept").text(data.imagesKept);
    $("#statDropped").text(data.imagesDropped);
    $("#statFaces").text(data.facesDetected);
    $("#statPlayers").text(data.playersDetected);
  } catch (err) {
    $("#summaryAlbumName").text("Album");
  }
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
    $("#exportStatus").text(
      `Copying photos: ${data.copiedImages}/${data.totalImages} · ` +
      `Faces: ${data.processedPlayers}/${data.totalPlayers} player(s)`
    );
  }
  if (!data.running) {
    clearInterval(exportPollTimer);
    exportPollTimer = null;
    setExportUIEnabled(true);
    if (data.error) {
      $("#exportStatus").text("Export failed: " + data.error);
    } else {
      $("#exportStatus").text(`Done — ${data.copiedImages} photo(s) exported.`);
    }
  }
}

$(function () {
  loadSummary();

  $("#exportBtn").on("click", async () => {
    setExportUIEnabled(false);
    $("#exportStatus").text("Choose a destination folder…");
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select export destination folder" });
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
    $("#exportStatus").text("Starting export…");
    $("#exportOutputSection").show();
    $("#exportOutput").val("");
    exportPollSince = 0;
    try {
      await apiPost("/api/apply/export", { destination: res.path, exportFaceTagging });
      exportPollTimer = setInterval(pollExportStatus, 500);
    } catch (err) {
      $("#exportStatus").text("Failed: " + err.message);
      setExportUIEnabled(true);
    }
  });
});

