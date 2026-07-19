"use strict";

// Page 2 — Import

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

  $("#startProcessingBtn").on("click", async () => {
    const path = $("#importPathInput").val().trim();
    if (!path) { $("#importStatus").text("Choose a folder first."); return; }
    $("#startProcessingBtn").prop("disabled", true);
    $("#importStatus").text("Saving import path…");
    try {
      await apiPost("/api/import-path", { path });
      $("#importStatus").text("Import path saved. Processing not yet implemented.");
    } catch (err) {
      $("#importStatus").text("Failed: " + err.message);
    } finally {
      $("#startProcessingBtn").prop("disabled", false);
    }
  });

  loadImportState().catch(() => {});
});
