"use strict";

// Page 2 — Select Album: create a new album (source folder, blur
// sensitivity, face recognition) in the box up top, or pick a previously
// processed one from the polaroid gallery below.

let pollTimer = null;
let pollSince = 0;

// Hover slideshow tick rate for polaroid preview cycling, in ms.
const SLIDESHOW_INTERVAL_MS = 700;

// ------------------------------------------------------------------ //
// Previously processed albums — polaroid gallery
// ------------------------------------------------------------------ //
function albumThumbUrl(album, file) {
  return `/api/albums/thumb?id=${encodeURIComponent(album.name)}&file=${encodeURIComponent(file)}`;
}

function buildPolaroidCard(album) {
  const urls = (album.previewImages || []).map((f) => albumThumbUrl(album, f));
  const card = $("<div>", { class: "polaroid" });
  const photo = $("<div>", { class: "polaroid-photo" });

  if (urls.length) {
    const startUrl = urls[Math.floor(Math.random() * urls.length)];
    const img = $("<img>", { src: startUrl, alt: album.name });
    img.data("original", startUrl);
    img.data("urls", urls);
    photo.append(img);
  } else {
    photo.append($("<div>", { class: "polaroid-empty muted" }).text("No preview"));
  }

  const deleteBtn = $("<button>", {
    type: "button",
    class: "polaroid-delete",
    title: "Delete album",
    html: "&times;",
  });
  photo.append(deleteBtn);

  const caption = $("<div>", { class: "polaroid-caption" });
  caption.append($("<div>", { class: "polaroid-name" }).text(album.name));
  caption.append($("<div>", { class: "polaroid-date" }).text(album.createdDisplay || ""));

  card.append(photo, caption);

  // Hover slideshow: cycle the thumbnail through a random sample of the
  // album's sharp images while hovered; reset to the original still on
  // mouse-out. (The delete button's fade in/out is pure CSS — see
  // .polaroid:hover .polaroid-delete in style.css.)
  let slideTimer = null;
  card.on("mouseenter", () => {
    const img = card.find("img");
    const imgUrls = img.data("urls");
    if (!imgUrls || imgUrls.length < 2) return;
    slideTimer = setInterval(() => {
      img.attr("src", imgUrls[Math.floor(Math.random() * imgUrls.length)]);
    }, SLIDESHOW_INTERVAL_MS);
  });
  card.on("mouseleave", () => {
    if (slideTimer) { clearInterval(slideTimer); slideTimer = null; }
    const img = card.find("img");
    const original = img.data("original");
    if (original) img.attr("src", original);
  });

  deleteBtn.on("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete album "${album.name}"? This permanently removes its output folder and cannot be undone.`)) {
      return;
    }
    deleteBtn.prop("disabled", true);
    try {
      await apiPost("/api/albums/delete", { id: album.name });
      if (slideTimer) clearInterval(slideTimer);
      card.fadeOut(180, function () {
        $(this).remove();
        if (!$("#albumGallery").children().length) $("#noAlbums").show();
      });
    } catch (err) {
      alert("Failed to delete: " + err.message);
      deleteBtn.prop("disabled", false);
    }
  });

  card.on("click", async () => {
    try {
      await apiPost("/api/albums/select", { id: album.name });
      window.location.href = "/cluster";
    } catch (err) {
      alert("Failed to resume: " + err.message);
    }
  });

  return card;
}

async function loadAlbums() {
  const data = await apiGet("/api/albums");
  const gallery = $("#albumGallery").empty();
  $("#noAlbums").toggle(data.albums.length === 0);
  data.albums.forEach((album) => gallery.append(buildPolaroidCard(album)));
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
// Import state (last-used folder / sensitivity / face-recognition choice)
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
    $("#startProcessingBtn").prop("disabled", false);
    $("#importStatus").text(
      data.returnCode === 0
        ? "Processing complete."
        : `Processing exited with code ${data.returnCode}.`
    );
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
    const path = $("#importPathInput").val().trim();
    if (!path) { $("#importStatus").text("Choose a folder first."); return; }
    const { mode, customValue } = readSensitivityFromUI();
    const recognizeFaces = $("#recognizeFacesInput").is(":checked");
    $("#startProcessingBtn").prop("disabled", true);
    $("#importStatus").text("Saving import path…");
    $("#processingSection").show();
    $("#processingOutput").val("");
    pollSince = 0;
    try {
      await apiPost("/api/import-path", { path });
      $("#importStatus").text("Processing…");
      await apiPost("/api/start-processing", {
        path,
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        recognizeFaces,
      });
      pollTimer = setInterval(pollProcessingOutput, 500);
    } catch (err) {
      $("#importStatus").text("Failed: " + err.message);
      $("#startProcessingBtn").prop("disabled", false);
    }
  });

  initDropZone();
  loadImportState().catch(() => {});
  loadAlbums().catch(() => {});
});
