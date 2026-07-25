"use strict";

// ------------------------------------------------------------------ //
// Client state
// ------------------------------------------------------------------ //
const App = {
  fixedAlbum: false,
  album: null,        // album name
  srcDir: null,
  names: [],          // autocomplete dictionary
  clusters: [],       // as returned by the server
  // Staged, un-saved operations keyed by `${clusterId}::${crop}`:
  //   { assignedName: string|null, pendingDelete: bool }
  staged: new Map(),
};

function key(clusterId, crop) { return clusterId + "::" + crop; }

function stateFor(clusterId, crop) {
  const k = key(clusterId, crop);
  if (!App.staged.has(k)) App.staged.set(k, { assignedName: null, pendingDelete: false });
  return App.staged.get(k);
}

function isDirty() {
  for (const st of App.staged.values()) {
    if (st.pendingDelete || st.assignedName) return true;
  }
  return false;
}

function refreshDirty() {
  const dirty = isDirty();
  $("#saveBtn").prop("disabled", !dirty);
  $("#dirtyBadge").toggle(dirty);
}

// ------------------------------------------------------------------ //
// API helpers
// ------------------------------------------------------------------ //
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

// ------------------------------------------------------------------ //
// Home view
// ------------------------------------------------------------------ //
async function loadHome() {
  const data = await apiGet("/api/albums");
  App.fixedAlbum = data.fixed;
  if (data.fixed && data.albums.length === 1) {
    openAlbum(data.albums[0]);
    return;
  }
  showHome();
  const list = $("#albumList").empty();
  $("#noAlbums").toggle(data.albums.length === 0);
  data.albums.forEach((name) => {
    $("<li>").text(name).on("click", () => openAlbum(name)).appendTo(list);
  });
}

function showHome() {
  $("#homeView").show();
  $("#albumView").hide();
  $("#backBtn").hide();
  $("#saveBtn").hide();
  $("#dirtyBadge").hide();
  $("#title").text("Face Tag UI");
}

// ------------------------------------------------------------------ //
// Album view
// ------------------------------------------------------------------ //
async function openAlbum(name) {
  const data = await apiGet("/api/albums/" + encodeURIComponent(name));
  App.album = data.album;
  App.srcDir = data.srcDir;
  App.names = data.names || [];
  App.clusters = data.clusters || [];
  App.staged.clear();

  $("#homeView").hide();
  $("#albumView").show();
  $("#backBtn").toggle(!App.fixedAlbum);
  $("#saveBtn").show();
  $("#title").text(App.album);
  $("#albumMeta").text(App.srcDir ? "Source: " + App.srcDir : "Source directory unknown");

  renderBands();
  refreshDirty();
}

function renderBands() {
  const container = $("#bands").empty();
  if (App.clusters.length === 0) {
    container.append($("<p class='muted'>").text("No clusters found in this album."));
    return;
  }
  App.clusters.forEach((cluster) => container.append(renderBand(cluster)));
}

function renderBand(cluster) {
  const band = $("<div class='band'>").attr("data-cluster", cluster.id);

  // Header (selectable)
  const kindText = cluster.pending
    ? "pending"
    : "matched" + (cluster.playernum != null ? " · #" + cluster.playernum : "");
  const header = $("<div class='band-header'>").append(
    $("<span class='band-title'>")
      .text(cluster.pending ? "Cluster " + cluster.id : cluster.name)
      .append($("<span>").addClass("kind").addClass(cluster.pending ? "pending" : "matched").text(kindText)),
    $("<span class='band-count'>").text(cluster.faces.length + " face(s)")
  );
  header.on("click", () => selectBand(band));
  band.append(header);

  // Tools (shown when band selected)
  const tools = $("<div class='band-tools'>");
  const selectAll = $("<input type='checkbox' class='select-all'>");
  selectAll.on("change", function () {
    band.find(".thumb-check").prop("checked", this.checked).each(function () {
      $(this).closest(".thumb").toggleClass("checked", this.checked);
    });
  });
  const nameInput = $("<input type='text' placeholder='Player name…'>");
  nameInput.autocomplete({
    source: App.names,
    minLength: 0,
    delay: 0,
    select: function (event, ui) {
      // Selecting an item (keyboard or mouse) applies immediately.
      nameInput.val(ui.item.value);
      applyBand(band, ui.item.value);
      return false;
    },
  }).on("focus", function () { $(this).autocomplete("search", $(this).val()); });
  nameInput.on("keydown", function (e) {
    if (e.key === "Enter") {
      const menu = nameInput.autocomplete("widget");
      const activeItem = menu.is(":visible") && menu.find(".ui-state-active").length > 0;
      if (!activeItem) {
        e.preventDefault();
        applyBand(band, nameInput.val());
      }
    }
  });
  const applyBtn = $("<button class='btn btn-primary'>").text("Apply")
    .on("click", () => applyBand(band, nameInput.val()));
  const deleteBtn = $("<button class='btn btn-danger'>").text("Delete")
    .on("click", () => setDeleteForChecked(band, true));
  const undeleteBtn = $("<button class='btn'>").text("Undelete")
    .on("click", () => setDeleteForChecked(band, false));

  tools.append(
    $("<label>").append(selectAll, $("<span>").text("Select all")),
    nameInput, applyBtn, deleteBtn, undeleteBtn
  );
  band.append(tools);

  // Thumbnails
  const thumbs = $("<div class='thumbs'>");
  cluster.faces.forEach((face) => thumbs.append(renderThumb(cluster, face)));
  band.append(thumbs);

  return band;
}

function renderThumb(cluster, face) {
  const st = App.staged.get(key(cluster.id, face.crop)) || {};
  const wrap = $("<div class='thumb'>")
    .attr("data-crop", face.crop)
    .toggleClass("pending-delete", !!st.pendingDelete);

  const check = $("<input type='checkbox' class='thumb-check'>");
  check.on("change", function () {
    wrap.toggleClass("checked", this.checked);
  });

  const thumbUrl = "/thumb/" + encodeURIComponent(App.album) + "/" +
    encodeURIComponent(cluster.id) + "/" + encodeURIComponent(face.crop);
  const originalUrl = "/original/" + encodeURIComponent(App.album) +
    "?file=" + encodeURIComponent(face.origFilename);

  const img = $("<img>").attr("src", thumbUrl).attr("title", face.origFilename)
    .on("click", () => window.open(originalUrl, "_blank"));

  const mask = $("<div class='mask'>").text("DELETE");
  const badge = $("<div class='assigned-badge'>").text(st.assignedName ? "→ " + st.assignedName : "");

  wrap.append(check, mask, img, badge);
  return wrap;
}

// ------------------------------------------------------------------ //
// Band interactions
// ------------------------------------------------------------------ //
let selectedBand = null;

function selectBand(band) {
  if (selectedBand && selectedBand[0] === band[0]) {
    band.removeClass("selected");
    selectedBand = null;
    return;
  }
  $(".band.selected").removeClass("selected");
  band.addClass("selected");
  selectedBand = band;
  // Selecting a band auto-selects all thumbnails in the cluster.
  band.find(".thumb-check").prop("checked", true).each(function () {
    $(this).closest(".thumb").addClass("checked");
  });
  band.find(".select-all").prop("checked", true);
}

function checkedThumbs(band) {
  return band.find(".thumb-check:checked").closest(".thumb");
}

function applyBand(band, name) {
  name = (name || "").trim();
  if (!name) return;
  const clusterId = band.attr("data-cluster");
  const thumbs = checkedThumbs(band);
  if (thumbs.length === 0) return;
  thumbs.each(function () {
    const crop = $(this).attr("data-crop");
    const st = stateFor(clusterId, crop);
    st.assignedName = name;
    $(this).find(".assigned-badge").text("→ " + name);
  });
  refreshDirty();
}

function setDeleteForChecked(band, pending) {
  const clusterId = band.attr("data-cluster");
  const thumbs = checkedThumbs(band);
  if (thumbs.length === 0) return;
  thumbs.each(function () {
    const crop = $(this).attr("data-crop");
    const st = stateFor(clusterId, crop);
    st.pendingDelete = pending;
    $(this).toggleClass("pending-delete", pending);
  });
  refreshDirty();
}

// [Del] key: delete checked thumbs in the selected band, unless focus is in an
// input (name textbox) or on a button.
$(document).on("keydown", function (e) {
  if (e.key !== "Delete") return;
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON") return;
  if (!selectedBand) return;
  e.preventDefault();
  setDeleteForChecked(selectedBand, true);
});

// ------------------------------------------------------------------ //
// Save
// ------------------------------------------------------------------ //
async function save() {
  const operations = [];
  for (const [k, st] of App.staged.entries()) {
    const sep = k.indexOf("::");
    const clusterId = k.slice(0, sep);
    const crop = k.slice(sep + 2);
    if (st.pendingDelete) {
      operations.push({ type: "delete", sourceCluster: clusterId, crop });
    } else if (st.assignedName) {
      operations.push({ type: "assign", sourceCluster: clusterId, crop, name: st.assignedName });
    }
  }
  if (operations.length === 0) return;

  $("#saveBtn").prop("disabled", true).text("Saving…");
  try {
    const res = await apiPost(
      "/api/albums/" + encodeURIComponent(App.album) + "/commit",
      { operations }
    );
    App.clusters = res.clusters || [];
    App.staged.clear();
    selectedBand = null;
    renderBands();
  } catch (err) {
    alert("Save failed: " + err.message);
  } finally {
    $("#saveBtn").text("Save");
    refreshDirty();
  }
}

// ------------------------------------------------------------------ //
// Heartbeat — tells the server this page is still open. The server exits
// automatically if it stops receiving these (e.g. the tab/browser was
// closed), so we don't leave orphaned local servers running.
//
// Sent from a dedicated Worker (heartbeat-worker.js) instead of a plain
// setInterval on this page, because browsers throttle a backgrounded tab's
// own timers (Chrome clamps to ~once/minute after ~5 min hidden) — longer
// than the server's heartbeat timeout — which was causing the server to
// shut itself down mid-run whenever the tab was left in the background.
// Worker timers aren't throttled that way. Falls back to a plain
// setInterval if Workers aren't available.
// ------------------------------------------------------------------ //
const HEARTBEAT_INTERVAL_MS = 10000;

function sendHeartbeat() {
  // Best-effort; a single dropped heartbeat is fine, the server tolerates
  // multiple missed intervals before exiting.
  fetch("/api/heartbeat", { method: "POST" }).catch(() => {});
}

function startHeartbeat() {
  if (window.Worker) {
    try {
      new Worker("/static/heartbeat-worker.js");
      return;
    } catch (err) {
      // fall through to main-thread heartbeat below
    }
  }
  sendHeartbeat();
  setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);
}

// ------------------------------------------------------------------ //
// Boot
// ------------------------------------------------------------------ //
$(function () {
  $("#backBtn").on("click", () => {
    if (isDirty() && !confirm("Discard unsaved changes and return to albums?")) return;
    loadHome();
  });
  $("#saveBtn").on("click", save);

  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });

  startHeartbeat();

  loadHome().catch((err) => alert("Failed to load: " + err.message));
});
