"use strict";

// Page 4 — Face clustering: native review UI (thumbnails / assign / delete)
// for the current album's .FaceReco clusters, ported from the standalone
// face_tag_ui.py tool so it looks and feels like every other step instead of
// being embedded from a separate server.

const ClusterApp = {
  album: null,
  names: [],
  clusters: [],
  // Staged, un-saved operations keyed by `${clusterId}::${crop}`:
  //   { assignedName: string|null, pendingDelete: bool }
  staged: new Map(),
};

function ckey(clusterId, crop) { return clusterId + "::" + crop; }

function stateFor(clusterId, crop) {
  const k = ckey(clusterId, crop);
  if (!ClusterApp.staged.has(k)) ClusterApp.staged.set(k, { assignedName: null, pendingDelete: false });
  return ClusterApp.staged.get(k);
}

function isDirty() {
  for (const st of ClusterApp.staged.values()) {
    if (st.pendingDelete || st.assignedName) return true;
  }
  return false;
}

function refreshDirty() {
  const dirty = isDirty();
  $("#clusterSaveBtn").prop("disabled", !dirty);
  $("#clusterDirtyBadge").toggle(dirty);
}

// ------------------------------------------------------------------ //
// Load
// ------------------------------------------------------------------ //
async function initCluster() {
  $("#clusterEmpty").hide();
  $("#clusterReady").hide();
  $("#clusterStatus").text("Loading\u2026");

  let current;
  try {
    current = await apiGet("/api/current-album");
  } catch (err) {
    $("#clusterStatus").text("Failed to load current album: " + err.message);
    return;
  }

  if (!current.album) {
    $("#clusterEmpty").show();
    $("#clusterStatus").text("");
    return;
  }

  if (!current.album.hasFaceReco) {
    $("#clusterStatus").text("This album has no face-recognition data (.FaceReco) to review.");
    return;
  }

  try {
    const data = await apiGet("/api/cluster/data");
    ClusterApp.album = data.album;
    ClusterApp.names = data.names || [];
    ClusterApp.clusters = data.clusters || [];
    ClusterApp.staged.clear();

    $("#clusterAlbumName").text(ClusterApp.album.name);
    $("#clusterAlbumSrc").text(ClusterApp.album.srcDir ? "Source: " + ClusterApp.album.srcDir : "");
    $("#clusterReady").show();
    $("#clusterStatus").text("");
    renderBands();
    refreshDirty();
  } catch (err) {
    $("#clusterStatus").text("Failed to load clusters: " + err.message);
  }
}

// ------------------------------------------------------------------ //
// Rendering
// ------------------------------------------------------------------ //
function renderBands() {
  const container = $("#clusterBands").empty();
  if (ClusterApp.clusters.length === 0) {
    container.append($("<p class='muted'>").text("No face clusters found in this album."));
    return;
  }
  ClusterApp.clusters.forEach((cluster) => container.append(renderBand(cluster)));
}

function renderBand(cluster) {
  const band = $("<div class='band'>").attr("data-cluster", cluster.id);

  const kindText = cluster.pending
    ? "pending"
    : "matched" + (cluster.playernum != null ? " \u00b7 #" + cluster.playernum : "");
  const header = $("<div class='band-header'>").append(
    $("<span class='band-title'>")
      .text(cluster.pending ? "Cluster " + cluster.id : cluster.name)
      .append($("<span>").addClass("kind").addClass(cluster.pending ? "pending" : "matched").text(kindText)),
    $("<span class='band-count'>").text(cluster.faces.length + " face(s)")
  );
  header.on("click", () => selectBand(band));
  band.append(header);

  const tools = $("<div class='band-tools'>");
  const selectAll = $("<input type='checkbox' class='select-all'>");
  selectAll.on("change", function () {
    band.find(".thumb-check").prop("checked", this.checked).each(function () {
      $(this).closest(".thumb").toggleClass("checked", this.checked);
    });
  });
  const nameInput = $("<input type='text' placeholder='Player name\u2026'>");
  nameInput.autocomplete({
    source: ClusterApp.names,
    minLength: 0,
    delay: 0,
    select: function (event, ui) {
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
  const applyBtn = $("<button class='btn btn-primary btn-sm'>").text("Apply")
    .on("click", () => applyBand(band, nameInput.val()));
  const deleteBtn = $("<button class='btn btn-danger btn-sm'>").text("Delete")
    .on("click", () => {
      setDeleteForChecked(band, true);
      focusNextBand(band);
    });
  const undeleteBtn = $("<button class='btn btn-sm'>").text("Undelete")
    .on("click", () => setDeleteForChecked(band, false));

  tools.append(
    $("<label>").append(selectAll, $("<span>").text("Select all")),
    nameInput, applyBtn, deleteBtn, undeleteBtn
  );
  band.append(tools);

  const thumbs = $("<div class='thumbs'>");
  cluster.faces.forEach((face) => thumbs.append(renderThumb(cluster, face)));
  band.append(thumbs);

  // Clicking anywhere in the band's content area (thumbnails) also selects
  // the band, without toggling it back off the way the header does. Clicks
  // inside .band-tools are excluded: those controls are only visible once
  // the band is already selected, and some of them (Apply/Delete) move
  // selection to a *different* band via focusNextBand() — letting this
  // click bubble up would immediately re-select the original band here and
  // stomp on that navigation.
  band.on("click", function (e) {
    if ($(e.target).closest(".band-header").length) return;
    if ($(e.target).closest(".band-tools").length) return;
    activateBand(band);
  });

  return band;
}

function renderThumb(cluster, face) {
  const st = ClusterApp.staged.get(ckey(cluster.id, face.crop)) || {};
  const wrap = $("<div class='thumb'>")
    .attr("data-crop", face.crop)
    .toggleClass("pending-delete", !!st.pendingDelete);

  const check = $("<input type='checkbox' class='thumb-check'>");
  check.on("change", function () {
    wrap.toggleClass("checked", this.checked);
  });

  const thumbUrl = "/api/cluster/thumb/" + encodeURIComponent(cluster.id) + "/" + encodeURIComponent(face.crop);
  const originalUrl = "/api/cluster/original?file=" + encodeURIComponent(face.origFilename);

  const img = $("<img>").attr("src", thumbUrl).attr("title", face.origFilename)
    .on("click", () => Viewport.showImageWindow(originalUrl));

  const mask = $("<div class='mask'>").text("DELETE");
  const badge = $("<div class='assigned-badge'>").text(st.assignedName ? "\u2192 " + st.assignedName : "");

  wrap.append(check, mask, img, badge);
  return wrap;
}

// ------------------------------------------------------------------ //
// Band interactions
// ------------------------------------------------------------------ //
let selectedBand = null;

// Selects `band`, replacing any previous selection. No-op if it is already
// the selected band (unlike selectBand(), this never toggles it back off —
// used for interactions that should always land on a specific band).
function activateBand(band) {
  if (selectedBand && selectedBand[0] === band[0]) return;
  $(".band.selected").removeClass("selected");
  band.addClass("selected");
  selectedBand = band;
  // Selecting a band auto-selects all thumbnails in the cluster.
  band.find(".thumb-check").prop("checked", true).each(function () {
    $(this).closest(".thumb").addClass("checked");
  });
  band.find(".select-all").prop("checked", true);
}

function selectBand(band) {
  if (selectedBand && selectedBand[0] === band[0]) {
    band.removeClass("selected");
    selectedBand = null;
    return;
  }
  activateBand(band);
}

// After a name is applied to a band, move on to the next band and focus its
// player-name box so a whole album can be tagged without touching the mouse.
// The page is scrolled by the exact amount needed to put the next band's top
// edge where the previous band's top edge used to be on screen (a "conveyor
// belt" effect), clamped so it never scrolls past the bottom of the page.
function focusNextBand(band) {
  const bands = $("#clusterBands > .band");
  const idx = bands.index(band[0]);
  if (idx === -1) return;
  const next = bands.eq(idx + 1);
  if (next.length === 0) return;

  const prevTop = band[0].getBoundingClientRect().top;
  activateBand(next);
  const nextInput = next.find("input[type='text']");
  nextInput.trigger("focus");

  const delta = next[0].getBoundingClientRect().top - prevTop;
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  const targetScroll = Math.max(0, Math.min(window.scrollY + delta, maxScroll));
  window.scrollTo({ top: targetScroll, behavior: "smooth" });
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
    $(this).find(".assigned-badge").text("\u2192 " + name);
  });
  refreshDirty();
  focusNextBand(band);
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

// [Del] key: delete checked thumbs in the selected band, unless focus is in
// an input (name textbox) or on a button.
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
async function saveCluster() {
  const operations = [];
  for (const [k, st] of ClusterApp.staged.entries()) {
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

  $("#clusterSaveBtn").prop("disabled", true).text("Saving\u2026");
  try {
    const res = await apiPost("/api/cluster/commit", { operations });
    ClusterApp.clusters = res.clusters || [];
    ClusterApp.staged.clear();
    selectedBand = null;
    renderBands();
  } catch (err) {
    alert("Save failed: " + err.message);
  } finally {
    $("#clusterSaveBtn").text("Save");
    refreshDirty();
  }
}

// ------------------------------------------------------------------ //
// Boot
// ------------------------------------------------------------------ //
$(function () {
  $("#clusterSaveBtn").on("click", saveCluster);

  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });

  initCluster();
});


$(initCluster);
