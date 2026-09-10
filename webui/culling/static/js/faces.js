"use strict";

// Page 4 — Faces: native review UI (thumbnails / assign / delete)
// for the current album's .FaceReco clusters.

const FacesApp = {
  album: null,
  names: [],
  clusters: [],
  // Staged, un-saved operations keyed by `${clusterId}::${crop}`:
  //   { assignedName: string|null, pendingDelete: bool }
  staged: new Map(),
  // Pagination over the flat, ordered list of faces across all clusters.
  // A cluster that straddles a page boundary is simply rendered twice,
  // once per page, each time with the subset of faces that landed there.
  page: 1,
  pageSize: 300,
};

function ckey(clusterId, crop) { return clusterId + "::" + crop; }

// Shared observer that lazy-loads thumbnail <img> src just before it
// scrolls into view, instead of every thumbnail across every cluster
// fetching its image up front.
const thumbLazyLoadObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    const img = entry.target;
    img.src = img.dataset.src;
    thumbLazyLoadObserver.unobserve(img);
  });
}, { rootMargin: "400px" });

function stateFor(clusterId, crop) {
  const k = ckey(clusterId, crop);
  if (!FacesApp.staged.has(k)) FacesApp.staged.set(k, { assignedName: null, pendingDelete: false });
  return FacesApp.staged.get(k);
}

function isDirty() {
  for (const st of FacesApp.staged.values()) {
    if (st.pendingDelete || st.assignedName) return true;
  }
  return false;
}

function refreshDirty() {
  const dirty = isDirty();
  $("#facesSaveBtn").prop("disabled", !dirty);
  $("#facesDirtyBadge").toggle(dirty);
}

// ------------------------------------------------------------------ //
// Load
// ------------------------------------------------------------------ //
async function initFaces() {
  $("#facesEmpty").hide();
  $("#facesReady").hide();
  $("#facesStatus").text("Loading\u2026");

  let current;
  try {
    current = await apiGet("/api/current-album");
  } catch (err) {
    $("#facesStatus").text("Failed to load current album: " + err.message);
    return;
  }

  if (!current.album) {
    $("#facesEmpty").show();
    $("#facesStatus").text("");
    return;
  }

  if (!current.album.hasFaceReco) {
    $("#facesStatus").text("This album has no face-recognition data (.FaceReco) to review.");
    return;
  }

  try {
    const data = await apiGet("/api/faces/data");
    FacesApp.album = data.album;
    FacesApp.names = data.names || [];
    FacesApp.clusters = data.clusters || [];
    FacesApp.staged.clear();
    FacesApp.page = 1;

    $("#facesAlbumName").text(FacesApp.album.name);
    $("#facesAlbumSrc").text(FacesApp.album.srcDir ? "Source: " + FacesApp.album.srcDir : "");
    $("#facesReady").show();
    $("#facesStatus").text("");
    renderBands();
    renderPagination();
    refreshDirty();
  } catch (err) {
    $("#facesStatus").text("Failed to load clusters: " + err.message);
  }
}

// ------------------------------------------------------------------ //
// Pagination — flattens all clusters' faces, in order, into one list and
// slices it into fixed-size pages. A cluster whose faces cross a page
// boundary is split: each page only renders the faces that landed on it.
// ------------------------------------------------------------------ //
function buildFlatFaceList() {
  const flat = [];
  FacesApp.clusters.forEach((cluster) => {
    cluster.faces.forEach((face) => flat.push({ cluster, face }));
  });
  return flat;
}

function totalPageCount() {
  return Math.max(1, Math.ceil(buildFlatFaceList().length / FacesApp.pageSize));
}

function currentPageItems() {
  const flat = buildFlatFaceList();
  const start = (FacesApp.page - 1) * FacesApp.pageSize;
  return { flat, start, items: flat.slice(start, start + FacesApp.pageSize) };
}

// Groups the page's flat (cluster, face) pairs back into per-cluster runs,
// preserving order, so a split cluster still renders as one band per page.
function groupItemsByCluster(items) {
  const groups = [];
  let current = null;
  items.forEach(({ cluster, face }) => {
    if (!current || current.cluster.id !== cluster.id) {
      current = { cluster, faces: [] };
      groups.push(current);
    }
    current.faces.push(face);
  });
  return groups;
}

function goToPage(n) {
  const total = totalPageCount();
  n = Math.min(Math.max(1, Math.trunc(n) || 1), total);
  if (n === FacesApp.page) {
    renderPagination();
    return;
  }
  FacesApp.page = n;
  selectedBand = null;
  renderBands();
  renderPagination();
  window.scrollTo({ top: 0, behavior: "auto" });
}

function renderPagination() {
  const total = totalPageCount();
  const { flat, start, items } = currentPageItems();
  $(".faces-page-input").val(FacesApp.page).attr("max", total);
  $(".faces-page-total").text(total);
  $(".faces-first-page-btn, .faces-prev-page-btn").prop("disabled", FacesApp.page <= 1);
  $(".faces-next-page-btn, .faces-last-page-btn").prop("disabled", FacesApp.page >= total);
  $(".faces-page-range").text(
    flat.length ? "Faces " + (start + 1) + "\u2013" + (start + items.length) + " of " + flat.length : ""
  );
}

// ------------------------------------------------------------------ //
// Rendering
// ------------------------------------------------------------------ //
function renderBands() {
  const container = $("#facesBands").empty();
  lastClickedThumb = null;
  if (FacesApp.clusters.length === 0) {
    container.append($("<p class='muted'>").text("No face clusters found in this album."));
    return;
  }
  const { items } = currentPageItems();
  groupItemsByCluster(items).forEach((group) => {
    // A continuation if this page's run doesn't start at the cluster's
    // very first face — i.e. the rest of it was shown on the prior page.
    const isContinuation = group.faces[0] !== group.cluster.faces[0];
    container.append(renderBand(group.cluster, group.faces, isContinuation));
  });
}

function renderBand(cluster, faces, isContinuation) {
  const band = $("<div class='band'>").attr("data-cluster", cluster.id);

  const kindText = cluster.pending
    ? "pending"
    : "matched" + (cluster.playernum != null ? " \u00b7 #" + cluster.playernum : "");
  const titleSpan = $("<span class='band-title'>")
    .text(cluster.pending ? "Cluster " + cluster.id : cluster.name)
    .append($("<span>").addClass("kind").addClass(cluster.pending ? "pending" : "matched").text(kindText));
  if (isContinuation) titleSpan.append($("<span class='kind'>").text(" (cont\u2019d)"));
  const header = $("<div class='band-header'>").append(
    titleSpan,
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
    source: FacesApp.names,
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
  const deleteBtn = $("<button class='btn btn-danger btn-sm'>").text("Ignore")
    .on("click", () => {
      setDeleteForChecked(band, true);
      focusNextBand(band);
    });
  const undeleteBtn = $("<button class='btn btn-sm'>").text("Un-ignore")
    .on("click", () => setDeleteForChecked(band, false));

  tools.append(
    $("<label>").append(selectAll, $("<span>").text("Select all")),
    nameInput, applyBtn, deleteBtn, undeleteBtn
  );
  band.append(tools);

  const thumbs = $("<div class='thumbs'>");
  faces.forEach((face) => thumbs.append(renderThumb(cluster, face)));
  band.append(thumbs);

  // Clicking anywhere in the band's content area (thumbnails) also selects
  // the band, without toggling it back off the way the header does. Clicks
  // inside .band-tools are excluded: those controls are only visible once
  // the band is already selected, and some of them (Apply/Ignore) move
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
  const st = FacesApp.staged.get(ckey(cluster.id, face.crop)) || {};
  const wrap = $("<div class='thumb'>")
    .attr("data-crop", face.crop)
    .toggleClass("pending-delete", !!st.pendingDelete);

  const check = $("<input type='checkbox' class='thumb-check'>");
  check.on("change", function () {
    wrap.toggleClass("checked", this.checked);
  });

  const thumbUrl = "/api/faces/thumb/" + encodeURIComponent(cluster.id) + "/" + encodeURIComponent(face.crop);
  const originalUrl = "/api/original/" + encodeURIComponent(face.origFilename);

  const img = $("<img>").attr("data-src", thumbUrl).attr("title", face.origFilename)
    .on("click", (e) => {
      if (e.ctrlKey || e.metaKey || e.shiftKey) return; // handled by the card-level handler below
      Viewport.showImageWindow(originalUrl, {
        originalPath: face.originalPath || face.origFilename,
        aiEditKey: face.origFilename,
      });
    });
  thumbLazyLoadObserver.observe(img[0]);

  const mask = $("<div class='mask'>").text("IGNORE");
  const badge = $("<div class='assigned-badge'>").text(st.assignedName ? "\u2192 " + st.assignedName : "");

  wrap.append(check, mask, img, badge);

  // Ctrl/Cmd-click toggles this face instead of opening the preview.
  // Shift-click checks the whole range back to the last ctrl/shift-clicked
  // face in this same band, without opening the preview either.
  wrap.on("click", function (e) {
    if ($(e.target).closest(".thumb-check").length) return; // checkbox handles its own click
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      toggleThumbChecked(wrap);
      lastClickedThumb = wrap;
    } else if (e.shiftKey) {
      e.preventDefault();
      selectThumbRangeTo(wrap);
      lastClickedThumb = wrap;
    }
  });

  return wrap;
}

// ------------------------------------------------------------------ //
// Band interactions
// ------------------------------------------------------------------ //
let selectedBand = null;
// Anchor for shift-click range selection; reset whenever bands are re-rendered.
let lastClickedThumb = null;

function toggleThumbChecked(thumb) {
  const check = thumb.find(".thumb-check");
  check.prop("checked", !check.prop("checked")).trigger("change");
}

function checkThumb(thumb) {
  const check = thumb.find(".thumb-check");
  if (!check.is(":checked")) check.prop("checked", true).trigger("change");
}

// Checks every face between the shift-click anchor and `thumb`, inclusive,
// but only if the anchor is still in the same band (cluster) as `thumb`.
// Otherwise there's no meaningful range, so just check `thumb` itself.
function selectThumbRangeTo(thumb) {
  const band = thumb.closest(".band");
  const anchorValid = lastClickedThumb
    && $.contains(document, lastClickedThumb[0])
    && lastClickedThumb.closest(".band")[0] === band[0];
  if (!anchorValid) {
    checkThumb(thumb);
    return;
  }
  const thumbs = band.find(".thumb");
  const from = thumbs.index(lastClickedThumb[0]);
  const to = thumbs.index(thumb[0]);
  const [lo, hi] = from <= to ? [from, to] : [to, from];
  thumbs.slice(lo, hi + 1).each(function () { checkThumb($(this)); });
}

// Selects `band`, replacing any previous selection. No-op if it is already
// the selected band (unlike selectBand(), this never toggles it back off —
// used for interactions that should always land on a specific band).
function activateBand(band) {
  if (selectedBand && selectedBand[0] === band[0]) return;
  $(".band.selected").removeClass("selected");
  band.addClass("selected");
  selectedBand = band;

  const clusterId = band.attr("data-cluster");
  const thumbs = band.find(".thumb");
  const alreadyChecked = thumbs.filter(function () { return $(this).find(".thumb-check").is(":checked"); }).length > 0;

  // Auto-select only if the band has no selection yet, and only faces that
  // haven't been triaged (assigned a name, or marked ignore/pending-delete)
  // so revisiting a cluster doesn't re-select faces already dealt with.
  if (!alreadyChecked) {
    thumbs.each(function () {
      const thumb = $(this);
      const st = FacesApp.staged.get(ckey(clusterId, thumb.attr("data-crop"))) || {};
      if (st.assignedName || st.pendingDelete) return;
      thumb.find(".thumb-check").prop("checked", true).trigger("change");
    });
  }

  const checkedCount = thumbs.filter(function () { return $(this).find(".thumb-check").is(":checked"); }).length;
  band.find(".select-all").prop("checked", thumbs.length > 0 && checkedCount === thumbs.length);
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
  const bands = $("#facesBands > .band");
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

// [Del] key: mark checked thumbs in the selected band as ignored, unless
// focus is in an input (name textbox) or on a button.
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
async function saveFaces() {
  const operations = [];
  for (const [k, st] of FacesApp.staged.entries()) {
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

  $("#facesSaveBtn").prop("disabled", true).text("Saving\u2026");
  try {
    const res = await apiPost("/api/faces/commit", { operations });
    FacesApp.clusters = res.clusters || [];
    FacesApp.staged.clear();
    selectedBand = null;
    FacesApp.page = Math.min(FacesApp.page, totalPageCount());
    renderBands();
    renderPagination();
  } catch (err) {
    alert("Save failed: " + err.message);
  } finally {
    $("#facesSaveBtn").text("Save");
    refreshDirty();
  }
}

// ------------------------------------------------------------------ //
// Boot
// ------------------------------------------------------------------ //
$(function () {
  $("#facesSaveBtn").on("click", saveFaces);

  $(".faces-first-page-btn").on("click", () => goToPage(1));
  $(".faces-prev-page-btn").on("click", () => goToPage(FacesApp.page - 1));
  $(".faces-next-page-btn").on("click", () => goToPage(FacesApp.page + 1));
  $(".faces-last-page-btn").on("click", () => goToPage(totalPageCount()));
  $(".faces-page-input").on("change", function () { goToPage(parseInt($(this).val(), 10)); });
  $(".faces-page-input").on("keydown", function (e) {
    if (e.key === "Enter") { goToPage(parseInt($(this).val(), 10)); $(this).trigger("blur"); }
  });

  window.addEventListener("beforeunload", (e) => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ""; }
  });

  initFaces();
});


$(initFaces);
