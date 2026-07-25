"use strict";

// Page 3 — Review: sort blur / sharp / skipped photos into keep / drop,
// grouped into time-based "bursts". Decisions are staged here in memory
// (across all 3 tabs) and only written to album.json when Apply is hit.

const CATEGORIES = ["blur", "sharp", "skipped"];

let currentCategory = "sharp";
let sortMode = "size";
let previewCount = 1;

// Zoom/pan controller for the preview pane (see static/js/viewport.js),
// shared across all currently previewed image cells (up to 4 at once) so
// scrolling zooms every visible image together, and persists across
// image/group navigation until reset (Esc).
const previewZoomCtl = Viewport.createZoomController({ min: 1, max: 6, step: 0.25 });
function previewImgs() {
  return $("#reviewPreview .review-preview-cell img");
}

// Locked "fit" viewport size for each currently displayed preview cell (in
// px, matching the CSS border width below). Computed once per image from
// its natural size vs. the cell's available space, then held fixed even as
// the zoom level changes, so zooming clips/scales the image inside a
// stationary frame instead of resizing the frame itself.
const PREVIEW_VIEWPORT_BORDER = 4;
let previewViewportLockers = [];

// categoryData[category] = { groups: [{images:[{file,anno,keep}, ...]}, ...], activeGroup, activeImage }
const categoryData = {};
// Pending client-side overrides, shared across all 3 tabs: {filename: keep}
const pendingOverrides = {};

function reviewThumbUrl(category, anno) {
  return `/api/review/thumb?category=${encodeURIComponent(category)}&file=${encodeURIComponent(anno)}`;
}

function firstKeepIndex(group) {
  if (!group || !group.images.length) return 0;
  const idx = group.images.findIndex((im) => im.keep);
  return idx >= 0 ? idx : 0;
}

// ------------------------------------------------------------------ //
// Load
// ------------------------------------------------------------------ //
async function initReview() {
  $("#reviewEmpty").hide();
  $("#reviewApp").hide();

  let current;
  try {
    current = await apiGet("/api/current-album");
  } catch (err) {
    $("#reviewStatus").text("Failed to load current album: " + err.message).show();
    return;
  }

  if (!current.album) {
    $("#reviewEmpty").show();
    return;
  }
  $("#reviewApp").show();

  try {
    const summary = await apiGet("/api/review/summary");
    $("#countBlur").text(summary.blurCount);
    $("#countSharp").text(summary.sharpCount);
    $("#countSkipped").text(summary.skippedCount);
  } catch (err) {
    $("#reviewStatus").text("Failed to load review summary: " + err.message).show();
  }

  try {
    await fetchCategory(currentCategory);
    renderNav();
    renderMain();
  } catch (err) {
    $("#reviewStatus").text("Failed to load review data: " + err.message).show();
  }
}

async function fetchCategory(category) {
  const url = `/api/review/data?category=${encodeURIComponent(category)}&sort=${encodeURIComponent(sortMode)}`;
  const data = await apiGet(url);
  // Re-apply any pending (not-yet-applied) overrides on top of the
  // persisted keep state the server just handed back.
  data.groups.forEach((group) => {
    group.images.forEach((im) => {
      if (Object.prototype.hasOwnProperty.call(pendingOverrides, im.file)) {
        im.keep = pendingOverrides[im.file];
      }
    });
  });
  categoryData[category] = {
    groups: data.groups,
    activeGroup: 0,
    activeImage: firstKeepIndex(data.groups[0]),
  };
}

async function ensureCategory(category) {
  if (!categoryData[category]) {
    await fetchCategory(category);
  }
}

// ------------------------------------------------------------------ //
// Nav pane (burst groups as rows of small keep/drop dots)
// ------------------------------------------------------------------ //
function renderNav() {
  const data = categoryData[currentCategory];
  const $nav = $("#reviewNav").empty();
  if (!data) return;

  data.groups.forEach((group, gi) => {
    const $row = $("<div>", { class: "review-nav-row" }).toggleClass("active", gi === data.activeGroup);
    group.images.forEach((im) => {
      $row.append($("<span>", { class: "review-dot" }).toggleClass("keep", !!im.keep).toggleClass("drop", !im.keep));
    });
    $row.on("click", () => {
      data.activeGroup = gi;
      data.activeImage = firstKeepIndex(group);
      renderNav();
      renderMain();
    });
    $nav.append($row);
  });

  const $active = $nav.find(".review-nav-row.active");
  if ($active.length) {
    $active[0].scrollIntoView({ block: "nearest" });
  }
}

// ------------------------------------------------------------------ //
// Main pane (multi-image preview + in-group thumbnail strip)
// ------------------------------------------------------------------ //
function computeWindow(count, activeIndex, n) {
  n = Math.max(1, Math.min(n, count));
  const targetPos = Math.floor((n + 1) / 2); // 1-based position of the active image within the window
  let start = activeIndex - (targetPos - 1);
  start = Math.max(0, Math.min(start, count - n));
  return { start, end: start + n };
}

// Locate the preview image under a pointer event, falling back to the first
// currently rendered preview image if the pointer isn't directly over one
// (e.g. over a cell's letterboxed padding).
function previewImgAt(e) {
  const $fromTarget = $(e.target).closest(".review-preview-cell").find("img");
  return $fromTarget.length ? $fromTarget : previewImgs().first();
}

// Hovering over a zoomed-in image pans it proportionally to the cursor
// position; entering/leaving the preview area does nothing on its own —
// zoom only changes via scroll, click, or the '1'/Esc hotkeys below, and
// persists across image/group navigation until reset.
$("#reviewPreview").on("mousemove", function (e) {
  const $img = previewImgAt(e);
  if (!$img.length) return;
  previewZoomCtl.panTo(previewImgs(), $img[0], e.clientX, e.clientY);
});

$("#reviewPreview").on("wheel", function (e) {
  const $img = previewImgAt(e);
  if (!$img.length) return;
  e.preventDefault();
  const oe = e.originalEvent;
  const delta = oe.deltaY < 0 ? previewZoomCtl.step : -previewZoomCtl.step;
  previewZoomCtl.adjustByStep(previewImgs(), delta, $img[0], oe.clientX, oe.clientY);
});

// Clicking a preview image toggles it between "fit" and 100% (actual pixel
// size), centered on the click point.
$("#reviewPreview").on("click", ".review-preview-cell img", function (e) {
  previewZoomCtl.toggleClick(previewImgs(), this, e.clientX, e.clientY);
});

// Zoom to 100% (actual pixel size) — used by the '1' hotkey.
function zoomToActualSize() {
  const $img = previewImgs().first();
  if (!$img.length) return;
  previewZoomCtl.zoomToActual(previewImgs(), $img[0]);
}

// Size a preview cell's viewport to exactly wrap the image at "fit" scale
// (i.e. the classic object-fit: contain box), based on its natural size vs.
// the available space in the cell. Locked once here; unaffected by the
// zoom level afterwards since zoom only transforms the <img> inside.
function lockPreviewViewport($cell, $viewport, imgEl) {
  const box = Viewport.computeContainFit(imgEl.naturalWidth, imgEl.naturalHeight, $cell.width(), $cell.height());
  if (!box) return;
  $viewport.css({
    width: box.width + PREVIEW_VIEWPORT_BORDER * 2 + "px",
    height: box.height + PREVIEW_VIEWPORT_BORDER * 2 + "px",
  });
}

// Re-lock viewports if the window is resized (the "fit" box depends on
// available space, unlike zoom which never resizes it).
$(window).on("resize", () => previewViewportLockers.forEach((fn) => fn()));

function renderMain() {
  const data = categoryData[currentCategory];
  const $preview = $("#reviewPreview").empty();
  const $rankInfo = $("#reviewRankInfo").empty().hide();
  const $strip = $("#reviewStrip").empty();
  if (!data) return;

  const group = data.groups[data.activeGroup];
  if (!group) return;
  const activeImage = group.images[data.activeImage];

  const { start, end } = computeWindow(group.images.length, data.activeImage, previewCount);
  const cellEntries = [];
  for (let i = start; i < end; i++) {
    const im = group.images[i];
    const $cell = $("<div>", { class: "review-preview-cell" });
    const $viewport = $("<div>", { class: "review-preview-viewport" }).toggleClass("selected", !!im.keep);
    const $img = $("<img>", { src: reviewThumbUrl(currentCategory, im.anno), alt: im.file });
    $viewport.append($img);
    if (im.keep) {
      $viewport.append($("<span>", { class: "keep-badge" }).html("&#10003;"));
    }
    if (im.burstRanking) {
      $viewport.append($("<span>", { class: "rank-badge rank-" + im.burstRanking.rank }).text("#" + im.burstRanking.rank));
    }
    $cell.append($viewport);
    $preview.append($cell);
    cellEntries.push({ $cell, $viewport, img: $img[0] });
  }
  previewViewportLockers = cellEntries.map(({ $cell, $viewport, img }) => () => lockPreviewViewport($cell, $viewport, img));
  previewViewportLockers.forEach((lockFn, idx) => {
    const img = cellEntries[idx].img;
    if (img.complete) lockFn();
    else $(img).on("load", lockFn);
  });
  previewZoomCtl.apply(previewImgs());

  if (activeImage.burstRanking) {
    const rank = activeImage.burstRanking.rank;
    $rankInfo
      .append($("<span>", { class: "rank-badge rank-" + rank }).text("#" + rank))
      .append($("<span>", { class: "rank-reason" }).text(activeImage.burstRanking.reason || ""))
      .show();
  }

  group.images.forEach((im, i) => {
    const $thumb = $("<div>", { class: "review-strip-thumb" }).toggleClass("active", i === data.activeImage);
    $thumb.toggleClass("keep", !!im.keep);
    $thumb.append($("<img>", { src: reviewThumbUrl(currentCategory, im.anno), alt: im.file }));
    if (im.keep) {
      $thumb.append($("<span>", { class: "keep-badge" }).html("&#10003;"));
    }
    if (im.burstRanking) {
      $thumb.append($("<span>", { class: "rank-badge rank-" + im.burstRanking.rank }).text("#" + im.burstRanking.rank));
      $thumb.attr("title", "#" + im.burstRanking.rank + ": " + (im.burstRanking.reason || ""));
    }
    $thumb.on("click", () => {
      data.activeImage = i;
      renderMain();
    });
    $strip.append($thumb);
  });

  const $activeThumb = $strip.find(".review-strip-thumb.active");
  if ($activeThumb.length) {
    $activeThumb[0].scrollIntoView({ inline: "nearest", block: "nearest" });
  }
}

function toggleActiveKeep() {
  const data = categoryData[currentCategory];
  if (!data) return;
  const group = data.groups[data.activeGroup];
  if (!group) return;
  const im = group.images[data.activeImage];
  im.keep = !im.keep;
  pendingOverrides[im.file] = im.keep;
  renderNav();
  renderMain();
}

// ------------------------------------------------------------------ //
// Hotkeys — w/s move between groups, a/d move within a group, space toggles.
// Arrow keys are aliases: Up=w, Down=s, Left=a, Right=d.
// ------------------------------------------------------------------ //
const ARROW_KEY_ALIASES = { ArrowUp: "w", ArrowDown: "s", ArrowLeft: "a", ArrowRight: "d" };

$(document).on("keydown", (e) => {
  const activeTag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
  if (activeTag === "select" || activeTag === "input" || activeTag === "textarea") return;
  if (!$("#reviewApp").is(":visible")) return;

  const data = categoryData[currentCategory];
  if (!data) return;
  const group = data.groups[data.activeGroup];

  const key = ARROW_KEY_ALIASES[e.key] || e.key;
  if (ARROW_KEY_ALIASES[e.key]) e.preventDefault();
  switch (key) {
    case "w":
      if (data.activeGroup > 0) {
        data.activeGroup--;
        data.activeImage = firstKeepIndex(data.groups[data.activeGroup]);
        renderNav();
        renderMain();
      }
      break;
    case "s":
      if (data.activeGroup < data.groups.length - 1) {
        data.activeGroup++;
        data.activeImage = firstKeepIndex(data.groups[data.activeGroup]);
        renderNav();
        renderMain();
      }
      break;
    case "a":
      if (group && data.activeImage > 0) {
        data.activeImage--;
        renderMain();
      } else if (data.activeGroup > 0) {
        data.activeGroup--;
        data.activeImage = data.groups[data.activeGroup].images.length - 1;
        renderNav();
        renderMain();
      }
      break;
    case "d":
      if (group && data.activeImage < group.images.length - 1) {
        data.activeImage++;
        renderMain();
      } else if (data.activeGroup < data.groups.length - 1) {
        data.activeGroup++;
        data.activeImage = 0;
        renderNav();
        renderMain();
      }
      break;
    case " ":
      e.preventDefault();
      toggleActiveKeep();
      break;
    case "1":
      zoomToActualSize();
      break;
    case "Escape":
      previewZoomCtl.resetToFit(previewImgs());
      break;
    default:
      return;
  }
});

// ------------------------------------------------------------------ //
// Toolbar controls
// ------------------------------------------------------------------ //
$("#reviewTabs").on("click", ".review-tab", async function () {
  const category = $(this).data("category");
  if (category === currentCategory) return;
  currentCategory = category;
  $(".review-tab").removeClass("active");
  $(this).addClass("active");
  await ensureCategory(category);
  renderNav();
  renderMain();
});

$("#reviewPreviewCount").on("change", function () {
  previewCount = parseInt($(this).val(), 10) || 1;
  renderMain();
});

$("#reviewSort").on("change", async function () {
  sortMode = $(this).val();
  Object.keys(categoryData).forEach((c) => delete categoryData[c]);
  await fetchCategory(currentCategory);
  renderNav();
  renderMain();
});

$("#reviewApplyBtn").on("click", async function () {
  const overrides = Object.assign({}, pendingOverrides);
  if (Object.keys(overrides).length === 0) {
    $("#reviewStatus").text("No changes to apply.").show();
    return;
  }
  $(this).prop("disabled", true);
  try {
    await apiPost("/api/review/apply", { overrides });
    Object.keys(pendingOverrides).forEach((k) => delete pendingOverrides[k]);
    $("#reviewStatus").text("Changes applied.").show();
  } catch (err) {
    $("#reviewStatus").text("Failed to apply changes: " + err.message).show();
  } finally {
    $(this).prop("disabled", false);
  }
});

$(function () {
  initReview();
});
