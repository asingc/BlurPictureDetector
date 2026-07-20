"use strict";

// Page 3 — Review: sort blur / sharp / skipped photos into keep / drop,
// grouped into time-based "bursts". Decisions are staged here in memory
// (across all 3 tabs) and only written to results.json when Apply is hit.

const CATEGORIES = ["blur", "sharp", "skipped"];

let currentCategory = "blur";
let sortMode = "size";
let previewCount = 1;

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

function attachHoverZoom($img) {
  $img.on("mousemove", function (e) {
    const rect = this.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * 100;
    const relY = ((e.clientY - rect.top) / rect.height) * 100;
    const pos = relX + "% " + relY + "%";
    $("#reviewPreview .review-preview-cell img").addClass("zoomed").css("object-position", pos);
  });
  $img.on("mouseleave", function () {
    $("#reviewPreview .review-preview-cell img").removeClass("zoomed").css("object-position", "");
  });
}

function renderMain() {
  const data = categoryData[currentCategory];
  const $preview = $("#reviewPreview").empty();
  const $strip = $("#reviewStrip").empty();
  if (!data) return;

  const group = data.groups[data.activeGroup];
  if (!group) return;
  const activeImage = group.images[data.activeImage];

  $preview.toggleClass("keep-bg", !!activeImage.keep).toggleClass("drop-bg", !activeImage.keep);

  const { start, end } = computeWindow(group.images.length, data.activeImage, previewCount);
  for (let i = start; i < end; i++) {
    const im = group.images[i];
    const $cell = $("<div>", { class: "review-preview-cell" });
    const $img = $("<img>", { src: reviewThumbUrl(currentCategory, im.anno), alt: im.file });
    attachHoverZoom($img);
    $cell.append($img);
    $preview.append($cell);
  }

  group.images.forEach((im, i) => {
    const $thumb = $("<div>", { class: "review-strip-thumb" }).toggleClass("active", i === data.activeImage);
    $thumb.toggleClass("keep", !!im.keep).toggleClass("drop", !im.keep);
    $thumb.append($("<img>", { src: reviewThumbUrl(currentCategory, im.anno), alt: im.file }));
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
      }
      break;
    case "d":
      if (group && data.activeImage < group.images.length - 1) {
        data.activeImage++;
        renderMain();
      }
      break;
    case " ":
      e.preventDefault();
      toggleActiveKeep();
      break;
    case "Escape":
      $("#reviewPreview .review-preview-cell img").removeClass("zoomed").css("object-position", "");
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
