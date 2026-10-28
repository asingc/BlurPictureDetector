"use strict";

// Shared helpers used by every page's own JS file (team.js, select-album.js, ...).

async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

// ------------------------------------------------------------------ //
// Tiny cookie helpers — used for small pieces of client-only UI memory
// (e.g. the Face Recognition checkbox's last state) that intentionally
// live in the browser rather than in server-side state.json.
// ------------------------------------------------------------------ //
function setCookie(name, value, days) {
  const maxAge = Math.round((days || 3650) * 86400);
  document.cookie = `${name}=${encodeURIComponent(value)};max-age=${maxAge};path=/;samesite=lax`;
}

function getCookie(name) {
  const match = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
  return match ? decodeURIComponent(match[1]) : null;
}

// ------------------------------------------------------------------ //
// Team picker widget — a row of selectable "team" blocks plus a trailing
// "add team" block with a plus icon. Shared by the Add Album page (pick
// which team's settings to use for a new album) and the Team Setup page
// (switch which team you're editing / start a brand-new one).
//
//   renderTeamPicker($container, teams, {
//     selectedId,       // id of the currently-selected team, or null/"" for none
//     onSelect(team),   // called with the full team object when a block is clicked
//     onAddNew(),       // called when the trailing "add" block is clicked
//     addLabel,         // text for the trailing block (default "Add Team")
//   });
// ------------------------------------------------------------------ //
function renderTeamPicker($container, teams, opts) {
  opts = opts || {};
  $container.empty();
  (teams || []).forEach((team) => {
    const isSelected = !!opts.selectedId && team.id === opts.selectedId;
    const block = $("<button>", {
      type: "button",
      class: "team-block" + (isSelected ? " selected" : ""),
    }).text(team.name || "Unnamed Team");
    block.on("click", () => { if (opts.onSelect) opts.onSelect(team); });
    $container.append(block);
  });
  const addBlock = $("<button>", { type: "button", class: "team-block team-block-add" });
  addBlock.append($("<span>", { class: "team-block-plus" }).text("+"));
  addBlock.append($("<span>").text(opts.addLabel || "Add Team"));
  addBlock.on("click", () => { if (opts.onAddNew) opts.onAddNew(); });
  $container.append(addBlock);
}

// ------------------------------------------------------------------ //
// Heartbeat — tells the server this page is still open. The server exits
// automatically if it stops receiving these (e.g. the tab/browser was
// closed), so we don't leave orphaned local servers running. Runs on every
// page since it's loaded from common.js via base.html.
//
// Sent from a dedicated Worker (heartbeat-worker.js) instead of a plain
// setInterval on this page, because browsers throttle a backgrounded tab's
// own timers (Chrome clamps to ~once/minute after ~5 min hidden) — longer
// than the server's heartbeat timeout — which was causing the server to
// shut itself down mid-run whenever the tab was left in the background
// (e.g. during a long Import/Apply job). Worker timers aren't throttled
// that way. Falls back to a plain setInterval if Workers aren't available.
// ------------------------------------------------------------------ //
const HEARTBEAT_INTERVAL_MS = 10000;

function sendHeartbeat() {
  // Best-effort; a single dropped heartbeat is fine, the server tolerates
  // multiple missed intervals before exiting.
  fetch("/api/heartbeat", { method: "POST" }).catch(() => {});
}

$(function () {
  let workerStarted = false;
  if (window.Worker) {
    try {
      new Worker("/static/js/heartbeat-worker.js");
      workerStarted = true;
    } catch (err) {
      workerStarted = false;
    }
  }
  if (!workerStarted) {
    sendHeartbeat();
    setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);
  }
});

// ------------------------------------------------------------------ //
// LogDialog — the shared "launch a background job, stream its log into a
// modal, react when it's done" UX used by almost every long-running action
// in this app (initial import, import-more, re-run face detection, deep
// regrade, jersey-colour LLM re-run, export, burst-video, face-tagging CSV,
// per-player videos, AI edit, ...). This used to be hand-copied into every
// page's own JS file (init*Dialog/open*Dialog/append*Lines/finish*Dialog/
// poll*Status, ~9 nearly-identical copies) — LogDialog.create() gives one
// implementation of the generic "dialog chrome + poll-into-textarea" parts;
// each caller still owns its own start/completion logic (what to re-enable,
// what message to show, whether to navigate away or open a compare window),
// since that genuinely differs per feature.
//
//   const job = LogDialog.create({
//     dialogId: "exportDialog",         // existing <div> (jQuery UI dialog target)
//     outputId: "exportOutput",         // <textarea> inside it that the log is appended to
//     statusUrl: (since) => `/api/album-tools/export-status?since=${since}`,
//     pollIntervalMs: 500,              // optional, default 500
//     dialogOptions: { width: 720 },    // optional, merged over the jQuery UI defaults
//     label: "export",                 // optional — see job.open() below
//   });
//
//   job.open("Exporting…");             // opens the (non-closable while running) dialog;
//                                        // if `label` was given, immediately appends
//                                        // "Starting export…" as the first log line —
//                                        // real output can take a while to show up (the
//                                        // launched script's own startup/library-import
//                                        // time), so this gives instant feedback instead
//                                        // of a dialog that just sits empty meanwhile.
//   job.reset();                        // clears the output box + poll cursor
//   job.startPolling((data) => {        // fires once per tick, incl. an immediate first
//     if (data.running) return;         // check so the dialog doesn't sit empty for a
//     job.setFinished("Export complete"); // whole poll interval before any text shows up
//     ...react to data...
//   });
//
// If `dialogId` isn't already a <div> on the page, one is created lazily
// (with a single <textarea outputId>) the first time create() runs, so new
// callers (like ai-edit.js) don't need to hand-author any dialog markup.
// ------------------------------------------------------------------ //
const LogDialog = (function () {
  function ensureDialog(dialogId, outputId, dialogOptions) {
    let $dialog = $("#" + dialogId);
    if ($dialog.length === 0) {
      $dialog = $("<div>", { id: dialogId }).append(
        $("<textarea>", { id: outputId, class: "output-box", rows: 16, readonly: true })
      );
      $("body").append($dialog);
    }
    $dialog.dialog($.extend({
      autoOpen: false,
      modal: true,
      closeOnEscape: false,
      draggable: false,
      resizable: false,
      width: 640,
    }, dialogOptions || {}));
    return $dialog;
  }

  function create(opts) {
    const outputId = opts.outputId;
    const pollIntervalMs = opts.pollIntervalMs || 500;
    const $dialog = ensureDialog(opts.dialogId, outputId, opts.dialogOptions);
    let timer = null;
    let since = 0;

    function appendLines(lines) {
      if (!lines || !lines.length) return;
      const box = document.getElementById(outputId);
      if (!box) return;
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
      box.value += (box.value ? "\n" : "") + lines.join("\n");
      if (atBottom) box.scrollTop = box.scrollHeight;
    }

    function open(title) {
      if (title) $dialog.dialog("option", "title", title);
      $dialog.dialog("option", "closeOnEscape", false);
      $dialog.dialog("open");
      // Non-closable while running: hide the titlebar's [x] button too.
      $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
      if (opts.label) appendLines([`Starting ${opts.label}…`]);
    }

    function setFinished(title) {
      if (title) $dialog.dialog("option", "title", title);
      $dialog.dialog("option", "closeOnEscape", true);
      $dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
    }

    function close() {
      $dialog.dialog("close");
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
    }

    function reset() {
      stop();
      since = 0;
      const box = document.getElementById(outputId);
      if (box) box.value = "";
    }

    // Polls opts.statusUrl(since) every pollIntervalMs, appending any new
    // lines and invoking onData(data) with the raw response each time —
    // the caller checks data.running/data.error/etc. to decide when (and
    // how) to react. Fires an immediate check instead of waiting for the
    // first interval tick, so the dialog doesn't sit empty that long.
    // Stops itself as soon as the job is no longer running, so onData's
    // "finished" branch fires exactly once instead of repeating forever
    // (every caller relies on this single-fire behavior).
    function startPolling(onData) {
      stop();
      async function tick() {
        let data;
        try {
          data = await apiGet(opts.statusUrl(since));
        } catch (err) {
          return; // transient — try again next tick
        }
        appendLines(data.lines);
        since = data.next;
        if (!data.running) stop();
        onData(data);
      }
      timer = setInterval(tick, pollIntervalMs);
      tick();
    }

    return { $dialog, appendLines, open, setFinished, close, reset, startPolling, stop };
  }

  return { create };
})();

// ------------------------------------------------------------------ //
// Publishes the topbar's real rendered height as --topbar-h so sticky
// elements further down the page (e.g. the cluster page's toolbar) can
// offset themselves below it without hardcoding a pixel guess.
// ------------------------------------------------------------------ //
function syncTopbarHeight() {
  const el = document.getElementById("topbar");
  if (el) document.documentElement.style.setProperty("--topbar-h", el.offsetHeight + "px");
}

// ------------------------------------------------------------------ //
// Reflects the currently-selected album's name in both the browser tab
// title (e.g. "Game vs Panthers — Photo Culling") and the topbar heading
// (e.g. "Photo Culling - Game vs Panthers"). Runs on every page since this
// file is loaded by base.html site-wide, so both stay in sync with
// whichever album is current without each page having to wire it up itself.
// ------------------------------------------------------------------ //
const BASE_TITLE = document.title;
async function syncAlbumTitle() {
  let name = null;
  try {
    const data = await apiGet("/api/current-album");
    name = data.album && (data.album.displayName || data.album.name);
  } catch (err) {
    // Non-fatal — leave the default title/heading in place.
  }
  document.title = name ? `${name} — ${BASE_TITLE}` : BASE_TITLE;
  const $titleAlbum = $("#titleAlbum");
  $titleAlbum.text(name ? ` - ${name}` : "");
  // Full un-truncated text as a native tooltip (CSS ellipsizes this span
  // on narrow windows — see .title-album in style.css).
  $titleAlbum.attr("title", name || "");
  syncTopbarHeight();
}

$(function () {
  syncTopbarHeight();
  $(window).on("resize", syncTopbarHeight);
  syncAlbumTitle();
});
