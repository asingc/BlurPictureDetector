"use strict";

// Shared helpers used by every page's own JS file (team.js, import.js, ...).

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
