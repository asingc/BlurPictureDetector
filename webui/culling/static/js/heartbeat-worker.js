"use strict";

// Runs in a dedicated Worker so the heartbeat keeps firing on schedule even
// when the tab is backgrounded. Browsers (Chrome in particular) throttle
// setInterval/setTimeout on a backgrounded page's own timers down to as
// little as once a minute after ~5 minutes hidden — longer than the
// server's heartbeat-watchdog timeout — which was causing the server to
// shut itself down mid-run whenever a tab was left in the background (e.g.
// during a long Import/Apply job). Worker timers aren't subject to that
// page-visibility throttling, so the ping stays on schedule regardless of
// tab focus/visibility.
const HEARTBEAT_INTERVAL_MS = 10000;

function sendHeartbeat() {
  fetch("/api/heartbeat", { method: "POST" }).catch(() => {});
}

sendHeartbeat();
setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);
