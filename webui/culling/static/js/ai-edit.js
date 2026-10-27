"use strict";

// Shared "AI edit" action (autoedit.py) — triggered from both the Culling
// page (next to the file-path box) and the Faces page (in the
// original-image preview window, see viewport.js). Streams the script's
// combined stdout/stderr into a non-closable jQuery UI modal via LogDialog
// (see common.js), reusing the same processing_state/`/api/processing-
// output` polling plumbing as import-more/rerun-facereco/deep-regrade (see
// static/js/album-tools.js) — only one such background job can run at a
// time. The dialog's DOM is built once, lazily (LogDialog creates it on
// first use since no #aiEditDialog markup exists on the page), so this
// file has zero footprint on pages that never use it.
//
// On success the log modal gives way to a side-by-side before/after compare
// window (Viewport.showCompareWindow) whose Accept/Reject buttons decide
// whether the generated image is kept — see openCompare() below.
const AiEdit = (function () {
  let job = null;

  function dialog() {
    if (!job) {
      job = LogDialog.create({
        dialogId: "aiEditDialog",
        outputId: "aiEditOutput",
        statusUrl: (since) => `/api/processing-output?since=${since}`,
        pollIntervalMs: 1000,
        label: "AI edit",
      });
    }
    return job;
  }

  function finish(returnCode) {
    const j = dialog();
    const success = returnCode === 0;
    if (success) {
      j.close();
      openCompare();
      return;
    }
    j.setFinished(`AI edit failed (exit code ${returnCode})`);
  }

  // Shows the generated image next to what it was made from and waits for
  // an explicit verdict: the edit is only recorded on the album entry once
  // accepted, and the generated file is deleted on reject (see
  // /api/edit/ai-edit/accept|reject in culling_app.py).
  async function openCompare() {
    let state;
    try {
      state = await apiGet("/api/edit/ai-edit/pending");
    } catch (err) {
      state = null;
    }
    if (!state || !state.pending) {
      reopenLog("AI edit finished, but no result was produced.", "AI edit produced no image");
      return;
    }
    const stamp = Date.now();
    Viewport.showCompareWindow({
      title: `AI edit \u2014 ${state.file}`,
      beforeUrl: `/api/edit/ai-edit/image?side=before&t=${stamp}`,
      afterUrl: `/api/edit/ai-edit/image?side=after&t=${stamp}`,
      beforeLabel: "Before",
      afterLabel: "After (AI edit)",
      acceptLabel: "Accept",
      rejectLabel: "Reject",
      onAccept: () => apiPost("/api/edit/ai-edit/accept", {}).catch((err) => {
        alert(`Failed to keep the edit: ${err.message}`);
      }),
      onReject: () => apiPost("/api/edit/ai-edit/reject", {}).catch((err) => {
        alert(`Failed to discard the edit: ${err.message}`);
      }),
    });
  }

  // Reopens the log modal in a terminal (closable) state with `message`
  // appended — used when the run ends without something to compare. Goes
  // straight to the "already finished" dialog chrome (skipping the usual
  // non-closable-while-running state LogDialog.open() sets up), so this
  // uses the raw jQuery UI dialog ($dialog, exposed by LogDialog) directly.
  function reopenLog(message, title) {
    const j = dialog();
    j.appendLines([message]);
    j.$dialog.dialog("option", "title", title);
    j.$dialog.dialog("option", "closeOnEscape", true);
    j.$dialog.dialog("open");
    j.$dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
  }

  // Launches autoedit.py against `key` (the Album's bookkeeping key / plain
  // filename — same identifier used by /api/original etc.).
  async function run(key) {
    const j = dialog();
    j.reset();
    j.open("AI editing\u2026");
    try {
      await apiPost("/api/edit/ai-edit", { file: key });
    } catch (err) {
      j.appendLines([`Failed to start: ${err.message}`]);
      finish(1);
      return;
    }
    j.startPolling((data) => {
      if (data.running) return;
      finish(data.returnCode);
    });
  }

  return { run };
})();
