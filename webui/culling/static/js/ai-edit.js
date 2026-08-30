"use strict";

// Shared "AI edit" action (autoedit.py) — triggered from both the Review
// page (next to the file-path box) and the Faces/cluster page (in the
// original-image preview window, see viewport.js). Streams the script's
// combined stdout/stderr into a non-closable jQuery UI modal, reusing the
// same processing_state/`/api/processing-output` polling plumbing as
// import-more/rerun-facereco/deep-regrade (see static/js/apply.js) — only
// one such background job can run at a time. The dialog's DOM is built
// once, lazily, so this file has zero footprint on pages that never use it.
const AiEdit = (function () {
  let $dialog = null;
  let pollTimer = null;
  let pollSince = 0;

  function dialog() {
    if ($dialog) return $dialog;
    $dialog = $("<div>", { id: "aiEditDialog" }).append(
      $("<textarea>", { id: "aiEditOutput", class: "output-box", rows: 16, readonly: true })
    );
    $("body").append($dialog);
    $dialog.dialog({
      autoOpen: false,
      modal: true,
      closeOnEscape: false,
      draggable: false,
      resizable: false,
      width: 640,
    });
    return $dialog;
  }

  function appendLines(lines) {
    if (!lines || !lines.length) return;
    const box = document.getElementById("aiEditOutput");
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    box.value += (box.value ? "\n" : "") + lines.join("\n");
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  function finish(returnCode) {
    const $d = dialog();
    const success = returnCode === 0;
    $d.dialog("option", "title", success ? "AI edit complete" : `AI edit failed (exit code ${returnCode})`);
    $d.dialog("option", "closeOnEscape", true);
    $d.dialog("widget").find(".ui-dialog-titlebar-close").show();
  }

  async function poll() {
    let data;
    try {
      data = await apiGet(`/api/processing-output?since=${pollSince}`);
    } catch (err) {
      return; // transient — retry on the next tick
    }
    appendLines(data.lines);
    pollSince = data.next;
    if (!data.running) {
      clearInterval(pollTimer);
      pollTimer = null;
      finish(data.returnCode);
    }
  }

  // Launches autoedit.py against `key` (album.json bookkeeping key / plain
  // filename — same identifier used by /api/original etc.).
  async function run(key) {
    const $d = dialog();
    $("#aiEditOutput").val("");
    pollSince = 0;
    $d.dialog("option", "title", "AI editing\u2026");
    $d.dialog("option", "closeOnEscape", false);
    $d.dialog("open");
    $d.dialog("widget").find(".ui-dialog-titlebar-close").hide();
    try {
      await apiPost("/api/edit/ai-edit", { file: key });
    } catch (err) {
      appendLines([`Failed to start: ${err.message}`]);
      finish(1);
      return;
    }
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 1000);
    poll();
  }

  return { run };
})();
