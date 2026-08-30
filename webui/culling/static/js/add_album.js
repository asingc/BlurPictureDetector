"use strict";

// Add Album page — create a new album (source folder, blur sensitivity,
// face recognition) and pick which team's settings to use for it.

let pollTimer = null;
let pollSince = 0;

// True while a processing run is in flight — used to disable the whole
// screen and to warn the user if they try to navigate away.
let isProcessing = false;

// Teams loaded from the server, and which one is currently selected for
// this album (used when Import Images is clicked). The server already
// returns them ordered most-recently-used-to-import first.
let teams = [];
let selectedTeamId = "";

// Which jersey color is pinned for this album ("" = Auto/detect).
let selectedTeamColor = "";

// Cookie names for the Face Recognition checkbox's client-only memory.
const RECOGNIZE_FACES_COOKIE = "recognizeFacesLastState";
const RECOGNIZE_FACES_CONSENT_COOKIE = "recognizeFacesConsentAck";

// Set once the user clicks "I Agree" in the consent dialog for the current
// open; used to tell an "agreed" close apart from a "cancelled" one.
let consentAgreed = false;

// ------------------------------------------------------------------ //
// Team picker
// ------------------------------------------------------------------ //
function renderTeamPickerUI() {
  renderTeamPicker($("#teamPicker"), teams, {
    selectedId: selectedTeamId,
    onSelect: (team) => {
      selectedTeamId = team.id;
      renderTeamPickerUI();
    },
    onAddNew: () => { window.location.href = "/team"; },
    addLabel: "Create a New Team",
  });
  renderTeamColorOptions();
}

// ------------------------------------------------------------------ //
// Team jersey color — options are the selected team's registered jersey
// colors (minus forced ones, e.g. goalie kits, which are always allowed
// regardless of this choice), plus "Auto" (default) which polls the
// dominant color from the photos instead of pinning one. Rendered as
// round chip buttons, same interaction pattern as the team picker above.
// ------------------------------------------------------------------ //
function renderTeamColorOptions() {
  const team = teams.find((t) => t.id === selectedTeamId);
  const colors = ((team && team.jerseyColors) || []).filter((jc) => !jc.forced);
  if (!colors.some((jc) => jc.color === selectedTeamColor)) selectedTeamColor = "";

  const $container = $("#teamColorPicker").empty();
  const disabled = $("#ignoreJerseyColorInput").is(":checked");

  const makeChip = (label, value) => {
    const chip = $("<button>", {
      type: "button",
      class: "color-chip" + (selectedTeamColor === value ? " selected" : ""),
      disabled: disabled,
    }).text(label);
    chip.on("click", () => { selectedTeamColor = value; renderTeamColorOptions(); });
    $container.append(chip);
  };

  makeChip("Auto", "");
  colors.forEach((jc) => makeChip(jc.color, jc.color));
}

async function loadTeams(preferredTeamId) {
  const data = await apiGet("/api/teams/query");
  teams = data.teams || [];
  const hasPreferred = preferredTeamId && teams.some((t) => t.id === preferredTeamId);
  selectedTeamId = hasPreferred ? preferredTeamId : (teams[0] ? teams[0].id : "");
  renderTeamPickerUI();
}

// ------------------------------------------------------------------ //
// Blur sensitivity
// ------------------------------------------------------------------ //
function applySensitivityToUI(mode, customValue) {
  $(`input[name="sensMode"][value="${mode || "medium"}"]`).prop("checked", true);
  const custom = customValue ?? 0.50;
  $("#sensitivityCustomSlider").val(custom);
  $("#sensitivityCustomValue").text(Number(custom).toFixed(2));
}

function readSensitivityFromUI() {
  const mode = $('input[name="sensMode"]:checked').val() || "medium";
  const customValue = parseFloat($("#sensitivityCustomSlider").val()) || 0;
  return { mode, customValue };
}

// ------------------------------------------------------------------ //
// Import state (sensitivity / selected team). The last-used import
// directory is intentionally not surfaced on this page — each import
// starts from a fresh folder-picker dialog (see the Import Images button).
// ------------------------------------------------------------------ //
async function loadImportState() {
  const state = await apiGet("/api/import-state");
  applySensitivityToUI(state.sensitivityMode, state.sensitivityCustomValue);
  return state.selectedTeamId || "";
}

// ------------------------------------------------------------------ //
// Face Recognition checkbox — default unchecked, remembered client-side
// via a cookie (not the server), with a one-time parental-consent-style
// disclaimer the first time it's ever checked.
// ------------------------------------------------------------------ //
function initRecognizeFacesCheckbox() {
  const lastState = getCookie(RECOGNIZE_FACES_COOKIE);
  $("#recognizeFacesInput").prop("checked", lastState === "1");

  const $consentDialog = $("#faceRecoConsentDialog");
  const $readCheckbox = $("#faceRecoConsentReadCheckbox");

  $consentDialog.dialog({
    autoOpen: false,
    modal: true,
    width: 460,
    resizable: false,
    buttons: [
      {
        text: "I Agree",
        class: "btn btn-primary",
        disabled: true,
        click: function () {
          consentAgreed = true;
          setCookie(RECOGNIZE_FACES_CONSENT_COOKIE, "1");
          setCookie(RECOGNIZE_FACES_COOKIE, "1");
          $(this).dialog("close");
        },
      },
      {
        text: "Cancel",
        click: function () { $(this).dialog("close"); },
      },
    ],
    open: function () {
      consentAgreed = false;
      $readCheckbox.prop("checked", false);
      $consentDialog.dialog("widget").find(".btn-primary").button("option", "disabled", true);
    },
    close: function () {
      if (!consentAgreed) {
        // Declined, dismissed, or closed without agreeing — leave the
        // checkbox unchecked and don't remember that consent was granted.
        $("#recognizeFacesInput").prop("checked", false);
        setCookie(RECOGNIZE_FACES_COOKIE, "0");
      }
    },
  });

  $readCheckbox.on("change", function () {
    // Must go through the jQuery UI button widget API here, not a plain
    // .prop("disabled", ...) — jQuery UI keeps its own "disabled" widget
    // state/`.ui-state-disabled` class (which sets `pointer-events: none`)
    // separate from the DOM attribute, so toggling only the DOM prop left
    // the button visually enabled but still unclickable with the mouse
    // (keyboard activation via Tab+Enter still worked, which is why this
    // only showed up for mouse users).
    $consentDialog.dialog("widget").find(".btn-primary").button("option", "disabled", !this.checked);
  });

  $("#recognizeFacesInput").on("change", function () {
    const checked = this.checked;
    if (checked && getCookie(RECOGNIZE_FACES_CONSENT_COOKIE) !== "1") {
      $consentDialog.dialog("open");
      return; // cookie is set by the dialog's buttons/close handler
    }
    setCookie(RECOGNIZE_FACES_COOKIE, checked ? "1" : "0");
  });
}

// ------------------------------------------------------------------ //
// Disable/re-enable the whole screen while a processing run is active.
// ------------------------------------------------------------------ //
function setImportUIEnabled(enabled) {
  isProcessing = !enabled;
  $("#createAlbumPanel").find("input, button").prop("disabled", !enabled);
  $("#teamPanel").find("button").prop("disabled", !enabled);
  $("#importImagesBtn").prop("disabled", !enabled);
  if (enabled && $("#ignoreJerseyColorInput").is(":checked")) {
    $("#teamColorPicker").find("button").prop("disabled", true);
  }
}

// ------------------------------------------------------------------ //
// Processing log dialog — a non-closable jQuery UI modal while a run is
// active (its dark overlay blocks every other UI element from being hit
// by accident), becoming closable once the run finishes.
// ------------------------------------------------------------------ //
let dismissTimer = null;

function initProcessingDialog() {
  $("#processingDialog").dialog({
    autoOpen: false,
    modal: true,
    closeOnEscape: false,
    draggable: false,
    resizable: false,
    width: 640,
  });
}

function openProcessingDialog() {
  const $dialog = $("#processingDialog");
  $dialog.dialog("option", "title", "Processing…");
  $dialog.dialog("option", "closeOnEscape", false);
  $dialog.dialog("open");
  // Non-closable while running: hide the titlebar's [x] button too.
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").hide();
}

function finishProcessingDialog(returnCode) {
  const $dialog = $("#processingDialog");
  const success = returnCode === 0;
  $dialog.dialog("option", "title", success ? "Processing complete" : `Processing failed (exit code ${returnCode})`);
  $dialog.dialog("option", "closeOnEscape", true);
  $dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();

  // (Re)bind the close handler for this run: navigate to Culling only on
  // success, whether the dialog is dismissed by the user or auto-closes.
  $dialog.off("dialogclose.processing").on("dialogclose.processing", () => {
    if (dismissTimer) { clearTimeout(dismissTimer); dismissTimer = null; }
    if (success) window.location.href = "/review";
  });

  if (success) {
    $("#importStatus").text("Processing complete. Continuing to Culling…");
    dismissTimer = setTimeout(() => { $dialog.dialog("close"); }, 5000);
  } else {
    $("#importStatus").text(`Processing exited with code ${returnCode}.`);
  }
}

// ------------------------------------------------------------------ //
// Processing output polling
// ------------------------------------------------------------------ //
function appendOutputLines(lines) {
  if (!lines.length) return;
  const box = document.getElementById("processingOutput");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.value += (box.value ? "\n" : "") + lines.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

async function pollProcessingOutput() {
  let data;
  try {
    data = await apiGet(`/api/processing-output?since=${pollSince}`);
  } catch (err) {
    return; // transient — try again on the next tick
  }
  appendOutputLines(data.lines);
  pollSince = data.next;
  if (!data.running) {
    clearInterval(pollTimer);
    pollTimer = null;
    setImportUIEnabled(true);
    finishProcessingDialog(data.returnCode);
  }
}

$(function () {
  $("#sensitivityCustomSlider").on("input", function () {
    $("#sensitivityCustomValue").text(Number($(this).val()).toFixed(2));
    $('input[name="sensMode"][value="custom"]').prop("checked", true);
  });

  $("#importImagesBtn").on("click", async () => {
    const albumName = $("#albumNameInput").val().trim();
    if (!albumName) { $("#importStatus").text("Album name is required."); $("#albumNameInput").trigger("focus"); return; }
    if (!selectedTeamId) { $("#importStatus").text("Select a team first."); return; }

    $("#importImagesBtn").prop("disabled", true);
    $("#importStatus").text("Choose a folder…");
    let path;
    try {
      const res = await apiPost("/api/browse-folder", { context: "import" });
      path = res.path;
    } catch (err) {
      $("#importStatus").text("Folder picker failed: " + err.message);
      $("#importImagesBtn").prop("disabled", false);
      return;
    }
    if (!path) {
      // Cancelled — fall back and do nothing.
      $("#importStatus").text("");
      $("#importImagesBtn").prop("disabled", false);
      return;
    }

    const { mode, customValue } = readSensitivityFromUI();
    const recognizeFaces = $("#recognizeFacesInput").is(":checked");
    const noTeam = $("#ignoreJerseyColorInput").is(":checked");
    const teamColor = noTeam ? "" : selectedTeamColor;
    setImportUIEnabled(false);
    $("#processingOutput").val("");
    pollSince = 0;
    openProcessingDialog();
    $("#importStatus").text("Processing…");
    try {
      await apiPost("/api/import-path", { path });
      await apiPost("/api/start-processing", {
        path,
        albumName,
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        recognizeFaces,
        noTeam,
        teamColor,
        teamId: selectedTeamId,
      });
      pollTimer = setInterval(pollProcessingOutput, 500);
    } catch (err) {
      $("#importStatus").text("Failed: " + err.message);
      setImportUIEnabled(true);
      $("#processingDialog").dialog("close");
    }
  });

  $("#ignoreJerseyColorInput").on("change", function () {
    $("#teamColorPicker").find("button").prop("disabled", this.checked);
  });

  window.addEventListener("beforeunload", (e) => {
    if (isProcessing) { e.preventDefault(); e.returnValue = ""; }
  });

  initRecognizeFacesCheckbox();
  initProcessingDialog();
  (async () => {
    const preferredTeamId = await loadImportState().catch(() => "");
    await loadTeams(preferredTeamId).catch(() => {});
  })();
});
