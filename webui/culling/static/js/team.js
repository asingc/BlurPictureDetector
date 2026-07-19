"use strict";

// Page 1 — Team Setup

const Team = {
  jerseyColors: [],
  players: [],
  openaiApiKey: "",
  sensitivity: { mode: "medium", customValue: 0.50 },
};

function renderJerseyList() {
  const list = $("#jerseyList").empty();
  Team.jerseyColors.forEach((jc, idx) => {
    const tag = $("<span class='tag'>").text(jc.color + (jc.forced ? " (Special/Goalie)" : ""));
    const remove = $("<button class='tag-remove'>&times;</button>").on("click", () => {
      Team.jerseyColors.splice(idx, 1);
      renderJerseyList();
    });
    tag.append(remove);
    list.append(tag);
  });
}

function renderPlayerTable() {
  const tbody = $("#playerTable tbody").empty();
  Team.players.forEach((p, idx) => {
    const row = $("<tr>");
    row.append($("<td>").text(p.name));
    row.append($("<td>").text(p.number));
    const removeBtn = $("<button class='btn btn-danger btn-sm'>Remove</button>").on("click", () => {
      Team.players.splice(idx, 1);
      renderPlayerTable();
    });
    row.append($("<td>").append(removeBtn));
    tbody.append(row);
  });
}

function applySensitivityToUI() {
  const mode = Team.sensitivity.mode || "medium";
  $(`input[name="sensMode"][value="${mode}"]`).prop("checked", true);
  const custom = Team.sensitivity.customValue ?? 0.50;
  $("#sensitivityCustomSlider").val(custom);
  $("#sensitivityCustomValue").text(Number(custom).toFixed(2));
}

function readSensitivityFromUI() {
  const mode = $('input[name="sensMode"]:checked').val() || "medium";
  const customValue = parseFloat($("#sensitivityCustomSlider").val()) || 0;
  return { mode, customValue };
}

async function loadTeam() {
  const data = await apiGet("/api/team");
  Object.assign(Team, data);
  renderJerseyList();
  renderPlayerTable();
  $("#openaiKeyInput").val(Team.openaiApiKey || "");
  applySensitivityToUI();
}

async function saveTeam() {
  Team.sensitivity = readSensitivityFromUI();
  Team.openaiApiKey = $("#openaiKeyInput").val();
  $("#saveTeamBtn").prop("disabled", true);
  $("#teamSaveStatus").text("Saving…");
  try {
    await apiPost("/api/team", Team);
    $("#teamSaveStatus").text("Saved.");
    window.location.href = "/import";
  } catch (err) {
    $("#teamSaveStatus").text("Save failed: " + err.message);
  } finally {
    $("#saveTeamBtn").prop("disabled", false);
  }
}

$(function () {
  $("#addJerseyBtn").on("click", () => {
    const color = $("#jerseyColorInput").val().trim();
    if (!color) return;
    Team.jerseyColors.push({ color, forced: $("#jerseyForcedInput").is(":checked") });
    $("#jerseyColorInput").val("");
    $("#jerseyForcedInput").prop("checked", false);
    renderJerseyList();
  });
  $("#jerseyColorInput").on("keydown", (e) => { if (e.key === "Enter") $("#addJerseyBtn").click(); });

  $("#addPlayerBtn").on("click", () => {
    const raw = $("#playerBulkInput").val();
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
    lines.forEach((line) => {
      const [namePart, numberPart] = line.split(",");
      const name = (namePart || "").trim();
      const number = (numberPart || "").trim();
      if (name) Team.players.push({ name, number });
    });
    $("#playerBulkInput").val("");
    renderPlayerTable();
  });

  $('input[name="sensMode"]').on("change", function () {
    // no-op: slider stays interactive regardless of selected mode
  });
  $("#sensitivityCustomSlider").on("input", function () {
    $("#sensitivityCustomValue").text(Number($(this).val()).toFixed(2));
    $('input[name="sensMode"][value="custom"]').prop("checked", true);
  });

  $("#toggleKeyVisibilityBtn").on("click", function () {
    const input = $("#openaiKeyInput");
    const isPwd = input.attr("type") === "password";
    input.attr("type", isPwd ? "text" : "password");
    $(this).text(isPwd ? "Hide" : "Show");
  });

  $("#saveTeamBtn").on("click", saveTeam);

  loadTeam().catch((err) => alert("Failed to load team.json: " + err.message));
});
