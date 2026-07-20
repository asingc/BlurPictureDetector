"use strict";

// Page 1 — Team Setup

const Team = {
  jerseyColors: [],
  players: [],
  openaiApiKey: "",
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

async function loadTeam() {
  const data = await apiGet("/api/team");
  Object.assign(Team, data);
  renderJerseyList();
  renderPlayerTable();
  $("#openaiKeyInput").val(Team.openaiApiKey || "");
}

async function saveTeam() {
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

  $("#toggleKeyVisibilityBtn").on("click", function () {
    const input = $("#openaiKeyInput");
    const isPwd = input.attr("type") === "password";
    input.attr("type", isPwd ? "text" : "password");
    $(this).text(isPwd ? "Hide" : "Show");
  });

  $("#saveTeamBtn").on("click", saveTeam);

  loadTeam().catch((err) => alert("Failed to load team.json: " + err.message));
});
