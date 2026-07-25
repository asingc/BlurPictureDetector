"use strict";

// Page 1 — Team Setup: a row of team blocks at the top lets you switch
// between existing teams or start a brand-new one; the panels below edit
// whichever team is currently selected.

let teams = [];
let selectedTeamId = null; // null => creating a brand-new (unsaved) team

function blankTeam() {
  return { id: "", name: "", jerseyColors: [], players: [], openaiApiKey: "" };
}

let Team = blankTeam();

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

function renderForm() {
  $("#teamNameInput").val(Team.name || "");
  renderJerseyList();
  renderPlayerTable();
  $("#openaiKeyInput").val(Team.openaiApiKey || "");
  const hasGoalkeeper = Team.jerseyColors.some((jc) => jc.forced);
  $("#hasGoalkeeperInput").prop("checked", hasGoalkeeper);
  $("#goalkeeperJerseyRow").toggle(hasGoalkeeper);
}

function renderPicker() {
  renderTeamPicker($("#teamPicker"), teams, {
    selectedId: selectedTeamId,
    onSelect: (team) => {
      selectedTeamId = team.id;
      Team = $.extend(true, {}, team);
      renderForm();
      renderPicker();
    },
    onAddNew: () => {
      selectedTeamId = null;
      Team = blankTeam();
      $("#teamSaveStatus").text("");
      renderForm();
      renderPicker();
    },
    addLabel: "Add new team",
  });
}

async function loadTeams() {
  const data = await apiGet("/api/teams/query");
  teams = data.teams || [];
  if (teams.length) {
    selectedTeamId = teams[0].id;
    Team = $.extend(true, {}, teams[0]);
  } else {
    selectedTeamId = null;
    Team = blankTeam();
  }
  renderForm();
  renderPicker();
}

async function saveTeam() {
  Team.name = $("#teamNameInput").val().trim();
  Team.openaiApiKey = $("#openaiKeyInput").val();
  $("#saveTeamBtn").prop("disabled", true);
  $("#teamSaveStatus").text("Saving…");
  try {
    const res = await apiPost("/api/team", Team);
    Team = res.team;
    selectedTeamId = Team.id;
    const idx = teams.findIndex((t) => t.id === Team.id);
    if (idx >= 0) teams[idx] = Team; else teams.push(Team);
    renderForm();
    renderPicker();
    $("#teamSaveStatus").text("Saved.");
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
    Team.jerseyColors.push({ color, forced: false });
    $("#jerseyColorInput").val("");
    renderJerseyList();
  });
  $("#jerseyColorInput").on("keydown", (e) => { if (e.key === "Enter") $("#addJerseyBtn").click(); });

  $("#hasGoalkeeperInput").on("change", function () {
    $("#goalkeeperJerseyRow").toggle($(this).is(":checked"));
  });

  $("#addGoalkeeperJerseyBtn").on("click", () => {
    const color = $("#goalkeeperJerseyInput").val().trim();
    if (!color) return;
    Team.jerseyColors.push({ color, forced: true });
    $("#goalkeeperJerseyInput").val("");
    renderJerseyList();
  });
  $("#goalkeeperJerseyInput").on("keydown", (e) => { if (e.key === "Enter") $("#addGoalkeeperJerseyBtn").click(); });

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

  loadTeams().catch((err) => alert("Failed to load teams: " + err.message));
});
