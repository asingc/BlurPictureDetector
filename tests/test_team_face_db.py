"""Per-team system face DB: team ids, autocomplete scoping, export sync, migration."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

import MigrateFaceDB
import culling_app as app
def write_album(album_dir: Path, entries: list[dict], team_id: str) -> None:
    album_dir.mkdir(parents=True, exist_ok=True)
    (album_dir / "album.json").write_text(
        json.dumps({"team_id": team_id, "import_status": "complete", "results": entries}), encoding="utf-8"
    )

TEAM_A = "11111111-1111-4111-8111-111111111111"
TEAM_B = "22222222-2222-4222-8222-222222222222"


def _team(team_id: str, name: str, players: list[str]) -> dict:
    return {"id": team_id, "name": name, "players": [{"name": p, "number": ""} for p in players]}


def _person(root: Path, folder: str, name: str | None = None) -> Path:
    d = root / folder
    (d / "Face").mkdir(parents=True)
    (d / "face.json").write_text(json.dumps({"name": name or folder, "faces": []}), encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    team_json = repo / "team.json"
    teams = {"Teams": [
        _team(TEAM_A, "Team A", ["Ansel Chang", "Ava A"]),
        _team(TEAM_B, "Team B", ["Ansel Chang", "Ben B"]),
    ]}
    team_json.write_text(json.dumps(teams), encoding="utf-8")
    monkeypatch.setattr(app, "REPO_ROOT", repo)
    monkeypatch.setattr(app, "TEAM_JSON_PATH", team_json)
    return repo


def test_legacy_team_id_is_upgraded_and_still_resolves(env):
    app.TEAM_JSON_PATH.write_text(
        json.dumps({"Teams": [_team("abc123def456", "Old", ["X"]), _team(TEAM_A, "Keep", [])]}),
        encoding="utf-8",
    )
    teams = app._load_teams_file().Teams
    old, kept = teams
    assert uuid.UUID(old.id) and old.legacyIds == ["abc123def456"]
    assert kept.id == TEAM_A and kept.legacyIds == []
    assert app._resolve_team("abc123def456").id == old.id
    assert app._canonical_team_id("abc123def456") == old.id
    # Persisted: a second load must not mint another id.
    assert app._load_teams_file().Teams[0].id == old.id


def test_rename_keeps_id_and_face_db_folder(env):
    renamed = app.TeamData(id=TEAM_A, name="Renamed", players=[])
    app._save_team(renamed)
    assert app._find_team(app._load_teams_file(), TEAM_A).name == "Renamed"
    assert app._team_face_db_dir(TEAM_A).name == TEAM_A


def test_autocomplete_limited_to_current_team(env, tmp_path):
    root = env / ".FaceReco"
    _person(root / TEAM_A, "Coach A")
    _person(root / TEAM_B, "Coach B")
    album = tmp_path / "album"
    write_album(album, [], team_id=TEAM_A)
    _person(album / ".FaceReco", "Tagged In Album")

    names = app._collect_names(album)
    assert "Coach A" in names and "Tagged In Album" in names
    assert "Ava A" in names and "Ansel Chang" in names
    assert "Coach B" not in names and "Ben B" not in names


def test_sync_puts_any_tagged_name_in_team_folder_only(env, tmp_path):
    album = tmp_path / "album"
    write_album(album, [], team_id=TEAM_B)
    fr = album / ".FaceReco"
    for name in ("Ansel Chang", "Random Coach"):
        d = _person(fr, name)
        (d / "Face" / "c.png").write_bytes(b"x")
        (d / "face.json").write_text(json.dumps({
            "name": name, "faces": [{"cropFileName": "c.png", "origFilename": "a.jpg"}],
        }), encoding="utf-8")

    assert app._sync_named_clusters_to_system_db(album, {"a.jpg"}) == 2

    team_dir = env / ".FaceReco" / TEAM_B
    assert (team_dir / "Ansel Chang" / "Face" / "c.png").is_file()
    assert (team_dir / "Random Coach" / "face.json").is_file()
    assert not (env / ".FaceReco" / TEAM_A).exists()


def test_migration_copies_into_every_rostered_team(tmp_path):
    root = tmp_path / ".FaceReco"
    _person(root, "Ansel Chang")
    _person(root, "Ava A")
    _person(root, "Ansel", name="Ansel Chang")  # alias folder, name from face.json
    _person(root, "Riley Shen")
    (root / "calibration.json").write_text("{}", encoding="utf-8")
    teams = [_team(TEAM_A, "A", ["Ansel Chang", "Ava A"]), _team(TEAM_B, "B", ["Ansel Chang"])]

    dry = MigrateFaceDB.migrate(root, teams, dry_run=True)
    assert len(dry.copied) == 5 and not (root / TEAM_A).exists() and (root / "Ansel Chang").exists()

    report = MigrateFaceDB.migrate(root, teams)
    assert report.backup and report.backup.is_dir()
    for tid in (TEAM_A, TEAM_B):
        assert (root / tid / "Ansel Chang" / "face.json").is_file()
        assert (root / tid / "Ansel" / "face.json").is_file()
        assert (root / tid / "calibration.json").is_file()
    assert (root / TEAM_A / "Ava A").is_dir() and not (root / TEAM_B / "Ava A").exists()
    assert report.unmatched == ["Riley Shen"] and (root / "Riley Shen").is_dir()
    assert not (root / "Ansel Chang").exists()

    # Idempotent: nothing flat left to move.
    again = MigrateFaceDB.migrate(root, teams, backup=False)
    assert not again.copied


def test_migration_rejects_pre_uuid_team_ids(tmp_path):
    team_json = tmp_path / "team.json"
    team_json.write_text(json.dumps({"Teams": [_team("abc123def456", "Old", [])]}), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-UUID"):
        MigrateFaceDB._load_teams(team_json)
