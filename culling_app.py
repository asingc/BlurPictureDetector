"""End-to-end photo culling & organization web UI.

Launches a local FastAPI server backing a jQuery/jQuery UI multi-page app —
each step of the workflow is its own template + own JS file (sharing one
Jinja2 base layout for the topbar/stepper look-and-feel), giving each page
natural code isolation instead of one giant single-page app:

  1. Team setup       — jersey colours, player roster, OpenAI key.
  2. Select album     — resume a previously processed album, or create a new
                         one (source folder, blur sensitivity, face
                         recognition) and run the detection pipeline.
  3. Image review      — sort blur / sharp / skipped.
  4. Face clustering    — group and label face clusters.
  5. Apply changes      — move files / write face-reco info, with progress.

This module currently implements the server-side pieces for pages 1 and 2
(team persistence + import-folder selection with "remember last" support).
Later iterations will add the processing/review/clustering/apply endpoints.

Usage::

    python culling_app.py [--host 127.0.0.1] [--port 8010]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from algo.utils import atomic_save_and_backup

try:
    from PIL import Image as _PILImage
    from PIL import ExifTags as _PILExifTags
except ImportError:  # Pillow not installed — EXIF timestamps just won't be available.
    _PILImage = None
    _PILExifTags = None

log = logging.getLogger("CullingApp")

REPO_ROOT = Path(__file__).resolve().parent
WEBUI_DIR = REPO_ROOT / "webui" / "culling"

TEAM_JSON_PATH = REPO_ROOT / "team.json"
STATE_JSON_PATH = REPO_ROOT / ".culling_state.json"
OUTPUT_DIR = REPO_ROOT / "albums"

# "high" recalibrated 2026-07-27 (0.70 -> 0.68) alongside the production
# sharpness-evaluator swap to WeightedGeometricMeanEvaluator (algo/sharpness.py)
# so "high" sensitivity keeps the same recall (~0.63) the old evaluator
# achieved at its old 0.70 threshold — see _setup_tmp/sharpness_eval/calibrate_high_threshold.py.
SENSITIVITY_PRESETS: dict[str, float] = {"low": 0.35, "medium": 0.50, "high": 0.68}

# Folder-name candidates for a face-DB dir, mirroring 1_prep_review.py / face_tag_ui.py.
FACE_DB_DIR_CANDIDATES: tuple[str, ...] = (".FaceReco", ".facereco", ".Facereco")


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class JerseyColor(BaseModel):
    color: str
    forced: bool = False


class Player(BaseModel):
    name: str = ""
    number: str = ""


class TeamData(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    jerseyColors: list[JerseyColor] = Field(default_factory=list)
    players: list[Player] = Field(default_factory=list)
    openaiApiKey: str = ""


class GlobalSettings(BaseModel):
    """App-wide settings that aren't tied to any one team. Empty for now —
    reserved for future cross-team settings; `extra="allow"` so hand-added
    fields round-trip instead of being dropped on save."""
    model_config = ConfigDict(extra="allow")


class TeamsFile(BaseModel):
    """On-disk shape of team.json: a list of teams plus shared Global
    settings."""
    Teams: list[TeamData] = Field(default_factory=list)
    Global: GlobalSettings = Field(default_factory=GlobalSettings)


class ImportPathRequest(BaseModel):
    path: str


class StartProcessingRequest(BaseModel):
    path: str
    albumName: str = ""                   # optional custom name for the output album folder
    sensitivityMode: str = "medium"       # "low" | "medium" | "high" | "custom"
    sensitivityCustomValue: float = 0.50  # used only when mode == "custom", 0-0.99
    recognizeFaces: bool = True
    teamId: str = ""                      # which team's jersey colors/roster/API key to use


class BrowseFolderRequest(BaseModel):
    title: str = "Select photo folder to import"
    # "import" seeds the dialog from the *parent* of the last-used import
    # source folder; "export" seeds it from the *parent* of the last export
    # destination — either way, picking a sibling folder for the next
    # album is one click.
    context: str = "import"


class ExportRequest(BaseModel):
    destination: str
    exportFaceTagging: bool = True


class AlbumSelectRequest(BaseModel):
    id: str  # album directory name, e.g. "20260719-030120-TestImages" — not a path


class AlbumDeleteRequest(BaseModel):
    id: str  # album directory name, e.g. "20260719-030120-TestImages" — not a path


class ReviewApplyRequest(BaseModel):
    # {original filename: keep} for every image the user actually toggled
    # this session, across all 3 review tabs at once. Anything not present
    # here keeps its current effective (explicit-or-default) keep state.
    overrides: dict[str, bool] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Processing (1_prep_review.py) — run in a background thread, buffer combined
# stdout/stderr lines so the browser can poll for them.
# --------------------------------------------------------------------------- #
class ProcessingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.return_code: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None


processing_state = ProcessingState()


# --------------------------------------------------------------------------- #
# Export (page 5 — Apply) — copy "keep" images and merge tagged-player faces
# into a .FaceReco database at a destination folder, run in a background
# thread with the same buffered-lines-for-polling pattern as ProcessingState.
# --------------------------------------------------------------------------- #
class ExportState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.done = False
        self.error: Optional[str] = None
        self.lines: list[str] = []
        self.total_images = 0
        self.copied_images = 0
        self.total_players = 0
        self.processed_players = 0
        self.dest_dir: Optional[str] = None


export_state = ExportState()


# --------------------------------------------------------------------------- #
# Heartbeat watchdog — the browser pings /api/heartbeat periodically from
# every page (common.js); if none arrives within the timeout, the server
# exits automatically so we don't leave orphaned local servers running after
# the browser tab/window is closed. Mirrors face_tag_ui.py's watchdog.
# --------------------------------------------------------------------------- #
class HeartbeatState:
    def __init__(self, timeout: float = 180.0) -> None:
        self.lock = threading.Lock()
        self.timeout = timeout
        self.last_heartbeat = time.time()


heartbeat_state = HeartbeatState()


def _heartbeat_watchdog() -> None:
    """Exit the process if no browser heartbeat arrives within the timeout.

    Runs as a daemon thread. The timer starts from server launch, so the
    browser has one full timeout window to load the page and send its first
    heartbeat before the watchdog can fire.
    """
    poll_interval = max(1.0, min(5.0, heartbeat_state.timeout / 3))
    last_poll = time.time()
    while True:
        time.sleep(poll_interval)
        now = time.time()
        poll_gap = now - last_poll
        last_poll = now
        if poll_gap > poll_interval * 3:
            # This watchdog thread itself missed multiple polls — wall-clock
            # time jumped far more than a sleeping thread should ever drift.
            # That only happens if the whole machine (this server process
            # included) was suspended, not just the browser tab/window being
            # closed — a closed tab doesn't affect our own thread's timing at
            # all. Since the browser suspends/resumes together with us on the
            # same machine, forgive this cycle instead of shutting down: it
            # will send a fresh heartbeat shortly after waking anyway.
            log.info(
                "Watchdog poll delayed %.0fs (expected ~%.0fs) — system likely "
                "resumed from sleep; resetting heartbeat timer instead of exiting.",
                poll_gap, poll_interval,
            )
            with heartbeat_state.lock:
                heartbeat_state.last_heartbeat = now
            continue
        with heartbeat_state.lock:
            elapsed = now - heartbeat_state.last_heartbeat
        if elapsed > heartbeat_state.timeout:
            log.info(
                "No heartbeat received for %.0fs (timeout %.0fs) — shutting down.",
                elapsed, heartbeat_state.timeout,
            )
            os._exit(0)


def _sensitivity_arg(mode: str, custom_value: float) -> str:
    if mode == "custom":
        return str(custom_value)
    return mode


def _jerseycolor_arg(team: TeamData) -> str:
    parts = []
    for jc in team.jerseyColors:
        prefix = "+" if jc.forced else ""
        parts.append(prefix + jc.color)
    return ";".join(parts)


def _run_processing(path: str, album_name: str, sensitivity_mode: str, sensitivity_custom_value: float, recognize_faces: bool, team_id: str) -> None:
    team = _resolve_team(team_id)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_stem = _sanitize_player_dirname(album_name.strip()) if album_name.strip() else Path(path).stem
    output_dir = OUTPUT_DIR / f"{ts}-{folder_stem}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "1_prep_review.py"),
        path,
        "--sensitivity", _sensitivity_arg(sensitivity_mode, sensitivity_custom_value),
        "--output", str(output_dir),
        "--no-tag-ui",
    ]
    if not recognize_faces:
        cmd += ["--skip-facereco"]
    jerseycolor = _jerseycolor_arg(team)
    if jerseycolor:
        cmd += ["--jerseycolor", jerseycolor]
    if team.id:
        cmd += ["--team-id", team.id]
    if team.openaiApiKey:
        cmd += ["--openaikey", team.openaiApiKey]

    log.info("Starting processing: %s", " ".join("***" if a == team.openaiApiKey else a for a in cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    with processing_state.lock:
        processing_state.process = proc

    assert proc.stdout is not None
    for line in proc.stdout:
        with processing_state.lock:
            processing_state.lines.append(line.rstrip("\n"))

    proc.wait()
    with processing_state.lock:
        processing_state.running = False
        processing_state.return_code = proc.returncode
        processing_state.process = None
    log.info("Processing finished (exit code %s)", proc.returncode)

    if proc.returncode == 0 and _is_album_complete(output_dir):
        state = _load_state()
        state["currentAlbum"] = str(output_dir.resolve())
        _save_state(state)
        log.info("Current album set from processing run: %s", output_dir.resolve())


# --------------------------------------------------------------------------- #
# team.json persistence — {"Teams": [...], "Global": {...}}. Multiple teams
# are supported; each is identified by a stable `id` assigned the first time
# it's loaded/saved without one. QueryTeams (below) exposes every team's
# settings; the Team Setup page can create/update any team by id.
# --------------------------------------------------------------------------- #
def _default_teams_file() -> TeamsFile:
    return TeamsFile(Teams=[TeamData(name="Team 1")])


def _load_teams_file() -> TeamsFile:
    """Load team.json, transparently migrating the legacy flat single-team
    format (no "Teams" key — the whole file *was* one team's data) into
    {"Teams": [...], "Global": {...}}, and backfilling a stable `id`/`name`
    for any team that predates those fields (re-saved immediately so the
    generated id doesn't change on the next load)."""
    if not TEAM_JSON_PATH.is_file():
        teams_file = _default_teams_file()
        _save_teams_file(teams_file)
        return teams_file
    try:
        with open(TEAM_JSON_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        if "Teams" not in payload:
            payload = {"Teams": [payload], "Global": {}}
        needs_resave = False
        # Defensive: coerce null player fields to "" so one malformed entry
        # (e.g. hand-edited JSON, or a future bulk-parse edge case) doesn't
        # discard the whole file back to defaults. Also backfill id/name.
        for i, team in enumerate(payload.get("Teams", [])):
            if not isinstance(team, dict):
                continue
            if not team.get("id"):
                team["id"] = uuid.uuid4().hex[:12]
                needs_resave = True
            if not team.get("name"):
                team["name"] = f"Team {i + 1}"
                needs_resave = True
            for player in team.get("players", []):
                if isinstance(player, dict):
                    player["name"] = player.get("name") or ""
                    player["number"] = player.get("number") or ""
        teams_file = TeamsFile.model_validate(payload)
        if not teams_file.Teams:
            teams_file.Teams = [TeamData(name="Team 1")]
            needs_resave = True
        if needs_resave:
            _save_teams_file(teams_file)
        return teams_file
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("Failed to load team.json (%s) — using defaults", exc)
        return _default_teams_file()


def _save_teams_file(teams_file: TeamsFile) -> None:
    with open(TEAM_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(teams_file.model_dump(), fh, indent=2)


def _find_team(teams_file: TeamsFile, team_id: str) -> Optional[TeamData]:
    for team in teams_file.Teams:
        if team.id == team_id:
            return team
    return None


def _resolve_team(team_id: str) -> TeamData:
    """Resolve a team by id for use by the processing pipeline, falling back
    to the first team on disk if no id was given or it no longer exists."""
    teams_file = _load_teams_file()
    if team_id:
        team = _find_team(teams_file, team_id)
        if team is not None:
            return team
        log.warning("Team id %r not found — falling back to the first team.", team_id)
    return teams_file.Teams[0]


def _save_team(team: TeamData) -> TeamData:
    """Create (if `team.id` is blank or unrecognized) or update (if it
    matches an existing team) a single team, preserving every other team /
    Global settings already on disk. Returns the saved team (with its id
    filled in if it was newly created)."""
    teams_file = _load_teams_file()
    if not team.id:
        team.id = uuid.uuid4().hex[:12]
    for i, existing in enumerate(teams_file.Teams):
        if existing.id == team.id:
            teams_file.Teams[i] = team
            break
    else:
        teams_file.Teams.append(team)
    _save_teams_file(teams_file)
    log.info("team.json saved (team %r: %d player(s), %d jersey colour(s))",
              team.name, len(team.players), len(team.jerseyColors))
    return team


# --------------------------------------------------------------------------- #
# App state persistence (last import path, etc.) — kept separate from
# team.json since it isn't "team" data.
# --------------------------------------------------------------------------- #
def _load_state() -> dict:
    if not STATE_JSON_PATH.is_file():
        return {"lastImportPath": "", "currentAlbum": "", "selectedTeamId": ""}
    try:
        with open(STATE_JSON_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"lastImportPath": "", "currentAlbum": "", "selectedTeamId": ""}


def _save_state(state: dict) -> None:
    with open(STATE_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# --------------------------------------------------------------------------- #
# Album discovery — list/select previously processed albums/ folders so the
# user can resume review/tagging without re-running 1_prep_review.py.
# --------------------------------------------------------------------------- #
def _album_has_facereco(path: Path) -> bool:
    return any((path / name).is_dir() for name in FACE_DB_DIR_CANDIDATES)


def _is_album_complete(path: Path) -> bool:
    """An album is "fully processed" once 1_prep_review.py has written both
    album.json and info.json - both are always written together (barring a
    completely empty input folder), regardless of --skip-facereco or whether
    any blurry images were found."""
    return (path / "album.json").is_file() and (path / "info.json").is_file()


# How many random sharp-image filenames to hand to the client per album for
# the polaroid preview / hover slideshow — capped so the /api/albums payload
# stays small even for albums with thousands of sharp photos.
PREVIEW_IMAGE_SAMPLE = 8


# Matches the "<yyyymmdd-hhmmss>-" prefix _run_processing() stamps onto every
# output folder name (see `output_dir = OUTPUT_DIR / f"{ts}-{folder_stem}"`).
_ALBUM_DIR_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}-\d{6}-")


def _album_display_name(dir_name: str) -> str:
    """Human-friendly album name for UI display: the directory name with its
    leading run-timestamp prefix stripped (that timestamp is already shown
    separately as `createdDisplay`). Falls back to the full directory name
    if it doesn't match the expected `1_prep_review.py` naming scheme."""
    return _ALBUM_DIR_TIMESTAMP_PREFIX_RE.sub("", dir_name) or dir_name


def _format_created(timestamp: str) -> str:
    """"20260719-030759" -> "Jul 19, 2026 03:07 AM"; falls back to the raw
    string (or "") if it doesn't match the expected 1_prep_review.py format."""
    if not timestamp:
        return ""
    try:
        return datetime.strptime(timestamp, "%Y%m%d-%H%M%S").strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return timestamp


def _read_album_summary(path: Path) -> dict:
    summary: dict = {
        "name": path.name,  # also serves as the album's id — never the absolute path
        "displayName": _album_display_name(path.name),  # human-friendly, for UI display only
        "srcDir": "",
        "timestamp": "",
        "createdDisplay": "",
        "ourJerseyColor": None,
        "sharpCount": 0,
        "blurCount": 0,
        "skippedCount": 0,
        "hasFaceReco": _album_has_facereco(path),
        "previewImages": [],
    }
    try:
        with open(path / "info.json", encoding="utf-8") as fh:
            info = json.load(fh)
        summary["srcDir"] = info.get("SrcDir", "")
        summary["timestamp"] = info.get("Timestamp", "")
        summary["createdDisplay"] = _format_created(info.get("Timestamp", ""))
        summary["ourJerseyColor"] = info.get("OurJerseyColor")
        sharp = info.get("Anno_Sharp", [])
        summary["sharpCount"] = len(sharp)
        summary["blurCount"] = len(info.get("Anno_Blur", []))
        summary["skippedCount"] = len(info.get("Anno_Skipped", []))
        results_by_name = _results_by_filename(_load_results_payload(path))
        sharp_previews = []
        for item in sharp:
            result = results_by_name.get(item.get("src"))
            preview_path = result.get("preview_path") if result else None
            if preview_path:
                sharp_previews.append(Path(preview_path).name)
        summary["previewImages"] = random.sample(sharp_previews, min(len(sharp_previews), PREVIEW_IMAGE_SAMPLE))
    except (json.JSONDecodeError, OSError):
        pass
    return summary


def _list_albums() -> list[dict]:
    if not OUTPUT_DIR.is_dir():
        return []
    albums = [
        _read_album_summary(child)
        for child in OUTPUT_DIR.iterdir()
        if child.is_dir() and _is_album_complete(child)
    ]
    albums.sort(key=lambda a: a["name"], reverse=True)
    return albums


# --------------------------------------------------------------------------- #
# Face clustering — ported directly from face_tag_ui.py so the Face
# Clustering page (4) is a native page of this workflow (own template + own
# JS, styled like every other step) instead of embedding a separate server
# via an iframe.
# --------------------------------------------------------------------------- #
FACE_SUBDIR = "Face"
FACE_ANNOTATED_SUBDIR = "Face.annotated"

# Matching tolerance for body_bbox floats when writing back to album.json.
BBOX_EPS = 1e-6


def _album_dir_by_id(album_id: str) -> Path:
    """Resolve an album's directory name (id) to its path under OUTPUT_DIR —
    _safe_component rejects separators/".." so this can only ever land on a
    direct child of OUTPUT_DIR, never an arbitrary path on disk."""
    path = OUTPUT_DIR / _safe_component(album_id)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Not a valid album directory.")
    return path


def _current_album_path() -> Path:
    state = _load_state()
    album_path_str = state.get("currentAlbum", "")
    if not album_path_str:
        raise HTTPException(status_code=400, detail="No album selected.")
    path = Path(album_path_str)
    if not path.is_dir() or not _is_album_complete(path):
        raise HTTPException(status_code=400, detail="Album not found or not fully processed.")
    return path


# --------------------------------------------------------------------------- #
# Review (page 3) — sort blur / sharp / skipped images into keep / drop,
# grouped into time-based "bursts". Decisions are staged client-side and only
# committed to album.json (as an explicit "keep" boolean per entry) when the
# user hits Apply — see api_review_apply().
# --------------------------------------------------------------------------- #
REVIEW_CATEGORIES = ("blur", "sharp", "skipped")
_REVIEW_INFO_KEY = {"blur": "Anno_Blur", "sharp": "Anno_Sharp", "skipped": "Anno_Skipped"}
# All annotated previews (blur/sharp/skipped alike) live in one shared folder
# — see algo/stages/annotation.py::AnnotationStage.
_PREVIEWS_SUBDIR = "previews"
# Effective "keep" when album.json has no explicit value yet (older albums,
# or entries the user hasn't touched this session): sharp images default to
# keep, blur/skipped default to drop — matching the pre-existing behaviour of
# manually deleting anno_* previews to "reject" an image.
_REVIEW_DEFAULT_KEEP = {"blur": False, "sharp": True, "skipped": False}

# Consecutive photos within this many seconds of each other are treated as
# the same "burst" for the nav-pane grouping.
BURST_GAP_SECONDS = 1.0

# EXIF tag ids (main IFD "DateTime", and Exif sub-IFD "DateTimeOriginal" /
# "DateTimeDigitized") — checked in that order.
_EXIF_TAG_DATETIME = 306
_EXIF_TAG_DATETIME_ORIGINAL = 36867
_EXIF_TAG_DATETIME_DIGITIZED = 36868


def _image_timestamp(path: Path) -> float:
    """Best-effort capture time (as a Unix timestamp) for burst grouping:
    EXIF capture time first, falling back to min(file create time, file
    modified time) when EXIF is missing/unreadable (non-JPEG source, no
    camera metadata, Pillow not installed, etc.)."""
    if _PILImage is not None:
        try:
            with _PILImage.open(path) as img:
                exif = img.getexif()
                raw = exif.get(_EXIF_TAG_DATETIME)
                if not raw:
                    sub = exif.get_ifd(_PILExifTags.IFD.Exif)
                    raw = sub.get(_EXIF_TAG_DATETIME_ORIGINAL) or sub.get(_EXIF_TAG_DATETIME_DIGITIZED)
                if raw:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass
    try:
        stat = path.stat()
        return min(stat.st_ctime, stat.st_mtime)
    except OSError:
        return 0.0


def _load_results_payload(album_path: Path) -> dict:
    with open(album_path / "album.json", encoding="utf-8") as fh:
        return json.load(fh)


def _results_by_filename(payload: dict) -> dict[str, dict]:
    return {Path(entry["file"]).name: entry for entry in payload.get("results", [])}


def _review_images(album_path: Path, category: str) -> list[dict]:
    """Per-image metadata for one review category: filename, annotated-preview
    filename, effective keep state, and best-effort capture timestamp."""
    with open(album_path / "info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    src_dir = Path(info.get("SrcDir", ""))
    results_by_name = _results_by_filename(_load_results_payload(album_path))
    default_keep = _REVIEW_DEFAULT_KEEP[category]
    previews_dir = album_path / _PREVIEWS_SUBDIR

    images = []
    for item in info.get(_REVIEW_INFO_KEY[category], []):
        src_name = item.get("src")
        if not src_name:
            continue
        result = results_by_name.get(src_name)
        preview_path = result.get("preview_path") if result else None
        if not preview_path:
            continue
        anno_name = Path(preview_path).name
        keep = bool(result.get("keep", default_keep)) if result else default_keep
        burst_ranking = result.get("burst_ranking") if result else None
        llm_grade = result.get("llm_grade") if result else None
        src_path = src_dir / src_name
        ts_path = src_path if src_path.is_file() else previews_dir / anno_name
        images.append({
            "file": src_name,
            "anno": anno_name,
            "keep": keep,
            "burstRanking": burst_ranking,
            "llmGrade": llm_grade,
            "timestamp": _image_timestamp(ts_path),
        })
    return images


def _kept_image_basenames(album_path: Path) -> set[str]:
    """Basenames of every image the user decided to keep, across all 3
    review categories — the same effective (explicit-or-default) keep state
    the Review page shows. Used by the Apply/export step to decide which
    original photos (and which face crops) to copy to the destination."""
    kept: set[str] = set()
    for category in REVIEW_CATEGORIES:
        for image in _review_images(album_path, category):
            if image["keep"]:
                kept.add(image["file"])
    return kept


def _group_bursts(images: list[dict]) -> list[list[dict]]:
    """Group images into "bursts": a new group starts whenever the gap since
    the chronologically-previous image exceeds BURST_GAP_SECONDS."""
    ordered = sorted(images, key=lambda im: (im["timestamp"], im["file"]))
    groups: list[list[dict]] = []
    for im in ordered:
        if groups and im["timestamp"] - groups[-1][-1]["timestamp"] <= BURST_GAP_SECONDS:
            groups[-1].append(im)
        else:
            groups.append([im])
    return groups


def _sort_groups(groups: list[list[dict]], sort_mode: str) -> list[list[dict]]:
    """Order in which burst groups are listed in the nav pane. Images within
    a group are always chronological, regardless of this setting."""
    if sort_mode == "old":
        return sorted(groups, key=lambda g: g[0]["timestamp"])
    if sort_mode == "new":
        return sorted(groups, key=lambda g: g[0]["timestamp"], reverse=True)
    return sorted(groups, key=lambda g: (-len(g), g[0]["timestamp"]))  # "size" (default)


def _facereco_dir(album_path: Path) -> Optional[Path]:
    for name in FACE_DB_DIR_CANDIDATES:
        candidate = album_path / name
        if candidate.is_dir():
            return candidate
    return None


def _global_face_db_dir() -> Optional[Path]:
    """Read-only name dictionary shared across albums (repo-root .FaceReco)."""
    for name in FACE_DB_DIR_CANDIDATES:
        candidate = REPO_ROOT / name
        if candidate.is_dir():
            return candidate
    return None


def _safe_component(name: str) -> str:
    """Reject a path component containing separators or traversal."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid path component")
    return name


_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_player_dirname(name: str) -> str:
    """Make a free-text player name safe to use as a destination folder
    name (Windows/POSIX invalid characters replaced, trailing dots/spaces
    trimmed — Windows rejects both)."""
    cleaned = _INVALID_FS_CHARS.sub("_", name).strip(" .")
    return cleaned or "Unnamed"


def _is_pending_cluster(folder_name: str) -> bool:
    return folder_name.isdigit()


def _load_face_json(cluster_dir: Path) -> dict:
    fp = cluster_dir / "face.json"
    if not fp.is_file():
        return {"name": "", "playernum": None, "cluster": cluster_dir.name,
                "provider": None, "aligned": None, "faces": []}
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def _write_face_json(cluster_dir: Path, payload: dict) -> None:
    with open(cluster_dir / "face.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _build_clusters(album_path: Path) -> list[dict]:
    fr = _facereco_dir(album_path)
    if fr is None:
        return []

    pending: list[dict] = []
    matched: list[dict] = []
    for child in sorted(fr.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        payload = _load_face_json(child)
        faces = []
        for face in payload.get("faces", []):
            crop = face.get("cropFileName")
            if not crop:
                continue
            faces.append({
                "crop": crop,
                "origFilename": face.get("origFilename", ""),
            })
        cluster = {
            "id": child.name,
            "pending": _is_pending_cluster(child.name),
            "name": payload.get("name") or "",
            "playernum": payload.get("playernum"),
            "faces": faces,
        }
        (pending if cluster["pending"] else matched).append(cluster)

    # Pending (numeric) clusters first, then matched (named) clusters.
    return pending + matched


def _collect_names(album_path: Path) -> list[str]:
    """Collect player names from each cluster's face.json (this album's own
    .FaceReco plus the shared repo-root face DB used for autocomplete)."""
    names: set[str] = set()
    for source in (_global_face_db_dir(), _facereco_dir(album_path)):
        if source is None or not source.is_dir():
            continue
        for child in source.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            name = (_load_face_json(child).get("name") or "").strip()
            if name:
                names.add(name)
    return sorted(names, key=str.casefold)


class ClusterOperation(BaseModel):
    type: str            # "assign" | "delete"
    sourceCluster: str
    crop: str
    name: Optional[str] = None


class ClusterCommitRequest(BaseModel):
    operations: list[ClusterOperation]


def _next_numeric_id(fr: Path) -> str:
    highest = 0
    for child in fr.iterdir():
        if child.is_dir() and child.name.isdigit():
            highest = max(highest, int(child.name))
    return f"{highest + 1:04d}"


def _ensure_target_cluster(fr: Path, name: str, template: dict, new_id: str) -> Path:
    """Return the target cluster dir for ``name`` inside ``fr``, creating it
    (folder + skeleton face.json) if it does not yet exist."""
    target = fr / name
    if not target.exists():
        (target / FACE_SUBDIR).mkdir(parents=True, exist_ok=True)
        (target / "Negative").mkdir(parents=True, exist_ok=True)
        _write_face_json(target, {
            "name": name,
            "playernum": None,
            "cluster": new_id,
            "provider": template.get("provider"),
            "aligned": template.get("aligned"),
            "faces": [],
        })
    else:
        (target / FACE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return target


def _unique_crop_name(face_dir: Path, crop: str) -> str:
    """Avoid clobbering an existing crop of the same name in the target."""
    if not (face_dir / crop).exists():
        return crop
    stem = Path(crop).stem
    suffix = Path(crop).suffix
    i = 1
    while (face_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return f"{stem}_{i}{suffix}"


def _move_crop_files(src_cluster: Path, dst_cluster: Path, crop: str) -> str:
    """Move the crop PNG (and its Face.annotated twin if present). Returns the
    possibly-renamed crop filename in the destination."""
    dst_face = dst_cluster / FACE_SUBDIR
    dst_face.mkdir(parents=True, exist_ok=True)
    new_name = _unique_crop_name(dst_face, crop)

    src_png = src_cluster / FACE_SUBDIR / crop
    if src_png.is_file():
        shutil.move(str(src_png), str(dst_face / new_name))

    src_anno = src_cluster / FACE_ANNOTATED_SUBDIR / crop
    if src_anno.is_file():
        dst_anno = dst_cluster / FACE_ANNOTATED_SUBDIR
        dst_anno.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_anno), str(dst_anno / new_name))
    return new_name


def _delete_crop_files(cluster: Path, crop: str) -> None:
    for sub in (FACE_SUBDIR, FACE_ANNOTATED_SUBDIR):
        fp = cluster / sub / crop
        if fp.is_file():
            fp.unlink()


def _pop_face_entry(payload: dict, crop: str) -> Optional[dict]:
    faces = payload.get("faces", [])
    for i, face in enumerate(faces):
        if face.get("cropFileName") == crop:
            return faces.pop(i)
    return None


def _boxes_match(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    return all(
        abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) <= BBOX_EPS
        for k in ("x1", "y1", "x2", "y2")
    )


def _update_results_json(album_path: Path, assignments: list[dict]) -> None:
    """Write player_name / player_number onto matching album.json bodies.

    ``assignments`` is a list of {origFilename, body_bbox, name, playernum}.
    """
    if not assignments:
        return
    results_fp = album_path / "album.json"
    if not results_fp.is_file():
        return
    with open(results_fp, encoding="utf-8") as fh:
        data = json.load(fh)

    # Index results entries by basename for quick lookup.
    by_name: dict[str, list[dict]] = {}
    for entry in data.get("results", []):
        base = Path(entry.get("file", "")).name
        by_name.setdefault(base, []).append(entry)

    changed = False
    for a in assignments:
        for entry in by_name.get(a["origFilename"], []):
            ann = entry.get("annotation_data")
            if not ann:
                continue
            for body in ann.get("evaluated", []):
                if _boxes_match(body.get("body_bbox"), a["body_bbox"]):
                    body["player_name"] = a["name"]
                    body["player_number"] = a["playernum"]
                    changed = True

    if changed:
        atomic_save_and_backup(json.dumps(data, indent=2), results_fp)


def _commit_cluster_operations(album_path: Path, req: ClusterCommitRequest) -> dict:
    fr = _facereco_dir(album_path)
    if fr is None:
        raise HTTPException(status_code=400, detail="Album has no .FaceReco folder")

    # Cache of cluster payloads we mutate, keyed by absolute path.
    payloads: dict[Path, dict] = {}

    def payload_for(cluster_dir: Path) -> dict:
        if cluster_dir not in payloads:
            payloads[cluster_dir] = _load_face_json(cluster_dir)
        return payloads[cluster_dir]

    # For fresh named clusters created in this batch, remember their dir so a
    # second assignment to the same name reuses it.
    target_dirs: dict[str, Path] = {}
    results_updates: list[dict] = []

    # Process deletions first, then assignments.
    ops = sorted(req.operations, key=lambda o: 0 if o.type == "delete" else 1)
    deleted_keys: set[tuple[str, str]] = set()

    for op in ops:
        src_name = _safe_component(op.sourceCluster)
        crop = _safe_component(op.crop)
        src_dir = fr / src_name
        if not src_dir.is_dir():
            continue
        src_payload = payload_for(src_dir)

        if op.type == "delete":
            _pop_face_entry(src_payload, crop)
            _delete_crop_files(src_dir, crop)
            deleted_keys.add((src_name, crop))
            continue

        if op.type == "assign":
            if (src_name, crop) in deleted_keys:
                continue  # delete wins
            name = (op.name or "").strip()
            if not name or name == src_name:
                continue  # nothing to do / already in this named cluster

            if name in target_dirs:
                dst_dir = target_dirs[name]
            else:
                dst_dir = _ensure_target_cluster(
                    fr, name, src_payload, _next_numeric_id(fr)
                )
                target_dirs[name] = dst_dir
            dst_payload = payload_for(dst_dir)

            entry = _pop_face_entry(src_payload, crop)
            new_crop = _move_crop_files(src_dir, dst_dir, crop)
            if entry is not None:
                entry["cropFileName"] = new_crop
                dst_payload.setdefault("faces", []).append(entry)
                results_updates.append({
                    "origFilename": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                    "name": name,
                    "playernum": dst_payload.get("playernum"),
                })

    # Rewrite every touched face.json.
    for cluster_dir, payload in payloads.items():
        if cluster_dir.is_dir():
            _write_face_json(cluster_dir, payload)

    # Dissolve emptied pending (numeric) clusters.
    for cluster_dir, payload in payloads.items():
        if (
            cluster_dir.is_dir()
            and _is_pending_cluster(cluster_dir.name)
            and not payload.get("faces")
        ):
            shutil.rmtree(cluster_dir, ignore_errors=True)

    _update_results_json(album_path, results_updates)

    return {"ok": True, "clusters": _build_clusters(album_path)}


# --------------------------------------------------------------------------- #
# Export (page 5 — Apply). Three steps, run in a background thread while the
# browser polls /api/apply/export-status the same way it polls processing
# output:
#
#   1. Sync every recognized/tagged player's face crops (from kept images
#      only) into the permanent, cross-album system face database at
#      REPO_ROOT/.FaceReco — uncapped; RebuildFaceDB.py's retire step (below)
#      is what curates it, not this step.
#   2. Run RebuildFaceDB.py against that system database to regenerate
#      embeddings and retire redundant crops.
#   3. Run apply_export.py to copy every kept photo (with edits + metadata
#      preserved) plus players.csv into the user-chosen destination — a flat
#      folder, no subdirectories. albums/<album>/ (and its own .FaceReco) is
#      the persistent "working album"; the destination is just the final
#      deliverable.
# --------------------------------------------------------------------------- #
def _export_log(msg: str) -> None:
    with export_state.lock:
        export_state.lines.append(msg)
    log.info("[Export] %s", msg)


def _stream_subprocess(cmd: list[str], on_line=None) -> int:
    """Run *cmd*, forwarding each combined stdout/stderr line to _export_log
    (and optionally *on_line* for progress-counter bookkeeping). Returns the
    process's exit code."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        _export_log(line)
        if on_line:
            on_line(line)
    proc.wait()
    return proc.returncode


def _sync_named_clusters_to_system_db(album_path: Path, kept: set[str]) -> int:
    """Copy every recognized/tagged player's face crops from *kept* images in
    this album's own .FaceReco into REPO_ROOT/.FaceReco (creating a person
    folder there the first time we see them). Copies everything uncapped —
    RebuildFaceDB.py's retire step decides afterward what's actually worth
    keeping. Returns the number of tagged players processed."""
    fr = _facereco_dir(album_path)
    if fr is None:
        _export_log("No .FaceReco folder in this album — nothing to sync.")
        return 0

    named_clusters = [
        child for child in sorted(fr.iterdir())
        if child.is_dir() and not child.name.startswith(".") and not _is_pending_cluster(child.name)
    ]
    if not named_clusters:
        _export_log("No tagged players found in this album — nothing to sync.")
        return 0

    system_fr = REPO_ROOT / ".FaceReco"
    system_fr.mkdir(parents=True, exist_ok=True)

    processed = 0
    for cluster_dir in named_clusters:
        payload = _load_face_json(cluster_dir)
        name = (payload.get("name") or cluster_dir.name).strip() or cluster_dir.name

        candidates = [
            f for f in payload.get("faces", [])
            if f.get("cropFileName") and Path(f.get("origFilename", "")).name in kept
        ]
        with export_state.lock:
            export_state.processed_players = processed + 1
            export_state.total_players = len(named_clusters)
        if not candidates:
            _export_log(f"  {name}: no kept faces to sync.")
            processed += 1
            continue

        dest_dir = system_fr / _sanitize_player_dirname(name)
        dest_face_dir = dest_dir / FACE_SUBDIR
        dest_face_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "Negative").mkdir(parents=True, exist_ok=True)

        if not (dest_dir / "face.json").is_file():
            _write_face_json(dest_dir, {
                "name": name,
                "playernum": payload.get("playernum"),
                "provider": payload.get("provider"),
                "cluster": cluster_dir.name,
                "faces": [],
                "negative_faces": [],
            })

        copied = 0
        skipped = 0
        for f in candidates:
            src_crop = cluster_dir / FACE_SUBDIR / f["cropFileName"]
            if not src_crop.is_file():
                continue
            dest_crop = dest_face_dir / f["cropFileName"]
            if dest_crop.is_file():
                # Already synced in a previous export run — skip instead of
                # copying a renamed duplicate (crop file names are
                # deterministic per source photo/body, so an existing file
                # of the same name means this exact face is already there).
                skipped += 1
                continue
            shutil.copy2(str(src_crop), str(dest_crop))
            src_anno = cluster_dir / FACE_ANNOTATED_SUBDIR / f["cropFileName"]
            if src_anno.is_file():
                dest_anno_dir = dest_dir / FACE_ANNOTATED_SUBDIR
                dest_anno_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_anno), str(dest_anno_dir / f["cropFileName"]))
            copied += 1
        _export_log(
            f"  {name}: {copied} new face crop(s) synced, {skipped} already present "
            f"(of {len(candidates)} kept face crop(s))."
        )
        processed += 1

    _export_log(f"Face-DB sync complete: {processed}/{len(named_clusters)} player(s) processed.")
    return processed


def _open_in_file_explorer(path: Path) -> None:
    """Open *path* in the OS's native file explorer. Only meaningful when the
    server and browser run on the same machine (see api_browse_folder's
    docstring for the same assumption)."""
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 — local-machine convenience only
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _run_export(album_path: Path, dest_dir: Path, export_face_tagging: bool) -> None:
    try:
        kept = _kept_image_basenames(album_path)
        with export_state.lock:
            export_state.total_images = len(kept)
            export_state.dest_dir = str(dest_dir)

        _export_log("Step 1/3: syncing tagged face crops into system face database (./FaceReco)...")
        _sync_named_clusters_to_system_db(album_path, kept)

        system_fr = REPO_ROOT / ".FaceReco"
        if system_fr.is_dir() and any(system_fr.iterdir()):
            _export_log("Step 2/3: rebuilding face database embeddings (RebuildFaceDB.py)...")
            rc = _stream_subprocess([
                sys.executable, str(REPO_ROOT / "RebuildFaceDB.py"), str(system_fr), "--skip-calibration",
            ])
            if rc != 0:
                raise RuntimeError(f"RebuildFaceDB.py failed with exit code {rc}")
        else:
            _export_log("Step 2/3: system face database is empty — skipping rebuild.")

        _export_log(f"Step 3/3: exporting {len(kept)} kept photo(s) + players.csv to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)

        def _bump_copied(line: str) -> None:
            # apply_export.py logs "  Copied: <name>" but the subprocess's
            # own logging.Formatter prepends "<time> [LEVEL   ] " to every
            # line, so match the message as a substring rather than a
            # prefix (startswith would never match and copied_images would
            # stay stuck at 0 despite the export succeeding).
            if "  Copied: " in line:
                with export_state.lock:
                    export_state.copied_images += 1

        cmd = [sys.executable, str(REPO_ROOT / "apply_export.py"), str(album_path), str(dest_dir)]
        if export_face_tagging:
            cmd.append("--export-face-tagging")
        rc = _stream_subprocess(cmd, on_line=_bump_copied)
        if rc != 0:
            raise RuntimeError(f"apply_export.py failed with exit code {rc}")

        _export_log("Export complete.")
        with export_state.lock:
            export_state.done = True
        state = _load_state()
        state["lastExportPath"] = str(dest_dir.resolve())
        _save_state(state)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[Export] failed")
        with export_state.lock:
            export_state.error = str(exc)
    finally:
        with export_state.lock:
            export_state.running = False


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Photo Culling UI")
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))


def _static_url(path: str) -> str:
    """Build a `/static/...` URL with a cache-busting `?v=<mtime>` query
    param, so browsers pick up edited JS/CSS immediately instead of serving
    a stale cached copy after the file changes on disk."""
    fp = WEBUI_DIR / "static" / path
    try:
        version = int(fp.stat().st_mtime)
    except OSError:
        version = 0
    return f"/static/{path}?v={version}"


templates.env.globals["static_url"] = _static_url

PAGE_STEPS: tuple[str, ...] = ("team", "import", "review", "cluster", "apply")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/import")


@app.post("/api/heartbeat")
def api_heartbeat() -> dict:
    with heartbeat_state.lock:
        heartbeat_state.last_heartbeat = time.time()
    return {"ok": True}


def _render_page(step: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, f"{step}.html", {"active_step": step})


for _step in PAGE_STEPS:
    def _make_route(step: str):
        def _route(request: Request) -> HTMLResponse:
            return _render_page(step, request)
        return _route
    app.add_api_route(f"/{_step}", _make_route(_step), methods=["GET"], response_class=HTMLResponse)


@app.get("/add-album")
def add_album_page(request: Request) -> HTMLResponse:
    # Sub-page of "Select Album" (reached via the "+" polaroid card there),
    # not a numbered step of its own — highlight the "Select Album" step.
    return templates.TemplateResponse(request, "add_album.html", {"active_step": "import"})


@app.post("/api/team")
def api_save_team(team: TeamData) -> dict:
    saved = _save_team(team)
    return {"ok": True, "team": saved.model_dump()}


@app.get("/api/teams/query")
def api_query_teams() -> dict:
    """QueryTeams — returns settings (jersey colours, roster, API key) for
    every team stored in team.json, plus the shared Global settings block."""
    teams_file = _load_teams_file()
    return {
        "teams": [team.model_dump() for team in teams_file.Teams],
        "global": teams_file.Global.model_dump(),
    }


@app.get("/api/sensitivity-presets")
def api_sensitivity_presets() -> dict:
    return SENSITIVITY_PRESETS


@app.get("/api/albums")
def api_list_albums() -> dict:
    return {"albums": _list_albums()}


@app.post("/api/albums/select")
def api_select_album(req: AlbumSelectRequest) -> dict:
    path = _album_dir_by_id(req.id)
    if not _is_album_complete(path):
        raise HTTPException(status_code=400, detail="Not a fully processed album.")
    state = _load_state()
    state["currentAlbum"] = str(path.resolve())
    _save_state(state)
    log.info("Current album set: %s", path.resolve())
    return {"ok": True, "album": _read_album_summary(path)}


@app.get("/api/albums/thumb")
def api_album_thumb(id: str = Query(...), file: str = Query(...)) -> FileResponse:
    album_path = _album_dir_by_id(id)
    fp = album_path / _PREVIEWS_SUBDIR / _safe_component(file)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fp)


@app.post("/api/albums/delete")
def api_delete_album(req: AlbumDeleteRequest) -> dict:
    album_path = _album_dir_by_id(req.id)
    shutil.rmtree(album_path)
    state = _load_state()
    if state.get("currentAlbum") == str(album_path.resolve()):
        state["currentAlbum"] = ""
        _save_state(state)
    log.info("Album deleted: %s", album_path.resolve())
    return {"ok": True}


@app.get("/api/current-album")
def api_current_album() -> dict:
    state = _load_state()
    album_path = state.get("currentAlbum", "")
    if not album_path or not _is_album_complete(Path(album_path)):
        return {"album": None}
    return {"album": _read_album_summary(Path(album_path))}


@app.get("/api/review/summary")
def api_review_summary() -> dict:
    album_path = _current_album_path()
    with open(album_path / "info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    return {
        "blurCount": len(info.get("Anno_Blur", [])),
        "sharpCount": len(info.get("Anno_Sharp", [])),
        "skippedCount": len(info.get("Anno_Skipped", [])),
    }


@app.get("/api/review/data")
def api_review_data(category: str = Query(...), sort: str = Query("size")) -> dict:
    if category not in REVIEW_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    album_path = _current_album_path()
    groups = _sort_groups(_group_bursts(_review_images(album_path, category)), sort)
    return {
        "category": category,
        "groups": [
            {"images": [
                {"file": im["file"], "anno": im["anno"], "keep": im["keep"], "burstRanking": im["burstRanking"], "llmGrade": im["llmGrade"]}
                for im in group
            ]}
            for group in groups
        ],
    }


@app.get("/api/review/thumb")
def api_review_thumb(category: str = Query(...), file: str = Query(...)) -> FileResponse:
    if category not in REVIEW_CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    album_path = _current_album_path()
    fp = album_path / _PREVIEWS_SUBDIR / _safe_component(file)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fp)


@app.post("/api/review/apply")
def api_review_apply(req: ReviewApplyRequest) -> dict:
    """Commit pending keep/drop decisions (staged client-side, across all 3
    review tabs at once) into album.json. Every reviewable entry gets an
    explicit "keep" field written — the user's override if they touched it
    this session, else its current effective (explicit-or-default) value —
    so album.json becomes fully self-describing going forward. Written
    atomically (temp file + os.replace, with the previous contents backed
    up) so a crash mid-write can never corrupt album.json."""
    album_path = _current_album_path()
    with open(album_path / "info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    payload = _load_results_payload(album_path)
    results_by_name = _results_by_filename(payload)

    for category in REVIEW_CATEGORIES:
        default_keep = _REVIEW_DEFAULT_KEEP[category]
        for item in info.get(_REVIEW_INFO_KEY[category], []):
            src_name = item.get("src")
            result = results_by_name.get(src_name)
            if result is None:
                continue
            current = bool(result.get("keep", default_keep))
            result["keep"] = bool(req.overrides.get(src_name, current))

    atomic_save_and_backup(json.dumps(payload, indent=2), album_path / "album.json")
    log.info("Review changes applied: %s (%d overrides)", album_path.resolve(), len(req.overrides))
    return {"ok": True}


@app.get("/api/import-state")
def api_get_import_state() -> dict:
    state = _load_state()
    last_path = state.get("lastImportPath", "")
    return {
        "lastImportPath": last_path,
        "lastImportPathExists": bool(last_path) and Path(last_path).is_dir(),
        "sensitivityMode": state.get("sensitivityMode", "medium"),
        "sensitivityCustomValue": state.get("sensitivityCustomValue", 0.50),
        "recognizeFaces": state.get("recognizeFaces", True),
        "selectedTeamId": state.get("selectedTeamId", ""),
    }


@app.post("/api/import-path")
def api_set_import_path(req: ImportPathRequest) -> dict:
    path = Path(req.path)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")
    state = _load_state()
    state["lastImportPath"] = str(path.resolve())
    _save_state(state)
    log.info("Import path set: %s", path.resolve())
    return {"ok": True, "path": str(path.resolve())}


@app.post("/api/browse-folder")
def api_browse_folder(req: BrowseFolderRequest = BrowseFolderRequest()) -> dict:
    """Open a native OS folder-picker dialog on the server machine.

    Only meaningful when the server and browser run on the same machine
    (the normal case for this tool — see InitPhotoProcessing.bat / launcher
    pattern). Returns {"path": null} if the user cancels.
    """
    state = _load_state()
    if req.context == "export":
        last_export = state.get("lastExportPath", "")
        initial_dir = str(Path(last_export).parent) if last_export else ""
    else:
        last_import = state.get("lastImportPath", "")
        initial_dir = str(Path(last_import).parent) if last_import else ""
    if not initial_dir or not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    result: dict[str, Optional[str]] = {"path": None}

    def _run_dialog() -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial_dir, title=req.title)
        root.destroy()
        result["path"] = chosen or None

    # tkinter must run on the main thread on some platforms; FastAPI's sync
    # endpoints already execute in a worker thread, so calling it directly
    # here is safe (no separate thread needed) and keeps the dialog modal.
    _run_dialog()

    if result["path"]:
        log.info("Folder picked via dialog: %s", result["path"])
    else:
        log.info("Folder picker cancelled")
    return result


@app.post("/api/start-processing")
def api_start_processing(req: StartProcessingRequest) -> dict:
    path = Path(req.path)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")

    with processing_state.lock:
        if processing_state.running:
            raise HTTPException(status_code=409, detail="Processing is already running.")
        processing_state.running = True
        processing_state.return_code = None
        processing_state.lines = []

    # Remember these choices for next time, same pattern as lastImportPath.
    state = _load_state()
    state["sensitivityMode"] = req.sensitivityMode
    state["sensitivityCustomValue"] = req.sensitivityCustomValue
    state["recognizeFaces"] = req.recognizeFaces
    state["selectedTeamId"] = req.teamId
    _save_state(state)

    thread = threading.Thread(
        target=_run_processing,
        args=(str(path.resolve()), req.albumName, req.sensitivityMode, req.sensitivityCustomValue, req.recognizeFaces, req.teamId),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.get("/api/processing-output")
def api_processing_output(since: int = 0) -> dict:
    with processing_state.lock:
        new_lines = processing_state.lines[since:]
        return {
            "lines": new_lines,
            "next": since + len(new_lines),
            "running": processing_state.running,
            "returnCode": processing_state.return_code,
        }


@app.get("/api/cluster/data")
def api_cluster_data() -> dict:
    album_path = _current_album_path()
    return {
        "album": _read_album_summary(album_path),
        "names": _collect_names(album_path),
        "clusters": _build_clusters(album_path),
    }


@app.get("/api/cluster/thumb/{cluster}/{crop}")
def api_cluster_thumb(cluster: str, crop: str) -> FileResponse:
    album_path = _current_album_path()
    fr = _facereco_dir(album_path)
    if fr is None:
        raise HTTPException(status_code=404, detail="No .FaceReco")
    fp = fr / _safe_component(cluster) / FACE_SUBDIR / _safe_component(crop)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(fp)


@app.get("/api/cluster/original")
def api_cluster_original(file: str = Query(...)) -> FileResponse:
    album_path = _current_album_path()
    src_dir = _read_album_summary(album_path).get("srcDir")
    if not src_dir:
        raise HTTPException(status_code=404, detail="Source directory unknown")
    fp = Path(src_dir) / _safe_component(file)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Original not found")
    return FileResponse(fp)


@app.post("/api/cluster/commit")
def api_cluster_commit(req: ClusterCommitRequest) -> dict:
    album_path = _current_album_path()
    return _commit_cluster_operations(album_path, req)


@app.get("/api/apply/summary")
def api_apply_summary() -> dict:
    album_path = _current_album_path()
    with open(album_path / "info.json", encoding="utf-8") as fh:
        info = json.load(fh)
    total_images = (
        len(info.get("Anno_Sharp", [])) + len(info.get("Anno_Blur", [])) + len(info.get("Anno_Skipped", []))
    )
    kept = len(_kept_image_basenames(album_path))
    clusters = _build_clusters(album_path)
    return {
        "name": album_path.name,
        "imagesKept": kept,
        "imagesDropped": max(0, total_images - kept),
        "facesDetected": sum(len(c["faces"]) for c in clusters),
        "playersDetected": sum(1 for c in clusters if not c["pending"]),
    }


@app.post("/api/apply/export")
def api_start_export(req: ExportRequest) -> dict:
    album_path = _current_album_path()
    destination = req.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination folder is required.")
    dest_path = Path(destination)
    if dest_path.is_file():
        raise HTTPException(status_code=400, detail="Destination is a file, not a folder.")

    with export_state.lock:
        if export_state.running:
            raise HTTPException(status_code=409, detail="Export is already running.")
        export_state.reset()
        export_state.running = True

    thread = threading.Thread(
        target=_run_export,
        args=(album_path, dest_path, req.exportFaceTagging),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.get("/api/apply/export-status")
def api_export_status(since: int = 0) -> dict:
    with export_state.lock:
        new_lines = export_state.lines[since:]
        return {
            "lines": new_lines,
            "next": since + len(new_lines),
            "running": export_state.running,
            "done": export_state.done,
            "error": export_state.error,
            "totalImages": export_state.total_images,
            "copiedImages": export_state.copied_images,
            "totalPlayers": export_state.total_players,
            "processedPlayers": export_state.processed_players,
            "destDir": export_state.dest_dir,
        }


@app.post("/api/apply/open-destination")
def api_open_export_destination() -> dict:
    """Open the just-exported destination folder in the OS's native file
    explorer (Windows Explorer / macOS Finder / Linux file manager)."""
    with export_state.lock:
        dest_dir = export_state.dest_dir
    if not dest_dir or not Path(dest_dir).is_dir():
        raise HTTPException(status_code=400, detail="Destination folder not found.")
    _open_in_file_explorer(Path(dest_dir))
    return {"ok": True}


if (WEBUI_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEBUI_DIR / "static"), name="static")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Photo culling & organization web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    parser.add_argument("--heartbeat-timeout", type=float, default=180.0,
                        help="Seconds without a browser heartbeat before the "
                             "server exits automatically. Use 0 to disable.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    heartbeat_state.timeout = args.heartbeat_timeout
    heartbeat_state.last_heartbeat = time.time()

    url = f"http://{args.host}:{args.port}/"
    log.info("team.json:  %s", TEAM_JSON_PATH)
    log.info("state file: %s", STATE_JSON_PATH)
    log.info("Open %s in your browser", url)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if heartbeat_state.timeout > 0:
        log.info("Heartbeat watchdog: exit after %.0fs without a browser ping", heartbeat_state.timeout)
        threading.Thread(target=_heartbeat_watchdog, daemon=True).start()
    else:
        log.info("Heartbeat watchdog disabled")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
