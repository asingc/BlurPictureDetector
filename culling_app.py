"""End-to-end photo culling & organization web UI.

Launches a local FastAPI server backing a jQuery/jQuery UI multi-page app —
each step of the workflow is its own template + own JS file (sharing one
Jinja2 base layout for the topbar/stepper look-and-feel), giving each page
natural code isolation instead of one giant single-page app:

  1. Team setup       — jersey colours, player roster, OpenAI key, sensitivity.
  2. Import           — choose a source photo folder (remembers last choice)
                         and run the detection pipeline.
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
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

log = logging.getLogger("CullingApp")

REPO_ROOT = Path(__file__).resolve().parent
WEBUI_DIR = REPO_ROOT / "webui" / "culling"

TEAM_JSON_PATH = REPO_ROOT / "team.json"
STATE_JSON_PATH = REPO_ROOT / ".culling_state.json"

SENSITIVITY_PRESETS: dict[str, float] = {"low": 0.35, "medium": 0.50, "high": 0.70}


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class JerseyColor(BaseModel):
    color: str
    forced: bool = False


class Player(BaseModel):
    name: str = ""
    number: str = ""


class Sensitivity(BaseModel):
    mode: str = "medium"          # "low" | "medium" | "high" | "custom"
    customValue: float = 0.50     # used only when mode == "custom", 0-0.99


class TeamData(BaseModel):
    jerseyColors: list[JerseyColor] = Field(default_factory=list)
    players: list[Player] = Field(default_factory=list)
    openaiApiKey: str = ""
    sensitivity: Sensitivity = Field(default_factory=Sensitivity)


class ImportPathRequest(BaseModel):
    path: str


# --------------------------------------------------------------------------- #
# team.json persistence
# --------------------------------------------------------------------------- #
def _default_team() -> TeamData:
    return TeamData()


def _load_team() -> TeamData:
    if not TEAM_JSON_PATH.is_file():
        return _default_team()
    try:
        with open(TEAM_JSON_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        return TeamData.model_validate(payload)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("Failed to load team.json (%s) — using defaults", exc)
        return _default_team()


def _save_team(team: TeamData) -> None:
    with open(TEAM_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(team.model_dump(), fh, indent=2)
    log.info("team.json saved (%d player(s), %d jersey colour(s))",
              len(team.players), len(team.jerseyColors))


# --------------------------------------------------------------------------- #
# App state persistence (last import path, etc.) — kept separate from
# team.json since it isn't "team" data.
# --------------------------------------------------------------------------- #
def _load_state() -> dict:
    if not STATE_JSON_PATH.is_file():
        return {"lastImportPath": ""}
    try:
        with open(STATE_JSON_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"lastImportPath": ""}


def _save_state(state: dict) -> None:
    with open(STATE_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Photo Culling UI")
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))

PAGE_STEPS: tuple[str, ...] = ("team", "import", "review", "cluster", "apply")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/team")


def _render_page(step: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, f"{step}.html", {"active_step": step})


for _step in PAGE_STEPS:
    def _make_route(step: str):
        def _route(request: Request) -> HTMLResponse:
            return _render_page(step, request)
        return _route
    app.add_api_route(f"/{_step}", _make_route(_step), methods=["GET"], response_class=HTMLResponse)


@app.get("/api/team")
def api_get_team() -> dict:
    return _load_team().model_dump()


@app.post("/api/team")
def api_save_team(team: TeamData) -> dict:
    _save_team(team)
    return {"ok": True}


@app.get("/api/sensitivity-presets")
def api_sensitivity_presets() -> dict:
    return SENSITIVITY_PRESETS


@app.get("/api/import-state")
def api_get_import_state() -> dict:
    state = _load_state()
    last_path = state.get("lastImportPath", "")
    return {
        "lastImportPath": last_path,
        "lastImportPathExists": bool(last_path) and Path(last_path).is_dir(),
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
def api_browse_folder() -> dict:
    """Open a native OS folder-picker dialog on the server machine.

    Only meaningful when the server and browser run on the same machine
    (the normal case for this tool — see InitPhotoProcessing.bat / launcher
    pattern). Returns {"path": null} if the user cancels.
    """
    state = _load_state()
    initial_dir = state.get("lastImportPath") or str(Path.home())
    if not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    result: dict[str, Optional[str]] = {"path": None}

    def _run_dialog() -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial_dir, title="Select photo folder to import")
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    url = f"http://{args.host}:{args.port}/"
    log.info("team.json:  %s", TEAM_JSON_PATH)
    log.info("state file: %s", STATE_JSON_PATH)
    log.info("Open %s in your browser", url)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
