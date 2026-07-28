"""Web UI for face-recognition cluster tagging (Iteration 1).

Launch a small local web app that lets a photographer review the face
clusters produced by ``1_prep_review.py`` (the ``.FaceReco/`` folder inside an
output album), assign player names to unmatched clusters, and delete bad face
crops.

Nothing is written to disk until the user presses **Save** in the browser.
Save then, in one batch:

  * relocates assigned face crops into the target person's cluster folder
    (creating it inside the album's own ``.FaceReco/`` if needed),
  * removes faces marked for deletion (crop PNG + ``Face.annotated`` PNG +
    ``face.json`` entry),
  * rewrites every affected ``face.json``,
  * deletes emptied numeric ("pending") clusters,
  * writes the assigned player name / number back into ``album.json`` for
    the matching body (matched by ``body_bbox``).

The ``--face-db`` folder is used **read-only**, purely as the source of the
name-autocomplete dictionary; it is never modified.

Usage::

    python face_tag_ui.py [--album PATH] [--face-db PATH]
                          [--output-dir ./albums] [--host 127.0.0.1]
                          [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from algo.facereco import record_manual_override
from algo.utils import load_album_source_index

log = logging.getLogger("FaceTagUI")

# Folder-name candidates for a ``.FaceReco`` DB, mirroring 1_prep_review.py.
FACE_DB_DIR_CANDIDATES: tuple[str, ...] = (".FaceReco", ".facereco", ".Facereco")

# Sub-folders inside a cluster directory (see algo/facereco.py).
FACE_SUBDIR = "Face"
FACE_ANNOTATED_SUBDIR = "Face.annotated"

# Cluster folders whose names start with "." are internal (.AllFaces, .debug).
# A cluster is "pending" (unmatched) when its folder name is purely numeric.

WEBUI_DIR = Path(__file__).resolve().parent / "webui"

# Matching tolerance for body_bbox floats when writing back to album.json.
BBOX_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Configuration resolved at launch and shared with the request handlers.
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(
        self,
        output_dir: Path,
        album: Path | None,
        face_db_dir: Path | None,
        heartbeat_timeout: float = 180.0,
    ) -> None:
        self.output_dir = output_dir
        self.album = album
        self.face_db_dir = face_db_dir
        # Heartbeat watchdog: the browser pings /api/heartbeat periodically;
        # if none arrives within heartbeat_timeout seconds, the server exits.
        # <= 0 disables the watchdog entirely.
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = time.time()
        self.heartbeat_lock = threading.Lock()


STATE: AppState  # populated in main()


def _heartbeat_watchdog() -> None:
    """Exit the process if no browser heartbeat arrives within the timeout.

    Runs as a daemon thread. The timer starts from server launch, so the
    browser has one full timeout window to load the page and send its first
    heartbeat before the watchdog can fire.
    """
    poll_interval = max(1.0, min(5.0, STATE.heartbeat_timeout / 3))
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
            with STATE.heartbeat_lock:
                STATE.last_heartbeat = now
            continue
        with STATE.heartbeat_lock:
            elapsed = now - STATE.last_heartbeat
        if elapsed > STATE.heartbeat_timeout:
            log.info(
                "No heartbeat received for %.0fs (timeout %.0fs) — shutting down.",
                elapsed, STATE.heartbeat_timeout,
            )
            os._exit(0)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _find_face_db_in_directory(directory: Path) -> Path | None:
    for name in FACE_DB_DIR_CANDIDATES:
        candidate = directory / name
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_face_db_dir(explicit: str | None, output_dir: Path) -> Path | None:
    """Resolve the read-only name dictionary, mirroring 1_prep_review.py.

    Order: explicit ``--face-db`` -> ``.FaceReco`` in CWD -> ``.FaceReco``
    walking up from the output directory.
    """
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.is_dir():
            log.info("Face DB resolved (explicit): %s", candidate)
            return candidate
        log.warning("--face-db directory not found: %s", candidate)
        return None

    found = _find_face_db_in_directory(Path.cwd())
    if found is not None:
        log.info("Face DB resolved (current directory): %s", found)
        return found

    current = output_dir.resolve()
    while True:
        found = _find_face_db_in_directory(current)
        if found is not None:
            log.info("Face DB resolved (output ancestry): %s", found)
            return found
        if current.parent == current:
            break
        current = current.parent
    return None


def _album_dir(album: str) -> Path:
    """Return the validated absolute path to an album directory.

    Rejects path traversal and, when ``--album`` was passed, restricts access
    to that single album.
    """
    if STATE.album is not None:
        if album != STATE.album.name:
            raise HTTPException(status_code=404, detail="Album not available")
        return STATE.album

    if not album or "/" in album or "\\" in album or album in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid album name")
    path = (STATE.output_dir / album).resolve()
    # Ensure the resolved path is still inside output_dir.
    if STATE.output_dir.resolve() not in path.parents and path != STATE.output_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid album path")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Album not found")
    return path


def _facereco_dir(album_path: Path) -> Path | None:
    return _find_face_db_in_directory(album_path)


def _safe_component(name: str) -> str:
    """Reject a path component containing separators or traversal."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid path component")
    return name


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


# --------------------------------------------------------------------------- #
# Read models for the album view
# --------------------------------------------------------------------------- #
def _read_src_dir(album_path: Path) -> str | None:
    info = album_path / "info.json"
    if not info.is_file():
        return None
    try:
        with open(info, encoding="utf-8") as fh:
            return json.load(fh).get("SrcDir")
    except (json.JSONDecodeError, OSError):
        return None


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
    """Collect player names from each cluster's ``face.json``.

    For every cluster directory under the face-DB and the album's own
    ``.FaceReco``, open ``face.json`` and read its ``name`` field, ignoring
    blanks (pending/numeric clusters and any folder without a real name).
    """
    names: set[str] = set()
    for source in (STATE.face_db_dir, _facereco_dir(album_path)):
        if source is None or not source.is_dir():
            continue
        for child in source.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            name = (_load_face_json(child).get("name") or "").strip()
            if name:
                names.add(name)
    return sorted(names, key=str.casefold)


# --------------------------------------------------------------------------- #
# Commit (Save) — apply all staged operations
# --------------------------------------------------------------------------- #
class Operation(BaseModel):
    type: str            # "assign" | "delete"
    sourceCluster: str
    crop: str
    name: str | None = None


class CommitRequest(BaseModel):
    operations: list[Operation]


def _next_numeric_id(fr: Path) -> str:
    highest = 0
    for child in fr.iterdir():
        if child.is_dir() and child.name.isdigit():
            highest = max(highest, int(child.name))
    return f"{highest + 1:04d}"


def _ensure_target_cluster(
    fr: Path,
    name: str,
    template: dict,
    new_id: str,
) -> Path:
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


def _pop_face_entry(payload: dict, crop: str) -> dict | None:
    faces = payload.get("faces", [])
    for i, face in enumerate(faces):
        if face.get("cropFileName") == crop:
            return faces.pop(i)
    return None


def _boxes_match(a: dict | None, b: dict | None) -> bool:
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

    # Index results entries by their disambiguated bookkeeping key (falling
    # back to basename for older albums written before "key" existed) --
    # plain basename indexing breaks once two source directories share a
    # filename (see algo/utils.py::make_unique_import_key).
    by_name: dict[str, list[dict]] = {}
    for entry in data.get("results", []):
        key = entry.get("key") or Path(entry.get("file", "")).name
        by_name.setdefault(key, []).append(entry)

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
        with open(results_fp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)


def _commit(album_path: Path, req: CommitRequest) -> dict:
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
            entry = _pop_face_entry(src_payload, crop)
            _delete_crop_files(src_dir, crop)
            deleted_keys.add((src_name, crop))
            if entry is not None:
                record_manual_override(fr, deleted={
                    "file": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                })
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
                record_manual_override(fr, assigned={
                    "file": entry.get("origFilename", ""),
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
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Face Tag UI")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEBUI_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/heartbeat")
def api_heartbeat() -> JSONResponse:
    with STATE.heartbeat_lock:
        STATE.last_heartbeat = time.time()
    return JSONResponse({"ok": True})


@app.get("/api/albums")
def api_albums() -> JSONResponse:
    if STATE.album is not None:
        return JSONResponse({"fixed": True, "albums": [STATE.album.name]})
    albums = []
    for child in sorted(STATE.output_dir.iterdir()):
        if child.is_dir() and _find_face_db_in_directory(child) is not None:
            albums.append(child.name)
    return JSONResponse({"fixed": False, "albums": albums})


@app.get("/api/albums/{album}")
def api_album(album: str) -> JSONResponse:
    album_path = _album_dir(album)
    return JSONResponse({
        "album": album_path.name,
        "srcDir": _read_src_dir(album_path),
        "names": _collect_names(album_path),
        "clusters": _build_clusters(album_path),
    })


@app.get("/thumb/{album}/{cluster}/{crop}")
def thumb(album: str, cluster: str, crop: str) -> FileResponse:
    album_path = _album_dir(album)
    fr = _facereco_dir(album_path)
    if fr is None:
        raise HTTPException(status_code=404, detail="No .FaceReco")
    fp = fr / _safe_component(cluster) / FACE_SUBDIR / _safe_component(crop)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(fp)


@app.get("/original/{album}")
def original(album: str, file: str = Query(...)) -> FileResponse:
    album_path = _album_dir(album)
    album_json = album_path / "album.json"
    if album_json.is_file():
        index = load_album_source_index(album_json)
        src_path = index.get(file)
        if src_path:
            fp = Path(src_path)
            if fp.is_file():
                return FileResponse(fp)
    # Fall back to the legacy single-SrcDir + basename join for albums
    # written before multi-source-directory import support existed.
    src_dir = _read_src_dir(album_path)
    if not src_dir:
        raise HTTPException(status_code=404, detail="Source directory unknown")
    fp = Path(src_dir) / _safe_component(file)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Original not found")
    return FileResponse(fp)


@app.post("/api/albums/{album}/commit")
def api_commit(album: str, req: CommitRequest) -> JSONResponse:
    album_path = _album_dir(album)
    return JSONResponse(_commit(album_path, req))


if (WEBUI_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEBUI_DIR / "static"), name="static")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Face-reco cluster tagging web UI")
    parser.add_argument("--album", type=str, default=None,
                        help="Restrict to a single album directory (path).")
    parser.add_argument("--face-db", type=str, default=None,
                        help="Face-DB folder used read-only for name autocomplete.")
    parser.add_argument("--output-dir", type=str, default="./albums",
                        help="Directory containing album folders (default: ./albums).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--heartbeat-timeout", type=float, default=180.0,
                        help="Seconds without a browser heartbeat before the "
                             "server exits automatically. Use 0 to disable.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = Path(args.output_dir).resolve()
    album = Path(args.album).resolve() if args.album else None
    if album is not None and not album.is_dir():
        parser.error(f"--album directory not found: {album}")

    global STATE
    STATE = AppState(
        output_dir=output_dir,
        album=album,
        face_db_dir=_resolve_face_db_dir(args.face_db, output_dir),
        heartbeat_timeout=args.heartbeat_timeout,
    )

    log.info("Output dir: %s", output_dir)
    if album is not None:
        log.info("Album (fixed): %s", album)
    log.info("Face DB (names): %s", STATE.face_db_dir)
    url = f"http://{args.host}:{args.port}/"
    log.info("Open %s in your browser", url)

    # Launch the default browser once the server is about to accept requests.
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if STATE.heartbeat_timeout > 0:
        log.info("Heartbeat watchdog: exit after %.0fs without a browser ping", STATE.heartbeat_timeout)
        threading.Thread(target=_heartbeat_watchdog, daemon=True).start()
    else:
        log.info("Heartbeat watchdog disabled")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
