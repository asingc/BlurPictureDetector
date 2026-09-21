#!/usr/bin/env python3
"""
BlurPictureDetector
-------------------
Detect blurry sport images by analysing the sharpness of the main subject.

Usage:
    python 1_prep_review.py <image_or_directory> [--sensitivity low|medium|high]

How it works:
    1. YOLOv8n detects persons in the image (model auto-downloaded on first run).
    2. The largest detected person is cropped out as the "main subject".
    3. Two classical sharpness metrics are computed on the greyscale crop:
       - Laplacian variance  (sensitive to fine detail / high-frequency content)
       - Tenengrad           (gradient-energy measure, robust to noise)
    4. The two metrics are combined into a single sharpness_score in [0, 1]
       where 1 = perfectly sharp and 0 = completely blurry.
    5. Images whose sharpness_score falls below the sensitivity threshold are
       flagged, given a 1-star rating, and logged in blurry.csv and blur.lst.
       Files are NOT moved — their paths are recorded instead.
    6. A copy of every processed image is saved to
       <image_dir>/annotated/ with the subject bounding box drawn (style via app_config).

If no person is detected in an image the file is left untouched and skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from algo.config import app_config
from algo import album
from algo.album import Album
from algo.album_info import AlbumInfo
from algo.frame import Frame
from algo.image_cache import ImageCache
from algo.results import baseline_stars, build_result_entries
from algo.stage import ProcessStage
from algo.stages.annotation import AnnotationStage
from algo.stages.auto_adjust import AutoAdjustStage
from algo.stages.face_reco import FaceRecoStage
from algo.stages.grading import GradingStage
from algo.stages.image_analysis import ImageAnalysisStage, MediaPipeImageAnalysisStage
from algo.stages.jersey_counting import JerseyCountingStage
from algo.stages.llm_culling import LLMCullingStage
from algo.llm.culling_provider import DEFAULT_OPENAI_MODEL
from algo.utils import make_unique_import_key

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# sharpness_score threshold per sensitivity level.
# A file is flagged as blurry when  sharpness_score <= threshold.
#   low    → only flag severely blurry images  (high tolerance)
#   medium → balanced default
#   high   → flag even slightly blurry images  (low tolerance)
# Recalibrated 2026-07-27 (low 0.35->0.16, medium 0.50->0.28, high 0.68->0.42)
# alongside the production face-crop size change to a fixed 96px long edge
# (clamp_long_edge / app_config.face_crop_{min,max}_long_edge_px in
# algo/scorers.py, replacing the old shrink-only ~4%-of-image-size cap) —
# normalizing crop size shifts the WeightedGeometricMeanEvaluator score scale,
# so each threshold was re-picked to preserve the RECALL the old
# variable/native crop size achieved at its old threshold — see
# culling_app.py's SENSITIVITY_PRESETS and
# _setup_tmp/sharpness_eval/calibrate_96px_thresholds.py.
#
# Recalibrated again 2026-08-31 (low 0.35->0.40, high 0.68->0.62; medium
# unchanged) alongside the swap to ContrastNormalizedEvaluator, which shifts
# the score scale slightly.  Each threshold was re-picked to preserve the
# recall the previous evaluator achieved at its old threshold.
SENSITIVITY_THRESHOLDS: dict[str, float] = {
    "low":    0.40,
    "medium": 0.50,
    "high":   0.62,
}

# Album folder names are "<timestamp>-<input stem>" (see _build_album_dir_name)
# — capped so a long source folder/file name can't blow past Windows' legacy
# 260-char MAX_PATH once nested album subpaths (previews/, .FaceReco/<cluster>/
# Face/<crop>.png, etc.) are appended on top.
ALBUM_DIR_NAME_MAX_LEN = 60

# COCO 17-keypoint skeleton: pairs of indices to connect with a line.
# Keypoint order: 0=nose 1=L-eye 2=R-eye 3=L-ear 4=R-ear
#   5=L-shoulder 6=R-shoulder 7=L-elbow 8=R-elbow 9=L-wrist 10=R-wrist
#   11=L-hip 12=R-hip 13=L-knee 14=R-knee 15=L-ankle 16=R-ankle
_COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),   # head
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12),                   # torso sides
    (11, 12),                           # hips
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
)



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("BlurPictureDetector")


def _build_album_dir_name(ts: str, stem: str) -> str:
    """Return "<ts>-<stem>", truncating *stem* so the whole name fits within
    ALBUM_DIR_NAME_MAX_LEN characters (the timestamp itself is never cut)."""
    prefix = f"{ts}-"
    max_stem_len = max(1, ALBUM_DIR_NAME_MAX_LEN - len(prefix))
    return prefix + stem[:max_stem_len].rstrip(" -_")


def _try_enable_windows_long_paths() -> None:
    """Best-effort, opt-in (only called when --enable-long-paths is passed):
    sets the machine-wide HKLM LongPathsEnabled=1 policy so paths beyond the
    legacy 260-char MAX_PATH work. No-ops on non-Windows. Requires admin —
    logs the manual command instead of failing if we don't have it, since we
    never attempt to self-elevate."""
    if os.name != "nt":
        return
    manual_cmd = (
        'New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
        "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
    )
    try:
        import winreg
        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
            try:
                current, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            except FileNotFoundError:
                current = 0
            if current == 1:
                log.debug("Windows long-path support already enabled.")
                return
            winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
            log.info("Enabled Windows long-path support (LongPathsEnabled=1).")
    except PermissionError:
        log.warning(
            "Could not enable Windows long-path support (needs an elevated/Administrator "
            "terminal). Run this once as Administrator, then re-run this script:\n  %s",
            manual_cmd,
        )
    except OSError as err:
        log.warning("Could not enable Windows long-path support: %s\n  Run manually: %s", err, manual_cmd)


def _setup_console_logging() -> None:
    """Configure a DEBUG-level console handler (called once at startup)."""
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(ch)


def _add_file_logging(log_path: Path) -> None:
    """Attach a DEBUG-level file handler once the output directory is known."""
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))
    log.addHandler(fh)
    log.debug("Log file opened: %s", log_path.resolve())


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

_FACE_MODEL_URL  = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
_FACE_MODEL_PATH = Path(__file__).parent / "yolov8n-face.pt"


def _ensure_face_model() -> Path:
    """Download yolov8n-face.pt next to this script if not already present."""
    if _FACE_MODEL_PATH.exists():
        log.debug("Face model already present: %s", _FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    log.info("Downloading yolov8n-face.pt from %s …", _FACE_MODEL_URL)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    log.info("Download complete: %s", _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


_TEAM_JSON_PATH = Path(__file__).resolve().parent / "team.json"


def _load_registered_team(team_id: str) -> dict | None:
    """Look up team_id's entry in team.json (the same file culling_app.py's
    Team Setup page reads/writes), returning its raw dict, or None if
    team_id doesn't match any registered team. Parsed directly (no pydantic
    dependency here, unlike culling_app.py) since this is a standalone CLI
    script."""
    try:
        with open(_TEAM_JSON_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    teams = payload.get("Teams") if isinstance(payload, dict) else None
    if not isinstance(teams, list):
        return None
    for team in teams:
        if isinstance(team, dict) and team.get("id") == team_id:
            return team
    return None


_FACE_DB_DIR_CANDIDATES: tuple[str, ...] = (".FaceReco", ".facereco", ".Facereco")


def _find_face_db_in_directory(directory: Path) -> Path | None:
    """Return the first face-DB directory under *directory*, if any."""
    for name in _FACE_DB_DIR_CANDIDATES:
        candidate = directory / name
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _walk_to_root(start: Path) -> list[Path]:
    """Return [start, parent, ..., root] for an existing path lineage."""
    chain: list[Path] = []
    current = start.resolve()
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    return chain


def _resolve_face_db_dir(
    explicit_path: str | None,
    output_dir: Path,
    input_path: Path,
) -> Path | None:
    """Resolve face DB path using ordered fallback search.

    Order:
      1) explicit --face-db path (authoritative)
      2) current working directory (.FaceReco/.facereco/.Facereco)
      3) walk upward from output directory
      4) walk upward from source directory (or parent for single file input)
    """
    if explicit_path:
        candidate = Path(explicit_path).resolve()
        if candidate.is_dir():
            log.info("Face DB resolved (explicit): %s", candidate)
            return candidate
        log.error("--face-db directory not found: %s", candidate)
        return None

    found = _find_face_db_in_directory(Path.cwd())
    if found is not None:
        log.info("Face DB resolved (current directory): %s", found)
        return found

    for directory in _walk_to_root(output_dir):
        found = _find_face_db_in_directory(directory)
        if found is not None:
            log.info("Face DB resolved (target ancestry): %s", found)
            return found

    source_start = input_path if input_path.is_dir() else input_path.parent
    for directory in _walk_to_root(source_start):
        found = _find_face_db_in_directory(directory)
        if found is not None:
            log.info("Face DB resolved (source ancestry): %s", found)
            return found

    return None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_MAX_BOXES = 8  # matches top-8 body selection in detect_qualified_persons

_CSV_FIELDS: list[str] = ["File", "Verdict", "Sharp Score", "# Boxes"] + [
    col
    for n in range(1, _MAX_BOXES + 1)
    for col in (
        f"Box {n} - Verdict",
        f"Box {n} - Orig Dimension",
        f"Box {n} - Face Box Sharp Score",
        f"Box {n} - Facial Boxes",
    )
]


def write_csv(frames: list[Frame], csv_path: Path, *, append: bool = False) -> None:
    write_header = not (append and csv_path.exists())
    with open(csv_path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for frame in frames:
            img_w = frame.img_w or 1
            img_h = frame.img_h or 1
            if not frame.bodies:
                verdict, score = "Skipped", None
            elif frame.is_sharp():
                verdict = "Sharp"
                score = max(b.sharpness_score for b in frame.bodies if b.passed)
            else:
                verdict = "Blur"
                score = max(b.sharpness_score for b in frame.bodies)
            row: dict = {
                "File":        frame.path.name,
                "Verdict":     verdict,
                "Sharp Score": f"{score:.2f}" if score is not None else "",
                "# Boxes":     len(frame.bodies),
            }
            for n, body in enumerate(frame.bodies, 1):
                b = body.bbox
                kp_confs = (
                    ", ".join(f"{lm.confidence:.2f}" for lm in body.best_face.landmarks)
                    if body.best_face else ""
                )
                row[f"Box {n} - Verdict"]              = "Sharp" if body.passed else "Blur"
                row[f"Box {n} - Orig Dimension"]       = f"{b.width * img_w:.0f} x {b.height * img_h:.0f}"
                row[f"Box {n} - Face Box Sharp Score"] = f"{body.sharpness_score:.2f}"
                row[f"Box {n} - Facial Boxes"]         = kp_confs
            writer.writerow(row)
    log.info("CSV report written to:    %s", csv_path)


def write_blur_lst(frames: list[Frame], lst_path: Path, *, append: bool = False) -> None:
    """Write a plain-text list of blurry image filenames and their blur scores."""
    with open(lst_path, "a" if append else "w", encoding="utf-8") as fh:
        for frame in frames:
            if frame.bodies and not frame.is_sharp():
                score = max(b.sharpness_score for b in frame.bodies)
                fh.write(f"{frame.path.name}\t{score}\n")
    log.info("Blur list written to:     %s", lst_path)


# Fields written onto an Album entry AFTER the analysis pipeline (by the
# Review page's Apply step, the LLM burst-culling stage, and the image-editing
# workflow). A deep regrade rebuilds every entry from freshly-computed
# detections, so these have to be carried over explicitly or the user's
# review/editing work would be wiped.
_REGRADE_PRESERVED_FIELDS = (
    "stars", "stars_manual", "keep", "burst_ranking", "llm_grade", "burst_caption",
    "edited_image",
)


def _merge_preserved_fields(
    new_entries: list[dict], old_entries: list[dict], threshold: float
) -> list[dict]:
    old_by_key = {
        (e.get("key") or Path(e.get("file", "")).name): e
        for e in old_entries if e.get("file")
    }
    for entry in new_entries:
        old = old_by_key.get(entry.get("key") or Path(entry.get("file", "")).name)
        if not old:
            continue
        for field_name in _REGRADE_PRESERVED_FIELDS:
            if field_name in old:
                entry[field_name] = old[field_name]
        # A photo that changed side of the keep line carries a star rating
        # that no longer reflects it, so reset to the baseline unless the
        # user rated it by hand (culling_app.py's "stars_manual" marker).
        if old.get("status") != entry.get("status") and not entry.get("stars_manual"):
            entry["stars"] = baseline_stars(
                entry.get("sharpness_score"), entry.get("status", "blurry"), threshold
            )
            entry["keep"] = entry["stars"] >= 3
    return new_entries


def write_info_json(
    frames: list[Frame],
    input_path: Path,
    timestamp: str,
    album_path: Path,
    our_jersey_color: str | None = None,
    *,
    rebuild: bool = False,
) -> None:
    """Record this run in the album's info.json (see algo/album_info.py).

    Each new frame is appended to its verdict's bucket, so importing more
    images into an existing album grows the lists rather than replacing
    them. *rebuild* replaces all three buckets instead -- for a deep
    regrade, which re-derives every photo's verdict from scratch.
    """
    by_category: dict[str, list[tuple[str, Path]]] = {"blur": [], "sharp": [], "skipped": []}
    for frame in frames:
        if not frame.bodies:
            category = "skipped"
        else:
            category = "sharp" if frame.is_sharp() else "blur"
        by_category[category].append((frame.output_key or frame.path.name, frame.path))

    info = AlbumInfo(album_path)
    info.record_import(input_path, timestamp)
    info.our_jersey_color = our_jersey_color
    if rebuild:
        info.replace_entries(by_category)
    else:
        for category, items in by_category.items():
            for key, source_path in items:
                info.add(category, key, source_path)
    info.save()
    log.info("Info JSON written to:     %s", info.info_json)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect blurry sport images by analysing subject sharpness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a single image file or a directory of images. Not needed with --rerun-facereco-only.",
    )
    def _sensitivity_type(value: str) -> str:
        if value in SENSITIVITY_THRESHOLDS:
            return value
        try:
            float(value)
            return value
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"sensitivity must be low/medium/high or a numeric threshold (0–1), got {value!r}"
            )

    parser.add_argument(
        "--sensitivity",
        type=_sensitivity_type,
        default="medium",
        metavar="low|medium|high|<threshold>",
        help=(
            "Detection sensitivity (default: medium).  "
            "Use low/medium/high, or supply a numeric threshold directly (0–1, "
            "e.g. 0.45).  Scores <= threshold are flagged as blurry."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Root directory for all output files (previews/, blurry.csv, "
            "blur.lst, run.log).  "
            "Defaults to albums/<timestamp>-<input_name>/."
        ),
    )
    parser.add_argument(
        "--jerseycolor",
        default="blue;white;+purple;+orange;+light blue;+pink",
        metavar="COLOR[;COLOR...]",
        help=(
            "Semicolon-separated list of jersey colours that qualify for "
            "evaluation and annotation (case-insensitive).  "
            "Each entry can be a plain colour (e.g. blue) or a full hue+shade "
            "label (e.g. navy).  "
            "Prefix a colour with '+' to make it forced-include — it will always "
            "be in the filter regardless of what other colours are listed "
            "(e.g. +light blue;+pink for goalies).  "
            "Pass an empty string to disable jersey filtering."
        ),
    )
    parser.add_argument(
        "--skip-facereco",
        action="store_true",
        help=(
            "Skip face recognition clustering. By default, .FaceReco/ is generated "
            "under the output directory after preview generation completes. "
            "Use this flag to disable it (e.g. if dlib is not installed)."
        ),
    )
    parser.add_argument(
        "--face-db",
        default=None,
        metavar="DIR",
        help=(
            "Path to a face-DB directory.  Each sub-directory must represent a "
            "person and contain a face.json with positive embeddings.  "
            "Matched clusters will be stored in a folder named after that person. "
            "If omitted, auto-discovery is used: current dir -> target dir ancestry -> "
            "source dir ancestry."
        ),
    )
    parser.add_argument(
        "--face-db-match-threshold",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Cosine similarity threshold for matching a cluster against the face DB "
            "(higher = stricter matching). Use RebuildFaceDB.py against your "
            "face DB to pick a data-driven value (it calibrates by default). "
            "If omitted, this script auto-loads the provider-specific "
            "recommended value from <face-db>/calibration.json when available."
        ),
    )
    parser.add_argument(
        "--face-db-match-margin",
        type=float,
        default=None,
        metavar="MARGIN",
        help=(
            "Minimum cosine-similarity gap required between the best-matching "
            "person and the best-matching DIFFERENT person. "
            "A face whose top-2 candidates are nearly tied is left unmatched "
            "instead of guessed -- this is what prevents similar-looking people "
            "from being mixed up. If omitted, this script auto-loads the "
            "provider-specific recommended margin from <face-db>/calibration.json "
            "when available."
        ),
    )
    parser.add_argument(
        "--face-db-prototype-threshold",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Cosine similarity used to split EACH PERSON's own positive "
            "embeddings into visually-cohesive prototypes when the face DB is "
            "loaded. If omitted, this script auto-loads the calibrated "
            "prototype threshold from <face-db>/calibration.json when available, "
            "else falls back to 0.62."
        ),
    )
    parser.add_argument(
        "--disable-face-db-calibration",
        action="store_true",
        help=(
            "Ignore <face-db>/calibration.json even when present.  Use only the "
            "explicit CLI thresholds (or hardcoded fallback defaults when omitted)."
        ),
    )
    parser.add_argument(
        "--min-face-crop-px",
        type=int,
        default=32,
        metavar="PX",
        help=(
            "Minimum short-edge size (pixels) of a face crop for its embedding "
            "to be trusted (default: 32). Smaller crops are skipped entirely."
        ),
    )
    parser.add_argument(
        "--debug-align",
        action="store_true",
        help=(
            "Write per-face alignment QA images (annotated crop + aligned face) "
            "to <output>/.FaceReco/.debug for visual inspection of landmark "
            "order and alignment quality."
        ),
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help=(
            "Force model inference onto CPU for benchmarking or environments "
            "without usable CUDA."
        ),
    )
    parser.add_argument(
        "--noteam",
        action="store_true",
        help=(
            "Disable jersey-colour filtering. When set, bodies are never "
            "disqualified for wearing the wrong colour — all detected persons "
            "are scored regardless of jersey colour. Overrides --jerseycolor."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help=(
            "Detection/pose/face-landmark engine (default: mediapipe). "
            "mediapipe is Apache-2.0 licensed; yolo is the legacy engine "
            "(AGPL-3.0/GPL-3.0 — see README licensing notes) kept for "
            "comparison/rollback."
        ),
    )
    parser.add_argument(
        "--autoadjust",
        action="store_true",
        help=(
            "Compute a simple auto-exposure (brightness) correction per image, "
            "shown in the annotated previews and stored on the Album. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--openaikey",
        default=None,
        metavar="KEY",
        help=(
            "OpenAI API key for LLM-assisted burst culling (falls back to the "
            "OPENAI_API_KEY environment variable). When a key is available "
            "(from either source), the LLM culling stage runs automatically "
            "after face recognition — use --skip-llm-cull to opt out."
        ),
    )
    parser.add_argument(
        "--team-id",
        default=None,
        metavar="ID",
        help="Id of the team (from team.json) this album was processed for. Stored on the Album.",
    )
    parser.add_argument(
        "--enable-long-paths",
        action="store_true",
        help=(
            "Windows only: best-effort opt-in to the OS-wide 'Enable Win32 long "
            "paths' policy (HKLM LongPathsEnabled=1), on top of the 60-char cap "
            "already applied to the album directory name, in case deeply nested "
            "album subpaths still exceed MAX_PATH. Requires an elevated/Administrator "
            "terminal to actually take effect; otherwise just logs the manual command."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_OPENAI_MODEL,
        metavar="MODEL",
        help=f"OpenAI model used for LLM-assisted burst culling (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--skip-llm-cull",
        action="store_true",
        help="Skip LLM-assisted burst culling even when an OpenAI API key is available.",
    )
    parser.add_argument(
        "--rerun-facereco-only",
        action="store_true",
        help=(
            "Skip image analysis/grading/annotation/LLM-culling entirely and "
            "just re-run face recognition clustering against an existing "
            "album's already-written Album (requires --output pointing "
            "at that album; no positional path needed). Re-run reclusters "
            "from scratch but replays manual_overrides.json on top, same as "
            "any other FaceRecoStage run."
        ),
    )
    parser.add_argument(
        "--team-color",
        default=None,
        metavar="HUE:SHADE",
        help=(
            "Pin the team's jersey colour to this 'Hue:Shade' label (e.g. "
            "'Blue:Navy') instead of polling the dominant colour from the "
            "photos. Pass an empty string to clear an album's existing pin "
            "and go back to auto-detection."
        ),
    )
    parser.add_argument(
        "--regrade-only",
        action="store_true",
        help=(
            "Deep regrade: re-run the FULL analysis pipeline (person/pose "
            "detection, face detection, sharpness scoring, jersey re-poll, "
            "preview regeneration) over an existing album's already-imported "
            "source photos at the --sensitivity threshold given, without "
            "importing anything new (requires --output pointing at that "
            "album; no positional path needed). Star ratings, keep flags and "
            "LLM burst-culling results are preserved; FaceReco clusters are "
            "left untouched."
        ),
    )
    args = parser.parse_args()
    if args.rerun_facereco_only or args.regrade_only:
        if not args.output:
            flag = "--rerun-facereco-only" if args.rerun_facereco_only else "--regrade-only"
            parser.error(f"{flag} requires --output <existing album directory>")
    elif not args.path:
        parser.error("the following arguments are required: path")

    _setup_console_logging()
    # This script rewrites album.json several times per run (analysis, then
    # FaceReco, then LLM culling). Backing up every intermediate version of a
    # 30 MB file would leave a pile of gzipped copies of states the user
    # never saw; the interactive app keeps its backups.
    album.album_json_backups_enabled = False
    if args.enable_long_paths:
        _try_enable_windows_long_paths()
    log.debug("Arguments: path=%s sensitivity=%s output=%s jerseycolor=%s skip_facereco=%s noteam=%s face_db=%s",
              args.path, args.sensitivity, args.output, args.jerseycolor, args.skip_facereco, args.noteam, args.face_db)
    if args.cpu_only:
        log.info("CPU-only mode enabled: forcing YOLO and FaceReco providers to CPU")

    input_path = Path(args.path).resolve() if args.path else None
    if input_path is not None and not input_path.exists():
        log.error("Path does not exist: %s", input_path)
        sys.exit(1)

    # --- Import-into-existing-album detection -----------------------------
    # Merge mode: --output points at a directory that already has an
    # Album (i.e. "import more images" into an existing album). Load
    # its prior state now so settings can be locked and new images
    # deduplicated before any expensive model inference runs.
    output_root = Path(args.output).resolve() if args.output else None
    merge_album = Album(output_root) if output_root is not None else None
    merge_mode = merge_album is not None and merge_album.exists

    existing_info = AlbumInfo(output_root) if output_root is not None else None
    existing_entries: list[dict] = []
    run_settings: dict = {}
    already_imported: frozenset[Path] = frozenset()
    used_keys: dict[str, Path] = {}

    if merge_mode:
        existing_entries = list(merge_album.results)
        run_settings = dict(merge_album.run_settings)

        already_imported = frozenset(
            Path(e["file"]).resolve() for e in existing_entries if e.get("file")
        )

        log.info(
            "Importing into existing album %s — %d image(s) already present",
            output_root, len(already_imported),
        )

        # Lock the settings that must stay consistent across every import
        # into this album (grading/filtering behaviour), so a later "import
        # more" run can't silently produce results inconsistent with the
        # ones already reviewed. Settings are only locked once the album has
        # actually recorded them (older albums without run_settings fall
        # back to whatever was passed on the CLI, same as before).
        if run_settings:
            def _lock(name: str, current):
                stored = run_settings.get(name)
                if stored is not None and str(stored) != str(current):
                    log.info(
                        "Import-more: locking --%s to the original album setting %r (ignoring %r)",
                        name, stored, current,
                    )
                return stored if stored is not None else current
            # A deep regrade exists precisely to change the sensitivity, so
            # that one setting is taken from the CLI instead of being locked.
            if not args.regrade_only:
                args.sensitivity = _lock("sensitivity", args.sensitivity)
            args.jerseycolor = _lock("jerseycolor", args.jerseycolor)
            args.engine = _lock("engine", args.engine)
            args.noteam = _lock("noteam", args.noteam)
            args.team_id = _lock("team_id", args.team_id)
            args.autoadjust = _lock("autoadjust", args.autoadjust)

    # Every album must be associated with exactly one registered team --
    # used for jersey-colour filtering (unless --noteam) and the OpenAI key
    # lookup. Prefer the album's own already-recorded team_id on merge
    # (covers albums written before run_settings existed, where
    # args.team_id was never locked above).
    team_id = merge_album.team_id if merge_mode and merge_album.team_id else args.team_id
    if not team_id:
        parser.error("--team-id is required (must match a team registered in team.json)")
    registered_team = _load_registered_team(team_id)
    if registered_team is None:
        parser.error(f"--team-id {team_id!r} does not match any team registered in team.json")
    # Roster names restrict face-DB matching to this team (see FaceRecoStage below).
    roster_names: frozenset[str] | None = frozenset(
        p.get("name", "").strip()
        for p in (registered_team.get("players") or [])
        if isinstance(p, dict) and p.get("name", "").strip()
    ) or None

    # Parse the semicolon-separated jersey colour list, normalise to title-case.
    # Colours prefixed with '+' are forced-include: they are always added to the
    # filter regardless of what other colours are listed (e.g. goalie colours).
    # --noteam overrides everything and disables all colour filtering.
    forced_colors: frozenset[str] = frozenset(
        c.strip().lstrip("+").strip().title()
        for c in (args.jerseycolor or "").split(";")
        if c.strip().startswith("+") and c.strip().lstrip("+").strip()
    )
    regular_colors: frozenset[str] = frozenset(
        c.strip().title()
        for c in (args.jerseycolor or "").split(";")
        if c.strip() and not c.strip().startswith("+")
    )
    jersey_colors: frozenset[str] = regular_colors | forced_colors
    if args.noteam:
        log.info("--noteam: jersey-colour filtering disabled")
    elif forced_colors:
        log.debug("Jersey colours — regular: %s  forced (+): %s",
                  sorted(regular_colors), sorted(forced_colors))
    else:
        log.debug("Jersey colours: %s", sorted(jersey_colors))

    try:
        sensitivity_threshold = float(args.sensitivity)
    except ValueError:
        sensitivity_threshold = SENSITIVITY_THRESHOLDS[args.sensitivity]

    # An explicit --team-color wins (empty string clears the pin); otherwise
    # inherit whatever the album already recorded, so a regrade for some
    # other reason doesn't silently revert a colour the user chose.
    team_color_override = (
        args.team_color if args.team_color is not None
        else run_settings.get("team_color_override")
    )
    team_color_override = (team_color_override or "").strip()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir  = output_root if output_root is not None else Path("albums") / _build_album_dir_name(ts, input_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    _add_file_logging(output_dir / "run.log")
    log.debug("Output directory: %s", output_dir.resolve())

    if args.rerun_facereco_only:
        if not merge_mode:
            log.error("--rerun-facereco-only requires an existing Album under --output: %s", output_dir)
            sys.exit(1)
        facereco_input_path = input_path or Path(
            (existing_info.src_dir if existing_info else "") or output_dir
        )
        face_db_dir = _resolve_face_db_dir(args.face_db, output_dir, facereco_input_path)
        if face_db_dir is None:
            log.warning("Re-running face detection without a face DB (no dictionary found) — clusters will be unnamed.")
        FaceRecoStage(
            output_dir,
            face_db_dir=face_db_dir,
            face_db_allowed_names=roster_names,
            face_db_match_threshold=args.face_db_match_threshold,
            face_db_match_margin=args.face_db_match_margin,
            face_db_prototype_threshold=args.face_db_prototype_threshold,
            use_face_db_calibration=not args.disable_face_db_calibration,
            min_face_crop_px=args.min_face_crop_px,
            debug_align=args.debug_align,
            cpu_only=args.cpu_only,
            engine=run_settings.get("engine", args.engine),
            sensitivity_threshold=sensitivity_threshold,
        ).process([], app_config)
        log.info("Face detection re-run complete.")
        return

    # Deep regrade: re-analyse the album's ALREADY-imported photos instead of
    # discovering new ones. Everything downstream (grading, jersey counting,
    # annotation, Album serialization) runs through the exact same
    # stages an import uses -- only the input file list and the final merge
    # differ.
    regrade_paths: list[Path] | None = None
    regrade_keys: dict[Path, str] = {}
    if args.regrade_only:
        if not merge_mode:
            log.error("--regrade-only requires an existing Album under --output: %s", output_dir)
            sys.exit(1)
        regrade_paths = []
        missing = 0
        for e in existing_entries:
            fp = e.get("file")
            if not fp:
                continue
            path = Path(fp)
            if not path.is_file():
                missing += 1
                continue
            regrade_paths.append(path)
            regrade_keys[path.resolve()] = e.get("key") or path.name
        log.info("Deep regrade: re-analysing %d image(s) at sensitivity=%s (threshold=%.2f)%s",
                 len(regrade_paths), args.sensitivity, sensitivity_threshold,
                 f" — {missing} source file(s) missing, skipped" if missing else "")
        if not regrade_paths:
            log.error("Deep regrade: none of this album's source images are reachable on disk.")
            sys.exit(1)

    # A normal "import more images" run stages the new photos in a temp
    # album under the target directory first (its own previews/.FaceReco/
    # results, with a clean local key namespace) -- Album.import_from folds
    # that staging album into the real target once the whole pipeline below
    # succeeds, resolving any bookkeeping-key collision with the target's
    # existing entries at that point. Deep regrade re-analyses the target's
    # OWN already-imported photos in place, so it never stages anything.
    use_temp_staging = merge_mode and not args.regrade_only
    working_dir = (output_dir / f".import_tmp_{ts}") if use_temp_staging else output_dir
    run_settings_out = run_settings or {
        "sensitivity": args.sensitivity,
        "jerseycolor": args.jerseycolor,
        "engine": args.engine,
        "noteam": args.noteam,
        "team_id": args.team_id,
        "autoadjust": args.autoadjust,
    }
    working_album = None if args.regrade_only else Album.create(
        working_dir, team_id=team_id, run_settings=run_settings_out, import_status="in_progress",
    )

    log.info("Loading models … (engine=%s)", args.engine)
    # Per-run scratch store for the pixel regions later stages need, so the
    # analysis loop can release each source image instead of holding the whole
    # album in memory (see algo/image_cache.py). Removed at the end of the run.
    image_cache = ImageCache(working_dir)
    cache_face_originals = not args.skip_facereco and not args.regrade_only
    # A deep regrade re-analyses an explicit file list and must NOT skip
    # already-imported paths (those ARE the files it exists to reprocess).
    analysis_skip = frozenset() if args.regrade_only else already_imported
    if args.engine == "yolo":
        pose_model = YOLO("yolov8n-pose.pt")
        face_model = YOLO(_ensure_face_model())
        if args.cpu_only:
            pose_model.to("cpu")
            face_model.to("cpu")
        analysis_stage: ProcessStage = ImageAnalysisStage(
            input_path, pose_model, face_model,
            skip_paths=analysis_skip, only_paths=regrade_paths,
            cache=image_cache, cache_face_originals=cache_face_originals,
        )
    else:
        from algo.mediapipe_provider import load_face_landmarker, load_pose_landmarker
        from algo.torchvision_provider import load_person_detector
        person_detector = load_person_detector(force_cpu=args.cpu_only)
        pose_landmarker = load_pose_landmarker(num_poses=1)
        face_landmarker = load_face_landmarker(num_faces=1)
        analysis_stage = MediaPipeImageAnalysisStage(
            input_path, person_detector, pose_landmarker, face_landmarker,
            skip_paths=analysis_skip, only_paths=regrade_paths,
            cache=image_cache, cache_face_originals=cache_face_originals,
        )

    # Not needed by a deep regrade (it returns before the FaceReco stage) and
    # input_path is allowed to be None in that mode. Resolved against
    # working_dir (not output_dir) so a temp staging album -- nested under
    # the target -- still finds the target's own .FaceReco via ancestry walk.
    face_db_dir = None if args.regrade_only else _resolve_face_db_dir(args.face_db, working_dir, input_path)

    # Run image analysis first (on its own) so each newly-discovered frame
    # can be assigned its disambiguated bookkeeping key -- see
    # algo/utils.py::make_unique_import_key -- before AnnotationStage (later
    # in the pipeline) uses that key to name preview files.
    frames: list[Frame] = analysis_stage.process([], app_config)
    for frame in frames:
        # A regraded frame keeps the key it was assigned at import time, so
        # its preview/FaceReco/review bookkeeping all stays addressable.
        frame.output_key = (
            regrade_keys.get(frame.path.resolve(), frame.path.name) if args.regrade_only
            else make_unique_import_key(frame.path.name, used_keys, frame.path.resolve())
        )

    jersey_stage = JerseyCountingStage(
        forced_colors, regular_colors, no_team=args.noteam,
        team_color_override=team_color_override,
    )
    stages: list[ProcessStage] = [
        GradingStage(sensitivity_threshold),
        jersey_stage,
    ]
    if args.autoadjust:
        stages.append(AutoAdjustStage())
    stages.append(AnnotationStage(working_dir))
    for stage in stages:
        frames = stage.process(frames, app_config)

    our_jersey_color = jersey_stage.our_color.label if jersey_stage.our_color else None

    if args.regrade_only:
        run_settings_out = dict(run_settings)
        run_settings_out["sensitivity"] = args.sensitivity
        run_settings_out["team_color_override"] = team_color_override
        # Preserved fields (stars, keep, edited_image, ...) are merged into
        # the freshly-built entries before the single Album write below --
        # a deep regrade never needs more than one write.
        merged_results = _merge_preserved_fields(
            build_result_entries(frames), existing_entries, sensitivity_threshold
        )
        Album(output_dir).write_results(
            merged_results,
            our_jersey_color=our_jersey_color, team_id=team_id,
            import_status="complete",
            run_settings=run_settings_out,
        )
        log.info("Results written to album:  %s", output_dir)

        # Anno_* lists describe THIS full re-analysis, so they're rebuilt
        # from scratch; SrcDir/SrcDirs/Timestamp are carried over by
        # AlbumInfo itself (it only seeds them when absent).
        write_info_json(
            frames, Path((existing_info.src_dir if existing_info else "") or output_dir),
            (existing_info.timestamp if existing_info else "") or ts,
            output_dir, our_jersey_color=our_jersey_color, rebuild=True,
        )
        write_csv(frames, output_dir / "blurry.csv")
        write_blur_lst(frames, output_dir / "blur.lst")

        blurry_n = sum(1 for f in frames if f.bodies and not f.is_sharp())
        sharp_n = sum(1 for f in frames if f.bodies and f.is_sharp())
        skipped_n = sum(1 for f in frames if not f.bodies)
        image_cache.clear()
        log.info("Deep regrade complete — Sharp: %d  |  Blurry: %d  |  No person: %d",
                 sharp_n, blurry_n, skipped_n)
        return

    if merge_mode and not frames:
        log.info("No new images found to import — album is already up to date.")

    working_album.write_results(
        build_result_entries(frames),
        our_jersey_color=our_jersey_color, team_id=team_id,
        import_status="in_progress" if frames else "complete",
        run_settings=run_settings_out,
    )

    if frames:
        write_csv(frames, output_dir / "blurry.csv", append=merge_mode)
        write_info_json(frames, input_path, datetime.now().strftime("%Y%m%d-%H%M%S"),
                        output_dir, our_jersey_color=our_jersey_color)
    if any(f.bodies and not f.is_sharp() for f in frames):
        write_blur_lst(frames, output_dir / "blur.lst", append=merge_mode)
    elif not merge_mode:
        log.info("No blurry images detected — blur.lst not written.")

    if frames:
        log.info("")
        log.info("Annotated previews saved to:")
        log.info("  %s", output_dir / "previews")
        log.info("")
        log.info("Review the previews, then delete images you want to override:")
        log.info("  previews/  delete a blurry preview → keep that original (not blurry after all)")
        log.info("             delete a sharp preview → exclude that original (move to Unselected/)")
        log.info("")

    if frames and not args.skip_facereco:
        if face_db_dir is None:
            log.info("Face recognition skipped: no face DB found.")
        else:
            FaceRecoStage(
                working_dir,
                face_db_dir=face_db_dir,
                face_db_allowed_names=roster_names,
                face_db_match_threshold=args.face_db_match_threshold,
                face_db_match_margin=args.face_db_match_margin,
                face_db_prototype_threshold=args.face_db_prototype_threshold,
                use_face_db_calibration=not args.disable_face_db_calibration,
                min_face_crop_px=args.min_face_crop_px,
                debug_align=args.debug_align,
                cpu_only=args.cpu_only,
                engine=args.engine,
                sensitivity_threshold=sensitivity_threshold,
            ).process(frames, app_config)

    openai_api_key = args.openaikey or os.environ.get("OPENAI_API_KEY")
    if frames and not args.skip_llm_cull and openai_api_key:
        try:
            from algo.llm.culling_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=openai_api_key, model=args.llm_model)
            LLMCullingStage(
                working_dir, provider=provider, threshold=sensitivity_threshold,
            ).process(frames, app_config)
        except Exception as exc:
            log.error("LLM-assisted burst culling failed: %s", exc, exc_info=True)
    elif frames and not openai_api_key:
        log.info("LLM-assisted burst culling skipped: no OpenAI API key (--openaikey or OPENAI_API_KEY).")

    if frames:
        # Import completed successfully (analysis + FaceReco + LLM culling,
        # whichever ran) -- flip the provisional "in_progress" flag written
        # earlier so consumers (culling_app.py's album list, etc.) can tell
        # a finished import apart from one interrupted mid-run.
        working_album.mark_import_complete()
        if use_temp_staging:
            summary = merge_album.import_from(working_album)
            merge_album.update_settings(our_jersey_color=our_jersey_color, run_settings=run_settings_out)
            merge_album.mark_import_complete()
            log.info(
                "Import complete — %d photo(s) merged into %s (%d renamed on key "
                "collision, %d FaceReco cluster(s) added, %d merged)",
                summary.added, output_dir, summary.renamed,
                summary.facereco_clusters_added, summary.facereco_clusters_merged,
            )

    image_cache.clear()

    if use_temp_staging:
        shutil.rmtree(working_dir, ignore_errors=True)

    if frames:
        log.info("When done, run:  python culling_app.py")


if __name__ == "__main__":
    main()
