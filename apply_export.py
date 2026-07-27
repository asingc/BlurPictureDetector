#!/usr/bin/env python3
"""apply_export.py — Export a culled album to its final destination.

Copies every "kept" photo from a 1_prep_review.py / culling_app.py output
directory (an album under albums/) to a destination folder, applying edits
(placeholder for now — real pixel editing lands later) while preserving all
metadata, writes a players.csv describing which tagged player appears in
which exported photo, and embeds each photo's culling "stars" rating (see
algo/stages/llm_culling.py) into the exported copy's EXIF Rating/
RatingPercent tags (recognised by Windows Explorer, Adobe Bridge/Lightroom).

The destination folder ends up FLAT: the edited kept images plus
players.csv, no subdirectories. The album's own albums/<album>/ folder (and
its .FaceReco) remains the persistent "working album" — this script never
touches it beyond reading it.

Usage:
    python apply_export.py <album_dir> <destination_dir> [--export-face-tagging]

<album_dir>       : output directory produced by 1_prep_review.py (contains
                     info.json + album.json).
<destination_dir> : where the final, edited photos + players.csv go.
--export-face-tagging : also populate players.csv rows (Player Name, Player
                     Number, Image Path) for every tagged player found in a
                     kept photo. Omit to still create players.csv with just
                     the header (no rows) — e.g. when sharing photos without
                     revealing who's who.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

try:
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:  # optional — star-rating metadata is skipped without it
    piexif = None
    _PIEXIF_AVAILABLE = False

log = logging.getLogger("apply_export")

_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"

# Mirrors culling_app.py's REVIEW_CATEGORIES / _kept_image_basenames: which
# info.json list backs each review category, and the effective "keep" value
# when album.json has no explicit override yet.
_REVIEW_INFO_KEY = {"blur": "Anno_Blur", "sharp": "Anno_Sharp", "skipped": "Anno_Skipped"}
_REVIEW_DEFAULT_KEEP = {"blur": False, "sharp": True, "skipped": False}

_PLAYERS_CSV_HEADER = ["Player Name", "Player Number", "Image Path"]

# File types piexif can rewrite the EXIF segment of in place (no re-encode,
# no pixel/quality loss). Other kept formats (PNG, WEBP, RAW) are silently
# left untagged — no safe non-destructive path for them here.
_RATING_TAG_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}
# Standard Windows Explorer / Adobe Bridge-Lightroom star -> (Rating,
# RatingPercent) EXIF tag values (System.Rating / System.Rating.Percent).
_STAR_RATING_PERCENT = {1: 1, 2: 25, 3: 50, 4: 75, 5: 99}

# Mirrors culling_app.py's FACE_DB_DIR_CANDIDATES / _facereco_dir /
# _is_pending_cluster / _load_face_json.
_FACE_DB_DIR_CANDIDATES: tuple[str, ...] = (".FaceReco", ".facereco", ".Facereco")


def _facereco_dir(album_dir: Path) -> Path | None:
    for name in _FACE_DB_DIR_CANDIDATES:
        candidate = album_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _is_pending_cluster(folder_name: str) -> bool:
    return folder_name.isdigit()


def _load_face_json(cluster_dir: Path) -> dict:
    fp = cluster_dir / "face.json"
    if not fp.is_file():
        return {"name": "", "playernum": None, "faces": []}
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def _setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    log.addHandler(handler)


def _kept_image_basenames(info: dict, results_by_name: dict[str, dict]) -> set[str]:
    """Basenames of every image the user decided to keep — the explicit
    "keep" flag written by the Review page's Apply step if present, else the
    per-category default. Matches culling_app.py's _kept_image_basenames()."""
    kept: set[str] = set()
    for category, info_key in _REVIEW_INFO_KEY.items():
        default_keep = _REVIEW_DEFAULT_KEEP[category]
        for item in info.get(info_key, []):
            src_name = item.get("src")
            if not src_name:
                continue
            result = results_by_name.get(src_name)
            keep = bool(result.get("keep", default_keep)) if result else default_keep
            if keep:
                kept.add(src_name)
    return kept


def _apply_edits(src_file: Path, dest_file: Path) -> None:
    """Apply the album's editing prescription to *src_file* and write the
    result to *dest_file*.

    Placeholder for now: no pixel-level editing pipeline exists yet, so this
    is a byte-for-byte copy — which trivially preserves every bit of
    metadata (EXIF, ICC profile, GPS, timestamps, ...) since nothing gets
    re-encoded. When real editing lands, this is the single place to hook it
    in; it will then need to explicitly carry metadata forward across the
    re-encode instead of getting it for free.
    """
    shutil.copy2(str(src_file), str(dest_file))


def _export_photos(info: dict, kept: set[str], dest_dir: Path) -> int:
    src_dir = Path(info.get("SrcDir", ""))
    log.info("Copying %d kept photo(s) to %s", len(kept), dest_dir)

    copied = 0
    for name in sorted(kept):
        src_file = src_dir / name
        if not src_file.is_file():
            log.warning("  Missing, skipped: %s", name)
            continue
        try:
            _apply_edits(src_file, dest_dir / name)
            copied += 1
            log.info("  Copied: %s", name)
        except OSError as exc:
            log.warning("  Failed to copy %s: %s", name, exc)

    log.info("Photos copied: %d/%d", copied, len(kept))
    return copied


def _write_star_ratings(dest_dir: Path, kept: set[str], results_by_name: dict[str, dict]) -> int:
    """Embed each kept photo's culling "stars" rating (assigned by
    algo/stages/llm_culling.py's ``_assign_star_ratings``) into the exported
    copy's EXIF metadata, as the standard Windows/Adobe "Rating" (0-5) and
    "RatingPercent" tags recognised by Explorer, Lightroom and Bridge.

    Uses piexif to rewrite only the EXIF APP1 segment in place — the
    compressed image data itself is never touched (no re-encode, no quality
    loss). Only .jpg/.jpeg/.tif/.tiff are supported (piexif's format
    coverage); other kept formats (PNG, WEBP, RAW) are silently left
    untagged. Entries with no "stars" field (e.g. blur/skipped photos, or
    albums processed before this feature existed) are left untouched too.
    """
    if not _PIEXIF_AVAILABLE:
        log.warning("piexif not installed — skipping star-rating metadata (pip install piexif)")
        return 0

    tagged = 0
    for name in sorted(kept):
        result = results_by_name.get(name)
        stars = result.get("stars") if result else None
        if not stars:
            continue
        dest_file = dest_dir / name
        if dest_file.suffix.lower() not in _RATING_TAG_EXTS or not dest_file.is_file():
            continue
        try:
            exif_dict = piexif.load(str(dest_file))
            exif_dict["0th"][piexif.ImageIFD.Rating] = int(stars)
            exif_dict["0th"][piexif.ImageIFD.RatingPercent] = _STAR_RATING_PERCENT.get(int(stars), 0)
            piexif.insert(piexif.dump(exif_dict), str(dest_file))
            tagged += 1
        except Exception as exc:  # noqa: BLE001 — one file's tagging failure must not abort the export
            log.warning("  Failed to write star rating for %s: %s", name, exc)

    log.info("Star ratings written: %d/%d kept photo(s)", tagged, len(kept))
    return tagged


def _collect_player_rows(album_dir: Path, kept: set[str]) -> list[list[str]]:
    """One row per tagged face per kept image, sourced from the album's own
    .FaceReco database.

    Each named (non-pending) cluster's face.json lists every face crop
    belonging to that person together with the original image it came
    from -- this is the authoritative source of "who's in which photo".
    album.json's player_name/player_number fields are NOT used here: they
    are only ever populated for faces explicitly touched by a Face
    Clustering "commit" action, so faces the recognition pipeline
    auto-matched straight to a named person would otherwise be silently
    missing from players.csv (this previously caused players.csv to contain
    far fewer rows than the number of tagged faces actually in the album)."""
    rows: list[list[str]] = []
    fr = _facereco_dir(album_dir)
    if fr is None:
        return rows
    for cluster_dir in sorted(fr.iterdir()):
        if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
            continue
        if _is_pending_cluster(cluster_dir.name):
            continue  # unnamed cluster -- nobody to tag
        payload = _load_face_json(cluster_dir)
        name = (payload.get("name") or "").strip()
        if not name:
            continue
        playernum = payload.get("playernum")
        number_str = "" if playernum is None else str(playernum)
        for face in payload.get("faces", []):
            orig = face.get("origFilename", "")
            if orig in kept:
                rows.append([name, number_str, orig])
    rows.sort(key=lambda row: (row[2], row[0]))
    return rows


def _write_players_csv(dest_dir: Path, rows: list[list[str]]) -> Path:
    csv_path = dest_dir / "players.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_PLAYERS_CSV_HEADER)
        writer.writerows(rows)
    return csv_path


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Export a culled album's kept photos + players.csv to a destination folder."
    )
    parser.add_argument("album_dir", help="Album output directory (contains info.json + album.json).")
    parser.add_argument("destination_dir", help="Destination folder (created if missing).")
    parser.add_argument(
        "--export-face-tagging",
        action="store_true",
        help="Populate players.csv rows. Without this flag, players.csv is still created but header-only.",
    )
    args = parser.parse_args()

    album_dir = Path(args.album_dir).resolve()
    dest_dir = Path(args.destination_dir).resolve()

    info_path = album_dir / "info.json"
    results_path = album_dir / "album.json"
    if not info_path.is_file() or not results_path.is_file():
        log.error("Not a valid album directory (missing info.json/album.json): %s", album_dir)
        sys.exit(1)

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)
    with open(results_path, encoding="utf-8") as fh:
        results = json.load(fh)

    results_by_name = {Path(r.get("file", "")).name: r for r in results.get("results", [])}
    kept = _kept_image_basenames(info, results_by_name)

    dest_dir.mkdir(parents=True, exist_ok=True)

    _export_photos(info, kept, dest_dir)
    _write_star_ratings(dest_dir, kept, results_by_name)

    rows = _collect_player_rows(album_dir, kept) if args.export_face_tagging else []
    csv_path = _write_players_csv(dest_dir, rows)
    log.info("players.csv written: %s (%d row(s))", csv_path, len(rows))


if __name__ == "__main__":
    main()
