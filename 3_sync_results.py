#!/usr/bin/env python3
"""
3_sync_results.py
Sync blur/skipped classification decisions from a "target" directory to all
sibling directories that share the same parent.

Usage:
    python 3_sync_results.py <target_dir>

target_dir : a directory that already has a "blur" and/or "skipped" sub-folder
             populated by 2_apply_changes.py (or manually).

How it works:
    1. Collect the file *stems* (name without extension) from
       <target_dir>/blur/  and  <target_dir>/skipped/.
    2. Scan every other subdirectory that shares the same parent as target_dir
       (i.e. siblings — the target itself is excluded).
    3. For each image file found directly inside a sibling directory whose stem
       matches a stem in the "blur" or "skipped" sets:
         - Create <sibling>/blur/  or  <sibling>/skipped/  as needed.
         - Move the matching image file there.
    4. Write a timestamped log file (sync_results.log) inside target_dir.

No files are ever deleted.  Already-classified files (already inside a blur/
skipped subfolder of a sibling) are not touched.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
        ".cr3", ".cr2",
    }
)

CATEGORY_DIRS: tuple[str, str] = ("blur", "skipped")

log = logging.getLogger("3_sync_results")

_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    log.addHandler(ch)


def _add_file_logging(log_path: Path) -> None:
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT))
    log.addHandler(fh)
    log.debug("Log file: %s", log_path.resolve())


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _collect_stems(category_dir: Path) -> set[str]:
    """Return the lowercased stems of every image file in *category_dir*."""
    if not category_dir.is_dir():
        log.debug("  Category dir not found, skipping: %s", category_dir)
        return set()
    stems: set[str] = set()
    for f in category_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(f.stem.lower())
    log.debug(
        "  Collected %d stem(s) from '%s'",
        len(stems),
        category_dir.name,
    )
    return stems


def _move(src: Path, dest_dir: Path) -> bool:
    """Move *src* into *dest_dir*.  Returns True on success."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        log.warning("  Destination already exists, skipping: %s", dest)
        return False
    shutil.move(str(src), str(dest))
    log.info("  Moved %-50s  →  %s/%s", src.name, dest_dir.parent.name, dest_dir.name)
    return True


def _image_files_in(directory: Path) -> list[Path]:
    """Return direct-child image files in *directory* (non-recursive)."""
    return [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def sync(target_dir: Path) -> None:
    log.info("=" * 70)
    log.info("Target directory : %s", target_dir)

    # -----------------------------------------------------------------------
    # Step 1 — Build stem sets from target's blur/ and skipped/ sub-folders
    # -----------------------------------------------------------------------
    blur_stems: dict[str, set[str]] = {}
    total_reference_stems = 0
    for category in CATEGORY_DIRS:
        stems = _collect_stems(target_dir / category)
        blur_stems[category] = stems
        total_reference_stems += len(stems)
        log.info("  [target] %-10s : %d file stem(s)", category, len(stems))

    if total_reference_stems == 0:
        log.warning(
            "No classified files found in blur/ or skipped/ under target. "
            "Nothing to sync."
        )
        return

    # -----------------------------------------------------------------------
    # Step 2 — Enumerate sibling directories
    # -----------------------------------------------------------------------
    parent_dir = target_dir.parent
    siblings: list[Path] = [
        d for d in parent_dir.iterdir()
        if d.is_dir() and d.resolve() != target_dir.resolve()
    ]
    log.info("Parent directory : %s", parent_dir)
    log.info("Sibling dirs found: %d", len(siblings))
    if not siblings:
        log.warning("No sibling directories found. Nothing to sync.")
        return

    # -----------------------------------------------------------------------
    # Step 3 — Process each sibling
    # -----------------------------------------------------------------------
    grand_total_moved: dict[str, int] = {c: 0 for c in CATEGORY_DIRS}
    grand_total_scanned = 0
    grand_total_no_match = 0

    for sibling in sorted(siblings):
        log.info("-" * 70)
        log.info("Sibling: %s", sibling.name)

        images = _image_files_in(sibling)
        log.info("  Image files found: %d", len(images))

        if not images:
            log.info("  No image files — nothing to do.")
            continue

        grand_total_scanned += len(images)
        sibling_moved: dict[str, int] = {c: 0 for c in CATEGORY_DIRS}
        no_match = 0

        for img in images:
            stem_lower = img.stem.lower()
            matched = False
            for category in CATEGORY_DIRS:
                if stem_lower in blur_stems[category]:
                    dest_dir = sibling / category
                    log.debug(
                        "  Match [%s] stem='%s' file='%s'",
                        category, stem_lower, img.name,
                    )
                    if _move(img, dest_dir):
                        sibling_moved[category] += 1
                        grand_total_moved[category] += 1
                    matched = True
                    break  # a stem can only belong to one category
            if not matched:
                no_match += 1
                log.debug("  No match for: %s", img.name)

        grand_total_no_match += no_match
        log.info(
            "  Summary — moved to blur: %d | moved to skipped: %d | unmatched: %d",
            sibling_moved["blur"],
            sibling_moved["skipped"],
            no_match,
        )

    # -----------------------------------------------------------------------
    # Step 4 — Grand summary
    # -----------------------------------------------------------------------
    log.info("=" * 70)
    log.info("SYNC COMPLETE")
    log.info("  Siblings processed : %d", len(siblings))
    log.info("  Images scanned     : %d", grand_total_scanned)
    log.info("  Moved → blur       : %d", grand_total_moved["blur"])
    log.info("  Moved → skipped    : %d", grand_total_moved["skipped"])
    log.info("  Unmatched (kept)   : %d", grand_total_no_match)
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync blur/skipped classification from a target directory "
            "to all sibling directories."
        )
    )
    parser.add_argument(
        "target_dir",
        help=(
            "Path to the directory whose blur/ and skipped/ sub-folders "
            "define which images should be moved in sibling directories."
        ),
    )
    args = parser.parse_args()

    _setup_logging()

    target_dir = Path(args.target_dir).resolve()
    log.info("3_sync_results.py  —  started %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not target_dir.is_dir():
        log.error("Target directory does not exist: %s", target_dir)
        sys.exit(1)

    if target_dir.parent == target_dir:
        log.error("Target directory has no parent (is it a filesystem root?)")
        sys.exit(1)

    # Attach file log inside the target directory
    log_path = target_dir / "sync_results.log"
    _add_file_logging(log_path)

    sync(target_dir)


if __name__ == "__main__":
    main()
