#!/usr/bin/env python3
"""
2_apply_changes.py
Apply the classification results from a 1_prep_review.py output directory to the
original source directory based on which annotated preview images you kept.

Usage:
    python 2_apply_changes.py <ref_dir>

ref_dir : output directory produced by 1_prep_review.py (must contain info.json).

Review workflow
---------------
After running 1_prep_review.py, open the three anno_* folders and delete any
preview images you want to override:

  anno_blur/    — delete a preview → KEEP the original (not blurry after all)
  anno_sharp/   — delete a preview → EXCLUDE the original (don't want this photo)
  anno_skipped/ — delete a preview → leave the original untouched

Then run this script.  It compares what previews remain against info.json:

  anno_blur    preview present  → move original to <SrcDir>/Blur/
  anno_blur    preview deleted  → leave original in place
  anno_sharp   preview present  → leave original in place (confirmed keeper)
  anno_sharp   preview deleted  → move original to <SrcDir>/Blur/
  anno_skipped preview present  → move original to <SrcDir>/Skipped/
  anno_skipped preview deleted  → leave original in place

No files are ever deleted.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

log = logging.getLogger("2_apply_changes")


_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"


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


def _move(src: Path, dest_dir: Path) -> bool:
    """Move *src* into *dest_dir*.  Returns True on success."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        log.warning("Destination already exists, skipping: %s", dest)
        return False
    shutil.move(str(src), str(dest))
    log.info("  Moved %-40s → %s", src.name, dest_dir)
    return True


def _anno_exists(anno_dir: Path, original_name: str) -> bool:
    """Check whether the annotated copy (saved as <stem>.jpg) exists in *anno_dir*."""
    anno_name = Path(original_name).stem + ".jpg"
    return (anno_dir / anno_name).exists()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move source images into Blur/ and Skipped/ based on "
            "1_prep_review.py results stored in info.json."
        ),
    )
    parser.add_argument(
        "ref_dir",
        help="Path to the 1_prep_review.py output directory (must contain info.json).",
    )
    args = parser.parse_args()

    _setup_logging()

    ref_dir = Path(args.ref_dir).resolve()
    if not ref_dir.is_dir():
        log.error("Reference directory does not exist: %s", ref_dir)
        sys.exit(1)

    json_path = ref_dir / "info.json"
    if not json_path.exists():
        log.error("info.json not found in: %s", ref_dir)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as fh:
        info = json.load(fh)

    src_dir  = Path(info["SrcDir"]).resolve()
    src_type = info.get("SrcType", "")

    if not src_dir.exists():
        log.error("SrcDir does not exist: %s", src_dir)
        sys.exit(1)
    if not src_dir.is_dir():
        log.error("SrcDir is not a directory: %s", src_dir)
        sys.exit(1)
    if src_type != "Directory":
        log.error("SrcType is '%s' — only 'Directory' is supported.", src_type)
        sys.exit(1)

    _add_file_logging(ref_dir / "apply.log")

    log.info("Source      : %s", src_dir)
    log.info("Reference   : %s", ref_dir)
    log.info("Timestamp   : %s", info.get("Timestamp", ""))

    blur_dest        = src_dir / "Blur"
    skipped_dest     = src_dir / "Skipped"
    anno_blur_dir    = ref_dir / "anno_blur"
    anno_sharp_dir   = ref_dir / "anno_sharp"
    anno_skipped_dir = ref_dir / "anno_skipped"

    moved_blur = moved_skipped = left_in_place = skipped_missing = 0

    # ------------------------------------------------------------------
    # Anno_Blur — preview kept → Blur/;  preview deleted → leave in place
    # ------------------------------------------------------------------
    blur_list = info.get("Anno_Blur", [])
    log.info("--- Anno_Blur (%d file(s)) ---", len(blur_list))
    for name in blur_list:
        src_file = src_dir / name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", name)
            skipped_missing += 1
            continue
        if _anno_exists(anno_blur_dir, name):
            if _move(src_file, blur_dest):
                moved_blur += 1
        else:
            log.info("  Preview deleted — leaving in place: %s", name)
            left_in_place += 1

    # ------------------------------------------------------------------
    # Anno_Sharp — preview kept → leave in place;  preview deleted → Blur/
    # ------------------------------------------------------------------
    sharp_list = info.get("Anno_Sharp", [])
    log.info("--- Anno_Sharp (%d file(s)) ---", len(sharp_list))
    for name in sharp_list:
        src_file = src_dir / name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", name)
            skipped_missing += 1
            continue
        if _anno_exists(anno_sharp_dir, name):
            log.debug("  Preview present — leaving in place: %s", name)
            left_in_place += 1
        else:
            log.info("   Preview deleted — moving to Blur: %s", name)
            if _move(src_file, blur_dest):
                moved_blur += 1

    # ------------------------------------------------------------------
    # Anno_Skipped — move to <SrcDir>/Skipped only when confirmed
    # ------------------------------------------------------------------
    skipped_list = info.get("Anno_Skipped", [])
    log.info("--- Anno_Skipped (%d file(s)) ---", len(skipped_list))
    for name in skipped_list:
        src_file = src_dir / name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", name)
            skipped_missing += 1
            continue
        if _anno_exists(anno_skipped_dir, name):
            if _move(src_file, skipped_dest):
                moved_skipped += 1
        else:
            log.debug("  Not confirmed in anno_skipped — leaving in place: %s", name)
            left_in_place += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info(
        "Done — moved to Blur: %d  |  moved to Skipped: %d  "
        "|  left in place: %d  |  source not found: %d",
        moved_blur, moved_skipped, left_in_place, skipped_missing,
    )


if __name__ == "__main__":
    main()
