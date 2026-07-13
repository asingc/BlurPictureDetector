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

import cv2

from algo.models import AutoAdjustment
from algo.stages.image_analysis import _read_image
from algo.utils import apply_auto_adjustment

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


def _load_auto_adjustments(ref_dir: Path) -> dict[str, AutoAdjustment]:
    """Read results.json and return {source filename: AutoAdjustment prescription}."""
    results_path = ref_dir / "results.json"
    if not results_path.exists():
        log.debug("results.json not found — auto adjustment disabled: %s", results_path)
        return {}
    with open(results_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    adjustments: dict[str, AutoAdjustment] = {}
    for entry in payload.get("results", []):
        adj = entry.get("auto_adjustment")
        if not adj:
            continue
        adjustments[Path(entry["file"]).name] = AutoAdjustment(ev=float(adj.get("ev", 0.0)))
    return adjustments


def _write_auto_adjusted(
    src_file: Path,
    adjustments: dict[str, AutoAdjustment],
    dest_dir: Path,
) -> bool:
    """Apply the stored auto-adjustment prescription to *src_file* and save the
    corrected result as a JPEG under *dest_dir*.  Returns True on success."""
    adjustment = adjustments.get(src_file.name)
    if adjustment is None or adjustment.is_noop:
        log.debug("  No auto-adjustment recorded for: %s", src_file.name)
        return False
    image = _read_image(src_file)
    if image is None:
        log.warning("  Cannot read for auto adjustment: %s", src_file.name)
        return False
    corrected = apply_auto_adjustment(image, adjustment)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (src_file.stem + ".jpg")
    cv2.imwrite(str(dest_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info("  Auto-adjusted (EV%+.1f) → %s", adjustment.ev, dest_path)
    return True



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

    blur_dest         = src_dir / "Blur"
    skipped_dest      = src_dir / "Skipped"
    auto_adjust_dest  = src_dir / "auto adjusted"
    anno_blur_dir     = ref_dir / "anno_blur"
    anno_sharp_dir    = ref_dir / "anno_sharp"
    anno_skipped_dir  = ref_dir / "anno_skipped"

    auto_adjustments = _load_auto_adjustments(ref_dir)
    log.info("Auto adjustment prescriptions loaded: %d", len(auto_adjustments))

    moved_blur = moved_skipped = left_in_place = skipped_missing = auto_adjusted = 0

    # ------------------------------------------------------------------
    # Anno_Blur — preview kept → Blur/;  preview deleted → leave in place
    # ------------------------------------------------------------------
    blur_list = info.get("Anno_Blur", [])
    log.info("--- Anno_Blur (%d file(s)) ---", len(blur_list))
    for entry in blur_list:
        src_name, anno_name = entry["src"], entry["anno"]
        src_file = src_dir / src_name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", src_name)
            skipped_missing += 1
            continue
        if (anno_blur_dir / anno_name).exists():
            if _move(src_file, blur_dest):
                moved_blur += 1
        else:
            log.info("  Preview deleted — leaving in place: %s", src_name)
            left_in_place += 1
            if _write_auto_adjusted(src_file, auto_adjustments, auto_adjust_dest):
                auto_adjusted += 1

    # ------------------------------------------------------------------
    # Anno_Sharp — preview kept → leave in place;  preview deleted → Blur/
    # ------------------------------------------------------------------
    sharp_list = info.get("Anno_Sharp", [])
    log.info("--- Anno_Sharp (%d file(s)) ---", len(sharp_list))
    for entry in sharp_list:
        src_name, anno_name = entry["src"], entry["anno"]
        src_file = src_dir / src_name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", src_name)
            skipped_missing += 1
            continue
        if (anno_sharp_dir / anno_name).exists():
            log.debug("  Preview present — leaving in place: %s", src_name)
            left_in_place += 1
            if _write_auto_adjusted(src_file, auto_adjustments, auto_adjust_dest):
                auto_adjusted += 1
        else:
            log.info("   Preview deleted — moving to Blur: %s", src_name)
            if _move(src_file, blur_dest):
                moved_blur += 1

    # ------------------------------------------------------------------
    # Anno_Skipped — move to <SrcDir>/Skipped only when confirmed
    # ------------------------------------------------------------------
    skipped_list = info.get("Anno_Skipped", [])
    log.info("--- Anno_Skipped (%d file(s)) ---", len(skipped_list))
    for entry in skipped_list:
        src_name, anno_name = entry["src"], entry["anno"]
        src_file = src_dir / src_name
        if not src_file.exists():
            log.debug("  Not found in SrcDir, skipping: %s", src_name)
            skipped_missing += 1
            continue
        if (anno_skipped_dir / anno_name).exists():
            if _move(src_file, skipped_dest):
                moved_skipped += 1
        else:
            log.debug("  Not confirmed in anno_skipped — leaving in place: %s", src_name)
            left_in_place += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info(
        "Done — moved to Blur: %d  |  moved to Skipped: %d  "
        "|  left in place: %d  |  auto-adjusted: %d  |  source not found: %d",
        moved_blur, moved_skipped, left_in_place, auto_adjusted, skipped_missing,
    )


if __name__ == "__main__":
    main()
