from __future__ import annotations

import gzip
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from algo.models import AutoAdjustment, Box, ColorLab, Face

try:
    from PIL import Image as _PILImage
    from PIL import ExifTags as _PILExifTags
except ImportError:  # Pillow not installed — EXIF timestamps just won't be available.
    _PILImage = None
    _PILExifTags = None


# COCO 17-keypoint head indices: nose, left-eye, right-eye, left-ear, right-ear.
_HEAD_KP_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)


def atomic_save_and_backup(content: str, path: Union[str, Path]) -> None:
    """Atomically overwrite `path` with `content`, gzip-backing-up whatever
    was there before.

    Sequence (each step below only starts once the previous one has fully
    succeeded, so a crash/exception at any point never corrupts or loses the
    existing file at `path`):

    1. Write `content` to a temp file beside `path` (flushed + fsync'd).
    2. If `path` already exists, atomically rename it aside (os.replace is an
       atomic rename on both Windows and POSIX) — the original bytes are now
       safely parked under a temp name, untouched.
    3. Atomically rename the new temp file into place at `path` — the new
       content is now live.
    4. If there was a previous file, gzip-compress the parked-aside original
       into ``<path.parent>/backup/<path.stem>_<yyyymmdd-hhmmss>.gz``, where
       the timestamp is the *old* file's creation time — then delete the
       parked-aside temp copy.
    """
    path = Path(path)
    tmp_new_path = path.with_name(path.name + ".new.tmp")
    tmp_old_path = path.with_name(path.name + ".old.tmp")

    with open(tmp_new_path, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())

    had_old = path.is_file()
    old_ctime = path.stat().st_ctime if had_old else None

    try:
        if had_old:
            os.replace(path, tmp_old_path)  # park original aside, untouched
        os.replace(tmp_new_path, path)  # swap new content into place
    except OSError:
        # Best-effort rollback so a mid-sequence failure doesn't strand the
        # original outside of `path`.
        if had_old and tmp_old_path.is_file() and not path.is_file():
            os.replace(tmp_old_path, path)
        raise

    if had_old:
        backup_dir = path.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.fromtimestamp(old_ctime).strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"{path.stem}_{timestamp}.gz"
        with open(tmp_old_path, "rb") as src, gzip.open(backup_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp_old_path.unlink()

# EXIF tag ids (main IFD "DateTime", and Exif sub-IFD "DateTimeOriginal" /
# "DateTimeDigitized") — checked in that order. Mirrors culling_app.py's
# review-page burst grouping so both places agree on "when was this taken".
_EXIF_TAG_DATETIME = 306
_EXIF_TAG_DATETIME_ORIGINAL = 36867
_EXIF_TAG_DATETIME_DIGITIZED = 36868


def image_capture_timestamp(path: Path) -> float:
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
                    from datetime import datetime
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass
    try:
        stat = path.stat()
        return min(stat.st_ctime, stat.st_mtime)
    except OSError:
        return 0.0


def apply_auto_adjustment(image: np.ndarray, adjustment: AutoAdjustment | None) -> np.ndarray:
    """Apply a simple EV (exposure/brightness) correction to a BGR *image*.

    ``adjustment.ev`` multiplies pixel values by ``2 ** ev`` (a stop-based
    exposure compensation). Returns *image* unchanged (same array, not a
    copy) when there is nothing to apply.
    """
    if adjustment is None or adjustment.is_noop:
        return image
    out = image.astype(np.float32)
    out *= 2.0 ** adjustment.ev
    return np.clip(out, 0, 255).astype(np.uint8)


def cap_long_edge(image: np.ndarray, max_long_edge: float) -> np.ndarray:
    """
    Downsize *image* so its long edge is at most *max_long_edge* pixels.
    Never upscales — returns the original array when it is already smaller.
    """
    h, w = image.shape[:2]
    scale = min(1.0, max_long_edge / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _narrow_face_box(
    face: Face,
    conf_threshold: float = 0.3,
    pad: int = 0,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> Box | None:
    """Return the minimal bounding box enclosing the confident face landmarks,
    expanded by *pad* pixels on every side and clamped to the image bounds.
    Returns None if fewer than 2 landmarks are detected."""
    xs = [lm.point.x for lm in face.landmarks if lm.confidence >= conf_threshold]
    ys = [lm.point.y for lm in face.landmarks if lm.confidence >= conf_threshold]
    if len(xs) < 2:
        return None
    if img_w is not None and img_h is not None:
        pad_x = pad / img_w
        pad_y = pad / img_h
        x1 = max(0.0, min(xs) - pad_x)
        y1 = max(0.0, min(ys) - pad_y)
        x2 = min(1.0, max(xs) + pad_x)
        y2 = min(1.0, max(ys) + pad_y)
    else:
        x1 = min(xs) - pad
        y1 = min(ys) - pad
        x2 = max(xs) + pad
        y2 = max(ys) + pad
    return Box(x1, y1, x2, y2)


def _color_from_label(label: str) -> "ColorLab | None":
    """Parse a 'Hue:Shade' label into a ColorLab, or return None for N/A / Unknown."""
    if not label or label in ("N/A", "Unknown"):
        return None
    hue, _, shade = label.partition(":")
    return ColorLab(hue.strip(), shade.strip())


def _matches_allowed_jersey_color(color: "ColorLab | None", allowed_colors: frozenset[str]) -> bool:
    """Match *color* against the allow-list.

    Shade match is preferred; hue match is accepted as fallback.
    Allow-list entries may be plain hue names, shade names, or "Hue:Shade" labels.
    """
    if color is None:
        return False
    c_hue_l   = color.hue.strip().lower()
    c_shade_l = color.shade.strip().lower()
    for allowed in allowed_colors:
        a = allowed.strip().lower()
        if not a:
            continue
        if ":" in a:
            a_hue, _, a_shade = a.partition(":")
            if a_shade == c_shade_l or a_hue == c_hue_l:
                return True
        else:
            if a == c_shade_l or a == c_hue_l:
                return True
    return False


def _colors_match(color: "ColorLab | None", reference: "ColorLab | None") -> bool:
    """Return True if *color* matches *reference*.

    Shade equality is preferred; hue equality is accepted as fallback.
    """
    if color is None or reference is None:
        return False
    if color.shade.strip().lower() == reference.shade.strip().lower():
        return True
    return color.hue.strip().lower() == reference.hue.strip().lower()
