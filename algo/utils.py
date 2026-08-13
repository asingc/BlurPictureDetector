from __future__ import annotations

import gzip
import json
import os
import shutil
import uuid
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


# ---------------------------------------------------------------------------
# Multi-source-directory import support (1_prep_review.py "import more images"
# into an existing album). Album entries are globally identified by their
# absolute source path, but a lot of on-disk bookkeeping (preview filenames,
# info.json's Anno_* lists, FaceReco crop provenance, export destination
# filenames) is keyed by plain filename for readability -- which is only
# safe as long as filenames are unique across every source directory ever
# imported into the album. These two helpers keep that bookkeeping key
# ("key" in album.json result entries) collision-free without ever renaming
# the original file on disk.
# ---------------------------------------------------------------------------

def make_unique_import_key(filename: str, used_keys: dict[str, Path], src_path: Path) -> str:
    """Return a bookkeeping key for *filename* that's unique within
    *used_keys* (a dict of already-claimed key -> absolute source path),
    mutating *used_keys* to claim the returned key.

    Disambiguates with a ``__2``, ``__3``, ... suffix inserted before the
    extension only when a DIFFERENT source file already claimed that exact
    filename (e.g. two different source directories both containing an
    ``IMG_0001.JPG``). The original file on disk is never touched -- this
    key only affects internal bookkeeping (preview filenames, album.json /
    info.json entries, face-DB origFilename, and export destination
    filenames).
    """
    src_path = Path(src_path)
    existing = used_keys.get(filename)
    if existing is None or existing == src_path:
        used_keys[filename] = src_path
        return filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 2
    while True:
        candidate = f"{stem}__{i}{suffix}"
        existing = used_keys.get(candidate)
        if existing is None or existing == src_path:
            used_keys[candidate] = src_path
            return candidate
        i += 1


def load_album_source_index(album_json_path: Union[str, Path]) -> dict[str, str]:
    """Map every result entry's bookkeeping ``key`` to its absolute source
    path, as recorded in album.json's ``results[].file``/``results[].key``.

    Falls back to the entry's plain basename for older albums written before
    the "key" field existed (single-source-directory albums, where filename
    collisions can't happen). Used by every consumer that needs to open the
    true original file behind a filename that may not be globally unique
    across multiple imported source directories.
    """
    album_json_path = Path(album_json_path)
    if not album_json_path.is_file():
        return {}
    try:
        with open(album_json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    index: dict[str, str] = {}
    for entry in payload.get("results", []):
        file_path = entry.get("file")
        if not file_path:
            continue
        key = entry.get("key") or Path(file_path).name
        index[key] = file_path
    return index


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


# Cached, small "cover crop" square thumbnails used by the review nav strip
# (culling_app.py) — generated eagerly here (algo/stages/annotation.py) from
# the in-memory annotated image, alongside the full-size preview.
THUMBNAILS_SUBDIR = "thumbnails"
THUMBNAIL_SIZE = 128


def write_cover_thumbnail(image: np.ndarray, out_path: Path, size: int = THUMBNAIL_SIZE, quality: int = 85) -> None:
    """Write a *size* x *size* "cover crop" JPEG thumbnail of *image* to
    *out_path* (CSS object-fit: cover — scale down so the shorter edge
    exactly fills the square, then center-crop the longer edge's excess).
    """
    h, w = image.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    cropped = resized[top:top + size, left:left + size]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per-call tmp name (concurrent writers can target the same
    # thumbnail) that still ends in out_path's suffix — cv2.imwrite() picks
    # its codec from the filename's extension, so a trailing ".tmp" (with
    # no image extension) makes it fail every time with "could not find a
    # writer for the specified extension".
    tmp_fp = out_path.with_name(f"{out_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{out_path.suffix}")
    if not cv2.imwrite(str(tmp_fp), cropped, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"cv2.imwrite failed to write thumbnail: {tmp_fp}")
    os.replace(tmp_fp, out_path)


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


def clamp_long_edge(image: np.ndarray, min_long_edge: float, max_long_edge: float) -> np.ndarray:
    """
    Resize *image* so its long edge falls within [min_long_edge, max_long_edge].

    Unlike cap_long_edge (which only ever shrinks), this also upscales images
    whose long edge is below min_long_edge — so passing the same value for
    both bounds normalizes every image to that exact long-edge size. Returns
    the original array unchanged when already within range.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_long_edge:
        target = max_long_edge
    elif long_edge < min_long_edge:
        target = min_long_edge
    else:
        return image
    scale = target / long_edge
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


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
