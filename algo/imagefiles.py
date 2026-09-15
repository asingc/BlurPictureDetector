"""Which files on disk count as photos, and how to find them.

The one definition of the supported-extension set and of directory
enumeration. Everything that turns a source directory into a list of photos
goes through `collect_images`/`images_to_import`, so an `Album` always owns
the complete set of images it was created from and nothing enumerates
behind its back.

Kept out of algo/album.py deliberately: album.py's job is one JSON file,
and image-format knowledge (extension allow-list, RAW handling, sort order)
doesn't belong there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

# RAW formats decoded via rawpy rather than OpenCV.
RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
) | RAW_EXTENSIONS


def is_image(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def collect_images(input_path: Union[str, Path]) -> list[Path]:
    """Every supported image at *input_path*, sorted by filename.

    Accepts a single file as well as a directory so callers can point at
    one photo without special-casing.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if is_image(input_path) else []
    if input_path.is_dir():
        return sorted(f for f in input_path.iterdir() if f.is_file() and is_image(f))
    return []


def images_to_import(
    source_dir: Union[str, Path], exclude: Iterable[Union[str, Path]] = ()
) -> list[Path]:
    """The photos in *source_dir* that aren't already in the album.

    *exclude* is the album's complete current set of source paths, compared
    resolved so the same file reached by a different route (relative path,
    symlink, drive-letter case) is still recognised as already imported.
    """
    excluded = {Path(p).resolve() for p in exclude}
    return [p for p in collect_images(source_dir) if p.resolve() not in excluded]
