from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("BlurPictureDetector")

CACHE_DIR_NAME = ".imgcache"

# Must stay >= FaceRecoConfig.face_buffer_ratio (algo/facereco.py), otherwise a
# cached crop would be tighter than the one FaceReco crops for itself and the
# embeddings would no longer match RebuildFaceDB's.
FACE_ORIGINAL_BUFFER_RATIO = 0.15


class ImageCache:
    """Scratch, per-run store for the small pixel regions later stages need.

    Exists so the analysis loop can drop each decoded source image as soon as
    it has been processed, instead of holding every frame of the album in
    memory at once. Read back through ``Body``/``Face.get_*_crop()``; the
    whole directory is deleted when the run finishes.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / CACHE_DIR_NAME
        self._seq = 0

    def store(self, stem: str, arr: np.ndarray | None) -> str | None:
        """Persist *arr* and return its path, or None when there is nothing to
        store. The name only has to be unique -- consumers read the path back
        off the Body/Face that owns it rather than reconstructing it."""
        if arr is None or arr.size == 0:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        self._seq += 1
        path = self.root / f"{stem}_{self._seq:06d}.npy"
        try:
            # ascontiguousarray: crops are numpy views, and a view keeps its
            # whole parent frame alive until it is copied out.
            np.save(str(path), np.ascontiguousarray(arr), allow_pickle=False)
        except OSError as exc:
            log.warning("[ImageCache] could not write %s: %s", path.name, exc)
            return None
        return str(path)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
