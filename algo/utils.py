from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from algo.models import Box, Face


# COCO 17-keypoint head indices: nose, left-eye, right-eye, left-ear, right-ear.
_HEAD_KP_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)


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


def _matches_allowed_jersey_color(predicted: str, allowed_colors: frozenset[str]) -> bool:
    """Match either the full label (Hue, Shade) or just the hue."""
    predicted_norm = (predicted or "").strip().lower()
    if not predicted_norm:
        return False
    predicted_hue = predicted_norm.split(",", 1)[0].strip()
    for allowed in allowed_colors:
        allowed_norm = allowed.strip().lower()
        if not allowed_norm:
            continue
        if allowed_norm == predicted_norm or allowed_norm == predicted_hue:
            return True
    return False
