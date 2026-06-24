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


def _matches_allowed_jersey_color(cloth_color: str, allowed_colors: frozenset[str]) -> bool:
    """Match cloth_color ("Hue:Shade") against the allow-list.

    Shade match is preferred; hue match is accepted as fallback.
    Allow-list entries may be plain hue names, shade names, or "Hue:Shade" labels.
    """
    if not cloth_color:
        return False
    c_hue, _, c_shade = cloth_color.partition(":")
    c_hue_l   = c_hue.strip().lower()
    c_shade_l = c_shade.strip().lower()
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


def _colors_match(cloth_color: str, reference: str) -> bool:
    """Return True if *cloth_color* matches *reference* ("Hue:Shade" format).

    Shade equality is preferred; hue equality is accepted as fallback.
    """
    if not cloth_color or not reference:
        return False
    c_hue, _, c_shade = cloth_color.partition(":")
    r_hue, _, r_shade = reference.partition(":")
    if c_shade.strip().lower() == r_shade.strip().lower():
        return True
    return c_hue.strip().lower() == r_hue.strip().lower()
